"""
utils/metrics.py — Shared metric accumulation and computation for PIXAR evaluation.

Provides:
  - MetricsAccumulator  : accumulates raw counts from one evaluation pass
  - GroupAccumulator    : lightweight accumulator for per-model/per-op breakdown
  - merge_raw_counts    : merges counts from multiple workers/chunks (element-wise sum)
  - compute_metrics     : computes final metrics dict from merged raw counts
  - compute_group_metrics: computes per-model/per-op recall + IoU metrics
  - print_metrics_report: prints the standard evaluation report to stdout
  - print_group_report  : prints per-model or per-op breakdown table
"""

from __future__ import annotations

import numpy as np
import torch

# Number of bins for pixel-level ROC-AUC histogram
ROC_BINS = 512


# ---------------------------------------------------------------------------
# Raw-count accumulator
# ---------------------------------------------------------------------------

class MetricsAccumulator:
    """
    Accumulates raw intermediate counts during one evaluation pass.

    All state is kept as plain Python scalars, numpy arrays, or CPU torch
    tensors so it can be serialised to JSON without GPU involvement.

    Usage (per sample):
        acc = MetricsAccumulator()
        ...
        acc.update_cls(predicted_class, gt_class)
        if gt_class == 2:  # tampered
            acc.update_seg(pred_mask_bin, gt_mask, pred_scores)
            acc.update_obj(pred_obj, gt_obj)
        ...
        raw = acc.to_dict()   # JSON-serialisable
    """

    def __init__(self, num_cls_classes: int = 3, num_obj_classes: int = 81):
        self.num_cls_classes = num_cls_classes

        # Classification
        self.confusion_matrix = torch.zeros(num_cls_classes, num_cls_classes, device="cpu")
        self.correct = 0
        self.total = 0

        # Segmentation (tampered only)
        self.intersection_sum = np.zeros(2, dtype=np.float64)
        self.union_sum        = np.zeros(2, dtype=np.float64)
        self.acc_iou_sum      = np.zeros(2, dtype=np.float64)
        self.acc_iou_count    = 0

        # Pixel TP/FP/FN (tampered only)
        self.pix_TP = 0
        self.pix_FP = 0
        self.pix_FN = 0

        # ROC-AUC histograms (tampered only)
        self.pos_hist = torch.zeros(ROC_BINS, dtype=torch.float64)
        self.neg_hist = torch.zeros(ROC_BINS, dtype=torch.float64)

        # Object multi-label (tampered only)
        self.obj_tp_total = 0.0
        self.obj_fp_total = 0.0
        self.obj_fn_total = 0.0
        self.obj_exact_match_total = 0
        self.obj_rows_total = 0
        self.obj_hit1_total = 0
        self.obj_hit5_total = 0
        self.obj_hit_den_total = 0
        self._obj_tp_per_class: torch.Tensor | None = None
        self._obj_fp_per_class: torch.Tensor | None = None
        self._obj_fn_per_class: torch.Tensor | None = None

    # ------------------------------------------------------------------
    # Per-sample update methods
    # ------------------------------------------------------------------

    def update_cls(self, predicted_class: int, gt_class: int) -> None:
        """Update classification counts for one sample."""
        self.correct += int(predicted_class == gt_class)
        self.total += 1
        self.confusion_matrix[int(gt_class), int(predicted_class)] += 1

    def update_seg(
        self,
        pred_mask_bin: torch.Tensor,            # shape [B, H, W] or [H, W], int32; threshold >0 (for IoU)
        gt_mask: torch.Tensor,                  # shape [B, H, W] or [H, W], int; ground-truth
        pred_scores: torch.Tensor | None = None,# shape [B, H, W] or [H, W], float32 in [0,1] (for pixel metrics)
    ) -> None:
        """
        Update segmentation accumulators for one tampered sample or batch.

        pred_mask_bin  : binary prediction at raw-logit threshold (>0); used for IoU.
        gt_mask        : ground-truth soft mask (int) — may be batched [B, H, W].
        pred_scores    : sigmoid-normalised scores in [0,1]; used for pixel TP/FP/FN and ROC.
                         Pass None (default) to skip pixel metrics (e.g. during training validate).
                         A second binary mask at threshold >=0.5 is derived internally.

        Returns:
            (inter_np, union_np, acc_iou_img_np, n, pix_TP_call, pix_FP_call, pix_FN_call)
            where the last three are the pixel TP/FP/FN aggregated over THIS call only
            (per-group bookkeeping). They are 0 when pred_scores is None.
        """
        from utils.utils import intersectionAndUnionGPU  # local import for worker safety

        # Normalise to [B, H, W]
        if gt_mask.dim() == 2:
            gt_mask = gt_mask.unsqueeze(0)
        if pred_mask_bin.dim() == 2:
            pred_mask_bin = pred_mask_bin.unsqueeze(0)
        if pred_scores is not None and pred_scores.dim() == 2:
            pred_scores = pred_scores.unsqueeze(0)

        # pixel binary prediction at 0.5 threshold
        pred_bin = (pred_scores >= 0.5).to(torch.int32) if pred_scores is not None else None

        intersection = union = acc_iou = 0.0
        for mask_i, output_i in zip(gt_mask, pred_mask_bin):
            intersection_i, union_i, _ = intersectionAndUnionGPU(
                output_i.contiguous().clone(), mask_i.contiguous(), 2, ignore_index=255
            )
            intersection += intersection_i
            union += union_i
            acc_iou += intersection_i / (union_i + 1e-5)
            acc_iou[union_i == 0] += 1.0

        n = gt_mask.shape[0]
        inter_np        = intersection.cpu().numpy()
        union_np        = union.cpu().numpy()
        acc_iou_img_np  = (acc_iou / n).cpu().numpy()   # per-image average

        self.intersection_sum += inter_np
        self.union_sum        += union_np
        self.acc_iou_sum      += acc_iou_img_np
        self.acc_iou_count    += n

        # Pixel TP / FP / FN (uses 0.5-thresholded pred_bin) and ROC histograms
        if pred_scores is None:
            return inter_np, union_np, acc_iou_img_np, n, 0, 0, 0
        pix_TP_call = pix_FP_call = pix_FN_call = 0
        for mask_i, score_i, bin_i in zip(gt_mask, pred_scores, pred_bin):
            m_flat = mask_i.flatten().to(torch.uint8)
            p_flat = bin_i.flatten().to(torch.uint8)
            s_flat = score_i.flatten().to(torch.float32)

            tp_i = int((p_flat.eq(1) & m_flat.eq(1)).sum().item())
            fp_i = int((p_flat.eq(1) & m_flat.eq(0)).sum().item())
            fn_i = int((p_flat.eq(0) & m_flat.eq(1)).sum().item())
            pix_TP_call += tp_i
            pix_FP_call += fp_i
            pix_FN_call += fn_i
            self.pix_TP += tp_i
            self.pix_FP += fp_i
            self.pix_FN += fn_i

            s_clamped = s_flat.clamp_(0, 1)
            bins_idx  = torch.clamp((s_clamped * (ROC_BINS - 1)).long(), 0, ROC_BINS - 1).cpu()
            m_bool    = m_flat.cpu() > 0
            if m_bool.any():
                self.pos_hist.index_add_(
                    0, bins_idx[m_bool],
                    torch.ones_like(bins_idx[m_bool], dtype=torch.float64)
                )
            if (~m_bool).any():
                self.neg_hist.index_add_(
                    0, bins_idx[~m_bool],
                    torch.ones_like(bins_idx[~m_bool], dtype=torch.float64)
                )

        return inter_np, union_np, acc_iou_img_np, n, pix_TP_call, pix_FP_call, pix_FN_call

    def update_obj(
        self,
        pred_probs: torch.Tensor,   # [K] or [1, K] logits/probs
        gt_vec: torch.Tensor,       # [K] or [1, K] binary ground truth
        threshold: float = 0.5,
    ) -> None:
        """Update object multi-label accumulators for one tampered sample."""
        # Normalise to [1, K]
        if pred_probs.dim() == 1:
            pred_probs = pred_probs.unsqueeze(0)
        if gt_vec.dim() == 1:
            gt_vec = gt_vec.unsqueeze(0)

        gt     = gt_vec.float()
        pred   = (pred_probs >= threshold).to(gt.dtype)
        gt_bool = (gt > 0).to(torch.bool)

        # Top-K hit
        valid_rows = gt_bool.any(dim=1)
        n_valid    = int(valid_rows.sum().item())
        if n_valid > 0:
            K  = gt.shape[1]
            k5 = min(5, K)
            topk_idx = pred_probs.topk(k5, dim=1).indices
            top1_idx = topk_idx[:, :1]
            hit1 = gt_bool.gather(1, top1_idx).any(dim=1)
            topk_mask = torch.zeros_like(gt_bool)
            topk_mask.scatter_(1, topk_idx, True)
            hit5 = (topk_mask & gt_bool).any(dim=1)
            self.obj_hit1_total    += int(hit1[valid_rows].sum().item())
            self.obj_hit5_total    += int(hit5[valid_rows].sum().item())
            self.obj_hit_den_total += n_valid

        # Micro counts
        tp = (pred * gt).sum().double()
        fp = (pred * (1 - gt)).sum().double()
        fn = ((1 - pred) * gt).sum().double()
        self.obj_tp_total += tp.item()
        self.obj_fp_total += fp.item()
        self.obj_fn_total += fn.item()
        self.obj_exact_match_total += int((pred == gt).all(dim=1).sum().item())
        self.obj_rows_total        += gt.shape[0]

        # Per-class counts
        K      = gt.shape[1]
        device = gt.device
        if self._obj_tp_per_class is None:
            self._obj_tp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
            self._obj_fp_per_class = torch.zeros(K, device=device, dtype=torch.float64)
            self._obj_fn_per_class = torch.zeros(K, device=device, dtype=torch.float64)
        self._obj_tp_per_class += (pred * gt).sum(dim=0).double()
        self._obj_fp_per_class += (pred * (1 - gt)).sum(dim=0).double()
        self._obj_fn_per_class += ((1 - pred) * gt).sum(dim=0).double()

    # ------------------------------------------------------------------
    # Distributed reduction (for DeepSpeed / multi-GPU training)
    # ------------------------------------------------------------------

    def all_reduce_seg(self) -> None:
        """
        Distributed all-reduce for segmentation meters.

        Mirrors the three AverageMeter.all_reduce() calls in the original validate().
        No-op when torch.distributed is not initialised (single-GPU or test_parallel).
        """
        import torch.distributed as dist
        if not (dist.is_available() and dist.is_initialized()):
            return
        device = "cuda" if torch.cuda.is_available() else "cpu"
        # Pack [inter[0], inter[1], union[0], union[1], acc_iou[0], acc_iou[1], count]
        data = torch.tensor(
            self.intersection_sum.tolist()
            + self.union_sum.tolist()
            + self.acc_iou_sum.tolist()
            + [float(self.acc_iou_count)],
            dtype=torch.float32,
            device=device,
        )
        dist.all_reduce(data, dist.ReduceOp.SUM, async_op=False)
        data = data.cpu().numpy()
        self.intersection_sum = data[0:2]
        self.union_sum        = data[2:4]
        self.acc_iou_sum      = data[4:6]
        self.acc_iou_count    = int(data[6])

    # ------------------------------------------------------------------
    # Serialisation
    # ------------------------------------------------------------------

    def to_dict(self) -> dict:
        """Return JSON-serialisable dict of all raw counts."""
        return {
            "confusion_matrix":      self.confusion_matrix.tolist(),
            "correct":               self.correct,
            "total":                 self.total,
            "intersection_sum":      self.intersection_sum.tolist(),
            "union_sum":             self.union_sum.tolist(),
            "acc_iou_sum":           self.acc_iou_sum.tolist(),
            "acc_iou_count":         int(self.acc_iou_count),
            "pix_TP":                int(self.pix_TP),
            "pix_FP":                int(self.pix_FP),
            "pix_FN":                int(self.pix_FN),
            "pos_hist":              self.pos_hist.tolist(),
            "neg_hist":              self.neg_hist.tolist(),
            "obj_tp_total":          self.obj_tp_total,
            "obj_fp_total":          self.obj_fp_total,
            "obj_fn_total":          self.obj_fn_total,
            "obj_exact_match_total": int(self.obj_exact_match_total),
            "obj_rows_total":        int(self.obj_rows_total),
            "obj_hit1_total":        int(self.obj_hit1_total),
            "obj_hit5_total":        int(self.obj_hit5_total),
            "obj_hit_den_total":     int(self.obj_hit_den_total),
            "obj_tp_per_class":      (self._obj_tp_per_class.cpu().tolist()
                                      if self._obj_tp_per_class is not None else None),
            "obj_fp_per_class":      (self._obj_fp_per_class.cpu().tolist()
                                      if self._obj_fp_per_class is not None else None),
            "obj_fn_per_class":      (self._obj_fn_per_class.cpu().tolist()
                                      if self._obj_fn_per_class is not None else None),
        }


# ---------------------------------------------------------------------------
# Merge multiple raw-count dicts (one per GPU chunk)
# ---------------------------------------------------------------------------

def merge_raw_counts(raws: list[dict]) -> dict:
    """
    Element-wise sum of raw count dicts produced by MetricsAccumulator.to_dict().
    Identical to the original merge_raw() in test_parallel.py.
    """
    m: dict = {}

    # Confusion matrix
    cm = np.array(raws[0]["confusion_matrix"], dtype=np.float64)
    for r in raws[1:]:
        cm += np.array(r["confusion_matrix"], dtype=np.float64)
    m["confusion_matrix"] = cm.tolist()
    m["correct"] = sum(r["correct"] for r in raws)
    m["total"]   = sum(r["total"]   for r in raws)

    # Segmentation meters
    inter = np.array(raws[0]["intersection_sum"], dtype=np.float64)
    union = np.array(raws[0]["union_sum"],         dtype=np.float64)
    acc_s = np.array(raws[0]["acc_iou_sum"],       dtype=np.float64)
    acc_c = raws[0]["acc_iou_count"]
    for r in raws[1:]:
        inter += np.array(r["intersection_sum"], dtype=np.float64)
        union += np.array(r["union_sum"],         dtype=np.float64)
        acc_s += np.array(r["acc_iou_sum"],       dtype=np.float64)
        acc_c += r["acc_iou_count"]
    m["intersection_sum"] = inter.tolist()
    m["union_sum"]        = union.tolist()
    m["acc_iou_sum"]      = acc_s.tolist()
    m["acc_iou_count"]    = acc_c

    # Pixel counts
    m["pix_TP"] = sum(r["pix_TP"] for r in raws)
    m["pix_FP"] = sum(r["pix_FP"] for r in raws)
    m["pix_FN"] = sum(r["pix_FN"] for r in raws)

    # AUC histograms
    pos_h = np.array(raws[0]["pos_hist"], dtype=np.float64)
    neg_h = np.array(raws[0]["neg_hist"], dtype=np.float64)
    for r in raws[1:]:
        pos_h += np.array(r["pos_hist"], dtype=np.float64)
        neg_h += np.array(r["neg_hist"], dtype=np.float64)
    m["pos_hist"] = pos_h.tolist()
    m["neg_hist"] = neg_h.tolist()

    # OBJ scalars
    for key in ("obj_tp_total", "obj_fp_total", "obj_fn_total",
                "obj_exact_match_total", "obj_rows_total",
                "obj_hit1_total", "obj_hit5_total", "obj_hit_den_total"):
        m[key] = sum(r[key] for r in raws)

    # OBJ per-class vectors
    tp_c = fp_c = fn_c = None
    for r in raws:
        if r["obj_tp_per_class"] is not None:
            t  = np.array(r["obj_tp_per_class"], dtype=np.float64)
            f_ = np.array(r["obj_fp_per_class"], dtype=np.float64)
            fn = np.array(r["obj_fn_per_class"], dtype=np.float64)
            if tp_c is None:
                tp_c, fp_c, fn_c = t, f_, fn
            else:
                tp_c += t; fp_c += f_; fn_c += fn
    m["obj_tp_per_class"] = tp_c.tolist() if tp_c is not None else None
    m["obj_fp_per_class"] = fp_c.tolist() if fp_c is not None else None
    m["obj_fn_per_class"] = fn_c.tolist() if fn_c is not None else None

    # Per-model and per-op group breakdown (present only when --mapping_json was used)
    m["per_model"] = merge_group_dicts(raws, "per_model")
    m["per_op"]    = merge_group_dicts(raws, "per_op")

    # Per-model × per-op cross-table
    m["per_model_per_op"] = merge_nested_group_dicts(raws, "per_model_per_op")

    return m


# ---------------------------------------------------------------------------
# ROC-AUC from histograms
# ---------------------------------------------------------------------------

def compute_roc_auc(pos_hist: torch.Tensor, neg_hist: torch.Tensor) -> tuple[float, float]:
    """
    Compute pixel-level PR-AUC and ROC-AUC from histogram bin counts.

    Args:
        pos_hist: 1-D tensor, counts of positive (tampered) pixel scores per bin
        neg_hist: 1-D tensor, counts of negative pixel scores per bin

    Returns:
        (pr_auc, roc_auc) as floats; both 0.0 if no samples.
    """
    if (pos_hist.sum() + neg_hist.sum()) == 0:
        return 0.0, 0.0

    pos_cum = torch.cumsum(pos_hist.flip(0), dim=0)
    neg_cum = torch.cumsum(neg_hist.flip(0), dim=0)
    P   = pos_cum[-1]
    N   = neg_cum[-1]
    fn_h  = P - pos_cum
    tn_h  = N - neg_cum
    precision_h = pos_cum / (pos_cum + neg_cum + 1e-12)
    recall_h    = pos_cum / (pos_cum + fn_h + 1e-12)
    dr    = recall_h[:-1] - recall_h[1:]
    pr_auc  = float(torch.sum(precision_h[1:] * dr).item())
    fpr = neg_cum / (neg_cum + tn_h + 1e-12)
    tpr = recall_h
    df  = fpr[1:] - fpr[:-1]
    roc_auc = float(torch.sum((tpr[1:] + tpr[:-1]) * 0.5 * df).item())
    return pr_auc, roc_auc


# ---------------------------------------------------------------------------
# Compute final metrics from merged raw counts
# ---------------------------------------------------------------------------

def compute_metrics(m: dict) -> dict:
    """
    Compute all final metrics from a merged raw-counts dict.

    This is a pure function (no side effects) that replaces the computation
    half of compute_and_print() in test_parallel.py.

    Returns a dict with the same keys as the original metrics.json.
    """
    num_classes = 3
    class_names = ["Real", "Full Synthetic", "Tampered"]

    # ---- Pixel P/R/F1 ----
    pix_TP, pix_FP, pix_FN = m["pix_TP"], m["pix_FP"], m["pix_FN"]
    pixel_precision = pix_TP / (pix_TP + pix_FP + 1e-12) if (pix_TP + pix_FP) > 0 else 0.0
    pixel_recall    = pix_TP / (pix_TP + pix_FN + 1e-12) if (pix_TP + pix_FN) > 0 else 0.0
    pixel_f1        = (2 * pixel_precision * pixel_recall
                       / (pixel_precision + pixel_recall + 1e-12)
                       if (pixel_precision + pixel_recall) > 0 else 0.0)

    # ---- ROC-AUC ----
    pos_hist = torch.tensor(m["pos_hist"], dtype=torch.float64)
    neg_hist = torch.tensor(m["neg_hist"], dtype=torch.float64)
    _pixel_pr_auc, pixel_roc_auc = compute_roc_auc(pos_hist, neg_hist)

    # ---- OBJ multi-label ----
    obj_tp, obj_fp, obj_fn = m["obj_tp_total"], m["obj_fp_total"], m["obj_fn_total"]
    obj_micro_prec = obj_tp / (obj_tp + obj_fp + 1e-12) if (obj_tp + obj_fp) > 0 else 0.0
    obj_micro_rec  = obj_tp / (obj_tp + obj_fn + 1e-12) if (obj_tp + obj_fn) > 0 else 0.0
    obj_micro_f1   = (2 * obj_micro_prec * obj_micro_rec
                      / (obj_micro_prec + obj_micro_rec + 1e-12)
                      if (obj_micro_prec + obj_micro_rec) > 0 else 0.0)
    obj_subset_acc = (m["obj_exact_match_total"] / m["obj_rows_total"]
                      if m["obj_rows_total"] > 0 else 0.0)
    obj_top1 = (m["obj_hit1_total"] / m["obj_hit_den_total"] * 100.0
                if m["obj_hit_den_total"] > 0 else 0.0)
    obj_top5 = (m["obj_hit5_total"] / m["obj_hit_den_total"] * 100.0
                if m["obj_hit_den_total"] > 0 else 0.0)

    if m["obj_tp_per_class"] is not None:
        tp_c = np.array(m["obj_tp_per_class"])
        fp_c = np.array(m["obj_fp_per_class"])
        fn_c = np.array(m["obj_fn_per_class"])
        prec_c = tp_c / (tp_c + fp_c + 1e-12)
        rec_c  = tp_c / (tp_c + fn_c + 1e-12)
        f1_c   = 2 * prec_c * rec_c / (prec_c + rec_c + 1e-12)
        obj_macro_prec = float(prec_c.mean())
        obj_macro_rec  = float(rec_c.mean())
        obj_macro_f1   = float(f1_c.mean())
    else:
        obj_macro_prec = obj_macro_rec = obj_macro_f1 = 0.0

    # ---- IoU ----
    inter      = np.array(m["intersection_sum"])
    union_arr  = np.array(m["union_sum"])
    iou_class  = inter / (union_arr + 1e-10)
    ciou       = float(iou_class[1]) if len(iou_class) > 1 else 0.0
    acc_iou_sum   = np.array(m["acc_iou_sum"])
    acc_iou_count = m["acc_iou_count"]
    giou = (float(acc_iou_sum[1] / acc_iou_count)
            if (acc_iou_count > 0 and len(acc_iou_sum) > 1) else 0.0)

    # ---- Classification ----
    correct, total = m["correct"], m["total"]
    accuracy = correct / total * 100.0 if total > 0 else 0.0

    cm = np.array(m["confusion_matrix"])
    per_class_metrics: dict = {}
    for i in range(num_classes):
        tp_i  = cm[i, i]
        fp_i  = cm[:, i].sum() - tp_i
        fn_i  = cm[i, :].sum() - tp_i
        tot_i = cm[i, :].sum()
        prec_i = float(tp_i / (tp_i + fp_i)) if (tp_i + fp_i) > 0 else 0.0
        rec_i  = float(tp_i / (tp_i + fn_i)) if (tp_i + fn_i) > 0 else 0.0
        f1_i   = float(2 * prec_i * rec_i / (prec_i + rec_i)) if (prec_i + rec_i) > 0 else 0.0
        per_class_metrics[class_names[i]] = {
            "accuracy":  float(tp_i / tot_i) if tot_i > 0 else 0.0,
            "precision": prec_i,
            "recall":    rec_i,
            "f1":        f1_i,
        }

    # ---- Combined F1 ----
    iou      = ciou
    f1_score = (2 * (iou * accuracy / 100) / (iou + accuracy / 100 + 1e-10)
                if (iou + accuracy / 100) > 0 else 0.0)

    # ---- Collapse detection fields (B5 criterion) ----
    # cm[i, :].sum() = ground-truth count for class i
    # cm[:, j].sum() = total samples predicted as class j
    n_gt_real      = int(cm[0, :].sum())
    n_gt_tampered  = int(cm[2, :].sum())
    n_pred_tampered = int(cm[:, 2].sum())
    predicted_tampered_fraction = float(n_pred_tampered / total) if total > 0 else 0.0

    return {
        "accuracy":                     accuracy,
        "giou":                         giou,
        "ciou":                         ciou,
        "pixel_precision":              pixel_precision,
        "pixel_recall":                 pixel_recall,
        "pixel_f1":                     pixel_f1,
        "pixel_roc_auc":                pixel_roc_auc,
        "obj_micro_f1":                 obj_micro_f1,
        "obj_macro_f1":                 obj_macro_f1,
        "obj_top1":                     obj_top1,
        "obj_top5":                     obj_top5,
        "per_class_metrics":            per_class_metrics,
        "total_samples":                total,
        "combined_f1":                  f1_score,
        "predicted_tampered_fraction":  predicted_tampered_fraction,
        "n_gt_real":                    n_gt_real,
        "n_gt_tampered":                n_gt_tampered,
        # Extra fields available to callers (not saved to metrics.json)
        "_obj_micro_prec":   obj_micro_prec,
        "_obj_micro_rec":    obj_micro_rec,
        "_obj_macro_prec":   obj_macro_prec,
        "_obj_macro_rec":    obj_macro_rec,
        "_obj_subset_acc":   obj_subset_acc,
        "_confusion_matrix": cm,
    }


# ---------------------------------------------------------------------------
# Print standard evaluation report
# ---------------------------------------------------------------------------

def print_metrics_report(metrics: dict, total_samples: int, num_chunks: int) -> None:
    """
    Print the standard evaluation report to stdout.

    Accepts the dict returned by compute_metrics() plus the total sample
    count and chunk count (for the header line).
    """
    num_classes  = 3
    class_names  = ["Real", "Full Synthetic", "Tampered"]
    accuracy     = metrics["accuracy"]
    giou         = metrics["giou"]
    ciou         = metrics["ciou"]
    cm           = np.array(metrics["_confusion_matrix"])

    per_class    = metrics["per_class_metrics"]

    print(f"\n{'='*70}")
    print(f"Parallel Test Results ({total_samples} samples, {num_chunks} chunks merged)")
    print(f"{'='*70}")

    print(f"\nClassification Accuracy: {accuracy:.4f}%")
    print(f"Predicted Tampered Fraction: {metrics['predicted_tampered_fraction']:.4f}"
          f"  (n_gt_real={metrics['n_gt_real']}, n_gt_tampered={metrics['n_gt_tampered']})")
    print("\nPer-Class Metrics:")
    for cn, met in per_class.items():
        print(f"  {cn}:")
        print(f"    Accuracy:  {met['accuracy']:.4f}")
        print(f"    Precision: {met['precision']:.4f}")
        print(f"    Recall:    {met['recall']:.4f}")
        print(f"    F1 Score:  {met['f1']:.4f}")

    print(f"\nConfusion Matrix:")
    print(f"{'':20}", end="")
    for name in class_names:
        print(f"{name:>15}", end="")
    print()
    for i, cn in enumerate(class_names):
        print(f"{cn:20}", end="")
        for j in range(num_classes):
            print(f"{cm[i, j]:15.0f}", end="")
        print()

    print(f"\nSegmentation Metrics (tampered only):")
    print(f"  gIoU: {giou:.4f}")
    print(f"  cIoU: {ciou:.4f}")
    print(f"  Pixel Precision: {metrics['pixel_precision']:.4f}")
    print(f"  Pixel Recall:    {metrics['pixel_recall']:.4f}")
    print(f"  Pixel F1:        {metrics['pixel_f1']:.4f}")
    print(f"  Pixel ROC-AUC:   {metrics['pixel_roc_auc']:.4f}")

    print(f"\n[OBJ] Multi-Label Metrics (tampered only):")
    print(f"  Micro  - P: {metrics['_obj_micro_prec']:.4f}, "
          f"R: {metrics['_obj_micro_rec']:.4f}, "
          f"F1: {metrics['obj_micro_f1']:.4f}")
    print(f"  Macro  - P: {metrics['_obj_macro_prec']:.4f}, "
          f"R: {metrics['_obj_macro_rec']:.4f}, "
          f"F1: {metrics['obj_macro_f1']:.4f}")
    print(f"  Subset Acc: {metrics['_obj_subset_acc']:.4f}")
    print(f"  Top-1 Acc:  {metrics['obj_top1']:.4f}%")
    print(f"  Top-5 Acc:  {metrics['obj_top5']:.4f}%")

    print(f"\nCombined F1: {metrics['combined_f1']:.4f}")


def metrics_for_json(metrics: dict) -> dict:
    """Strip private keys (prefixed with '_') before saving to metrics.json."""
    return {k: v for k, v in metrics.items() if not k.startswith("_")}


# ---------------------------------------------------------------------------
# Per-model / per-operation breakdown (GroupAccumulator)
# ---------------------------------------------------------------------------

# Sample-count threshold below which a group is annotated as low-n in reports
GROUP_LOW_N_THRESHOLD = 30


class GroupAccumulator:
    """
    Lightweight accumulator for per-model or per-operation breakdown.

    Tracks: tampered-class recall (correct / total), IoU metrics, and
    pixel-level TP/FP/FN (added 21May 2026). OBJ multi-label metrics are NOT
    tracked per-group.
    """

    def __init__(self):
        self.correct           = 0    # samples predicted as tampered (TP for detection)
        self.total             = 0    # total gt-tampered samples in this group
        self.intersection_sum  = np.zeros(2, dtype=np.float64)
        self.union_sum         = np.zeros(2, dtype=np.float64)
        self.acc_iou_sum       = np.zeros(2, dtype=np.float64)
        self.acc_iou_count     = 0
        # Pixel TP/FP/FN aggregated over tampered samples in this group
        self.pix_TP            = 0
        self.pix_FP            = 0
        self.pix_FN            = 0

    def update(
        self,
        predicted_tampered: bool,
        intersection_np:    np.ndarray | None = None,
        union_np:           np.ndarray | None = None,
        acc_iou_per_img_np: np.ndarray | None = None,
        n:                  int = 1,
        pix_TP:             int | None = None,
        pix_FP:             int | None = None,
        pix_FN:             int | None = None,
    ) -> None:
        """
        Update for one tampered sample.

        predicted_tampered : True if model predicted class == 2 (tampered)
        intersection_np / union_np / acc_iou_per_img_np : returned by
            MetricsAccumulator.update_seg(); pass None to skip IoU tracking.
        pix_TP / pix_FP / pix_FN : pixel-level counts for THIS sample's seg
            update_seg call; pass None to skip pixel tracking (cls-only path).
        """
        self.total += 1
        if predicted_tampered:
            self.correct += 1
        if intersection_np is not None:
            self.intersection_sum += intersection_np
            self.union_sum        += union_np
            self.acc_iou_sum      += acc_iou_per_img_np
            self.acc_iou_count    += n
        if pix_TP is not None:
            self.pix_TP += int(pix_TP)
            self.pix_FP += int(pix_FP)
            self.pix_FN += int(pix_FN)

    def to_dict(self) -> dict:
        return {
            "correct":          self.correct,
            "total":            self.total,
            "intersection_sum": self.intersection_sum.tolist(),
            "union_sum":        self.union_sum.tolist(),
            "acc_iou_sum":      self.acc_iou_sum.tolist(),
            "acc_iou_count":    self.acc_iou_count,
            "pix_TP":           int(self.pix_TP),
            "pix_FP":           int(self.pix_FP),
            "pix_FN":           int(self.pix_FN),
        }

    @classmethod
    def from_dict(cls, d: dict) -> "GroupAccumulator":
        g = cls()
        g.correct          = d["correct"]
        g.total            = d["total"]
        g.intersection_sum = np.array(d["intersection_sum"], dtype=np.float64)
        g.union_sum        = np.array(d["union_sum"],         dtype=np.float64)
        g.acc_iou_sum      = np.array(d["acc_iou_sum"],       dtype=np.float64)
        g.acc_iou_count    = d["acc_iou_count"]
        g.pix_TP           = int(d["pix_TP"])
        g.pix_FP           = int(d["pix_FP"])
        g.pix_FN           = int(d["pix_FN"])
        return g


def merge_group_dicts(raws: list[dict], key: str) -> dict:
    """
    Merge per_model or per_op dicts from multiple raw-count chunks.

    Args:
        raws : list of raw-count dicts (one per GPU chunk)
        key  : "per_model" or "per_op"

    Returns:
        Merged dict mapping group name → raw group dict.
        Empty dict if none of the chunks had the key.
    """
    merged: dict = {}
    for r in raws:
        for name, gd in r.get(key, {}).items():
            if name not in merged:
                merged[name] = {
                    "correct":          0,
                    "total":            0,
                    "intersection_sum": np.zeros(2, dtype=np.float64),
                    "union_sum":        np.zeros(2, dtype=np.float64),
                    "acc_iou_sum":      np.zeros(2, dtype=np.float64),
                    "acc_iou_count":    0,
                    "pix_TP":           0,
                    "pix_FP":           0,
                    "pix_FN":           0,
                }
            m = merged[name]
            m["correct"]          += gd["correct"]
            m["total"]            += gd["total"]
            m["intersection_sum"] += np.array(gd["intersection_sum"], dtype=np.float64)
            m["union_sum"]        += np.array(gd["union_sum"],         dtype=np.float64)
            m["acc_iou_sum"]      += np.array(gd["acc_iou_sum"],       dtype=np.float64)
            m["acc_iou_count"]    += gd["acc_iou_count"]
            m["pix_TP"]           += int(gd["pix_TP"])
            m["pix_FP"]           += int(gd["pix_FP"])
            m["pix_FN"]           += int(gd["pix_FN"])
    # Convert numpy arrays back to lists for JSON serialisability
    for m in merged.values():
        for k in ("intersection_sum", "union_sum", "acc_iou_sum"):
            if isinstance(m[k], np.ndarray):
                m[k] = m[k].tolist()
    return merged


def compute_group_metrics(groups_raw: dict) -> dict:
    """
    Compute recall + IoU + pixel-level metrics for each group.

    Args:
        groups_raw : dict mapping group name → raw group dict (from merge_group_dicts)

    Returns:
        dict mapping group name → {
          "n", "recall", "giou", "ciou",
          "pixel_precision", "pixel_recall", "pixel_f1",
          # Raw counts retained to enable offline subset recomputation:
          "correct", "intersection_sum", "union_sum", "acc_iou_sum", "acc_iou_count",
          "pix_TP", "pix_FP", "pix_FN",
        }
    """
    result: dict = {}
    for name, g in groups_raw.items():
        inter = np.array(g["intersection_sum"])
        union = np.array(g["union_sum"])
        acc_s = np.array(g["acc_iou_sum"])
        acc_c = int(g["acc_iou_count"])
        ciou   = float(inter[1] / (union[1] + 1e-10)) if union[1] > 0 else 0.0
        giou   = float(acc_s[1] / acc_c)              if acc_c > 0    else 0.0
        recall = g["correct"] / g["total"]             if g["total"] > 0 else 0.0

        # pixel metrics from per-group TP/FP/FN
        tp, fp, fn = int(g["pix_TP"]), int(g["pix_FP"]), int(g["pix_FN"])
        pix_p = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else 0.0
        pix_r = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else 0.0
        pix_f = (2 * pix_p * pix_r / (pix_p + pix_r + 1e-12)
                 if (pix_p + pix_r) > 0 else 0.0)

        result[name] = {
            "n":              g["total"],
            "recall":         recall,
            "giou":           giou,
            "ciou":           ciou,
            "pixel_precision": pix_p,
            "pixel_recall":    pix_r,
            "pixel_f1":        pix_f,
            # Raw counts retained for offline subset recomputation
            "correct":          g["correct"],
            "intersection_sum": list(g["intersection_sum"]),
            "union_sum":        list(g["union_sum"]),
            "acc_iou_sum":      list(g["acc_iou_sum"]),
            "acc_iou_count":    int(acc_c),
            "pix_TP":           tp,
            "pix_FP":           fp,
            "pix_FN":           fn,
        }
    return result


def merge_nested_group_dicts(raws: list[dict], key: str) -> dict:
    """
    Merge per_model_per_op dicts from multiple raw-count chunks.

    Args:
        raws : list of raw-count dicts (one per GPU chunk)
        key  : "per_model_per_op"

    Returns:
        Nested dict: model → op → merged group raw dict.
        Empty dict if none of the chunks had the key.
    """
    merged: dict = {}
    for r in raws:
        for model_name, ops_dict in r.get(key, {}).items():
            if model_name not in merged:
                merged[model_name] = {}
            for op_name, gd in ops_dict.items():
                if op_name not in merged[model_name]:
                    merged[model_name][op_name] = {
                        "correct":          0,
                        "total":            0,
                        "intersection_sum": np.zeros(2, dtype=np.float64),
                        "union_sum":        np.zeros(2, dtype=np.float64),
                        "acc_iou_sum":      np.zeros(2, dtype=np.float64),
                        "acc_iou_count":    0,
                        "pix_TP":           0,
                        "pix_FP":           0,
                        "pix_FN":           0,
                    }
                m = merged[model_name][op_name]
                m["correct"]          += gd["correct"]
                m["total"]            += gd["total"]
                m["intersection_sum"] += np.array(gd["intersection_sum"], dtype=np.float64)
                m["union_sum"]        += np.array(gd["union_sum"],         dtype=np.float64)
                m["acc_iou_sum"]      += np.array(gd["acc_iou_sum"],       dtype=np.float64)
                m["acc_iou_count"]    += gd["acc_iou_count"]
                m["pix_TP"]           += int(gd["pix_TP"])
                m["pix_FP"]           += int(gd["pix_FP"])
                m["pix_FN"]           += int(gd["pix_FN"])
    # Convert numpy arrays back to lists for JSON serialisability
    for ops in merged.values():
        for m in ops.values():
            for k in ("intersection_sum", "union_sum", "acc_iou_sum"):
                if isinstance(m[k], np.ndarray):
                    m[k] = m[k].tolist()
    return merged


def compute_cross_table_metrics(
    nested_raw: dict,
    all_models: list,
    all_ops: list,
) -> dict:
    """
    Build a model × operation cross-table of recall + IoU + pixel metrics.

    Args:
        nested_raw : nested dict model → op → group raw dict
        all_models : ordered list of model names (rows)
        all_ops    : ordered list of operation names (columns)

    Returns:
        Nested dict model → op → {"n", "recall", "giou", "ciou",
                                  "pixel_precision", "pixel_recall", "pixel_f1"}.
        Empty cells (n == 0) are all-None scalars.
    """
    result: dict = {}
    EMPTY = {"n": 0, "recall": None, "giou": None, "ciou": None,
             "pixel_precision": None, "pixel_recall": None, "pixel_f1": None}
    for model in all_models:
        result[model] = {}
        for op in all_ops:
            g = nested_raw.get(model, {}).get(op)
            if g is None or g["total"] == 0:
                result[model][op] = dict(EMPTY)
                continue
            inter = np.array(g["intersection_sum"])
            union = np.array(g["union_sum"])
            acc_s = np.array(g["acc_iou_sum"])
            acc_c = int(g["acc_iou_count"])
            ciou   = float(inter[1] / (union[1] + 1e-10)) if union[1] > 0 else 0.0
            giou   = float(acc_s[1] / acc_c)              if acc_c > 0    else 0.0
            recall = g["correct"] / g["total"]
            tp, fp, fn = int(g["pix_TP"]), int(g["pix_FP"]), int(g["pix_FN"])
            pix_p = tp / (tp + fp + 1e-12) if (tp + fp) > 0 else 0.0
            pix_r = tp / (tp + fn + 1e-12) if (tp + fn) > 0 else 0.0
            pix_f = (2 * pix_p * pix_r / (pix_p + pix_r + 1e-12)
                     if (pix_p + pix_r) > 0 else 0.0)
            result[model][op] = {
                "n":      g["total"],
                "recall": recall,
                "giou":   giou,
                "ciou":   ciou,
                "pixel_precision": pix_p,
                "pixel_recall":    pix_r,
                "pixel_f1":        pix_f,
            }
    return result


def print_group_report(
    label: str,
    group_metrics: dict,
    low_n_threshold: int = GROUP_LOW_N_THRESHOLD,
) -> None:
    """
    Print a per-model or per-operation breakdown table.

    Rows are sorted by descending sample count.
    Groups with n < low_n_threshold are annotated with "← low n".
    """
    if not group_metrics:
        return

    total_n = sum(m["n"] for m in group_metrics.values())
    print(f"\n{label} (total tampered evaluated: {total_n}):")
    w = max(len(k) for k in group_metrics) + 2
    w = max(w, 12)
    print(f"  {'Name':<{w}} {'n':>6}  {'Recall':>7}  {'gIoU':>7}  {'cIoU':>7}")
    print(f"  {'─'*w}  {'─'*6}  {'─'*7}  {'─'*7}  {'─'*7}")
    for name, m in sorted(group_metrics.items(), key=lambda x: -x[1]["n"]):
        warn = "  ← low n" if m["n"] < low_n_threshold else ""
        print(
            f"  {name:<{w}} {m['n']:>6}  "
            f"{m['recall']:>7.4f}  {m['giou']:>7.4f}  {m['ciou']:>7.4f}"
            f"{warn}"
        )
