import argparse
import json
import os
import shutil
import sys
import time
from functools import partial
os.environ["TOKENIZERS_PARALLELISM"] = "false"
import deepspeed
import numpy as np
import torch
import tqdm
import transformers
from peft import LoraConfig, get_peft_model
from torch.utils.tensorboard import SummaryWriter
from model.PIXAR import PIXARForCausalLM
from model.llava import conversation as conversation_lib
from utils.PIXAR_Set import collate_fn, CustomDataset
from utils.batch_sampler import (
    BatchSampler,
    BalancedBatchSampler,
    BalancedSourceWeightedSampler,
    parse_weights_schedule,
)
import torch.distributed as dist
from utils.utils import (DEFAULT_IM_END_TOKEN, DEFAULT_IM_START_TOKEN,
                         AverageMeter, ProgressMeter, Summary, dict_to_cuda,
                         intersectionAndUnionGPU)
from utils.metrics import MetricsAccumulator, compute_metrics
import random
import torch.nn.functional as F
import warnings
warnings.filterwarnings("ignore")

def parse_args(args):
    parser = argparse.ArgumentParser(description="PIXAR Model Training")
    parser.add_argument("--local_rank", default=0, type=int, help="node rank")
    parser.add_argument(
        "--version", default="liuhaotian/llava-llama-2-13b-chat-lightning-preview"
    )
    parser.add_argument("--vis_save_path", default="./vis_output", type=str)
    parser.add_argument(
        "--precision",
        default="fp16",
        type=str,
        choices=["fp32", "bf16", "fp16"],
        help="precision for inference",
    )
    parser.add_argument("--image_size", default=1024, type=int, help="image size")
    parser.add_argument("--model_max_length", default=512, type=int)
    parser.add_argument("--lora_r", default=8, type=int)
    parser.add_argument(
        "--vision-tower", default="openai/clip-vit-large-patch14", type=str
    )
    parser.add_argument("--load_in_8bit", action="store_true", default=False)
    parser.add_argument("--load_in_4bit", action="store_true", default=False)

    parser.add_argument("--val_dataset", default="val", type=str)
    parser.add_argument("--dataset_dir", default="./dataset", type=str)
    parser.add_argument("--log_base_dir", default="./runs", type=str)
    parser.add_argument("--exp_name", default="pixar", type=str)
    parser.add_argument("--epochs", default=10, type=int)
    parser.add_argument("--steps_per_epoch", default=500, type=int)
    parser.add_argument(
        "--batch_size", default=2, type=int, help="batch size per device per step"
    )
    parser.add_argument(
        "--grad_accumulation_steps",
        default=10,
        type=int,
    )
    parser.add_argument("--val_batch_size", default=1, type=int)
    parser.add_argument("--workers", default=4, type=int)
    parser.add_argument("--lr", default=0.00001, type=float)

    # LR scheduler overrides; default is WarmupDecayLR(total = epochs * steps_per_epoch, min_lr=0).
    parser.add_argument("--scheduler_total_steps", type=int, default=None,
        help="Override total_num_steps for LR scheduler. If None, falls back to "
             "epochs * steps_per_epoch.")
    parser.add_argument("--scheduler_type", type=str, default="warmup_decay",
        choices=["warmup_decay", "constant"],
        # cosine intentionally omitted — needs explicit review before adding.
        help="Scheduler family. 'warmup_decay'=WarmupDecayLR (default). "
             "'constant'=WarmupLR (warmup to lr_max, then hold).")
    parser.add_argument("--warmup_min_lr", type=float, default=0.0,
        help="Floor LR for WarmupDecayLR decay. Default 0.0. "
             "IGNORED when scheduler_type=constant (a WARN is printed if non-zero).")

    parser.add_argument("--num_classes", type=int, default=3,
                       help="Number of classes for classification in stage 1")
    parser.add_argument("--use_stage1_cls", action="store_true", default=True,
                   help="Whether to use Stage 1 CLS token in Stage 2")
    parser.add_argument("--ce_loss_weight", default=1.0, type=float)
    parser.add_argument("--dice_loss_weight", default=1.0, type=float)
    parser.add_argument("--bce_loss_weight", default=1.0, type=float)
    parser.add_argument("--cls_loss_weight", default=1.0, type=float)
    parser.add_argument("--mask_loss_weight", default=1.0, type=float)
    parser.add_argument("--text_loss_weight", default=1.0, type=float, help="Weight for text generation loss")
    parser.add_argument("--lora_alpha", default=16, type=int)
    parser.add_argument("--lora_dropout", default=0.05, type=float)
    parser.add_argument("--lora_target_modules", default="q_proj,v_proj", type=str)
    parser.add_argument("--explanatory", default=0.1, type=float)
    parser.add_argument("--beta1", default=0.9, type=float)
    parser.add_argument("--beta2", default=0.95, type=float)
    parser.add_argument("--num_classes_per_sample", default=3, type=int)
    parser.add_argument("--exclude_val", action="store_true", default=False)
    parser.add_argument("--no_eval", action="store_true", default=False)
    parser.add_argument("--num_saves", default=10, type=int,
                        help="Number of evenly-spaced checkpoints (and validations) during training")
    parser.add_argument("--eval_only", action="store_true", default=False)
    parser.add_argument("--vision_pretrained", default="PATH_TO_SAM_ViT-H", type=str)
    parser.add_argument("--out_dim", default=256, type=int)
    parser.add_argument("--resume", default="", type=str)
    parser.add_argument("--print_freq", default=1, type=int)
    parser.add_argument("--start_epoch", default=0, type=int)
    parser.add_argument("--gradient_checkpointing", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--train_mask_decoder", action="store_true", default=True)
    parser.add_argument("--use_mm_start_end", action="store_true", default=True)
    parser.add_argument("--auto_resume", action="store_true", default=True)
    parser.add_argument("--no_auto_resume", dest="auto_resume", action="store_false",
                        help="Disable auto-resume from ckpt_model (use for clean/smoke runs)")
    parser.add_argument(
        "--conv_type",
        default="llava_v1",
        type=str,
        choices=["llava_v1", "llava_llama_2"],
    )
    
    parser.add_argument("--obj_pos_weight", type=float, default=60.0,
                        help="Scalar pos_weight for BCEWithLogits (None => auto-compute from batch).")
    parser.add_argument("--obj_pos_weight_max", type=float, default=100.0,
                        help="Clamp upper bound for auto-computed pos_weight.")

    parser.add_argument("--num_obj_classes", type=int, default=81,
                        help="Number of object categories for <OBJ> image-level classification")
    parser.add_argument("--obj_loss_weight", type=float, default=1.0,
                        help="Loss weight for <OBJ> image-level classification head")
    
    parser.add_argument("--obj_threshold", type=float, default=0.5,
                        help="Threshold for multi-label prediction on OBJ head")
    parser.add_argument("--log_obj_prefix", type=str, default="obj",
                        help="TensorBoard tag prefix for OBJ multi-label metrics")
    
    parser.add_argument(
        "--seg_prompt_mode",
        type=str,
        default="fuse",
        choices=["seg_only", "fuse", "text_only"],
        help="SAM prompt embedding mode for segmentation ablation."
    )

    parser.add_argument(
        "--mask_type",
        type=str,
        default="ours",
        choices=["ours", "others"],
        help="Mask type for loss computation: 'ours' uses gt_soft_mask, 'others' uses gt_mask."
    )

    parser.add_argument(
        "--balance_training",
        action="store_true",
        default=False,
        help=(
            "Use BalancedBatchSampler: each batch is real_ratio%% real + (1-real_ratio)%% tampered. "
            "Prevents CLS-head collapse when real:tampered ratio is severely skewed "
            "(e.g. balanced-all dataset ~1:4). Default: off."
        ),
    )

    parser.add_argument(
        "--real_ratio",
        type=float,
        default=0.5,
        help=(
            "Fraction of real samples per batch when --balance_training is active. "
            "E.g. 0.5 = 50%% real / 50%% tampered (default). "
            "With batch_size=2 any value in (0.25, 0.75) still yields 1:1. "
            "Ignored when --balance_training is not set."
        ),
    )

    parser.add_argument(
        "--source_weights_schedule",
        type=str,
        default=None,
        help=(
            "Epoch-interval source weights schedule (JSON). "
            "E.g. '{\"0-2\": {\"gemini\": 0}, \"2-end\": {\"gemini\": 33}}': "
            "epochs 0-1 exclude gemini, epochs 2-4 oversample gemini 33x. "
            "Intervals are [start, end) and must cover [0, epochs) contiguously. "
            "Combine with --balance_training for BT + Late Injection (mixed batch + "
            "source-weighted tampered slot). Requires mapping.json (auto-detected "
            "from dataset_dir or set via --mapping_json)."
        ),
    )

    parser.add_argument(
        "--mapping_json",
        type=str,
        default=None,
        help=(
            "Path to mapping.json for source label lookup used by "
            "--source_weights_schedule. Auto-detected from dataset_dir/mapping.json "
            "if not specified."
        ),
    )

    parser.add_argument(
        "--save_each_epoch",
        action="store_true",
        default=False,
        help=(
            "Save a DeepSpeed checkpoint after every N epochs to "
            "checkpoint_epoch_{N}/ under log_dir. N is controlled by --save_epoch_interval. "
            "Enables offline per-epoch evaluation (e.g. Exp-7 learning-curve analysis). "
            "Each checkpoint is ~31 GB; ensure sufficient disk space before enabling. "
            "Default: off."
        ),
    )

    parser.add_argument(
        "--save_epoch_interval",
        type=int,
        default=1,
        help=(
            "Save a checkpoint every this many epochs when --save_each_epoch is set. "
            "E.g. --save_epoch_interval 2 saves at epochs 2,4,6,... halving storage cost. "
            "Default: 1 (every epoch). Overridden by --save_epoch_list if set."
        ),
    )

    parser.add_argument(
        "--save_epoch_list",
        type=str,
        default=None,
        help=(
            "Comma-separated 1-indexed epoch numbers to save, e.g. '2,5,10'. "
            "When set, overrides --save_epoch_interval and saves only the listed "
            "epochs (plus final_checkpoint at end). Used by LR-Series Stage 1/2/3 "
            "(3-point eval at ep2/5/10) and Stage 4 (5-point eval at ep2/4/6/8/10). "
            "Default: None (use --save_epoch_interval)."
        ),
    )

    parser.add_argument(
        "--train_seed",
        type=int,
        default=None,
        help=(
            "Global random seed for reproducibility (sets torch, numpy, random). "
            "If not set, the default PyTorch/Python random state is used. "
            "E.g. --train_seed 42"
        ),
    )

    return parser.parse_args(args)


def _build_source_labels(dataset, mapping_json_path: str, local_rank: int = 0):
    """Per-sample source string from mapping.json ``type`` (strip "_coco_val_*"); missing => "unknown"."""
    with open(mapping_json_path) as f:
        mapping = json.load(f)

    labels = []
    for img_path in dataset.images:
        fname = os.path.basename(img_path)
        entry = mapping.get(fname, {})
        type_str = entry.get("type", "")
        source = type_str.split("_coco_val_", 1)[0] if type_str else "unknown"
        labels.append(source)

    if local_rank == 0:
        counts: dict = {}
        for s in labels:
            counts[s] = counts.get(s, 0) + 1
        print(f"[source_labels] distribution: {counts}")

    return labels


def main(args):
    args = parse_args(args)
    deepspeed.init_distributed()
    if args.train_seed is not None:
        random.seed(args.train_seed)
        np.random.seed(args.train_seed)
        torch.manual_seed(args.train_seed)
        torch.cuda.manual_seed_all(args.train_seed)
        if args.local_rank == 0:
            print(f"[seed] train_seed={args.train_seed} applied to random/numpy/torch")

    # Parse --save_epoch_list once (1-indexed epochs to save). None => use interval.
    args.save_epoch_set = None
    if args.save_epoch_list is not None:
        try:
            args.save_epoch_set = {int(x) for x in args.save_epoch_list.split(",") if x.strip()}
        except ValueError:
            raise ValueError(
                f"--save_epoch_list must be comma-separated integers, got: {args.save_epoch_list!r}"
            )
        if args.local_rank == 0:
            if args.save_epoch_interval != 1:
                print(f"[save_each_epoch] WARN: --save_epoch_list is set "
                      f"({sorted(args.save_epoch_set)}), overriding "
                      f"--save_epoch_interval={args.save_epoch_interval}")
            print(f"[save_each_epoch] save only at epochs: {sorted(args.save_epoch_set)}")
    args.log_dir = os.path.join(args.log_base_dir, args.exp_name)
    if args.local_rank == 0:
        os.makedirs(args.log_dir, exist_ok=True)
        writer = SummaryWriter(args.log_dir)
        # Persist all hyperparameters to a file alongside checkpoints
        hparam_path = os.path.join(args.log_dir, "hparams.txt")
        with open(hparam_path, "w") as _f:
            for k, v in sorted(vars(args).items()):
                _f.write(f"{k}: {v}\n")
        print("========== Hyperparameters ==========")
        for k, v in sorted(vars(args).items()):
            print(f"  {k}: {v}")
        print(f"=====================================")
        print(f"[log_dir] {args.log_dir}")
    else:
        writer = None

    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args.version,
        cache_dir=None,
        model_max_length=args.model_max_length,
        padding_side="right",
        use_fast=False,
    )

    tokenizer.pad_token = tokenizer.unk_token
    num_added_token = tokenizer.add_tokens("[CLS]")
    num_added_token = tokenizer.add_tokens("[SEG]")
    num_added_token = tokenizer.add_tokens("[OBJ]")
    num_added_token = tokenizer.add_tokens("[END]")

    args.cls_token_idx = tokenizer("[CLS]", add_special_tokens=False).input_ids[0]
    args.seg_token_idx = tokenizer("[SEG]", add_special_tokens=False).input_ids[0]
    args.obj_token_idx = tokenizer("[OBJ]", add_special_tokens=False).input_ids[0]
    args.end_token_idx = tokenizer("[END]", add_special_tokens=False).input_ids[0]
    if args.use_mm_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )

    model_args = {
        "train_mask_decoder": args.train_mask_decoder,
        "out_dim": args.out_dim,
        "cls_loss_weight": args.cls_loss_weight,
        "mask_loss_weight": args.mask_loss_weight,
        "ce_loss_weight": args.ce_loss_weight,
        "dice_loss_weight": args.dice_loss_weight,
        "bce_loss_weight": args.bce_loss_weight,
        "text_loss_weight": args.text_loss_weight,
        "cls_token_idx": args.cls_token_idx,
        "seg_token_idx": args.seg_token_idx,
        "obj_token_idx": args.obj_token_idx,
        "num_obj_classes": args.num_obj_classes,
        "obj_loss_weight": args.obj_loss_weight,
        "obj_pos_weight": args.obj_pos_weight,
        "obj_pos_weight_max": args.obj_pos_weight_max,
        "vision_pretrained": args.vision_pretrained,
        "vision_tower": args.vision_tower,
        "use_mm_start_end": args.use_mm_start_end,
        "seg_prompt_mode": args.seg_prompt_mode,
        "mask_type": args.mask_type,
    }
    torch_dtype = torch.float32
    if args.precision == "bf16":
        torch_dtype = torch.bfloat16
    elif args.precision == "fp16":
        torch_dtype = torch.half
    model = PIXARForCausalLM.from_pretrained(
        args.version, torch_dtype=torch_dtype, low_cpu_mem_usage=True, **model_args
    )

    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.bos_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    model.enable_input_require_grads()
    if args.gradient_checkpointing:
        model.gradient_checkpointing_enable()
    model.get_model().initialize_vision_modules(model.get_model().config)
    vision_tower = model.get_model().get_vision_tower()
    vision_tower.to(dtype=torch_dtype, device=args.local_rank)
    if not args.eval_only:
        model.get_model().initialize_pixar_modules(model.get_model().config)

    for p in vision_tower.parameters():
        p.requires_grad = False

    for p in model.get_model().mm_projector.parameters():
        p.requires_grad = False


    conversation_lib.default_conversation = conversation_lib.conv_templates[
        args.conv_type
    ]

    lora_r = args.lora_r
    if lora_r > 0:
        def find_linear_layers(model, lora_target_modules):
            cls = torch.nn.Linear
            lora_module_names = set()
            for name, module in model.named_modules():
                if (
                    isinstance(module, cls)
                    and all(
                        [
                            x not in name
                            for x in [
                                "visual_model",
                                "vision_tower",
                                "mm_projector",
                                "cls_head",
                                "obj_head",
                                "seg_proj",
                                "text_proj",
                                "gate_mlp",
                            ]
                        ]
                    )
                    and any([x in name for x in lora_target_modules])
                ):
                    lora_module_names.add(name)
            return sorted(list(lora_module_names))
        lora_alpha = args.lora_alpha
        lora_dropout = args.lora_dropout
        lora_target_modules = find_linear_layers(
                model, args.lora_target_modules.split(",")
        )
        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            target_modules=lora_target_modules,
            lora_dropout=lora_dropout,
            bias="none",
            task_type="CAUSAL_LM",
        )
        model = get_peft_model(model, lora_config)
        model.print_trainable_parameters()
    model.resize_token_embeddings(len(tokenizer))

    for n, p in model.named_parameters():
        if "lm_head" in n:
            p.requires_grad = False

    for n, p in model.named_parameters():
        if any(
            [
                x in n
                for x in ["embed_tokens", "mask_decoder", "cls_head", "obj_head", "seg_proj", "text_proj", "gate_mlp"]
            ]
        ):
            p.requires_grad = True

    print("Checking trainable parameters:")
    total_params = 0
    for n, p in model.named_parameters():
        if p.requires_grad:
            print(f"Trainable: {n} with {p.numel()} parameters")
            total_params += p.numel()
    print(f"Total trainable parameters: {total_params}")

    world_size = dist.get_world_size()
    args.distributed = world_size > 1
    train_dataset = CustomDataset(
        base_image_dir=args.dataset_dir,
        tokenizer=tokenizer,
        vision_tower=args.vision_tower,
        split="train",
        precision=args.precision,
        image_size=args.image_size,

    )
    print(f"\nInitializing datasets:")
    print(f"Training split size: {len(train_dataset)}")

    if args.no_eval == False:
        val_dataset = CustomDataset(
            base_image_dir=args.dataset_dir,
            tokenizer=tokenizer,
            vision_tower=args.vision_tower,
            split="validation",
            precision=args.precision,
            image_size=args.image_size,
    )
        print(
            f"Training with {len(train_dataset)} examples and validating with {len(val_dataset)} examples."
        )
    else:
        val_dataset = None
        print(f"Training with {len(train_dataset)} examples.")
    # ===== LR scheduler dispatch =====
    scheduler_total_num_steps = (
        args.scheduler_total_steps
        if args.scheduler_total_steps is not None
        else args.epochs * args.steps_per_epoch
    )
    if args.scheduler_type == "warmup_decay":
        scheduler_cfg = {
            "type": "WarmupDecayLR",
            "params": {
                "total_num_steps": scheduler_total_num_steps,
                "warmup_min_lr": 0 if args.warmup_min_lr == 0.0 else args.warmup_min_lr,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        }
    elif args.scheduler_type == "constant":
        if args.warmup_min_lr != 0.0 and args.local_rank == 0:
            print(f"[scheduler] WARN: scheduler_type=constant ignores warmup_min_lr "
                  f"(you passed {args.warmup_min_lr}). WarmupLR holds lr_max after warmup.")
        scheduler_cfg = {
            "type": "WarmupLR",
            "params": {
                "warmup_min_lr": 0.0,
                "warmup_max_lr": args.lr,
                "warmup_num_steps": 100,
                "warmup_type": "linear",
            },
        }
    if args.local_rank == 0:
        print("=" * 50)
        print(f"[scheduler] type={args.scheduler_type}")
        print(f"[scheduler] total_num_steps={scheduler_total_num_steps}  "
              f"(epochs={args.epochs} × steps_per_epoch={args.steps_per_epoch}; "
              f"overridden={args.scheduler_total_steps is not None})")
        print(f"[scheduler] warmup_max_lr={args.lr}  warmup_min_lr={args.warmup_min_lr}  "
              f"warmup_num_steps=100")
        print("=" * 50)

    ds_config = {
        "train_micro_batch_size_per_gpu": args.batch_size,
        "gradient_accumulation_steps": args.grad_accumulation_steps,
        "optimizer": {
            "type": "AdamW",
            "params": {
                "lr": args.lr,
                "weight_decay": 0.0,
                "betas": (args.beta1, args.beta2),
            },
        },
        "scheduler": scheduler_cfg,
        "fp16": {
            "enabled": args.precision == "fp16",
            "loss_scale": 0,  # Dynamic loss scaling
            "initial_scale_power": 12,
            "loss_scale_window": 1000,
            "min_loss_scale": 1,
            "hysteresis": 2
        },
        "bf16": {
            "enabled": args.precision == "bf16",
        },
        "gradient_clipping": 1.0,
        "zero_optimization": {
            "stage": 2,
            "contiguous_gradients": True,
            "overlap_comm": True,
            "reduce_scatter": True,
            "reduce_bucket_size": 5e8,
            "allgather_bucket_size": 5e8,
        },
    }
    _schedule = None
    _source_labels = None
    if args.source_weights_schedule:
        _schedule = parse_weights_schedule(args.source_weights_schedule, args.epochs)
        mapping_path = args.mapping_json or os.path.join(args.dataset_dir, "mapping.json")
        _source_labels = _build_source_labels(train_dataset, mapping_path, args.local_rank)
        known_sources = set(_source_labels)
        for _, _, w in _schedule:
            unknown = set(w.keys()) - known_sources
            if unknown:
                raise ValueError(
                    f"--source_weights_schedule references unknown sources: {unknown}. "
                    f"Known sources in dataset: {known_sources}"
                )
        if not args.balance_training:
            raise ValueError(
                "--source_weights_schedule currently requires --balance_training."
            )
        _initial_weights = _schedule[0][2]
        batch_sampler = BalancedSourceWeightedSampler(
            dataset=train_dataset,
            source_labels=_source_labels,
            source_weights=_initial_weights,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            world_size=dist.get_world_size(),
            rank=args.local_rank,
            real_ratio=args.real_ratio,
        )
        if args.local_rank == 0:
            print(f"[INFO] BT + source_weights_schedule active: {_schedule}")
    elif args.balance_training:
        batch_sampler = BalancedBatchSampler(
            dataset=train_dataset,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            world_size=dist.get_world_size(),
            rank=args.local_rank,
            real_ratio=args.real_ratio,
        )
        if args.local_rank == 0:
            print(f"[INFO] balance_training=True: using BalancedBatchSampler "
                  f"(real_ratio={args.real_ratio:.2f})")
    else:
        batch_sampler = BatchSampler(
            dataset=train_dataset,
            batch_size=ds_config["train_micro_batch_size_per_gpu"],
            world_size=dist.get_world_size(),
            rank=args.local_rank,
        )

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_sampler=batch_sampler,
        num_workers=args.workers,
        pin_memory=True,
        collate_fn=partial(
            collate_fn,
            tokenizer=tokenizer,
            conv_type=args.conv_type,
            use_mm_start_end=args.use_mm_start_end,
            local_rank=args.local_rank,
            cls_token_idx=args.cls_token_idx,
            obj_token_idx=args.obj_token_idx,
            seg_token_idx=args.seg_token_idx,
        ),
    )
    model_engine, optimizer, _, scheduler = deepspeed.initialize(
        model=model,
        model_parameters=model.parameters(),
        config=ds_config,
        training_data=None,  # Set to None since we're providing our own loader
    )

    if args.auto_resume and len(args.resume) == 0:
        resume = os.path.join(args.log_dir,  "ckpt_model")
        if os.path.exists(resume):
            args.resume = resume

    if args.resume:
        load_path, client_state = model_engine.load_checkpoint(args.resume)
        with open(os.path.join(args.resume, "latest"), "r") as f:
            ckpt_dir = f.readlines()[0].strip()
        args.start_epoch = (
            int(ckpt_dir.replace("global_step", "")) // args.steps_per_epoch
        )
        if args.start_epoch >= args.epochs:
            if args.local_rank == 0:
                print(f"[WARN] Checkpoint at epoch {args.start_epoch} already >= total epochs "
                      f"{args.epochs}. Training would be a no-op. Pass --no_auto_resume to "
                      f"run a fresh experiment instead.")
            args.start_epoch = 0
            args.resume = ""
        print(
            "resume training from {}, start from epoch {}".format(
                args.resume, args.start_epoch
            )
        )

    # validation dataset
    if val_dataset is not None:
        val_sampler = BatchSampler(
            dataset=val_dataset,
            batch_size=args.val_batch_size,
            world_size=dist.get_world_size(),
            rank=args.local_rank
        )
        val_loader = torch.utils.data.DataLoader(
            val_dataset,
            batch_sampler=val_sampler,
            num_workers=args.workers,
            pin_memory=True,
            collate_fn=partial(
                 collate_fn,
                 tokenizer=tokenizer,
                 conv_type=args.conv_type,
                 use_mm_start_end=args.use_mm_start_end,
                 local_rank=args.local_rank,
                 cls_token_idx=args.cls_token_idx,
                 obj_token_idx=args.obj_token_idx,
                 seg_token_idx=args.seg_token_idx,
             ),
        )

    train_iter = iter(train_loader)

    best_acc, best_score, cur_ciou = 0.0, 0.0, 0.0

    if args.eval_only:
        acc, giou, ciou, _ = validate(val_loader, model_engine, 0, writer, args)  # Classification validation
        exit()

    num_saves = args.num_saves
    step = max(1, args.epochs // num_saves)
    validation_epochs = list(range(step, args.epochs, step))
    if args.epochs not in validation_epochs:
        validation_epochs.append(args.epochs)  # always include last epoch
    if args.local_rank == 0:
        print(f"\nTraining Configuration:")
        print(f"Total epochs: {args.epochs}")
        print(f"Validation will be performed after epochs: {validation_epochs}")
    for epoch in range(args.start_epoch, args.epochs):
        batch_sampler.set_epoch(epoch)
        if _schedule is not None:
            new_weights = next(w for s, e, w in _schedule if s <= epoch < e)
            if new_weights != batch_sampler.source_weights:
                batch_sampler.set_source_weights(new_weights)
                # rebuild iter so the new pool size takes effect
                train_iter = iter(train_loader)
                if args.local_rank == 0:
                    print(f"[schedule] epoch {epoch}: source_weights → {new_weights}  "
                          f"(tampered_expanded_len={batch_sampler._tampered_expanded_len})")
        epoch_start = time.time()
        if args.local_rank == 0:
            print(f"\n[Epoch {epoch + 1}/{args.epochs}] started at {time.strftime('%Y-%m-%d %H:%M:%S')}")
        # train for one epoch
        train_iter = train(
            train_loader,
            model_engine,
            epoch,
            scheduler,
            writer,
            train_iter,
            args,
        )
        epoch_elapsed = time.time() - epoch_start
        # total optimizer steps this epoch × samples processed per step
        samp_per_step = args.batch_size * dist.get_world_size() * args.grad_accumulation_steps
        total_samp = args.steps_per_epoch * samp_per_step
        epoch_throughput = total_samp / epoch_elapsed
        if args.local_rank == 0:
            print(f"[Epoch {epoch + 1}/{args.epochs}] done  "
                  f"elapsed={epoch_elapsed:.0f}s  "
                  f"~{epoch_throughput:.1f} samples/s  "
                  f"({total_samp} samples / {args.steps_per_epoch} opt-steps)")
            writer.add_scalar("epoch/time_s", epoch_elapsed, epoch + 1)
            writer.add_scalar("epoch/throughput_samp_per_s", epoch_throughput, epoch + 1)
        if (epoch + 1) in validation_epochs:  # +1 because epoch starts from 0
            if args.local_rank == 0:
                print(f"\nPerforming validation after epoch {epoch + 1}")

            if args.no_eval == False:
                acc, giou, ciou, _ = validate(val_loader, model_engine, epoch, writer, args)
                is_best_iou = giou > best_score
                is_best_acc = acc > best_acc
                best_score = max(giou, best_score)
                best_acc = max(acc, best_acc)
                cur_ciou = ciou if is_best_iou else cur_ciou
                cur_acc = acc if is_best_acc else cur_acc
                is_best = is_best_iou or is_best_acc
            else:
                acc, giou, ciou = -1.0, -1.0, -1.0
                is_best = False

            if args.local_rank == 0:
                print(f"Current accuracy: {acc:.2f}%, Best accuracy: {best_acc:.2f}%")
                print(f"Current iou: {cur_ciou:.2f}%, Best score: {best_score:.2f}%")
            # Save best checkpoint only when validation is enabled (no_eval=True means
            # no metric to judge "best"; final_checkpoint already captures the final state)
            if not args.no_eval and is_best:
                save_dir = os.path.join(args.log_dir, "ckpt_model")
                if args.local_rank == 0:
                    torch.save(
                                {"epoch": epoch},
                                os.path.join(
                                    args.log_dir,
                                    f"meta_log_acc{best_acc:.3f}_iou{best_score:.3f}.pth"
                                ),
                    )
                    if os.path.exists(save_dir):
                        shutil.rmtree(save_dir)
                torch.distributed.barrier()
                model_engine.save_checkpoint(save_dir)
                torch.distributed.barrier()
        else:
            if args.local_rank == 0:
                print(f"Epoch {epoch + 1} completed. Skipping validation.")

        # Per-epoch checkpoint (--save_each_epoch, e.g. for Exp-7 learning-curve analysis)
        if args.save_each_epoch:
            if args.save_epoch_set is not None:
                should_save = (epoch + 1) in args.save_epoch_set
            else:
                should_save = (epoch + 1) % args.save_epoch_interval == 0
        else:
            should_save = False
        if should_save:
            epoch_save_dir = os.path.join(args.log_dir, f"checkpoint_epoch_{epoch + 1}")
            if args.local_rank == 0:
                if os.path.exists(epoch_save_dir):
                    shutil.rmtree(epoch_save_dir)
                print(f"[save_each_epoch] saving checkpoint_epoch_{epoch + 1} ...")
            torch.distributed.barrier()
            model_engine.save_checkpoint(epoch_save_dir)
            torch.distributed.barrier()

        # Save final epoch regardless of validation
        if epoch == args.epochs - 1:
            save_dir = os.path.join(args.log_dir, "final_checkpoint")
            if args.local_rank == 0:
                if os.path.exists(save_dir):
                    shutil.rmtree(save_dir)
            torch.distributed.barrier()
            model_engine.save_checkpoint(save_dir)
            if args.local_rank == 0:
                print(f"\nTraining completed. Final checkpoint saved to {save_dir}")

def train(
    train_loader,
    model,
    epoch,
    scheduler,
    writer,
    train_iter,
    args,
):
    """Main training loop."""
    batch_time = AverageMeter("Time", ":6.3f")
    data_time = AverageMeter("Data", ":6.3f")
    throughput = AverageMeter("Samp/s", ":7.1f")
    losses = AverageMeter("Loss", ":.4f")
    cls_losses = AverageMeter("ClsLoss", ":.4f")
    mask_bce_losses = AverageMeter("MaskBCELoss", ":.4f")
    mask_dice_losses = AverageMeter("MaskDICELoss", ":.4f")
    mask_losses = AverageMeter("MaskLoss", ":.4f")
    obj_losses = AverageMeter("ObjLoss", ":.4f")
    text_losses = AverageMeter("TextLoss", ":.4f")
    # samples processed per optimizer step across all GPUs (including grad accum)
    _samp_per_step = args.batch_size * dist.get_world_size() * args.grad_accumulation_steps
    progress = ProgressMeter(
        args.steps_per_epoch,
        [batch_time, throughput, losses, cls_losses, mask_bce_losses, mask_dice_losses, mask_losses, obj_losses, text_losses],
        prefix="Epoch: [{}]".format(epoch),
    )
    model.train()
    end = time.time()
    # total micro-batches per epoch = steps_per_epoch * grad_accumulation_steps
    # DeepSpeed handles gradient accumulation internally: it accumulates gradients
    # over grad_accumulation_steps micro-batches and then does all-reduce + optimizer step.
    total_micro_steps = args.steps_per_epoch * args.grad_accumulation_steps
    for global_step in range(total_micro_steps):
        try:
            input_dict = next(train_iter)
        except:
            train_iter = iter(train_loader)
            input_dict = next(train_iter)

        data_time.update(time.time() - end)
        input_dict = dict_to_cuda(input_dict)
        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        output_dict = model(**input_dict)
        loss = output_dict["loss"]
        cls_loss = output_dict["cls_loss"]
        mask_bce_loss = output_dict["mask_bce_loss"]
        mask_dice_loss = output_dict["mask_dice_loss"]
        mask_loss = output_dict["mask_loss"]
        obj_loss = output_dict.get("obj_loss", torch.tensor(0.0, device=loss.device))
        text_loss = output_dict.get("text_loss", torch.tensor(0.0, device=loss.device))
        losses.update(loss.item(), input_dict["images"].size(0))
        cls_losses.update(cls_loss.item(), input_dict["images"].size(0))
        # Use tampered count as denominator: mask/obj losses are computed only over
        # tampered samples, so the meter must reflect that — not the full batch size.
        n_tampered = int((input_dict['cls_labels'] == 2).sum().item())
        if n_tampered > 0:
            mask_bce_losses.update(mask_bce_loss.item(), n_tampered)
            mask_dice_losses.update(mask_dice_loss.item(), n_tampered)
            mask_losses.update(mask_loss.item(), n_tampered)
            obj_losses.update(obj_loss.item(), n_tampered)
        if text_loss.item() > 0:
            text_losses.update(text_loss.item(), input_dict["images"].size(0))

        model.backward(loss)
        model.step()

        # Log at optimizer step boundaries (every grad_accumulation_steps micro-batches)
        optimizer_step = global_step // args.grad_accumulation_steps
        is_optimizer_step = (global_step + 1) % args.grad_accumulation_steps == 0

        if is_optimizer_step:
            step_time = time.time() - end
            batch_time.update(step_time)
            throughput.update(_samp_per_step / max(step_time, 1e-6))
            end = time.time()

            if optimizer_step % args.print_freq == 0:
                if args.distributed:
                    batch_time.all_reduce()
                    data_time.all_reduce()
                    throughput.all_reduce()
                    losses.all_reduce()
                    cls_losses.all_reduce()
                    mask_bce_losses.all_reduce()
                    mask_dice_losses.all_reduce()
                    mask_losses.all_reduce()
                    obj_losses.all_reduce()
                    text_losses.all_reduce()

                if args.local_rank == 0:
                    progress.display(optimizer_step + 1)
                    writer.add_scalar("train/loss", losses.avg, optimizer_step)
                    writer.add_scalar("train/cls_loss", cls_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_bce_loss", mask_bce_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_dice_loss", mask_dice_losses.avg, optimizer_step)
                    writer.add_scalar("train/mask_loss", mask_losses.avg, optimizer_step)
                    writer.add_scalar("train/obj_loss", obj_losses.avg, optimizer_step)
                    writer.add_scalar("train/text_loss", text_losses.avg, optimizer_step)
                    writer.add_scalar("perf/step_time_s", batch_time.avg, optimizer_step)
                    writer.add_scalar("perf/data_time_s", data_time.avg, optimizer_step)
                    writer.add_scalar("perf/throughput_samp_per_s", throughput.avg, optimizer_step)
                batch_time.reset()
                data_time.reset()
                throughput.reset()
                losses.reset()
                cls_losses.reset()
                mask_bce_losses.reset()
                mask_dice_losses.reset()
                mask_losses.reset()
                obj_losses.reset()
                text_losses.reset()

            if optimizer_step != 0:
                curr_lr = scheduler.get_last_lr()
                if args.local_rank == 0:
                    writer.add_scalar("train/lr", curr_lr[0], optimizer_step)

    return train_iter
import random

def validate(val_loader, model_engine, epoch, writer, args, sample_ratio=None):
    """
    Validate the model with option for random sampling.
    Args:
        sample_ratio: if None, use all data; if float (e.g., 0.1), randomly sample that portion
    """
    model_engine.eval()
    acc = MetricsAccumulator()
    num_classes = 3
    class_names = ['Real', 'Full Synthetic', 'Tampered']

    # Calculate total number of batches and samples to use
    total_batches = len(val_loader)
    if sample_ratio is not None:
        num_batches = max(1, int(total_batches * sample_ratio))
        sample_indices = set(random.sample(range(total_batches), num_batches))
        print(f"\nValidating on {num_batches}/{total_batches} randomly sampled batches...")

    for batch_idx, input_dict in enumerate(tqdm.tqdm(val_loader)):
        if sample_ratio is not None and batch_idx not in sample_indices:
            continue
        if batch_idx == 0:
            print("\nFirst validation batch details:")
            for key, value in input_dict.items():
                if isinstance(value, torch.Tensor):
                    print(f"{key} shape: {value.shape}")
                elif isinstance(value, list):
                    print(f"{key} length: {len(value)}")

        torch.cuda.empty_cache()
        input_dict = dict_to_cuda(input_dict)

        if acc.total == 0:
            print("\nProcessing first batch:")
            print("Input dict keys:", input_dict.keys())

        if args.precision == "fp16":
            input_dict["images"] = input_dict["images"].half()
            input_dict["images_clip"] = input_dict["images_clip"].half()
        elif args.precision == "bf16":
            input_dict["images"] = input_dict["images"].bfloat16()
            input_dict["images_clip"] = input_dict["images_clip"].bfloat16()
        else:
            input_dict["images"] = input_dict["images"].float()
            input_dict["images_clip"] = input_dict["images_clip"].float()
        input_dict['inference'] = True

        with torch.no_grad():
            output_dict = model_engine(**input_dict)

        # Classification
        logits = output_dict["logits"]
        probs_cls = F.softmax(logits, dim=1)
        preds = torch.argmax(probs_cls, dim=1)
        cls_labels = input_dict["cls_labels"]
        for pred_i, gt_i in zip(preds, cls_labels):
            acc.update_cls(int(pred_i.item()), int(gt_i.item()))

        # OBJ multi-label (tampered samples only)
        if ("obj_logits" in output_dict) and ("obj_labels" in input_dict):
            tampered_mask = (cls_labels == 2)
            if tampered_mask.any():
                gt_obj   = input_dict["obj_labels"][tampered_mask]
                probs_obj = output_dict["obj_logits"][tampered_mask].sigmoid()
                acc.update_obj(probs_obj, gt_obj, threshold=args.obj_threshold)

        # Segmentation (first sample in batch, tampered only — batch_size=1 assumption)
        if cls_labels[0] == 2:
            pred_masks = output_dict["pred_masks"]
            masks_list = output_dict["gt_soft_masks"][0].int()
            output_list = (pred_masks[0] > 0).int()
            assert len(pred_masks) == 1
            # pred_scores=None → skip pixel TP/FP/FN and ROC (not needed during training)
            acc.update_seg(output_list, masks_list, pred_scores=None)

    # Distributed reduction of segmentation meters
    acc.all_reduce_seg()

    # Compute final metrics
    raw     = acc.to_dict()
    metrics = compute_metrics(raw)

    accuracy         = metrics["accuracy"]
    giou             = metrics["giou"]
    ciou             = metrics["ciou"]
    per_class_metrics = metrics["per_class_metrics"]
    iou              = ciou
    f1_score         = metrics["combined_f1"]
    obj_micro_prec   = metrics["_obj_micro_prec"]
    obj_micro_rec    = metrics["_obj_micro_rec"]
    obj_micro_f1     = metrics["obj_micro_f1"]
    obj_macro_prec   = metrics["_obj_macro_prec"]
    obj_macro_rec    = metrics["_obj_macro_rec"]
    obj_macro_f1     = metrics["obj_macro_f1"]
    obj_subset_acc   = metrics["_obj_subset_acc"]

    # Pixel accuracy (ciou proxy, TensorBoard only)
    pixel_accuracy = ciou * 100.0

    # Approximate AUC (TensorBoard only)
    avg_precision = np.mean([m['precision'] for m in per_class_metrics.values()])
    avg_recall    = np.mean([m['recall']    for m in per_class_metrics.values()])
    auc_approx    = avg_precision * avg_recall

    # Log metrics (rank-0 only)
    if args.local_rank == 0:
        writer.add_scalar("val/accuracy",      accuracy,      epoch)
        writer.add_scalar("val/giou",          giou,          epoch)
        writer.add_scalar("val/ciou",          ciou,          epoch)
        writer.add_scalar("val/pixel_accuracy", pixel_accuracy, epoch)
        writer.add_scalar("val/iou",           iou,           epoch)
        writer.add_scalar("val/f1_score",      f1_score,      epoch)
        writer.add_scalar("val/auc_approx",    auc_approx,    epoch)
        pfx = args.log_obj_prefix
        writer.add_scalar(f"val/{pfx}_micro_precision", obj_micro_prec, epoch)
        writer.add_scalar(f"val/{pfx}_micro_recall",    obj_micro_rec,  epoch)
        writer.add_scalar(f"val/{pfx}_micro_f1",        obj_micro_f1,   epoch)
        writer.add_scalar(f"val/{pfx}_subset_acc",      obj_subset_acc, epoch)
        writer.add_scalar(f"val/{pfx}_macro_precision", obj_macro_prec, epoch)
        writer.add_scalar(f"val/{pfx}_macro_recall",    obj_macro_rec,  epoch)
        writer.add_scalar(f"val/{pfx}_macro_f1",        obj_macro_f1,   epoch)
        for class_name, m in per_class_metrics.items():
            for metric_name, value in m.items():
                writer.add_scalar(
                    f"val/{class_name.lower().replace('/', '_')}_{metric_name}", value, epoch
                )

        validation_type = "Full" if sample_ratio is None else f"Sampled ({sample_ratio*100}%)"
        print(f"\n{validation_type} Validation Results:")
        print(f"giou: {giou:.4f}, ciou: {ciou:.4f}")
        print(f"Classification Accuracy: {accuracy:.4f}%")
        print(f"Pixel Accuracy: {pixel_accuracy:.4f}%")
        print(f"IoU: {iou:.4f}")
        print(f"F1 Score: {f1_score:.4f}")
        print(f"Approximate AUC: {auc_approx:.4f}")
        print(f"Total correct classifications: {acc.correct}")
        print(f"Total classification samples: {acc.total}")
        print("\n[OBJ] Multi-Label Metrics:")
        print(f"  threshold: {args.obj_threshold:.2f}")
        print(f"  micro  - P: {obj_micro_prec:.4f}, R: {obj_micro_rec:.4f}, F1: {obj_micro_f1:.4f}")
        print(f"  macro  - P: {obj_macro_prec:.4f}, R: {obj_macro_rec:.4f}, F1: {obj_macro_f1:.4f}")
        print(f"  subset - Acc: {obj_subset_acc:.4f}")
        print("\nPer-Class Metrics:")
        for class_name, m in per_class_metrics.items():
            print(f"\n{class_name}:")
            print(f"  Accuracy:  {m['accuracy']:.4f}")
            print(f"  Precision: {m['precision']:.4f}")
            print(f"  Recall:    {m['recall']:.4f}")
            print(f"  F1 Score:  {m['f1']:.4f}")

        cm = np.array(metrics["_confusion_matrix"])
        print("\nConfusion Matrix:")
        print("Predicted ")
        print("Actual ")
        print(f"{'':20}", end="")
        for name in class_names:
            print(f"{name:>12}", end="")
        print()
        for i, class_name in enumerate(class_names):
            print(f"{class_name:20}", end="")
            for j in range(num_classes):
                print(f"{cm[i, j]:12.0f}", end="")
            print()

    return accuracy, giou, ciou, per_class_metrics

if __name__ == "__main__":
    main(sys.argv[1:])
