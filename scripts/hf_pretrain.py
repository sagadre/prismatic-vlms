"""
hf_pretrain.py

Pretraining script for Prismatic VLM pretraining in native PyTorch, using Fully-Sharded Data Parallel (FSDP) to run
distributed training across GPUs. By default, assumes that CUDA toolkit is >= 11.0 (to support BF16 mixed precision).

Notes & Prerequisites:
    - We're loading Llama-2 (and possibly other) gated models from HuggingFace (HF Hub); these require an auth_token.
      For Llama-2, make sure to first get Meta approval, then fill out the form at the top of the HF Llama-2 page:
        => Link: https://huggingface.co/meta-llama/Llama-2-7b-chat-hf
        => Generate Token (from `huggingface.co`): Settings / Access Tokens / New "Read" Token
        => Set `cfg.hf_token` to file path with token (as single line text file) or raw string

    - If you want to set a custom location for all HF / TIMM artifacts --> `export HF_HOME="<PATH>"` *before* running!
        => For example (add to end of .bashrc): `export HF_HOME="/mnt/fsx/skaramcheti/cache"`

Run with:
    - [Single Node One-GPU (Debug)] : torchrun --standalone --nnodes 1 --nproc-per-node 1 scripts/pretrain.py
    - [Single Node Multi-GPU (= $K)]: torchrun --standalone --nnodes 1 --nproc-per-node $K scripts/pretrain.py
    - [Multi-Node/AWS Sagemaker] Depends on your individual setup; file an issue if you have trouble!
"""

import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Tuple, Union

import draccus
import torch
import torch.distributed as dist
import yaml

from prismatic.conf import DatasetConfig, DatasetRegistry, ModelConfig, ModelRegistry
from prismatic.data import get_dataset_and_collator
from prismatic.models import PrismaticConfig, PrismaticForVision2Seq
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_prismatic_processor, get_prompt_builder_fn
from prismatic.training import Metrics, get_train_strategy
from prismatic.util import set_global_seed

# Disable Tokenizers Parallelism to Play Nice w/ PyTorch Multiprocessing DataLoaders
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


@dataclass
class PretrainConfig:
    # fmt: off

    # ModelConfig (`prismatic/conf/models.py`); override with --model.type `ModelRegistry.<MODEL>.model_id`
    model: ModelConfig = field(
        # default_factory=ModelConfig.get_choice_class(ModelRegistry.EXP_SIGLIP_224PX.model_id)
        default_factory=ModelConfig.get_choice_class(ModelRegistry.PRISM_DINOSIGLIP_224PX_CONTROLLED_7B.model_id)
    )

    # DatasetConfig (`prismatic/conf/datasets.py`); override with --dataset.type `DatasetRegistry.<DATASET>.dataset_id`
    dataset: DatasetConfig = field(
        default_factory=DatasetConfig.get_choice_class(DatasetRegistry.LLAVA_V15.dataset_id)
    )

    # Pretraining Stage in < align (projector-only) | finetune (projector + LLM) | full-finetune (all) >
    # ---
    stage: str = "finetune"                                         # Pretraining Stage in < align | finetune >
    pretrained_checkpoint: Optional[Path] = None                    # Pretrained "align" Checkpoint to Load (if any)

    # Run Arguments
    run_id: Optional[str] = None                                    # Run ID for logging, Weights & Biases
    run_root_dir: Path = Path("/mnt/fsx/x-prismatic-vlms/hf-runs")  # Path to directory to store logs & checkpoints
    seed: int = 7                                                   # Random seed (for reproducibility)

    # HF Hub Credentials (for any gated models)
    hf_token: Union[str, Path] = Path(".hf_token")                  # Path to HF Token (or token as string)

    # Tracking Parameters
    trackers: Tuple[str, ...] = ("jsonl", "wandb")                  # Trackers to initialize (if W&B, add config!)
    wandb_project: str = "onyx-vlms"                                # Name of W&B project (default: `prismatic`)
    wandb_entity: Optional[str] = "stanford-voltron"                # Name of W&B entity (default: None)

    def __post_init__(self) -> None:
        """Set optimization parameters based on `stage` in {"align", "finetune"}."""
        if self.stage == "align":
            self.epochs = self.model.align_epochs
            self.max_steps = self.model.align_max_steps
            self.global_batch_size = self.model.align_global_batch_size
            self.per_device_batch_size = self.model.align_per_device_batch_size

            self.learning_rate = self.model.align_learning_rate
            self.weight_decay = self.model.align_weight_decay
            self.max_grad_norm = self.model.align_max_grad_norm
            self.lr_scheduler_type = self.model.align_lr_scheduler_type
            self.warmup_ratio = self.model.align_warmup_ratio

            self.train_strategy = self.model.align_train_strategy

        elif self.stage.endswith("finetune"):
            self.epochs = self.model.finetune_epochs
            self.max_steps = self.model.finetune_max_steps
            self.global_batch_size = self.model.finetune_global_batch_size
            self.per_device_batch_size = self.model.finetune_per_device_batch_size

            self.learning_rate = self.model.finetune_learning_rate
            self.weight_decay = self.model.finetune_weight_decay
            self.max_grad_norm = self.model.finetune_max_grad_norm
            self.lr_scheduler_type = self.model.finetune_lr_scheduler_type
            self.warmup_ratio = self.model.finetune_warmup_ratio

            self.train_strategy = self.model.finetune_train_strategy

        else:
            raise ValueError(f"Stage `{self.stage}` is not supported!")

    # fmt: on


@draccus.wrap()
def hf_pretrain(cfg: PretrainConfig) -> None:
    overwatch.info("Prismatic VLM Training :: Gathering Light")

    # Note => Under `torchrun` initializing `overwatch` will automatically set up `torch.distributed`
    torch.cuda.set_device(device_id := (overwatch.local_rank()))
    torch.cuda.empty_cache()

    # Create Unique Run Name & Save Directory
    model_id, dataset_id = cfg.model.model_id, cfg.dataset.dataset_id
    if dataset_id == "llava-v15":
        cfg.run_id = f"{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id
    else:
        cfg.run_id = f"{dataset_id}+{model_id}+stage-{cfg.stage}+x{cfg.seed}" if cfg.run_id is None else cfg.run_id

    # Start =>> Build Directories and Set Randomness
    overwatch.info('"Life is like a prism; what you see depends on how you turn the glass."', ctx_level=1)
    os.environ["HF_TOKEN"] = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else cfg.hf_token
    worker_init_fn = set_global_seed(cfg.seed, get_worker_init_fn=True)
    os.makedirs(run_dir := (cfg.run_root_dir / cfg.run_id), exist_ok=True)
    os.makedirs(cfg.run_root_dir / cfg.run_id / "checkpoints", exist_ok=True)
    if overwatch.is_rank_zero():
        draccus.dump(cfg, open(run_dir / "pretrain-config.yaml", "w"))
        with open(run_dir / "pretrain-config.yaml", "r") as f, open(run_dir / "pretrain-config.json", "w") as g:
            yaml_cfg = yaml.safe_load(f)
            json.dump(yaml_cfg, g, indent=2)

    # Create PrismaticConfig (`transformers.PretrainedConfig`) =>> used to fully specify a VLM instance
    overwatch.info(f"Building VLM Config with `{cfg.model.vision_backbone_id = }` and `{cfg.model.llm_backbone_id = }")
    vlm_config = PrismaticConfig(
        vision_backbone_id=cfg.model.vision_backbone_id,
        llm_backbone_id=cfg.model.llm_backbone_id,
        llm_max_length=cfg.model.llm_max_length,
        attn_implementation="flash_attention_2",
    )

    # Instantiate VLM =>> automatically loads pretrained weights for `vision_backbone` and `language_model`
    overwatch.info(f"Instantiating VLM `{model_id}` for Training Stage `{cfg.stage}`")
    vlm = (
        PrismaticForVision2Seq(vlm_config, load_pretrained_backbones=True)
        if cfg.stage != "align"
        else PrismaticForVision2Seq.from_pretrained(cfg.pretrained_checkpoint)
    )

    # Create PrismaticProcessor (initializes and wraps both `image_processor` and `tokenizer`)
    overwatch.info("Building `PrismaticProcessor` =>> Wraps Prismatic Image Tokenizer and Base LLM Tokenizer")
    processor = get_prismatic_processor(
        use_fused_vision_backbone=vlm_config.use_fused_vision_backbone,
        image_resize_strategy=cfg.model.image_resize_strategy,
        vision_backbone_cfgs=vlm.get_vision_backbone_cfgs(),
        llm_hf_hub_path=vlm_config.llm_hf_hub_path,
        llm_max_length=vlm_config.llm_max_length,
    )

    # [Explicit] Call to `freeze_backbones` here for clarity => will log exactly what is frozen / what's not!
    overwatch.info(f"Invoking `VLM.freeze_backbones()` for `{model_id}` => Training Stage: `{cfg.stage}`")
    vlm.freeze_backbones(cfg.stage)

    # Get Dataset for Specified Stage
    overwatch.info(f"Creating Dataset `{cfg.dataset.dataset_id}` => Stage: `{cfg.stage}`")
    train_dataset, collator = get_dataset_and_collator(
        cfg.stage,
        cfg.dataset,
        processor.image_processor.apply_transform,
        processor.tokenizer,
        prompt_builder_fn=get_prompt_builder_fn(vlm_config.llm_backbone_id),
        default_image_resolution=vlm.vision_backbone.default_image_resolution,
    )

    # Create Train Strategy
    overwatch.info(f"Initializing Train Strategy `{cfg.train_strategy}`")
    train_strategy = get_train_strategy(
        cfg.train_strategy,
        vlm,
        device_id,
        cfg.epochs,
        cfg.max_steps,
        cfg.global_batch_size,
        cfg.per_device_batch_size,
        cfg.learning_rate,
        cfg.weight_decay,
        cfg.max_grad_norm,
        cfg.lr_scheduler_type,
        cfg.warmup_ratio,
        enable_gradient_checkpointing=cfg.model.enable_gradient_checkpointing,
        enable_mixed_precision_training=cfg.model.enable_mixed_precision_training,
        reduce_in_full_precision=cfg.model.reduce_in_full_precision,
        worker_init_fn=worker_init_fn,
    )
    train_strategy.run_setup(run_dir, n_train_examples=len(train_dataset))

    # Create Metrics =>> Handles on the fly tracking, logging to specified trackers (e.g., JSONL, Weights & Biases)
    overwatch.info(f"Creating Metrics with Active Trackers => `{cfg.trackers}`")
    metrics = Metrics(
        cfg.trackers,
        cfg.run_id,
        run_dir,
        draccus.encode(cfg),
        cfg.stage,
        wandb_project=cfg.wandb_project,
        wandb_entity=cfg.wandb_entity,
        grad_accumulation_steps=train_strategy.grad_accumulation_steps,
    )

    # Run Training
    overwatch.info("Starting Training Loop")
    train_strategy.run_training(train_dataset, collator, metrics, stage=cfg.stage, seed=cfg.seed)

    # Finalize
    overwatch.info("Done with Training =>> Finalizing Metrics")
    metrics.finalize()

    # Invoke HF `.save_pretrained()` on Config, Processor, and VLM for easy export/import
    overwatch.info(f"Saving HF Config, Processor, and VLM to `{run_dir / 'hf-export'}`")
    full_vlm_state_dict = train_strategy.get_state_dict()
    if overwatch.is_rank_zero():
        processor.save_pretrained(run_dir / "hf-export")
        vlm.save_pretrained(run_dir / "hf-export", state_dict=full_vlm_state_dict)

    # And... we're done!
    overwatch.info("... and that's all, folks!")
    dist.barrier()
    dist.destroy_process_group()


if __name__ == "__main__":
    hf_pretrain()
