"""
launch.py

Utility script for launching VLM pretraining jobs (`scripts/pretrain.py`) via Sagemaker, with multi-node support.

Run with: `python scripts/sagemaker/launch.py <ARGS>`
"""

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import sagemaker
import wandb
from sagemaker.inputs import FileSystemInput
from sagemaker.pytorch import PyTorch

from prismatic.conf import DatasetRegistry, ModelRegistry

# === Constants ===
ROLE_ARN = "arn:aws:iam::124224456861:role/service-role/SageMaker-SageMakerAllAccess"
SUBNETS = ["subnet-07bf42d7c9cb929e4", "subnet-05f1115c7d6ccbd07", "subnet-0e260ba29726b9fbb"]
SECURITY_GROUP_IDS = ["sg-0afb9fb0e79a54061", "sg-0333993fea1aeb948", "sg-0c4b828f4023a04cc"]

S3_LOG_PATH = "s3://tri-ml-sandbox-16011-us-east-1-datasets/sagemaker/prismatic-vlms/"
LUSTRE_PARAMETERS = {
    "file_system_type": "FSxLustre",
    "file_system_access_mode": "rw",
    "file_system_id": "fs-0ee5fb54e88f9dd00",
    "directory_path": "/kxvmdbev",
}


@dataclass
class LaunchConfig:
    # fmt: off
    job_name: str = "sk-prismatic-vlm"                                  # Base Name for Job in Sagemaker Dashboard
    instance_count: int = 1                                             # Number of Nodes for Multi-Node Training
    instance_type: str = "ml.p4de.24xlarge"                             # Instance Type (default: p4de.24xlarge)
    instance_n_gpus: int = 8                                            # Number of GPUs per Instance

    # Prismatic VLM Pretraining Parameters
    model_type: str = (                                                 # Unique Model ID (specifies config)
        ModelRegistry.EXT_EXP_MISTRAL_V1_7B.model_id
    )
    dataset_type: str = (                                               # Unique Dataset ID (specifies config)
        DatasetRegistry.LLAVA_V15.dataset_id
    )

    # Stage & Batch Size Parameters =>> Set dynamically based on instance count!
    stage: str = "finetune"
    global_batch_size: int = 128                                        # Global Batch Size (across all nodes)
    per_device_max: int = 16                                            # Maximum Per Device Batch Size

    # Updated Paths for Data / Runs (on Sagemaker Volume)
    dataset_root_dir: str = (                                           # Dataset Root Directory (in Sagemaker Volume)
        "/opt/ml/input/data/training/skaramcheti/datasets/prismatic-vlms"
    )
    run_root_dir: str = (                                               # Run/Logs Root Directory (in Sagemaker Volume)
        "/opt/ml/input/data/training/x-prismatic-vlms/runs"
    )

    # Sagemaker Job Parameters
    entry_point: str = "scripts/pretrain.py"                            # Entry Point for Training
    input_source: str = "lustre"                                        # Data source in < lustre >
    image_uri: str = (                                                  # Path to Sagemaker Docker Image (in AWS ECR)
        "124224456861.dkr.ecr.us-east-1.amazonaws.com/prismatic-vlms:latest"
    )
    max_days: int = 7                                                   # Cutoff for Training Time

    # Weights & Biases API Key
    wandb_api_key: Union[str, Path] = Path(".wandb_api_key")            # W&B API Key (for real-time logging)

    # Local Debugging
    debug: bool = False                                                 # Launch Sagemaker Debugging (on `localhost`)
    # fmt: on


@draccus.wrap()
def launch(cfg: LaunchConfig) -> None:
    print("[*] Configuring Sagemaker Launch =>> Prismatic VLM Training!")

    # Parse & Verify W&B API Key
    print("[*] Verifying W&B API Key")
    wandb_api_key = cfg.wandb_api_key.read_text().strip() if isinstance(cfg.wandb_api_key, Path) else cfg.wandb_api_key
    assert wandb.login(key=wandb_api_key, verify=True), "Invalid W&B API Key!"

    # Initialize Sagemaker Session
    print(f"[*] Initializing Sagemaker Session\n\t=>> Role ARN: `{ROLE_ARN}`")
    sagemaker_session = sagemaker.Session() if not cfg.debug else sagemaker.LocalSession()

    # Assemble Job Hyperparameters
    #   =>> Note: For future `S3` support, make sure to set `input_mode = "FastFile"` in Pytorch Estimator init
    print(f"[*] Assembling Job Parameters =>> Model Type: `{cfg.model_type}`")
    assert cfg.input_source == "lustre", f"Found `{cfg.input_source = }`; we currently only support `lustre`!"
    train_fs = FileSystemInput(**LUSTRE_PARAMETERS)

    # Compute Batch Size Parameters
    per_device_batch_size = cfg.global_batch_size // (world_size := cfg.instance_count * cfg.instance_n_gpus)
    assert cfg.global_batch_size % world_size == 0, f"World Size `{world_size}` does not divide global batch size!"
    assert 1 <= per_device_batch_size <= cfg.per_device_max, f"Invalid per-device batch size {per_device_batch_size}!"

    assert cfg.stage.endswith("finetune"), "We only support `finetune` stages for now; `align` support is pending!"
    hyperparameters = {
        "model.type": cfg.model_type,
        "dataset.type": cfg.dataset_type,
        "dataset.dataset_root_dir": cfg.dataset_root_dir,
        "run_root_dir": cfg.run_root_dir,
        "stage": cfg.stage,
        "model.finetune_global_batch_size": cfg.global_batch_size,
        "model.finetune_per_device_batch_size": per_device_batch_size,
    }

    # Launch!
    print("[*] Creating Sagemaker Estimator =>> Launching!")
    estimator = PyTorch(
        role=ROLE_ARN,
        base_job_name=cfg.job_name,
        instance_count=cfg.instance_count,
        instance_type=cfg.instance_type if not cfg.debug else "local_gpu",
        entry_point=cfg.entry_point,
        image_uri=cfg.image_uri,
        hyperparameters=hyperparameters,
        environment={
            "PYTHONPATH": "/opt/ml/code",
            "WANDB_API_KEY": wandb_api_key,
            "HF_HOME": "/opt/ml/input/data/training/skaramcheti/cache",
        },
        sagemaker_session=sagemaker_session,
        subnets=SUBNETS,
        security_group_ids=SECURITY_GROUP_IDS,
        keep_alive_period_in_seconds=3600,
        max_run=60 * 60 * 24 * cfg.max_days,
        distribution={"torch_distributed": {"enabled": True}},
        disable_profiler=True,
    )
    estimator.fit(inputs={"training": train_fs if not cfg.debug else "file:///mnt/fsx/"})


if __name__ == "__main__":
    launch()
