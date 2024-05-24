"""
load.py

Entry point for loading pretrained VLMs for inference; exposes functions for listing available models (with canonical
IDs, mappings to paper experiments, and short descriptions), as well as for loading models (from disk or HF Hub).
"""
import json
import os
from pathlib import Path
from typing import List, Optional, Union

from huggingface_hub import hf_hub_download

from prismatic.models.materialize import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.llm.openlm import get_vision_state_dict, get_projector_state_dict
from prismatic.models.registry import GLOBAL_REGISTRY, MODEL_REGISTRY
from prismatic.models.vlms import PrismaticVLM
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# === HF Hub Repository ===
HF_HUB_REPO = "TRI-ML/prismatic-vlms"


# === Available Models ===
def available_models() -> List[str]:
    return list(MODEL_REGISTRY.keys())


def available_model_names() -> List[str]:
    return list(GLOBAL_REGISTRY.items())


def get_model_description(model_id_or_name: str) -> str:
    if model_id_or_name not in GLOBAL_REGISTRY:
        raise ValueError(f"Couldn't find `{model_id_or_name = }; check `prismatic.available_model_names()`")

    # Print Description & Return
    print(json.dumps(description := GLOBAL_REGISTRY[model_id_or_name]["description"], indent=2))

    return description


# === Load Pretrained Model ===
def load(
    model_id_or_path: Union[str, Path], hf_token: Optional[str] = None, cache_dir: Optional[Union[str, Path]] = None, strict: bool = True
) -> PrismaticVLM:
    """Loads a pretrained PrismaticVLM from either local disk or the HuggingFace Hub."""
    if os.path.isdir(model_id_or_path):
        overwatch.info(f"Loading from local path `{(run_dir := Path(model_id_or_path))}`")
        # Get paths for `config.json` and pretrained checkpoint
        assert (config_json := run_dir / "config.json").exists(), f"Missing `config.json` for `{run_dir = }`"
        assert (checkpoint_pt := run_dir / "checkpoints" / "latest-checkpoint.pt").exists(), "Missing checkpoint!"
    elif str(model_id_or_path).startswith("(openlm)") or str(model_id_or_path).startswith("(openvlm)"):
        model_id_or_path = str(model_id_or_path)
        not_from_pretrained_vlm = not model_id_or_path.startswith("(openvlm)")
        vision_backbone, image_transform = get_vision_backbone_and_transform(
            "dinosiglip-vit-so-384px",
            "resize-naive",
            dino_first=not_from_pretrained_vlm,
            pretrained=not_from_pretrained_vlm
        )
        if model_id_or_path.startswith("(openvlm)"):
            overwatch.info(f"Loading vision state dict from OpenVLM")
            vision_state_dict = get_vision_state_dict(model_id_or_path)
            vision_backbone.load_state_dict(vision_state_dict, strict=True)

        llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
            model_id_or_path,
            llm_max_length=1024,
            inference_mode=False,
        )

        vlm = get_vlm(
            model_id_or_path,
            "no-align+fused-gelu-mlp",
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=False,
            touch_arch_specifier="fused-gelu-mlp",
        )

        if model_id_or_path.startswith("(openvlm)"):
            overwatch.info("Loading projector state dict from OpenVLM")
            projector_state_dict = get_projector_state_dict(model_id_or_path)
            vlm.projector.load_state_dict(projector_state_dict, strict=True)
        return vlm
    else:
        if model_id_or_path not in GLOBAL_REGISTRY:
            raise ValueError(f"Couldn't find `{model_id_or_path = }; check `prismatic.available_model_names()`")

        overwatch.info(f"Downloading `{(model_id := GLOBAL_REGISTRY[model_id_or_path]['model_id'])} from HF Hub")
        config_json = hf_hub_download(repo_id=HF_HUB_REPO, filename=f"{model_id}/config.json", cache_dir=cache_dir)
        checkpoint_pt = hf_hub_download(
            repo_id=HF_HUB_REPO, filename=f"{model_id}/checkpoints/latest-checkpoint.pt", cache_dir=cache_dir
        )

    # Load Model Config from `config.json`
    with open(config_json, "r") as f:
        model_cfg = json.load(f)["model"]

    # = Load Individual Components necessary for Instantiating a VLM =
    #   =>> Print Minimal Config
    overwatch.info(
        f"Found Config =>> Loading & Freezing [bold blue]{model_cfg['model_id']}[/] with:\n"
        f"             Vision Backbone =>> [bold]{model_cfg['vision_backbone_id']}[/]\n"
        f"             LLM Backbone    =>> [bold]{model_cfg['llm_backbone_id']}[/]\n"
        f"             Arch Specifier  =>> [bold]{model_cfg['arch_specifier']}[/]\n"
        f"             Checkpoint Path =>> [underline]`{checkpoint_pt}`[/]"
    )

    # Load Vision Backbone
    overwatch.info(f"Creating Vision Backbone [bold]{model_cfg['vision_backbone_id']}[/]")
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        model_cfg["vision_backbone_id"],
        model_cfg["image_resize_strategy"],
        pretrained=False,  # We cannot really know if the vision backbone should be loaded or not here, the weight might or might not be in the checkpoint
        dino_first=not model_cfg['llm_backbone_id'].startswith("(openvlm)"),
    )

    # Load LLM Backbone --> note `inference_mode = True` by default when calling `load()`
    overwatch.info(f"Creating Pretrained LLM [bold]{model_cfg['llm_backbone_id']}[/]")
    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_cfg["llm_backbone_id"],
        llm_max_length=model_cfg.get("llm_max_length", 2048),
        hf_token=hf_token,
        inference_mode=True,
    )

    # Load VLM using `from_pretrained` (clobbers HF syntax... eventually should reconcile)
    overwatch.info(f"Loading VLM [bold blue]{model_cfg['model_id']}[/] from Checkpoint; Freezing Weights 🥶")
    vlm = PrismaticVLM.from_pretrained(
        checkpoint_pt,
        model_cfg["model_id"],
        vision_backbone,
        llm_backbone,
        arch_specifier=model_cfg["arch_specifier"],
        strict=strict,
    )

    return vlm
