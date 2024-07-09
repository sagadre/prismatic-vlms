"""
convert_prismatic_to_hf.py

Utility script for converting "old-format" (`prismatic < 1.0.0`) weights to the HuggingFace-derived format
(`PrismaticForVision2Seq`). In addition, exports the corresponding `PrismaticProcessor` (wrapper around image transform
and tokenizer as well), so that each checkpoint directory is the single-source of truth for a given VLM instance.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import draccus
import torch
from huggingface_hub import ModelCard, hf_hub_download

from prismatic.models import PrismaticConfig, PrismaticForVision2Seq
from prismatic.preprocessing import get_prismatic_processor

# === Model Card Template ===
MODEL_CARD_TEMPLATE = """
---
language: en
license: mit
library_name: prismatic
tags:
- vlm
- image-text-to-text
- multimodal
- pretraining
---

# {model_name}

**Model ID**: `{model_id}`

**Model Description**:
```json
{model_json}
```
"""

# === Conversion Constants ===
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
    "projector.4.weight": "projector.fc3.weight",
    "projector.4.bias": "projector.fc3.bias",
}


def remap_state_dicts(
    projector_state_dict: Dict[str, torch.Tensor], llm_backbone_state_dict: Dict[str, torch.Tensor]
) -> Dict[str, torch.Tensor]:
    """Iterate through the (deprecated) Prismatic component state dictionaries and unify / fix key mapping."""
    remapped_state_dict = {}

    # Iterate through Projector =>> use `PROJECTOR_KEY_MAPPING`
    for key, value in projector_state_dict.items():
        remapped_state_dict[PROJECTOR_KEY_MAPPING[key]] = value

    # Iterate through LLM Backbone =>> replace `llm.` with `language_model.`
    for key, value in llm_backbone_state_dict.items():
        remapped_state_dict[key.replace("llm.", "language_model.")] = value

    return remapped_state_dict


@dataclass
class HFConvertConfig:
    # fmt: off
    prismatic_model_path_or_id: Union[str, Path] = (                    # Path to "deprecated-format" VLM (disk/HF Hub)
        "prism-dinosiglip-224px+7b"
    )
    output_hf_model_local_path: Path = Path(                            # Local Path to save HF Model
        "/mnt/fsx/x-prismatic-vlms/hf-convert/prism-dinosiglip-224px-7b"
    )
    output_hf_model_hub_path: str = (                                   # Path to HF Hub Path for "internal" HF model
        "TRI-ML/prism-dinosiglip-224px-7b"                              #   =>> huggingface.co/TRI-ML/{...}
    )

    # Registry of (Deprecated) Models w/ Configurations (for low-budget Model Cards)
    model_registry: Path = Path("scripts/hf-export/registry.json")

    # HF Hub Credentials (required for auth and accessing gated models like Llama-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Path to HF Token (or token as string)

    def __post_init__(self) -> None:
        os.environ["HF_TOKEN"] = self.hf_token.read_text().strip() if isinstance(self.hf_token, Path) else self.hf_token

    # fmt: on


@draccus.wrap()
def convert_prismatic_to_hf(cfg: HFConvertConfig) -> None:
    print(f"[*] Converting Prismatic Model `{cfg.prismatic_model_path_or_id}` to HF Transformers Format")
    torch.set_default_dtype(torch.bfloat16)

    # Get `config.json` and `checkpoint_pt` -- mirrors logic in `prismatic.models.load.py`
    model_card = None
    if os.path.isdir(cfg.prismatic_model_path_or_id):
        print(f"[*] Loading from Local Path `{(run_dir := Path(cfg.prismatic_model_path_or_id))}`")
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"

        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
    else:
        print(f"[*] Downloading Prismatic Checkpoint from HF Hub :: `TRI-ML/{cfg.prismatic_model_path_or_id}`")
        config_json = hf_hub_download("TRI-ML/prismatic-vlms", f"{cfg.prismatic_model_path_or_id}/config.json")
        checkpoint_pt = hf_hub_download(
            "TRI-ML/prismatic-vlms", f"{cfg.prismatic_model_path_or_id}/checkpoints/latest-checkpoint.pt"
        )

        # Attempt to Load `model_card_data` from Deprecated Model Registry
        with open(cfg.model_registry, "r") as f:
            model_card_data = json.load(f)[cfg.prismatic_model_path_or_id]
            if model_card_data.get("complete", False):
                print(f"[*] Model `{cfg.prismatic_model_path_or_id}` has already been converted & verified!")
                return

        # Build Model Card
        model_card = ModelCard(
            content=MODEL_CARD_TEMPLATE.format(
                model_name=model_card_data["names"][0],
                model_id=model_card_data["model_id"],
                model_json=json.dumps(model_card_data, ensure_ascii=False, indent=4),
            )
        )

    # Load "Native" Config JSON =>> Create LLM Config & Instantiate Tokenizer
    with open(config_json, "r") as f:
        deprecated_config = json.load(f)["model"]

    # Create HF-Style PrismaticConfig (`transformers.PretrainedConfig`)
    vlm_config = PrismaticConfig(
        vision_backbone_id=deprecated_config["vision_backbone_id"],
        llm_backbone_id=deprecated_config["llm_backbone_id"],
        llm_max_length=deprecated_config["llm_max_length"],
        torch_dtype=torch.bfloat16,
    )

    # Instantiate VLM =>> automatically loads pretrained weights for `vision_backbone` and `language_model`
    #   =>> Note =>> Loading the LLM weights from scratch is often redundant -- but simpler code!
    print(f"[*] Instantiating VLM `{cfg.prismatic_model_path_or_id}` with Pretrained Backbones")
    vlm = PrismaticForVision2Seq(vlm_config, load_pretrained_backbones=True)

    # Create `PrismaticProcessor` =>> Set `tokenizer` flags to suppress any warnings on reload (and set for `.eval()`)
    print("[*] Building `PrismaticProcessor` =>> Wraps Prismatic Image Tokenizer and Base LLM Tokenizer")
    processor = get_prismatic_processor(
        vlm_config.use_fused_vision_backbone,
        deprecated_config["image_resize_strategy"],
        vlm.get_vision_backbone_cfgs(),
        vlm_config.llm_hf_hub_path,
        vlm_config.llm_max_length,
    )
    processor.tokenizer.init_kwargs.pop("add_prefix_space", None)  # Pop to prevent unnecessary warning on reload...
    processor.tokenizer.padding_side = "left"  # Set "left" for *inference*

    # Load (Deprecated) Prismatic Model State Dictionary (in preparation for conversion)
    print("[*] Loading (Deprecated) Prismatic VLM State Dictionary from Checkpoint")
    model_state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
    assert ("downsampler" not in model_state_dict) or (len(model_state_dict["downsampler"]) == 0), "Downsampler?"
    assert ("projector" in model_state_dict) and ("llm_backbone" in model_state_dict), "Missing keys!"
    if "vision_backbone" in model_state_dict:
        raise ValueError("Unexpected Vision Backbone in (Deprecated) State Dictionary!")

    # Convert =>> add "vlm.vision_backbone" parameters as well (they've already been loaded/are frozen)
    print("[*] Running Conversion")
    converted_state_dict = remap_state_dicts(model_state_dict["projector"], model_state_dict["llm_backbone"])
    converted_state_dict.update({f"vision_backbone.{k}": v for k, v in vlm.vision_backbone.state_dict().items()})
    vlm.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Cast VLM to BF16 before Saving / Uploading
    vlm.to(dtype=torch.bfloat16)

    # Save Pretrained VLM & Processor to Local Path
    print(f"[*] Saving Processor & Model to Local Path :: `{cfg.output_hf_model_local_path}`")
    processor.save_pretrained(cfg.output_hf_model_local_path)
    vlm.save_pretrained(cfg.output_hf_model_local_path, max_shard_size="7GB")
    if model_card is not None:
        model_card.save(cfg.output_hf_model_local_path / "README.md")

    # Push to *PRIVATE* HF Hub
    print(f"[*] Pushing Config, Processor, and Model to HF Hub :: `{cfg.output_hf_model_hub_path}`")
    vlm_config.push_to_hub(cfg.output_hf_model_hub_path, private=True)
    processor.push_to_hub(cfg.output_hf_model_hub_path, private=True)
    vlm.push_to_hub(cfg.output_hf_model_hub_path, max_shard_size="7GB", private=True)
    if model_card is not None:
        model_card.push_to_hub(cfg.output_hf_model_hub_path)


if __name__ == "__main__":
    convert_prismatic_to_hf()
