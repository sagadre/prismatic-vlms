"""
convert_openvla_weights_to_hf.py

Utility script for converting full OpenVLA VLA weights (from this repository, in the default "Prismatic" format) to
the HuggingFace "AutoClasses" (e.g., those defined in `prismatic.extern.hf_*`) for "native" use in `transformers``
via `trust_remote_code = True`.

Theoretically, these changes should be fully compatible with directly merging the models into `transformers` down the
line, with first-class support.
"""

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Union

import draccus
import timm
import torch
from huggingface_hub import hf_hub_download
from transformers import AutoTokenizer

from prismatic.conf import ModelConfig
from prismatic.extern.hf_prismatic.configuration_prismatic import OpenVLAConfig
from prismatic.extern.hf_prismatic.modeling_prismatic import OpenVLAForActionPrediction
from prismatic.extern.hf_prismatic.processing_prismatic import PrismaticImageProcessor, PrismaticProcessor


@dataclass
class HFConvertConfig:
    # fmt: off
    openvla_model_path_or_id: Union[str, Path] = (                      # Path to Pretrained VLA (on disk or HF Hub)
        "pretrained/siglip-224px+mx-oxe-magic-soup+n8+b32+x7"
    )
    output_hf_model_local_path: Path = Path(                            # Path to Local Path to save HF model
        "/raid/users/karl/models/huggingface/openvla-7b-v01"
    )
    output_hf_model_hub_path: str = "openvla/openvla-7b-v01"            # Path to HF Hub Path for "final" HF model
                                                                        #   => huggingface.co/openvla/openvla-{...}

    # HF Hub Credentials (required for Gated Models like LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    def __post_init__(self) -> None:
        self.hf_token = self.hf_token.read_text().strip() if isinstance(self.hf_token, Path) else self.hf_token

    # fmt: on


# === Conversion Constants ===
PROJECTOR_KEY_MAPPING = {
    "projector.0.weight": "projector.fc1.weight",
    "projector.0.bias": "projector.fc1.bias",
    "projector.2.weight": "projector.fc2.weight",
    "projector.2.bias": "projector.fc2.bias",
}


def remap_state_dicts_for_hf(
    vision_backbone_state_dict: Dict[str, torch.Tensor],
    projector_state_dict: Dict[str, torch.Tensor],
    llm_backbone_state_dict: Dict[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Iterate through Prismatic component state dictionaries and unify / fix key mapping for HF conversion."""
    hf_state_dict = {}

    # Iterate through Vision Backbone =>> add "vision_backbone." prefix
    for key, value in vision_backbone_state_dict.items():
        hf_state_dict[f"vision_backbone.{key}"] = value

    # Iterate through Projector =>> use `PROJECTOR_KEY_MAPPING`
    for key, value in projector_state_dict.items():
        hf_state_dict[PROJECTOR_KEY_MAPPING[key]] = value

    # Iterate through LLM Backbone =>> replace `llm.` with `language_model.`
    for key, value in llm_backbone_state_dict.items():
        hf_state_dict[key.replace("llm.", "language_model.")] = value

    return hf_state_dict


@draccus.wrap()
def convert_prismatic_weights_to_hf(cfg: HFConvertConfig) -> None:
    print(f"[*] Converting OpenVLA Model `{cfg.openvla_model_path_or_id}` to HF Transformers Format")
    torch.set_default_dtype(torch.bfloat16)

    # Get `config.json`, 'dataset_statistics.json' and `checkpoint_pt` -- mirrors logic in `prismatic.models.load.py`
    if os.path.isdir(cfg.openvla_model_path_or_id):
        print(f"[*] Loading from Local Path `{(run_dir := Path(cfg.openvla_model_path_or_id))}`")
        config_json, checkpoint_pt = run_dir / "config.json", run_dir / "checkpoints" / "latest-checkpoint.pt"
        dataset_statistics_json = run_dir / "dataset_statistics.json"

        assert config_json.exists(), f"Missing `config.json` for `{run_dir = }`"
        assert checkpoint_pt.exists(), f"Missing checkpoint for `{run_dir = }`"
        assert dataset_statistics_json.exists(), f"Missing `dataset_statistics.json` for `{run_dir = }`"
    else:
        print(f"[*] Downloading Prismatic Checkpoint from HF Hub :: `TRI-ML/{cfg.openvla_model_path_or_id}`")
        config_json = hf_hub_download("openvla/openvla-dev", f"{cfg.openvla_model_path_or_id}/config.json")
        checkpoint_pt = hf_hub_download(
            "openvla/openvla-dev", f"{cfg.openvla_model_path_or_id}/checkpoints/latest-checkpoint.pt"
        )
        dataset_statistics_json = hf_hub_download(
            "openvla/openvla-dev", f"{cfg.openvla_model_path_or_id}/dataset_statistics.json"
        )

    # Load "Native" Config JSON =>> Create LLM Config & Instantiate Tokenizer
    with open(config_json, "r") as f:
        vla_cfg = json.load(f)["vla"]
        prismatic_config = ModelConfig.get_choice_class(vla_cfg["base_vlm"])().__dict__

    # Load normalization statistics
    with open(dataset_statistics_json, "r") as f:
        norm_stats = json.load(f)

    # Create HF OpenVLAConfig (`transformers.PretrainedConfig`)
    hf_config = OpenVLAConfig(
        vision_backbone_id=prismatic_config["vision_backbone_id"],
        llm_backbone_id=prismatic_config["llm_backbone_id"],
        arch_specifier=prismatic_config["arch_specifier"],
        image_resize_strategy=prismatic_config["image_resize_strategy"],
        llm_max_length=prismatic_config["llm_max_length"],
        torch_dtype=torch.bfloat16,
        norm_stats=norm_stats,
    )

    # Instantiate & Add Pad to Tokenizer =>> following `prismatic.models.materialize.get_llm_backbone_and_tokenizer`
    #   TODO (siddk) :: Implement batched generation -- in which case this should set `padding_side = "left"`!
    print("[*] Instantiating and Patching Tokenizer, LLM Config")
    tokenizer = AutoTokenizer.from_pretrained(
        hf_config.hf_llm_id, model_max_length=hf_config.llm_max_length, token=cfg.hf_token, padding_side="right"
    )
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.init_kwargs.pop("add_prefix_space")  # Pop to prevent unnecessary warning on reload...
    assert tokenizer.pad_token_id == hf_config.pad_token_id, "Incorrect Pad Token ID!"
    assert len(tokenizer) > hf_config.text_config.vocab_size, "Tokenizer vocabulary must be larger than LLM vocabulary!"

    # Patch LLM Config in `hf_config` with vocab_size (+ `hf_config.pad_to_multiple_of`), pad_token_id + validate
    hf_config.text_config.vocab_size += hf_config.pad_to_multiple_of
    hf_config.text_config.pad_token_id = hf_config.pad_token_id
    hf_config.text_config.torch_dtype = torch.bfloat16
    assert hf_config.text_config.use_cache, "LLM config `use_cache` should be True for inference (set default)!"

    # Create Vision Backbone & Transform =>> following `prismatic.models.materialize.get_vision_backbone_and_transform`
    #   =>> Deviates a bit from existing code; as such, explicitly tested in `tests/test_image_transforms.py`
    print("[*] Loading TIMM Vision Backbone and Image Transform =>> Initializing PrismaticImageProcessor")
    timm_vision_backbone = timm.create_model(
        hf_config.timm_model_id,
        pretrained=True,
        num_classes=0,
        img_size=hf_config.image_size,
        act_layer=hf_config.timm_override_act_layer,
    )
    data_cfg = timm.data.resolve_model_data_config(timm_vision_backbone)

    # Create PrismaticImageProcessor (`transformers.ImageProcessingMixin`)
    hf_image_processor = PrismaticImageProcessor(
        image_resize_strategy=hf_config.image_resize_strategy,
        input_size=data_cfg["input_size"],
        interpolation=data_cfg["interpolation"],
        mean=data_cfg["mean"],
        std=data_cfg["std"],
    )

    # Create top-level PrismaticProcessor (`transformers.ProcessorMixin` =>> enables registry w/ AutoProcessor)
    print("[*] Creating PrismaticProcessor Instance from Tokenizer and PrismaticImageProcessor")
    hf_processor = PrismaticProcessor(image_processor=hf_image_processor, tokenizer=tokenizer)

    # Load Prismatic Model State Dictionary (in preparation for conversion)
    print("[*] Loading Prismatic VLM State Dictionary from Checkpoint")
    model_state_dict = torch.load(checkpoint_pt, map_location="cpu")["model"]
    assert ("downsampler" not in model_state_dict) or (len(model_state_dict["downsampler"]) == 0), "Downsampler?"
    assert ("projector" in model_state_dict) and ("llm_backbone" in model_state_dict), "Missing keys!"

    # Convert
    print("[*] Running Conversion")
    converted_state_dict = remap_state_dicts_for_hf(
        timm_vision_backbone.state_dict(), model_state_dict["projector"], model_state_dict["llm_backbone"]
    )

    # Create PrismaticForConditionalGeneration =>> Note that we can't initialize on `meta` device because TIMM
    print("[*] Building (Randomly Initialized) Model =>> OpenVLAForActionPrediction")
    hf_model = OpenVLAForActionPrediction(hf_config)
    hf_model.load_state_dict(converted_state_dict, strict=True, assign=True)

    # Case model to bf16 before saving
    hf_model.to(torch.bfloat16)

    # Save Pretrained Versions to Local Path
    print("[*] Saving Model & Processor to Local Path")
    hf_model.save_pretrained(cfg.output_hf_model_local_path)
    hf_image_processor.save_pretrained(cfg.output_hf_model_local_path)
    hf_processor.save_pretrained(cfg.output_hf_model_local_path)

    # Register AutoClasses
    OpenVLAConfig.register_for_auto_class()
    PrismaticImageProcessor.register_for_auto_class("AutoImageProcessor")
    PrismaticProcessor.register_for_auto_class("AutoProcessor")
    OpenVLAForActionPrediction.register_for_auto_class("AutoModelForVision2Seq")

    # Push to Hub
    print("[*] Pushing Model & Processor to HF Hub")
    hf_model.push_to_hub(cfg.output_hf_model_hub_path, private=True)
    hf_image_processor.push_to_hub(cfg.output_hf_model_hub_path, private=True)
    hf_processor.push_to_hub(cfg.output_hf_model_hub_path, private=True)


if __name__ == "__main__":
    convert_prismatic_weights_to_hf()
