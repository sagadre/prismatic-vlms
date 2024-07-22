"""
configuration_prismatic.py

Standalone classes for fully-specifying a Prismatic VLM configuration (derived from `transformers.PretrainedConfig`).
Defines full set of "registries" for various supported vision backbones and LLM backbones.
"""

from typing import Any, Dict, Optional

from transformers import CONFIG_MAPPING, AutoConfig, PretrainedConfig

from prismatic.overwatch import initialize_overwatch
import yaml
import argparse

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)

# === Registries =>> Maps ID --> {cls(), kwargs} :: Different Registries for Vision Backbones, LLM Backbones, VLMs ===
# fmt: off

# === Vision Backbone Registry ===
HF_VISION_BACKBONES = {
    # === 224px Backbones ===
    "clip-vit-l": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_large_patch14_clip_224.openai"],
        "timm_override_act_layers": ["quick_gelu"],
        "image_sizes": [224],
    },
    "dinov2-vit-l": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_large_patch14_reg4_dinov2.lvd142m"],
        "timm_override_act_layers": [None],
        "image_sizes": [224],
    },
    "in1k-vit-l": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_large_patch16_224.augreg_in21k_ft_in1k"],
        "timm_override_act_layers": [None],
        "image_sizes": [224],
    },
    "siglip-vit-so400m": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_so400m_patch14_siglip_224"],
        "timm_override_act_layers": [None],
        "image_sizes": [224],
    },

    # === CLIP ===
    "clip-vit-l-336px": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_large_patch14_clip_336.openai"],
        "timm_override_act_layers": ["quick_gelu"],
        "image_sizes": [336],
    },

    # === SigLIP ===
    "siglip-vit-so400m-384px": {
        "use_fused_vision_backbone": False,
        "timm_model_ids": ["vit_so400m_patch14_siglip_384"],
        "timm_override_act_layers": [None],
        "image_sizes": [384],
    },

    # === Fused Backbones ===
    "dinoclip-vit-l-336px": {
        "use_fused_vision_backbone": True,
        "timm_model_ids": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_large_patch14_clip_336.openai"],
        "timm_override_act_layers": [None, "quick_gelu"],
        "image_sizes": [336, 336],
    },

    "dinosiglip-vit-so-224px": {
        "use_fused_vision_backbone": True,
        "timm_model_ids": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_224"],
        "timm_override_act_layers": [None, None],
        "image_sizes": [224, 224],
    },
    "dinosiglip-vit-so-384px": {
        "use_fused_vision_backbone": True,
        "timm_model_ids": ["vit_large_patch14_reg4_dinov2.lvd142m", "vit_so400m_patch14_siglip_384"],
        "timm_override_act_layers": [None, None],
        "image_sizes": [384, 384]
    },
}

# === Language Model Registry ===
HF_LLM_BACKBONES = {
    # === Llama-2 Pure (Non-Chat) Backbones ===
    "llama2-7b-pure": {
        "llm_family": "llama2", "hf_meta": "llama", "hf_hub_path": "meta-llama/Llama-2-7b-hf"
    },
    "llama2-13b-pure": {
        "llm_family": "llama2", "hf_meta": "llama", "hf_hub_path": "meta-llama/Llama-2-13b-hf"
    },

    # === Llama-2 Chat Backbones ===
    "llama2-7b-chat": {
        "llm_family": "llama2", "hf_meta": "llama", "hf_hub_path": "meta-llama/Llama-2-7b-chat-hf"
    },
    "llama2-13b-chat": {
        "llm_family": "llama2", "hf_meta": "llama", "hf_hub_path": "meta-llama/Llama-2-13b-chat-hf"
    },

    # === Vicuna-v1.5 Backbones ===
    "vicuna-v15-7b": {
        "llm_family": "vicuna", "hf_meta": "llama", "hf_hub_path": "lmsys/vicuna-7b-v1.5"
    },
    "vicuna-v15-13b": {
        "llm_family": "vicuna", "hf_meta": "llama", "hf_hub_path": "lmsys/vicuna-13b-v1.5"
    },

    # === Mistral v0.1 Backbones ===
    "mistral-v0.1-7b-pure": {
        "llm_family": "mistral", "hf_meta": "mistral", "hf_hub_path": "mistralai/Mistral-7B-v0.1"
    },
    "mistral-v0.1-7b-instruct": {
        "llm_family": "mistral", "hf_meta": "mistral", "hf_hub_path": "mistralai/Mistral-7B-Instruct-v0.1"
    },
    # === OpenLM Backbones ===
    "openlm": {
        "llm_family": "openlm", "hf_meta": "openlm", "hf_hub_path": ""
    },
    "openvlm": {
        "llm_family": "openvlm", "hf_meta": "openvlm", "hf_hub_path": ""
    },
}

# fmt: on


class PrismaticConfig(PretrainedConfig):
    model_type: str = "prismatic"
    is_composition: bool = False

    def __init__(
        self,
        vision_backbone_id: str = "siglip-vit-so400m",
        llm_backbone_id: str = "vicuna-v15-7b",
        llm_max_length: int = 2048,
        pad_to_multiple_of: int = 64,
        pad_token_id: Optional[int] = None,
        image_token_id: Optional[int] = None,
        text_config: Optional[Dict[str, Any]] = None,
        **kwargs: str,
    ) -> None:
        if vision_backbone_id not in HF_VISION_BACKBONES:
            raise ValueError(f"Vision backbone `{vision_backbone_id} not in `{HF_VISION_BACKBONES.keys()}`")
        
        # Derive Vision Backbone Parameters
        self.use_fused_vision_backbone = HF_VISION_BACKBONES[vision_backbone_id]["use_fused_vision_backbone"]
        self.timm_model_ids = HF_VISION_BACKBONES[vision_backbone_id]["timm_model_ids"]
        self.timm_override_act_layers = HF_VISION_BACKBONES[vision_backbone_id]["timm_override_act_layers"]
        self.image_sizes = HF_VISION_BACKBONES[vision_backbone_id]["image_sizes"]
        self.llm_max_length = llm_max_length
        self.pad_to_multiple_of = pad_to_multiple_of
        
        if llm_backbone_id.startswith("(openlm)"):
            source = llm_backbone_id.replace("(openlm)", "")
            overwatch.info(f"OpenLM detected; loading OpenLM configuration from {source}")
            HF_LLM_BACKBONES["openlm"]["hf_hub_path"] = source
            self.llm_family = "openlm"
            self.llm_backbone_id = llm_backbone_id
            self.llm_hf_hub_path = llm_backbone_id
        elif llm_backbone_id.startswith("(openvlm)"):
            source = llm_backbone_id.replace("(openvlm)", "")
            overwatch.info(f"OpenVLM detected; loading OpenVLM configuration from {source}")
            HF_LLM_BACKBONES["openlm"]["hf_hub_path"] = source
            self.llm_family = "openvlm"
            self.llm_backbone_id = llm_backbone_id
            self.llm_hf_hub_path = llm_backbone_id
        elif llm_backbone_id not in HF_LLM_BACKBONES:
            raise ValueError(f"LLM backbone `{llm_backbone_id}` not in {HF_LLM_BACKBONES.keys()}")
        else:
            # Set Prismatic Configuration Fields
            self.vision_backbone_id = vision_backbone_id
            self.llm_backbone_id = llm_backbone_id

            # Derive LLM Backbone Parameters
            #   =>> [IMPORTANT] HF Utilities actually look for a `text_config` field... we need to use that naming!
            self.llm_family = HF_LLM_BACKBONES[llm_backbone_id]["llm_family"]
            self.llm_hf_hub_path = HF_LLM_BACKBONES[llm_backbone_id]["hf_hub_path"]

        if text_config is None:
            if self.llm_family in ["openlm", "openvlm"]:
                self.text_config = yaml.load(open(f"{source}/params.txt", "r"), Loader=yaml.FullLoader)
                self.text_config = argparse.Namespace(**self.text_config)
            else:
                self.text_config = AutoConfig.from_pretrained(self.llm_hf_hub_path)
        else:
            self.text_config = CONFIG_MAPPING[HF_LLM_BACKBONES[llm_backbone_id]["hf_meta"]](**text_config)

        # [CONTRACT] Prismatic VLMs explicitly create a *new* `pad_token` and `image_token` at the end of vocabulary!
        self.pad_token_id = pad_token_id if pad_token_id is not None else self.text_config.vocab_size
        self.image_token_id = image_token_id if image_token_id is not None else self.text_config.vocab_size + 1

        # Dispatch **kwargs to super() =>> note that `pad_token_id` collides, so we pass it in here as well...
        super().__init__(pad_token_id=self.pad_token_id, **kwargs)
