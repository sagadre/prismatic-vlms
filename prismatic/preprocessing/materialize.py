"""
materialize.py

Factory class for initializing Image Processors and Tokenizers on a per-VLM basis; provides and exports individual
functions for clear control flow.
"""

from typing import Any, Dict, List, Type

from timm.data import resolve_data_config
from transformers import AddedToken, AutoTokenizer

from prismatic.models.backbones.llm.prompting import (
    Llama2ChatPromptBuilder,
    MistralInstructPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
)
from prismatic.preprocessing.hf_processor import PrismaticImageProcessor, PrismaticProcessor


def get_prompt_builder_fn(llm_backbone_id: str) -> Type[PromptBuilder]:
    if llm_backbone_id.endswith("-pure"):
        return PurePromptBuilder

    elif llm_backbone_id.startswith("llama2-") and llm_backbone_id.endswith("-chat"):
        return Llama2ChatPromptBuilder

    elif llm_backbone_id.startswith("vicuna"):
        return VicunaV15ChatPromptBuilder

    elif llm_backbone_id.startswith("mistral-") and llm_backbone_id.endswith("-instruct"):
        return MistralInstructPromptBuilder

    raise ValueError(f"No PromptBuilder defined for LLM Backbone `{llm_backbone_id}`")


def get_image_processor(
    use_fused_vision_backbone: bool, image_resize_strategy: str, vision_backbone_cfgs: List[Dict[str, Any]]
) -> PrismaticImageProcessor:
    """Use the `vision_backbone_cfgs` to instantiate the appropriate TIMM Transforms."""
    input_sizes, interpolations, means, stds = [], [], [], []
    for cfg in vision_backbone_cfgs:
        transform_cfg = resolve_data_config(cfg)

        input_sizes.append(transform_cfg["input_size"])
        interpolations.append(transform_cfg["interpolation"])
        means.append(transform_cfg["mean"])
        stds.append(transform_cfg["std"])

    return PrismaticImageProcessor(
        use_fused_vision_backbone=use_fused_vision_backbone,
        image_resize_strategy=image_resize_strategy,
        input_sizes=input_sizes,
        interpolations=interpolations,
        means=means,
        stds=stds,
    )


def get_prismatic_processor(
    use_fused_vision_backbone: bool,
    image_resize_strategy: str,
    vision_backbone_cfgs: List[Dict[str, Any]],
    llm_hf_hub_path: str,
    llm_max_length: int,
) -> PrismaticProcessor:
    image_processor = get_image_processor(use_fused_vision_backbone, image_resize_strategy, vision_backbone_cfgs)

    # Tokenizer =>> by default, perform the necessary resizing/token addition!
    #   =>> Note that we assume Tokenizer with `padding_side="right"` during *training*; left pad during INFERENCE!
    tokenizer = AutoTokenizer.from_pretrained(llm_hf_hub_path, model_max_length=llm_max_length, padding_side="right")
    tokenizer.add_special_tokens({"pad_token": "<PAD>"})
    tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    return PrismaticProcessor(image_processor, tokenizer)
