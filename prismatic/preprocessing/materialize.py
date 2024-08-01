"""
materialize.py

Factory class for initializing Image Processors and Tokenizers on a per-VLM basis; provides and exports individual
functions for clear control flow.
"""

from typing import Any, Dict, List, Type

from timm.data import resolve_data_config
from transformers import AddedToken, AutoTokenizer

from prismatic.preprocessing.processors import PrismaticImageProcessor, PrismaticProcessor
from prismatic.preprocessing.prompting import (
    Llama2ChatPromptBuilder,
    MistralInstructPromptBuilder,
    PromptBuilder,
    PurePromptBuilder,
    VicunaV15ChatPromptBuilder,
    OpenlmPromptBuilder
)
from prismatic.models.backbones.llm.openlm import CustomTokenizer
from functools import partial


def get_prompt_builder_fn(llm_family: str, system_prompt: str = None) -> Type[PromptBuilder]:
    if llm_family.endswith("pure"):
        return partial(PurePromptBuilder, model_family=llm_family, system_prompt=system_prompt)

    elif llm_family.startswith("llama2") and llm_family.endswith("-chat"):
        return partial(Llama2ChatPromptBuilder, model_family=llm_family, system_prompt=system_prompt)

    elif llm_family.startswith("vicuna"):
        return partial(VicunaV15ChatPromptBuilder, model_family=llm_family, system_prompt=system_prompt)

    elif llm_family.startswith("mistral") and llm_family.endswith("-instruct"):
        return partial(MistralInstructPromptBuilder, model_family=llm_family, system_prompt=system_prompt)
    
    elif llm_family.startswith("openlm") or llm_family.startswith("openvlm"):
        return partial(OpenlmPromptBuilder, model_family=llm_family, system_prompt=system_prompt)

    raise ValueError(f"No PromptBuilder defined for LLM Backbone `{llm_family}`")


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
    if "openlm" in llm_hf_hub_path or "openvlm" in llm_hf_hub_path:
        tokenizer = CustomTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", model_max_length=llm_max_length
        )
    else:
        tokenizer = AutoTokenizer.from_pretrained(llm_hf_hub_path, model_max_length=llm_max_length, padding_side="right")
        tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        tokenizer.add_tokens(AddedToken("<image>", special=True, normalized=False), special_tokens=True)

    return PrismaticProcessor(image_processor, tokenizer)
