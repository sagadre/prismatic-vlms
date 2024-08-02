"""
vision_backbone.py

PyTorch nn.Module definition for a Vision Backbone (Visual Featurizer), with native support for (fused) TIMM
`VisionTransformer` pretrained models.

Provides implementations of several utility functions and "monkey-patching" logic for HuggingFace API compatibility.
"""

from functools import partial
from typing import Any, Callable, List, Optional, Tuple

import timm
import torch
import torch.nn as nn
from timm.models.vision_transformer import Block, LayerScale, VisionTransformer
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy


# === Utility Functions for Monkey-Patching ===
def unpack_tuple(fn: Callable[[Any], Tuple[Any]]) -> Callable[[Any], Any]:
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        result = fn(*args, **kwargs)
        return result[0] if isinstance(result, tuple) else result

    return wrapper


# [IMPORTANT] HF Transformers overwrites parameters with names containing `gamma` and `beta` AND DON'T EVEN DOCUMENT!!!
#             As a result, we need to patch any instance of `TIMM :: LayerScale` (especially relevant for DINOv2)
#   =>> HF :: https://github.com/huggingface/transformers/blob/v4.40.1/src/transformers/modeling_utils.py#L655-L658
#   =>> TIMM :: https://github.com/huggingface/pytorch-image-models/blob/main/timm/models/vision_transformer.py#L109


def _ls_new_forward(self, x: torch.Tensor) -> torch.Tensor:
    return x.mul_(self.scale_factor) if self.inplace else x * self.scale_factor


def ls_apply_patch(ls_module: LayerScale):
    ls_module.scale_factor = nn.Parameter(ls_module.gamma.clone())
    ls_module.forward = _ls_new_forward.__get__(ls_module, LayerScale)
    del ls_module.gamma


class PrismaticVisionBackbone(nn.Module):
    def __init__(
        self,
        use_fused_vision_backbone: bool,
        timm_model_ids: List[str],
        timm_override_act_layers: List[Optional[str]],
        image_sizes: List[int],
        load_pretrained_backbones: bool = False,
        fused_first: bool = False,
    ) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.image_sizes = image_sizes
        self.default_dtype = torch.bfloat16
        self.fused_first = fused_first

        # [Validation] Lightweight Validate on TIMM Dependency Version
        if timm.__version__ not in {"0.9.10", "0.9.11", "0.9.12", "0.9.16"}:
            raise NotImplementedError(
                "TIMM Version must be >= 0.9.10 and < 1.0.0 (breaking); please raise a GitHub Issue "
                "if you urgently need support for latest TIMM versions."
            )

        # [Contract] Validate number of (fused) vision backbones, create "alpha" featurizer and Instantiate
        #   =>> Note :: Monkey-Patch the `forward()` function of the backbone to ensure FSDP-compatibility
        #               Hardcodes `get_intermediate_layers` to return the **SECOND-TO-LAST** layer patches!
        assert len(timm_model_ids) <= 2, "Prismatic models only support up to 2 (fused) vision backbones!"
        self.featurizer: VisionTransformer = timm.create_model(
            timm_model_ids[0],
            pretrained=load_pretrained_backbones,
            num_classes=0,
            img_size=image_sizes[0],
            act_layer=timm_override_act_layers[0],  # Pretrained ViTs like CLIP use custom activations (`quick_gelu`)
        )
        self.featurizer.forward = unpack_tuple(
            partial(self.featurizer.get_intermediate_layers, n={len(self.featurizer.blocks) - 2})
        )
        self.embed_dim = self.featurizer.embed_dim
        self.num_patches = self.featurizer.patch_embed.num_patches

        # If `use_fused_vision_backbone` =>> create "secondary" featurizer (`fused_featurizer`)
        if self.use_fused_vision_backbone:
            self.fused_featurizer: VisionTransformer = timm.create_model(
                timm_model_ids[1],
                pretrained=load_pretrained_backbones,
                num_classes=0,
                img_size=image_sizes[1],
                act_layer=timm_override_act_layers[1],
            )
            self.fused_featurizer.forward = unpack_tuple(
                partial(self.fused_featurizer.get_intermediate_layers, n={len(self.fused_featurizer.blocks) - 2})
            )
            self.embed_dim += self.fused_featurizer.embed_dim

        # Patch `vision_backbone.featurizer` and `vision_backbone.fused_featurizer` with HF-Compatible LayerScale
        for module in self.modules():
            if isinstance(module, LayerScale):
                ls_apply_patch(module)

        # [HF-Compatibility] Set `_is_hf_initialized` *recursively* so that weights do not get re-initialized
        for module in self.modules():
            module._is_hf_initialized = True

    @staticmethod
    def get_fsdp_wrapping_policy() -> Callable:
        """Return a simple FSDP policy that wraps the _entire_ featurizer."""
        vit_wrap_policy = partial(_module_wrap_policy, module_classes={VisionTransformer})
        transformer_block_policy = partial(transformer_auto_wrap_policy, transformer_layer_cls={Block})
        return partial(_or_policy, policies=[vit_wrap_policy, transformer_block_policy])

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        """Run image (`pixel_values`) through featurizer; if channel-stacked, then dispatch and sequence stack."""
        if not self.use_fused_vision_backbone:
            return self.featurizer(pixel_values)

        # Split `pixel_values :: [bsz, 2 * 3, resolution, resolution]` =>> featurize =>> channel stack
        img, img_fused = torch.split(pixel_values, [3, 3], dim=1)
        patches, patches_fused = self.featurizer(img), self.fused_featurizer(img_fused)

        if self.fused_first:
            return torch.cat([patches_fused, patches], dim=2)
        else:
            return torch.cat([patches, patches_fused], dim=2)

    @property
    def default_image_resolution(self) -> Tuple[int, int, int]:
        return 3, self.image_sizes[0], self.image_sizes[0]
