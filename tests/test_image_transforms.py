"""
test_image_transforms.py

Verify that `timm` image transforms match the expected structure, namely creating a `torchvision.transforms.Compose`
container, with Resize --> Optional[Crop] --> ToTensor() --> Normalize(...); all should expect a PIL.Image.
"""

import pytest
import timm
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor

from prismatic.models.backbones.vision.clip_vit import CLIP_VISION_BACKBONES
from prismatic.models.backbones.vision.dinov2_vit import DINOv2_VISION_BACKBONES
from prismatic.models.backbones.vision.in1k_vit import IN1K_VISION_BACKBONES
from prismatic.models.backbones.vision.siglip_vit import SIGLIP_VISION_BACKBONES

# === Aggregate all TIMM Model IDs ===
TIMM_MODELS = [
    *CLIP_VISION_BACKBONES.values(),
    *DINOv2_VISION_BACKBONES.values(),
    *IN1K_VISION_BACKBONES.values(),
    *SIGLIP_VISION_BACKBONES.values(),
]


@pytest.mark.parametrize("timm_path_or_url", TIMM_MODELS)
def test_timm_resolve_data_config_structure(timm_path_or_url: str) -> None:
    """Verify all `timm.data.resolve_model_data_config` calls return config objects with the appropriate keys/values."""
    featurizer = timm.create_model(timm_path_or_url, pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_model_data_config(featurizer)

    assert data_cfg.keys() == {"input_size", "interpolation", "mean", "std", "crop_pct", "crop_mode"}
    assert len(data_cfg["input_size"]) == 3
    assert data_cfg["input_size"][0] == 3
    assert data_cfg["interpolation"] == "bicubic"


@pytest.mark.parametrize("timm_path_or_url", TIMM_MODELS)
def test_timm_transform_structure(timm_path_or_url: str) -> None:
    """Verify that `timm.data.create_transform` returns the correction Torchvision transform structure."""
    featurizer = timm.create_model(timm_path_or_url, pretrained=True, num_classes=0)
    data_cfg = timm.data.resolve_model_data_config(featurizer)

    timm_image_transform = timm.data.create_transform(**data_cfg, is_training=False)
    assert isinstance(timm_image_transform, Compose)
    assert len(timm_image_transform.transforms) == 4

    assert isinstance(timm_image_transform.transforms[0], Resize)
    assert timm_image_transform.transforms[0].antialias is True
    assert isinstance(timm_image_transform.transforms[1], CenterCrop)
    assert isinstance(timm_image_transform.transforms[2], ToTensor)
    assert isinstance(timm_image_transform.transforms[3], Normalize)
