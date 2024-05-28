"""
processing_prismatic.py

HuggingFace-style preprocessor definitions for Prismatic VLMs, inheriting from `ProcessorMixin`. Default configuration
specifies `siglip-224px+7b`.
"""

from dataclasses import dataclass
from typing import Any, ClassVar, List, Optional, Tuple, Union

import timm.data
import torch
import torchvision.transforms.functional as TVF
from PIL import Image
from torchvision.transforms import CenterCrop, Compose, Normalize, Resize, ToTensor
from transformers import PreTrainedTokenizerBase
from transformers.image_processing_utils import BatchFeature, ImageProcessingMixin
from transformers.processing_utils import ProcessorMixin
from transformers.tokenization_utils import PaddingStrategy, PreTokenizedInput, TextInput, TruncationStrategy
from transformers.utils import TensorType


# === Image Processing ===
def letterbox_pad_transform(image: Image.Image, padding_fill_value: Tuple[int, int, int]) -> Image.Image:
    """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
    (w, h), max_wh = image.size, max(image.size)
    horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
    padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)

    return TVF.pad(image, padding, fill=padding_fill_value, padding_mode="constant")


@dataclass
class LetterboxPadTransform:
    padding_fill_value: Tuple[int, int, int]

    def __call__(self, image: Image) -> Image:
        """Given a PIL.Image, pad to square by adding a symmetric border around the height/width."""
        (w, h), max_wh = image.size, max(image.size)
        horizontal_pad, vertical_pad = int((max_wh - w) / 2), int((max_wh - h) / 2)
        padding = (horizontal_pad, vertical_pad, horizontal_pad, vertical_pad)
        return TVF.pad(image, padding, fill=self.padding_fill_value, padding_mode="constant")


class PrismaticImageProcessor(ImageProcessingMixin):
    model_input_names: ClassVar[List[str]] = ["pixel_values"]

    def __init__(
        self,
        image_resize_strategy: str = "letterbox",
        input_size: Tuple[int, int, int] = (3, 224, 224),
        interpolation: str = "bicubic",
        mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        std: Tuple[float, float, float] = (0.5, 0.5, 0.5),
        **kwargs: str,
    ) -> None:
        """
        Initialize a PrismaticImageProcessor as a wrapper around a torchvision transform; this transform will be
        created by TIMM, and edited to follow our custom `image_resize_strategy` logic.

        @param image_resize_strategy: Prismatic image resize strategy in < resize-naive | resize-crop | letterbox >
        @param input_size: [TIMM :: `data_cfg`] Input image size as tuple (channels, width, height)
        @param interpolation: [TIMM :: `data_cfg`] Interpolation as string (default: "bicubic")
        @param mean: [TIMM :: `data_cfg`] Per-channel normalization mean as float tuple
        @param std: [TIMM :: `data_cfg`] Per-channel normalization std as float tuple
        """
        self.image_resize_strategy = image_resize_strategy

        # TIMM `data_cfg` Parameters
        self.input_size, self.interpolation, self.mean, self.std = input_size, interpolation, mean, std

        # Grab torchvision transform via TIMM =>> need to parse for specific "functional" transform values!
        # fmt: off
        transform = timm.data.create_transform(
            input_size=self.input_size,
            interpolation=self.interpolation,
            mean=self.mean,
            std=self.std,
            crop_pct=1.0,           # Set to 1.0 to ignore cropping (initial Resize sets `input_size`)
            crop_mode="center",     # Default crop mode -- no-op when `crop_pct == 1.0`
            is_training=False,      # No image augmentations when loading the transform!
        )
        # fmt: on

        # [Validation] Ensure appropriate transform structure, expected sizes
        if not (
            isinstance(transform, Compose)
            and (len(transform.transforms) == 4)
            and isinstance(transform.transforms[0], Resize)
            and isinstance(transform.transforms[1], CenterCrop)
            and isinstance(transform.transforms[2], ToTensor)
            and isinstance(transform.transforms[3], Normalize)
            and (transform.transforms[0].size == self.input_size[-1])
            and (transform.transforms[1].size == self.input_size[-2:])
        ):
            raise ValueError(f"Unexpected TIMM image transformation structure/sizes: `{transform}`")

        # HF Image Processors *must* be JSON-serializable; as such, cannot have torchvision. as an attribute.
        #   => Instead, we're going to parse the transform and call "torchvision.transforms.functional" (`tvf`)
        resize_t, crop_t, norm_t = transform.transforms[0], transform.transforms[1], transform.transforms[3]
        self.tvf_resize_params = {
            "size": resize_t.size,
            "interpolation": TVF.pil_modes_mapping[resize_t.interpolation],
            "max_size": None,
            "antialias": True,
        }
        self.tvf_crop_params = {"output_size": crop_t.size}
        self.tvf_normalize_params = {
            "mean": norm_t.mean.float().numpy().tolist(),
            "std": norm_t.std.float().numpy().tolist(),
            "inplace": False,
        }
        self.tvf_do_letterbox, self.tvf_letterbox_fill = False, None

        # Handle Prismatic `image_resize_strategy`
        if self.image_resize_strategy == "resize-naive":
            self.tvf_resize_params["size"] = (self.tvf_resize_params["size"], self.tvf_resize_params["size"])
        elif self.image_resize_strategy == "letterbox":
            self.tvf_do_letterbox, self.tvf_letterbox_fill = True, tuple([int(x * 255) for x in self.mean])
        elif self.image_resize_strategy == "resize-crop":
            pass
        else:
            raise ValueError(f"Image resize strategy `{self.image_resize_strategy}` is not supported!")

        # Dispatch **kwargs to super()
        super().__init__(**kwargs)

    def apply_transform(self, img: Image.Image) -> torch.Tensor:
        """Apply `functional` variant of TIMM's Transform = Compose([Resize -> CenterCrop -> ToTensor -> Normalize])"""
        if self.tvf_do_letterbox:
            img = letterbox_pad_transform(img, self.tvf_letterbox_fill)

        img = TVF.resize(img, **self.tvf_resize_params)
        img = TVF.center_crop(img, **self.tvf_crop_params)
        img_t = TVF.to_tensor(img)
        img_t = TVF.normalize(img_t, **self.tvf_normalize_params)

        return img_t

    def preprocess(
        self,
        images: Union[Image.Image, List[Image.Image]],
        return_tensors: Optional[Union[str, TensorType]] = None,
        **_: str,
    ) -> BatchFeature:
        """
        Preprocess an image (or batch of images); note that unlike the `transformers :: BaseImageProcessor` we
        explicitly only handle PIL.Image.Image instances for simplicity.

        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param return_tensors: BatchFeature default Tensor format (e.g., "pt" for torch); if None, returns np.ndarray

        @return: Instance of `transformers :: BatchFeature` with a single key "pixel_values"
        """
        if not isinstance(images, list):
            images = [images]

        # Apply `self.img_transform` to each image (will return list of torch.Tensors); stack into "batched" Tensor
        pixel_values = torch.stack([self.apply_transform(img.convert("RGB")) for img in images])

        # Return BatchFeature =>> note that for compatibility, constructor expects Dict[str, np.ndarray], so we convert
        return BatchFeature(data={"pixel_values": pixel_values.numpy()}, tensor_type=return_tensors)

    def __call__(self, images: Union[Image.Image, List[Image.Image]], **kwargs) -> BatchFeature:
        return self.preprocess(images, **kwargs)


# === PrismaticProcessor =>> Wraps both ImageProcessor and Tokenizer ===
#   =>> https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/processing_llava.py
class PrismaticProcessor(ProcessorMixin):
    attributes: ClassVar[List[str]] = ["image_processor", "tokenizer"]
    image_processor_class: str = "AutoImageProcessor"
    tokenizer_class: str = "AutoTokenizer"

    def __init__(
        self,
        image_processor: Optional[ImageProcessingMixin] = None,
        tokenizer: Optional[PreTrainedTokenizerBase] = None,
    ) -> None:
        super().__init__(image_processor, tokenizer)

    def __call__(
        self,
        text: Union[TextInput, PreTokenizedInput, List[TextInput], List[PreTokenizedInput]],
        images: Union[Image.Image, List[Image.Image]],
        padding: Union[bool, str, PaddingStrategy] = False,
        truncation: Optional[Union[bool, str, TruncationStrategy]] = None,
        max_length: Optional[int] = None,
        return_tensors: Optional[Union[str, TensorType]] = TensorType.PYTORCH,
    ) -> BatchFeature:
        """
        Preprocess a given (batch) of text/images for a Prismatic VLM; forwards text to the underlying LLM's tokenizer,
        forwards images to PrismaticImageProcessor.

        @param text: The (batch) of text to encode; must be a string or list of strings.
        @param images: A (batch of) PIL.Image.Image instance(s) to preprocess.
        @param padding: Sequence padding strategy (if multiple specified) in < True = "longest" | "max_length" | False >
        @param truncation: Truncation strategy for the output sequences; requires `max_length` to be specified
        @param max_length: Maximum length (in tokens) to truncate
        @param return_tensors: Type of return tensors (usually "pt" or TensorType.PYTORCH)

        @return: BatchFeature with keys for `input_ids`, `attention_mask` and `pixel_values`.
        """
        pixel_values = self.image_processor(images, return_tensors=return_tensors)["pixel_values"]
        text_inputs = self.tokenizer(
            text, return_tensors=return_tensors, padding=padding, truncation=truncation, max_length=max_length
        )

        # [Validate] Need same number of images and text inputs!
        if pixel_values.shape[0] != text_inputs.input_ids.shape[0]:
            raise ValueError("Batch is malformed; expected same number of images and text inputs!")

        return BatchFeature(data={**text_inputs, "pixel_values": pixel_values})

    # === Tokenizer Dispatch Utilities =>> check `PreTrainedTokenizerBase` for documentation ===
    def batch_decode(
        self,
        sequences: Union[List[int], List[List[int]], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> List[str]:
        return self.tokenizer.batch_decode(
            sequences=sequences,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    def decode(
        self,
        token_ids: Union[int, List[int], torch.Tensor, Any],  # `Any` = np.ndarray | tf.Tensor
        skip_special_tokens: bool = False,
        clean_up_tokenization_spaces: Optional[bool] = None,
        **kwargs: str,
    ) -> str:
        return self.tokenizer.decode(
            token_ids=token_ids,
            skip_special_tokens=skip_special_tokens,
            clean_up_tokenization_spaces=clean_up_tokenization_spaces,
            **kwargs,
        )

    @property
    def model_input_names(self) -> List[str]:
        tokenizer_input_names = self.tokenizer.model_input_names
        image_processor_input_names = self.image_processor.model_input_names

        return list(dict.fromkeys(tokenizer_input_names + image_processor_input_names))
