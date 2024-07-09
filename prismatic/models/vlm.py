"""
vlm.py

Prismatic VLM model definitions (derived from `transformers.PreTrainedModel`) providing a compositional interface
for defining different VLM instances (e.g., combinations of vision backbones, LLMs, architectures). Unlike v1 of the
Prismatic codebase, these classes are designed to implement the respective HuggingFace Transformers APIs, with full
compatibility with the rest of the HF ecosystem (e.g., Generation Utilities, PEFT finetuning, etc.).

References [LLaVa, IDEFICS-2]:
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/llava/modeling_llava.py
    => https://github.com/huggingface/transformers/blob/main/src/transformers/models/idefics2/modeling_idefics2.py
"""

from functools import partial
from typing import Any, Callable, ClassVar, Dict, List, Optional, Tuple, Union

import tokenizers
import torch
import torch.nn as nn
import transformers
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy, transformer_auto_wrap_policy
from transformers import AutoModelForCausalLM, LlamaForCausalLM, MistralForCausalLM, PretrainedConfig, PreTrainedModel
from transformers.modeling_outputs import CausalLMOutputWithPast
from transformers.models.llama.modeling_llama import LlamaDecoderLayer
from transformers.models.mistral.modeling_mistral import MistralDecoderLayer

from prismatic.models.backbones import PrismaticVisionBackbone
from prismatic.models.configuration import PrismaticConfig
from prismatic.overwatch import initialize_overwatch

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# PyTorch / HuggingFace Default IGNORE_INDEX (for labels)
IGNORE_INDEX = -100

# === Language Model Explicit Class Mappings ===
LLM_META = {
    "llama2": (LlamaForCausalLM, LlamaDecoderLayer),
    "vicuna": (LlamaForCausalLM, LlamaDecoderLayer),
    "mistral": (MistralForCausalLM, MistralDecoderLayer),
}


# === Prismatic Projector (nn.Module) Definition ===
class PrismaticProjector(nn.Module):
    def __init__(self, use_fused_vision_backbone: bool, vision_dim: int, llm_dim: int) -> None:
        super().__init__()
        self.use_fused_vision_backbone = use_fused_vision_backbone
        self.vision_dim, self.llm_dim = vision_dim, llm_dim

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(self.vision_dim)

        # Switch on `use_fused_vision_backbone` =>> use slightly different MLPs and projection factors!
        if not self.use_fused_vision_backbone:
            self.fc1 = nn.Linear(self.vision_dim, self.llm_dim, bias=True)
            self.fc2 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
        else:
            initial_projection_dim = 4 * vision_dim
            self.fc1 = nn.Linear(self.vision_dim, initial_projection_dim, bias=True)
            self.fc2 = nn.Linear(initial_projection_dim, self.llm_dim, bias=True)
            self.fc3 = nn.Linear(self.llm_dim, self.llm_dim, bias=True)
            self.act_fn1 = nn.GELU()
            self.act_fn2 = nn.GELU()

    def forward(self, img_patches: torch.Tensor) -> torch.Tensor:
        if not self.use_fused_vision_backbone:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
        else:
            projected_features = self.fc1(img_patches)
            projected_features = self.act_fn1(projected_features)
            projected_features = self.fc2(projected_features)
            projected_features = self.act_fn2(projected_features)
            projected_features = self.fc3(projected_features)

        return projected_features


# === Core VLM (HF-Compatible) Class Definitions ===
class PrismaticPreTrainedModel(PreTrainedModel):
    config_class: PretrainedConfig = PrismaticConfig
    base_model_prefix: str = "model"
    supports_gradient_checkpointing: bool = True

    _no_split_modules: ClassVar[List[str]] = ["PrismaticProjector"]
    _skip_keys_device_placement: str = "past_key_values"

    @property
    def _supports_flash_attn_2(self) -> bool:
        return self.language_model._supports_flash_attn_2

    @property
    def _supports_sdpa(self) -> bool:
        return self.language_model._supports_sdpa

    def _init_weights(self, module: nn.Module) -> None:
        # [CONTRACT] :: Each individual submodule of `PrismaticPreTrainedModel` is responsible for initializing its own
        #               weights; this function is *meant* to do nothing!
        pass


class PrismaticForVision2Seq(PrismaticPreTrainedModel):
    def __init__(self, config: PrismaticConfig, load_pretrained_backbones: bool = False) -> None:
        super().__init__(config)

        # Create Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "projector", "language_model"]
        self.trainable_module_keys = []

        # [Validation] Lightweight Validate on Transformers/Tokenizers Dependency Versions
        if (transformers.__version__ != "4.40.1") or (tokenizers.__version__ != "0.19.1"):
            overwatch.warning(
                f"Expected `transformers==4.40.1` and `tokenizers==0.19.1` but got "
                f"`transformers=={transformers.__version__}` and `tokenizers=={tokenizers.__version__}`; "
                f"there might be inference-time regressions due to dependency changes. If in doubt, please"
                f"use the above versions."
            )

        # Instantiate PrismaticVisionBackbone (w/ Potential Fused Backbone)
        self.vision_backbone = PrismaticVisionBackbone(
            self.config.use_fused_vision_backbone,
            self.config.timm_model_ids,
            self.config.timm_override_act_layers,
            self.config.image_sizes,
            load_pretrained_backbones=load_pretrained_backbones,
        )

        # Instantiate Multimodal Projector
        self.projector = PrismaticProjector(
            self.config.use_fused_vision_backbone, self.vision_backbone.embed_dim, self.config.text_config.hidden_size
        )

        # Instantiate LLM Backbone (via HF AutoModelForCausalLM)
        self.llm_cls, self.llm_transformer_layer_cls = LLM_META[config.llm_family]
        if not load_pretrained_backbones:
            self.language_model = AutoModelForCausalLM.from_config(
                self.config.text_config,
                attn_implementation=self.config._attn_implementation,
                torch_dtype=self.config.torch_dtype,
            )
        elif config.llm_family in {"llama2", "llama2-chat", "vicuna", "mistral"}:
            # Vicuña v1.5 Default Configuration is "Broken" --> need the following to suppress `UserWarnings`
            llm_kwargs = (
                {"do_sample": False, "temperature": 1.0, "top_p": 1.0} if self.config.llm_family in {"vicuna"} else {}
            )
            self.language_model = self.llm_cls.from_pretrained(
                self.config.llm_hf_hub_path,
                attn_implementation=self.config._attn_implementation,
                low_cpu_mem_usage=True,
                **llm_kwargs,
            )

            # When loading an LLM (for VLM pretraining) we add two special tokens =>> <PAD> and <image>
            #   =>> NOTE :: This change should already be reflected in `config` -- validate!
            assert self.language_model.config.to_dict() == self.config.text_config.to_dict(), "Configuration mismatch!"
            assert self.config.pad_token_id == self.language_model.vocab_size, "Unexpected <PAD> token!"
            assert self.config.image_token_id == self.language_model.vocab_size + 1, "Unexpected <image> token!"

            # Resize LLM Embeddings
            self.resize_token_embeddings(
                self.language_model.vocab_size + 2, pad_to_multiple_of=self.config.pad_to_multiple_of
            )

            # Update `language_model.config` and `self.config.text_config`
            self.language_model.config.pad_token_id = self.config.pad_token_id
            self.language_model.config.torch_dtype = torch.bfloat16
            self.config.text_config = self.language_model.config

        else:
            raise ValueError(f"Unsupported LLM Family `{config.llm_family}` with HF Path `{config.llm_hf_hub_path}`")

        # HF Boilerplate =>> initializes weights via `_init_weights()` and sets gradient checkpointing
        self.post_init()

    # === General Utilities ===
    def get_vision_backbone_cfgs(self) -> List[Dict[str, Any]]:
        vision_cfgs = [self.vision_backbone.featurizer.pretrained_cfg]
        if len(self.config.timm_model_ids) > 1:
            vision_cfgs.append(self.vision_backbone.fused_featurizer.pretrained_cfg)

        # [CONTRACT] Input Image Size will *always* be `self.config.image_sizes`
        for idx, cfg in enumerate(vision_cfgs):
            cfg["input_size"] = (3, self.config.image_sizes[idx], self.config.image_sizes[idx])

        return vision_cfgs

    def freeze_backbones(self, stage: str) -> None:
        """
        Set `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.language_model.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info("[Frozen]    🥶 =>> Vision Backbone", ctx_level=1)
            overwatch.info("[Frozen]    🥶 =>> LLM Backbone", ctx_level=1)
            overwatch.info("[TRAINABLE] 🔥 =>> Projector", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.language_model.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "language_model"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info("[Frozen]    🥶 =>> Vision Backbone", ctx_level=1)
            overwatch.info("[TRAINABLE] 🔥 =>> LLM Backbone", ctx_level=1)
            overwatch.info("[TRAINABLE] 🔥 =>> Projector", ctx_level=1)

        elif stage == "full-finetune":
            self.vision_backbone.default_dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.language_model.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "language_model"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info("[TRAINABLE] 🔥 =>> Vision Backbone", ctx_level=1)
            overwatch.info("[TRAINABLE] 🔥 =>> LLM Backbone", ctx_level=1)
            overwatch.info("[TRAINABLE] 🔥 =>> Projector", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for Prismatic VLM Training! Try < align | finetune >")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={self.llm_transformer_layer_cls}
        )

        # Get Prismatic VLM (Top-Level) Wrapping Policy =>> for now, just wraps `self.projector`
        prismatic_fsdp_wrapping_policy = partial(_module_wrap_policy, module_classes={PrismaticProjector})

        # Return union (_or_) over constituent policies =>> this gets applied recursively, bottom-up (tree-map)
        #   => Note: there is *not* a fall-through policy; any module that isn't covered by the above constituents will
        #            automatically be folded into the root VLM FSDP instance.
        return partial(
            _or_policy,
            policies=[
                vision_fsdp_wrapping_policy,
                llm_fsdp_wrapping_policy,
                prismatic_fsdp_wrapping_policy,
            ],
        )

    # === `PreTrainedModel` Boilerplate ===
    def get_input_embeddings(self) -> nn.Module:
        return self.language_model.get_input_embeddings()

    def set_input_embeddings(self, value: nn.Module) -> None:
        self.language_model.set_input_embeddings(value)

    def get_output_embeddings(self) -> nn.Module:
        return self.language_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings: nn.Module) -> None:
        self.language_model.set_output_embeddings(new_embeddings)

    def get_decoder(self) -> nn.Module:
        return self.language_model.get_decoder()

    def set_decoder(self, decoder: nn.Module) -> None:
        self.language_model.set_decoder(decoder)

    def tie_weights(self) -> None:
        self.language_model.tie_weights()  # Note: `Llama-2` and `Mistral` don't tie weights (no-op)

    def resize_token_embeddings(
        self, new_num_tokens: Optional[int] = None, pad_to_multiple_of: Optional[int] = None
    ) -> nn.Embedding:
        updated_embeddings = self.language_model.resize_token_embeddings(new_num_tokens, pad_to_multiple_of)

        # Update config/instance variables (language model variables get automatically updated in above call)
        self.vocab_size = updated_embeddings.num_embeddings
        self.config.vocab_size = updated_embeddings.num_embeddings
        self.config.text_config.vocab_size = updated_embeddings.num_embeddings

        return updated_embeddings

    # === Core Prismatic VLM `forward()` Logic ===
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[Tuple, CausalLMOutputWithPast]:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        # Respect `use_cache` only if not training (even if `gradient_checkpointing` is off)
        use_cache = use_cache if not self.training else False

        # [CONTRACT] `input_embeds` should always be None (we don't support custom handling)
        assert inputs_embeds is None, f"Unexpected `{inputs_embeds = }`"

        # Handle Inference (leverage cache, short-circuit on just LLM `forward()`)
        if input_ids.shape[1] == 1 and past_key_values is not None:
            # Attention Mask and Position IDs do *not* currently reflect the `num_patches` tokens we spliced in; fix!
            past_length = past_key_values[0][0].shape[2]
            extended_attention_mask = torch.ones(
                (attention_mask.shape[0], past_length + 1), dtype=attention_mask.dtype, device=attention_mask.device
            )
            extended_attention_mask[torch.where(attention_mask == 0)] = 0

            # Update Position IDs w/ "extended" length
            position_ids = torch.sum(extended_attention_mask, dim=1).unsqueeze(-1) - 1

            # We're leveraging the cache, so just redirect to `self.llm_backbone` with `input_ids` and `past_key_values`
            return self.language_model(
                input_ids=input_ids,
                attention_mask=extended_attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=None,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Handle `pixel_values` is None (no images) --> simple unimodal `forward()`
        elif pixel_values is None:
            return self.language_model(
                input_ids=input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                past_key_values=past_key_values,
                inputs_embeds=None,
                labels=labels,
                use_cache=use_cache,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )

        # Parse `bsz` and `seq_len` & compute `padding_side` ("right" during training and "left" during generation)
        bsz, seq_len = input_ids.shape
        is_left_padding = torch.any(input_ids[:, 0] == self.config.pad_token_id)

        # Identify where all <image> tokens are located --> should be no more than 1 per "row" (input example)
        image_token_mask = input_ids == self.config.image_token_id
        image_tokens_per_example = image_token_mask.sum(dim=-1)

        # Compute the maximum "fused" sequence length (after adding `patch_embeddings`)
        fused_seq_len = (image_tokens_per_example.max() * (self.vision_backbone.num_patches - 1)) + seq_len

        # Compute Positions to Insert Image Patches
        #   =>> `image_token_mask` gives us offset; <image> replaced w/ `num_patches` "tokens" (+ `num_patches - 1`)
        #   =>> `torch.cumsum` tells us how each <image> token shifts text token positions
        new_position_idxs = torch.cumsum(image_token_mask * (self.vision_backbone.num_patches - 1) + 1, dim=-1) - 1
        num_img_pad = fused_seq_len - 1 - new_position_idxs[:, -1]
        if is_left_padding:
            new_position_idxs += num_img_pad[:, None]

        # Get Positions for copying original `input_ids`
        batch_idxs, text_idxs = torch.where(input_ids != self.config.image_token_id)
        text_copy_position_idxs = new_position_idxs[batch_idxs, text_idxs]  # Shape: [bsz * (input) seq_len]

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            patch_features = self.vision_backbone(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim]
        projected_patch_embeddings = self.projector(patch_features)

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.get_input_embeddings()(input_ids)

        # Create "Fused" Embeddings, Attention Mask, Labels
        fused_embeddings = torch.zeros(
            bsz, fused_seq_len, input_embeddings.shape[-1], dtype=input_embeddings.dtype, device=input_embeddings.device
        )
        fused_attention_mask = torch.zeros(bsz, fused_seq_len, dtype=attention_mask.dtype, device=attention_mask.device)
        fused_labels = None
        if labels is not None:
            fused_labels = torch.full(
                (bsz, fused_seq_len), fill_value=IGNORE_INDEX, dtype=labels.dtype, device=labels.device
            )

        # Copy over original `input_ids`, `attention_mask`, `labels` (optional)
        fused_embeddings[batch_idxs, text_copy_position_idxs] = input_embeddings[batch_idxs, text_idxs]
        fused_attention_mask[batch_idxs, text_copy_position_idxs] = attention_mask[batch_idxs, text_idxs]
        if labels is not None:
            fused_labels[batch_idxs, text_copy_position_idxs] = labels[batch_idxs, text_idxs]

        # Compute `patch_embeddings` insertion positions in `fused_*` -- match on ZERO/EMPTY values!
        #   =>> Theoretically supports more than one <image> --> hence the filter on `num_img_pad`
        img_insert_mask = torch.all(fused_embeddings == 0, dim=-1)
        img_insert_mask = img_insert_mask & (img_insert_mask.cumsum(-1) - 1 >= num_img_pad[:, None])
        if img_insert_mask.sum() != projected_patch_embeddings.shape[:-1].numel():
            raise ValueError(f"Mismatch: {image_token_mask.sum()} <image> tokens != {len(pixel_values)} images!")

        # Insert `patch_embeddings` & create `fused_embeddings`, `fused_attention_mask`
        fused_embeddings[img_insert_mask] = projected_patch_embeddings.reshape(-1, projected_patch_embeddings.shape[-1])
        fused_attention_mask |= img_insert_mask

        # Recompute `position_ids` (only necessary for generation / left padding)!
        if is_left_padding or position_ids is not None:
            position_ids = (fused_attention_mask.cumsum(dim=-1) - 1).masked_fill_(fused_attention_mask == 0, 1)

        # Run LLM `forward()`
        return self.language_model(
            input_ids=None,
            attention_mask=fused_attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=fused_embeddings,
            labels=fused_labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

    # === GenerationMixin Methods ===
    def prepare_inputs_for_generation(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        use_cache: Optional[bool] = None,
        **kwargs: torch.Tensor,
    ) -> Dict[str, torch.Tensor]:
        if past_key_values is not None:
            if not isinstance(past_key_values, tuple):
                raise ValueError(f"Unexpected type for past_key_values: {type(past_key_values)} - expected tuple!")

            # Compute Cache / Past Length =>> Note `past_key_values` has the following structure:
            #   => [[# Transformer Blocks]] =>> [[2 (keys, values)]] =>> torch.Tensor [bsz, n_heads, seq_len, d]
            past_length = past_key_values[0][0].shape[2]

            # === Magic Rules from Llama-2/Mistral/HF Models :: Keep only unprocessed tokens! ===

            # 1 =>> If length of the attention_mask exceeds input_ids, then we are in a setting where some inputs
            #       are *exclusively* passed as part of the cache (e.g., when passing `input_embeds`).
            # NOTE :: This should never happen for Prismatic models (no `input_embeds` allowed!)
            if attention_mask is not None and attention_mask.shape[1] > input_ids.shape[1]:
                raise ValueError("Past Key Values Case (1) =>> Unsupported for Prismatic!")

            # 2 =>> If the past_length is smaller than input_ids, then input_ids holds all input tokens; discard
            #       based on past_length.
            elif past_length < input_ids.shape[1]:
                raise ValueError("Past Key Values Case (2) =>> Unsupported for Prismatic!")

            # 3 =>> Otherwise (past_length >= input_ids.shape[1]) --> we've eaten an <image>! Assume `input_ids` only
            #       has unprocessed tokens (at the end)
            elif self.config.image_token_id in input_ids:
                input_ids = input_ids[:, -1:]

            else:
                raise ValueError("Past Key Values Fall-Through Case =>> Unsupported for Prismatic!")

        # Validate Padding Side = LEFT
        if input_ids.shape[0] > 1 and (attention_mask is None or torch.any(attention_mask[:, -1] == 0)):
            raise ValueError("Batched Generation requires `input_ids` with <LEFT> Padding!")

        # Populate Position IDs based on Attention Mask --> left padding means we need to shift!
        position_ids = kwargs.get("position_ids", None)
        if attention_mask is not None:
            position_ids = attention_mask.long().cumsum(dim=-1) - 1
            position_ids.masked_fill_(attention_mask == 0, 1)
            if past_key_values is not None:
                position_ids = position_ids[:, -input_ids.shape[1] :]

        # If `inputs_embeds` are passed, we only want to use them in the 1st generation step
        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        # Make sure `pixel_values` are preserved in `model_inputs`
        model_inputs.update(
            {
                "attention_mask": attention_mask,
                "pixel_values": pixel_values,
                "position_ids": position_ids,
                "past_key_values": past_key_values,
                "use_cache": use_cache,
            }
        )

        return model_inputs
