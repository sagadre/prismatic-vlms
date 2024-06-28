"""
prismatic.py

PyTorch Module defining a PrismaticVLM, our general interface for defining the various different VLMs in our work.

Notes:
    - For now, we don't subclass `transformers.PretrainedModel` (or CausalLM). Instead, we assume a very limited subset
      of the {Model}ForCausalLM API that enables dispatch to the underlying LLM's `generate` utilities (feeding inputs
      through our custom projection shim).
"""

from __future__ import annotations

from functools import partial
from pathlib import Path
from typing import Callable, Dict, List, Optional, Type, Union

import torch
from PIL import Image
from torch.distributed.fsdp.wrap import _module_wrap_policy, _or_policy
from transformers.modeling_outputs import CausalLMOutputWithPast

from prismatic.models.backbones.llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import PromptBuilder
from prismatic.models.backbones.vision import VisionBackbone
from prismatic.models.vlms.base_vlm import VLM
from prismatic.overwatch import initialize_overwatch
from prismatic.util.nn_utils import FusedMLPProjector, LinearProjector, MLPProjector

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# HuggingFace Default / LLaMa-2 IGNORE_INDEX (for labels)
IGNORE_INDEX = -100


class PrismaticVLM(VLM):
    def __init__(
        self,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> None:
        super().__init__(
            "prismatic",
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
        )

        # Set Weight Initialization Seed for Projector Consistency
        torch.manual_seed(vision_backbone.embed_dim)

        # Initialize Projection (Adapter) based on `arch_specifier`
        self.arch_specifier = arch_specifier
        if arch_specifier == "linear":
            self.projector = LinearProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("fused-gelu-mlp"):
            self.projector = FusedMLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        elif arch_specifier.endswith("gelu-mlp"):
            self.projector = MLPProjector(vision_backbone.embed_dim, llm_backbone.embed_dim)
        else:
            raise ValueError(f"PrismaticVLM with `{arch_specifier = }` is not supported!")

        # Trackers
        self.vision_backbone_requires_grad = False

        # Set Module Keys =>> used in Checkpoint Saving / Model Loading
        self.all_module_keys = ["vision_backbone", "llm_backbone", "projector"]
        self.trainable_module_keys = []

        # === Generation Utilities ===
        #   => For computing likelihoods --> get tokens corresponding to "True", "False" and "Yes", "No"
        self.string2idx = {}
        for trigger_string in ["True", "False", "Yes", "No"] + [chr(ord("A") + i) for i in range(26)]:
            token_idx_list = self.llm_backbone.tokenizer.encode(trigger_string, add_special_tokens=False)
            assert len(token_idx_list) == 1, f'String "{trigger_string}" is tokenized as more than one token!'
            self.string2idx[trigger_string] = token_idx_list[0]

    @classmethod
    def from_pretrained(
        cls,
        pretrained_checkpoint: Path,
        model_id: str,
        vision_backbone: VisionBackbone,
        llm_backbone: LLMBackbone,
        enable_mixed_precision_training: bool = True,
        arch_specifier: str = "gelu-mlp",
    ) -> PrismaticVLM:
        """Initialize a PrismaticVLM from a pretrained checkpoint, freezing all weights, tailored for inference."""
        vlm = cls(
            model_id,
            vision_backbone,
            llm_backbone,
            enable_mixed_precision_training=enable_mixed_precision_training,
            arch_specifier=arch_specifier,
        )

        # Load from Checkpoint (Custom --> should load both *projector* and *llm* weights)
        model_state_dict = torch.load(pretrained_checkpoint, map_location="cpu")["model"]
        assert (
            "projector" in model_state_dict and "llm_backbone" in model_state_dict
        ), "PrismaticVLM `from_pretrained` expects checkpoint with keys for `projector` AND `llm_backbone`!"

        vlm.projector.load_state_dict(model_state_dict["projector"])
        vlm.llm_backbone.load_state_dict(model_state_dict["llm_backbone"])

        # Freeze Weights
        vlm.requires_grad_(False)
        vlm.eval()

        return vlm

    def get_prompt_builder(self, system_prompt: Optional[str] = None) -> PromptBuilder:
        prompt_initializer: Type[PromptBuilder] = self.llm_backbone.prompt_builder_fn
        return prompt_initializer(self.model_family, system_prompt=system_prompt)

    def freeze_backbones(self, stage: str) -> None:
        """
        This function sets `requires_grad_` on each of the component modules explicitly, depending on stage.

        We support two separate stages --> "align" and "finetune".
            => "align" --> vision_backbone*, llm_backbone* are frozen; only the `projector` is trained.
            => "finetune" --> vision_backbone* is frozen; both `projector` and `llm_backbone` are trained.

        :param stage: Pretraining stage in < "align" | "finetune" | "full-finetune" >
        """
        if stage == "align":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(False)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Trainable Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[Frozen]    🥶 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "finetune":
            self.vision_backbone.requires_grad_(False)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = False

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[Frozen]    🥶 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        elif stage == "full-finetune":
            self.vision_backbone.dtype = torch.float32
            self.vision_backbone.requires_grad_(True)
            self.llm_backbone.requires_grad_(True)
            self.projector.requires_grad_(True)

            # Add to `self.trainable_module_keys`
            self.trainable_module_keys = ["vision_backbone", "projector", "llm_backbone"]

            # Update Trackers
            self.vision_backbone_requires_grad = True

            # Explicitly Log Frozen / Unfrozen Components
            overwatch.info(f"[TRAINABLE] 🔥 =>> Vision Backbone `{self.vision_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> LLM Backbone `{self.llm_backbone.identifier}`", ctx_level=1)
            overwatch.info(f"[TRAINABLE] 🔥 =>> Projector `{self.arch_specifier}`", ctx_level=1)

        else:
            raise ValueError(f"Stage `{stage}` is not supported for LLaVa! Try < align | finetune >")

    def load_from_checkpoint(self, stage: str, run_dir: Path, pretrained_checkpoint: Optional[Path] = None) -> None:
        """Load weights from checkpoint (if required by the given stage)."""
        assert stage in {"align", "finetune", "full-finetune"}, f"Stage {stage} is not supported!"

        # If we're running a `no-align` architecture, we're good!
        if self.arch_specifier.startswith("no-align"):
            overwatch.info(
                f"PrismaticVLM with `{self.arch_specifier = }` does not require pretrained weights!", ctx_level=1
            )
            return

        # Otherwise, handle stage-specific logic!
        if stage == "align":
            overwatch.info("Stage `align` does not require pretrained weights =>> Starting Training", ctx_level=1)
            return

        # Otherwise, load from `pretrained_checkpoint` or match on `run_dir` (s/+stage-finetune/+stage-align/g)
        overwatch.info("Stage `finetune` requires `align` pretrained weights", ctx_level=1)

        # Config specifies path to a checkpoint to load
        if pretrained_checkpoint is not None:
            overwatch.info(f"Loading from Provided Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])

            return

        # [Contract] If no `pretrained_checkpoint`, assume `align` lives in the run directory; string substitution!
        model, scale, _, seed = run_dir.name.split("+")
        align_dirs = [
            d
            for d in run_dir.parent.iterdir()
            if (d.name.startswith(f"{model}+{scale}") and d.name.endswith(f"+stage-align+{seed}"))
        ]
        assert len(align_dirs) == 1, "Multiple or No Valid Pretrained Directories Exist -- Double Check `runs`!"
        if (pretrained_checkpoint := (align_dirs[0] / "checkpoints" / "latest-checkpoint.pt")).exists():
            overwatch.info(f"Loading from Discovered Checkpoint `{pretrained_checkpoint}`", ctx_level=1)
            model_state_dict = torch.load(pretrained_checkpoint)["model"]
            self.projector.load_state_dict(model_state_dict["projector"])
        else:
            raise ValueError(f"Could not find valid `align` checkpoint at {pretrained_checkpoint}!")

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return an FSDP _or_policy over the policies returned by each individual backbone (and our VLM policy)."""
        vision_fsdp_wrapping_policy = self.vision_backbone.get_fsdp_wrapping_policy()
        llm_fsdp_wrapping_policy = self.llm_backbone.get_fsdp_wrapping_policy()

        # Get Prismatic Wrapping Policy =>> just a module wrapping policy around `self.projector`
        prismatic_fsdp_wrapping_policy = partial(
            _module_wrap_policy,
            module_classes={LinearProjector, MLPProjector, FusedMLPProjector},
        )

        # Return union (_or_) over constituent policies
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

    # Note =>> We're not explicitly subclassing `PreTrainedModel` because we don't need the bloat; however, `forward()`
    #          *must* match the signature of a `{Model}ForCausalLM` so that we can inherit from `GenerationMixin`

    # ruff: noqa: C901
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
        # multimodal_indices: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        """Run a forward pass through the VLM, returning a CausalLMOutputWithPast instance (contains loss)."""

        # Handle Inference (leverage cache, short-circuit on just LLM forward)
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
            output = self.llm_backbone(
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
            return output

        # Handle `pixel_values` is None (no images) --> simple unimodal forward
        elif pixel_values is None:
            return self.llm_backbone(
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
        is_left_padding = torch.any(input_ids[:, 0] == self.llm_backbone.llm.config.pad_token_id)

        # Identify where all <image> tokens are located --> should be no more than 1 per "row" (input example)
        image_token_mask = input_ids == self.llm_backbone.llm.config.image_token_id
        image_tokens_per_example = image_token_mask.sum(dim=-1)

        # Compute the maximum "fused" sequence length (after adding `patch_embeddings`)
        fused_seq_len = (image_tokens_per_example.max() * (self.vision_backbone.num_patches - 1)) + seq_len

        # Compute Positions to Insert Image Patches
        #   =>> `image_token_mask` gives us offset; <image> replaced w/ `num_patches` "tokens" (add `num_patches - 1`)
        #   =>> `torch.cumsum` tells us how each <image> token shifts text token positions
        new_position_idxs = torch.cumsum(image_token_mask * (self.vision_backbone.num_patches - 1) + 1, dim=-1) - 1
        num_img_pad = fused_seq_len - 1 - new_position_idxs[:, -1]
        if is_left_padding:
            new_position_idxs += num_img_pad[:, None]

        # Get Positions for copying original `input_ids`
        batch_idxs, text_idxs = torch.where(input_ids != self.llm_backbone.llm.config.image_token_id)
        text_copy_position_idxs = new_position_idxs[batch_idxs, text_idxs]  # Shape: [bsz * (input) seq_len]

        # Run Visual Feature Extraction
        with torch.set_grad_enabled(self.vision_backbone_requires_grad):
            if isinstance(pixel_values, dict):
                patch_features = self.vision_backbone({k: pixel_values[k] for k in pixel_values})
            else:
                patch_features = self.vision_backbone(pixel_values)

        # Projection Logic :: [bsz, num_patches, llm_embed_dim]
        projected_patch_embeddings = self.projector(patch_features)

        # Get Input Embeddings from LLM Backbone :: [bsz, input_seq_len, llm_embed_dim]
        input_embeddings = self.llm_backbone.embed_input_ids(input_ids)

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

        # Run LLM Forward --> returns CausalLMOutputWithPast!
        return self.llm_backbone(
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
    #   => Note: The following methods override the functionality of `transformers.GenerationMixin`; these expect the
    #            contract in each of the function signatures, and also expect our `forward` function to roughly take
    #            the same arguments as the underlying LLM (see `LlamaModelForCausalLM` as an example)

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
            elif self.llm_backbone.llm.config.image_token_id in input_ids:
                input_ids = input_ids[:, -1:]

            else:
                raise ValueError("Past Key Values Fall-Through Case =>> Unsupported for Prismatic!")

        # Validate Padding Side = LEFT
        if input_ids.shape[0] > 1 and (attention_mask is None or torch.any(attention_mask[:, -1] == 0)):
            raise ValueError("PrismaticVLM Batched Generation requires `input_ids` with <LEFT> Padding!")

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

    # === Note :: Generate Batch w/ `return_string_probabilities` is necessary for `vlm-evaluation` ===
    @torch.inference_mode()
    def generate_batch(
        self,
        pixel_values: Union[torch.Tensor, Dict[str, torch.Tensor]],
        texts: List[str],
        return_string_probabilities: Optional[List[str]] = None,
        **kwargs: str,
    ) -> Union[List[str], List[List[float]]]:
        # For now, only support generation with a batch size of 1 for simplicity
        tokenizer = self.llm_backbone.tokenizer

        # Prepare Inputs
        batch_input_ids = [
            tokenizer(text, truncation=True, return_tensors="pt").input_ids.to(self.device) for text in texts
        ]
        if isinstance(pixel_values, torch.Tensor):
            pixel_values = pixel_values[None, ...].to(self.device)
        elif isinstance(pixel_values, dict):
            pixel_values = {k: v[None, ...].to(self.device) for k, v in pixel_values.items()}
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

        # Create Output Lists
        gen_texts, gen_probabilities = [], []

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            for idx, input_ids in enumerate(batch_input_ids):
                if isinstance(pixel_values, torch.Tensor):
                    pixel_values = pixel_values[idx]
                elif isinstance(pixel_values, dict):
                    pixel_values = {k: pixel_values[k][idx] for k in pixel_values}
                else:
                    raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values)}")

                # Handle `return_string_probabilities`
                if return_string_probabilities is None:
                    full_out_ids = super().generate(input_ids=input_ids, pixel_values=pixel_values, **kwargs)
                    gen_ids = full_out_ids[0, input_ids.shape[1] :]

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                else:
                    full_out_dict = super().generate(
                        input_ids=input_ids,
                        pixel_values=pixel_values,
                        output_scores=True,
                        return_dict_in_generate=True,
                        **kwargs,
                    )

                    # Generation pattern should usually be [TOKEN] <EOS> for True/False and Yes/No Generations
                    gen_ids = full_out_dict.sequences[0, input_ids.shape[1] :]

                    # [Debug] Verify that the first token generated is in `self.string2idx.values()`
                    # assert gen_ids[0] in self.string2idx.values(), "Generated ID not in mapping!"

                    # Decode `gen_ids` and strip any <EOS> tokens
                    gen_texts.append(tokenizer.decode(gen_ids, skip_special_tokens=True).strip())

                    # Get all token probabilities --> softmax over logits
                    token_probs = torch.softmax(full_out_dict.scores[0][0], dim=0)

                    # Get *normalized* probabilities for all values in `return_token_probabilities`
                    slice_idxs = torch.tensor([self.string2idx[s] for s in return_string_probabilities])
                    string_probs_unnormalized = token_probs[slice_idxs]
                    string_probs = string_probs_unnormalized / string_probs_unnormalized.sum()
                    gen_probabilities.append(string_probs.cpu().numpy().tolist())

        return gen_texts if return_string_probabilities is None else gen_probabilities

    @torch.inference_mode()
    def generate(self, images: Union[Image, List[Image]], prompt_texts: Union[str, List[str]], **kwargs) -> List[str]:
        if (not isinstance(images, list)) or (not isinstance(prompt_texts, list)):
            images, prompt_texts = [images], [prompt_texts]

        # Explicitly check that same number of images and prompts during batched generation
        if len(images) != len(prompt_texts):
            raise ValueError("Batched Generation expects homogenous batches of (image, text) pairs!")

        # Set `tokenizer.padding_side = LEFT` for generation!
        image_transform, tokenizer = self.vision_backbone.image_transform, self.llm_backbone.tokenizer
        tokenizer.padding_side = "left"

        # Prepare Inputs
        inputs = tokenizer(prompt_texts, truncation=True, padding=True, return_tensors="pt").to(self.device)
        input_ids, attention_mask = inputs.input_ids, inputs.attention_mask

        pixel_values = [image_transform(img) for img in images]
        if isinstance(pixel_values[0], torch.Tensor):
            pixel_values = torch.stack(pixel_values).to(self.device)
        elif isinstance(pixel_values[0], dict):
            pixel_values = {
                k: torch.stack([pixel_values[idx][k] for idx in range(len(pixel_values))]).to(self.device)
                for k in pixel_values[0]
            }
        else:
            raise ValueError(f"Unsupported `pixel_values` type = {type(pixel_values[0])}")

        # Invoke super().generate --> taps into `GenerationMixin` which (redirects) to `forward()`
        autocast_dtype = self.llm_backbone.half_precision_dtype
        with torch.autocast("cuda", dtype=autocast_dtype, enabled=self.enable_mixed_precision_training):
            # fmt: off
            generated_ids = super().generate(
                input_ids=input_ids,            # Shape: [bsz, seq]
                attention_mask=attention_mask,  # Shape: [bsz, seq]
                pixel_values=pixel_values,      # Shape: [bsz, 3, res, res] or Dict[str, Shape[1, 3, res, res]]
                **kwargs,
            )
            # fmt: on

        generated_text = tokenizer.batch_decode(generated_ids[:, input_ids.shape[1] :], skip_special_tokens=True)

        return generated_text
