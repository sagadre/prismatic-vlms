"""
openlm.py

Class definition for all LLMs derived from https://github.com/mlfoundations/open_lm.
"""
from functools import partial
from typing import Callable, List, Optional, Type
import os

import torch
from torch import nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
import yaml
from argparse import Namespace

from prismatic.models.backbones.llm.base_llm import LLMBackbone
from prismatic.models.backbones.llm.prompting import (
    PromptBuilder,
    OpenlmPromptBuilder
)
from prismatic.models.backbones.llm.base_llm import overwatch

try:
    from open_lm.model import Block
    from open_lm.utils.transformers.hf_model import OpenLMforCausalLM
    from open_lm.utils.transformers.hf_config import OpenLMConfig
    from open_lm.model import create_params
    from open_lm.main import get_latest_checkpoint, load_model
    from transformers import GPTNeoXTokenizerFast
except ImportError:
    overwatch.info("open_lm not installed. Install with `pip install git+https://github.com/mlfoundations/open_lm.git`")


# fmt: off
OPENLM_MODELS = {
    "vicuna-v15-13b": {
        "llm_family": "open_lm", "llm_cls": OpenLMforCausalLM, "hf_hub_path": ""
    },
}


class OpenlmLLMBackbone(LLMBackbone):
    """
    OpenLM LLM Backbone class.

    llm_backbone_id: str
        - should contain "openlm" to indicate that this is an OpenLM model, either by being in the path or using "[openlm]" as a prefix
        - (if not inference_mode) should be the path to a directory with the following structure:
            [llm_backbone_id]: a directory containing the model's files
                - params.txt: a file containing the model's parameters
                - [checkpoints]: a directory containing model checkpoints
                    {any_name}.pt: a model checkpoint file with .pt extension
    inference_mode: bool = False
        - whether to load the model in inference mode 
        /!\ in inference mode, we don't load the base weights at initialization, they should be loaded later
    """
    def __init__(
        self,
        llm_backbone_id: str,  # open_lm model.json
        inference_mode: bool = False,
        **kwargs,
        # use_flash_attention_2: bool = False,
    ) -> None:
        if llm_backbone_id.startswith("[openlm]"):
            llm_backbone_id = llm_backbone_id.replace("[openlm]", "")
        super().__init__(llm_backbone_id)
        self.llm_family = "open_lm"
        self.inference_mode = inference_mode

        # Read params.txt in the model directory
        # [Contract] `params.txt` must be in the `llm_backbone_id` directory
        with open(llm_backbone_id + "/params.txt", "r") as f:
            open_lm_args = yaml.safe_load(f)
        open_lm_args = Namespace(**open_lm_args)
        self.llm_max_length = int(open_lm_args.seq_len)
        open_lm_args.fsdp = True
        open_lm_args.distributed = True

        self.llm = OpenLMforCausalLM(OpenLMConfig(create_params(open_lm_args)))

        # Initialize LLM
        # [Contract] `inference_mode` means we're loading from a pretrained checkpoint; no need to load base weights now.
        if not self.inference_mode:
            checkpoint = get_latest_checkpoint(llm_backbone_id + "/checkpoints")
            if checkpoint is None and os.path.exists(llm_backbone_id + "/checkpoints/latest-checkpoint.pt"):
                checkpoint = llm_backbone_id + "/checkpoints/latest-checkpoint.pt"
            if checkpoint is None:
                overwatch.error(f"No checkpoint found in {llm_backbone_id}/checkpoints. No weights loaded.")
            else:
                open_lm_args.resume = checkpoint
                load_model(open_lm_args, self.llm.model)

        # Lightweight Handling (with extended explanation) for setting some LLM Parameters
        #   => Set `decoder.use_cache = False` --> incompatible with gradient checkpointing (+ training in general)
        #
        #      Reference: https://discuss.huggingface.co/t/what-is-the-purpose-of-use-cache-in-decoder/958
        self.llm.config.use_cache = False if not self.inference_mode else True

        #   => Turns out that when gradient checkpointing is on and the underlying LLM has no "trainable" parameters
        #      (requires_grad is False), backprop will fail; setting `enable_input_requires_grad()` registers a new
        #      forward hook that fixes this =>> also totally safe for the "full finetuning" setting!
        if not self.inference_mode:
            self.llm.enable_input_require_grads()

        # Load Tokenizer
        overwatch.info(f"Loading [bold]{self.llm_family}[/] GPTNeoXTokenizerFast", ctx_level=1)
        self.tokenizer = GPTNeoXTokenizerFast.from_pretrained(
            "EleutherAI/gpt-neox-20b", model_max_length=self.llm_max_length
        )

        self.bos_exists = (
            self.tokenizer("Testing 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id
        ) and (self.tokenizer("Testing 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id)

        # Additionally, explicitly verify that Tokenizer padding_side is set to right for training!
        assert self.tokenizer.padding_side == "right", "Tokenizer `padding_side` is not set to `right`!"

        # add pad token, note gptneox has vocab size of 50257, but open_lm by default has an lm_head with size 50432
        self.tokenizer.add_special_tokens({"pad_token": "<PAD>"})
        self.llm.config.pad_token_id = self.tokenizer.pad_token_id
        # self.llm.resize_token_embeddings(len(self.tokenizer), pad_to_multiple_of=64)

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # If the model comes from an OpenLM checkpoint this should be removed
        state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
        # If the model comes from a Prismatic checkpoint this should be removed
        state_dict = {x.replace("llm.model.", ""): y for x, y in state_dict.items()}
        # Load the state dict
        self.llm.model.load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def embed_dim(self) -> int:
        return self.llm.config.dim

    def get_fsdp_wrapping_policy(self) -> Callable:
        """Return a `transformer_auto_wrap_policy` where we wrap each instance of `self.transformer_layer_cls`"""
        transformer_block_policy = partial(
            transformer_auto_wrap_policy, transformer_layer_cls={self.transformer_layer_cls}
        )

        return transformer_block_policy

    def enable_gradient_checkpointing(self) -> None:
        """Dispatch to underlying LLM instance's `gradient_checkpointing_enable`; defined for all `PretrainedModel`."""
        self.llm.gradient_checkpointing_enable()

    def embed_input_ids(self, input_ids: torch.LongTensor) -> torch.Tensor:
        return self.llm.get_input_embeddings()(input_ids)

    # [Contract] Should match the `forward` call of the underlying `llm` instance!
    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[List[torch.FloatTensor]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        **kwargs,
    ) -> CausalLMOutputWithPast:

        output: CausalLMOutputWithPast = self.llm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds.to(self.llm.device) if inputs_embeds is not None else None,
            labels=labels,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        return output

    @property
    def prompt_builder_fn(self) -> Type[PromptBuilder]:
        return OpenlmPromptBuilder

    @property
    def transformer_layer_cls(self) -> Type[nn.Module]:
        return Block

    @property
    def half_precision_dtype(self) -> torch.dtype:
        return torch.bfloat16
