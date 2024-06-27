"""
openlm.py

Class definition for all LLMs derived from https://github.com/mlfoundations/open_lm.
"""
from functools import partial, cache
from typing import Callable, List, Optional, Type, Any
import os
import sys

from numpy import ndarray
import torch
from torch import nn as nn
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from transformers.modeling_outputs import CausalLMOutputWithPast
from composer.utils import dist, get_device
import yaml
import json
import argparse
import fsspec
import subprocess
import io

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
    from open_lm.file_utils import pt_load
    from open_lm.params import add_model_args
    from transformers import GPTNeoXTokenizerFast
except ImportError:
    overwatch.info("open_lm not installed. Install with `pip install git+https://github.com/mlfoundations/open_lm.git`")


# fmt: off
OPENLM_MODELS = {
    "vicuna-v15-13b": {
        "llm_family": "open_lm", "llm_cls": OpenLMforCausalLM, "hf_hub_path": ""
    },
}

class CustomTokenizer(GPTNeoXTokenizerFast):
    """
    This class handles special tokens at special indices for OpenLM models.
    This is mainly a hack
    """
    SPECIAL_STRS = ["<|img_patch|>", "<|a|>", "<|/a|>", "<|h|>", "<|/h|>"]
    SPECIAL_TOKENS = [50277, 50278, 50279, 50280, 50281]
    HUMAN_STOP = 50281
    HUMAN_START = 50280
    AGENT_START = 50278
    AGENT_STOP = 50279
    def __init__(self, *args, **kwargs):
        self.encoded_to_replaced = {}
        self.replaced_to_encoded = {}
        self._name_to_replaced = {}
        super().__init__(*args, **kwargs)
        print("Adding special tokens")
        special_tokens_dict = {"additional_special_tokens": self.SPECIAL_STRS}
        self.add_special_tokens(special_tokens_dict)
        assert [e[0] for e in self(self.SPECIAL_STRS)["input_ids"]] == self.SPECIAL_TOKENS
        self.pad_token_id = self.eos_token_id
        print("Tokenizer initialized")
    

class OpenlmLLMBackbone(LLMBackbone):
    """
    OpenLM LLM Backbone class.

    llm_backbone_id: str
        - should contain "openlm" to indicate that this is an OpenLM model, either by being in the path or using "(openlm)" as a prefix
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
        if llm_backbone_id.startswith("(openlm)"):
            llm_backbone_id = llm_backbone_id.replace("(openlm)", "")
            ignore_keys = []
        if llm_backbone_id.startswith("(openvlm)"):
            llm_backbone_id = llm_backbone_id.replace("(openvlm)", "")
            ignore_keys = ["image_extractor", "image_projector"]

        super().__init__(llm_backbone_id)
        # self.strange_input_logger = StrangeInputLogger()
        self.llm_family = "open_lm"
        self.inference_mode = inference_mode

        model_args = argparse.ArgumentParser()
        # For retro-compatability, we need to add the model args to the parser 
        # so that default values are set for new args that might not figure in the params.txt
        add_model_args(model_args)
        open_lm_args = model_args.parse_args([]).__dict__

        # Read params.txt in the model directory
        # [Contract] `params.txt` must be in the `llm_backbone_id` directory

        if llm_backbone_id.startswith("s3"):
            while llm_backbone_id.endswith("/"):
                llm_backbone_id = llm_backbone_id[:-1]
            cmd = f"aws s3 cp {llm_backbone_id}/params.txt -"
            proc = subprocess.Popen(cmd, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            stdout, stderr = proc.communicate()
            if proc.returncode != 0:
                raise Exception(f"Failed to fetch model from s3. stderr: {stderr.decode()}")
            open_lm_args.update(yaml.safe_load(io.BytesIO(stdout)))
        else:
            with fsspec.open(llm_backbone_id + "/params.txt", "rb") as f:
                open_lm_args.update(yaml.safe_load(f))

        open_lm_args = argparse.Namespace(**open_lm_args)

        ### TODO: This is a hack to avoid having to point to an exact existing model json file but it will be hard to maintain
        model_json = {}
        additional_keys = ["attn_name", "attn_activation", "attn_seq_scalar", "attn_seq_scalar_alpha", ]
        temp_params = create_params(open_lm_args)
        for key, value in temp_params.__dict__.items():
            if hasattr(open_lm_args, key):
                model_json[key] = open_lm_args.__dict__[key]
                temp_params.__dict__[key] = open_lm_args.__dict__[key]
        for key in additional_keys:
            model_json[key] = open_lm_args.__dict__[key]
        model_json["weight_tying"] = temp_params.weight_tying
        model_json["seq_len"] = temp_params.seq_len
        if hasattr(temp_params, "dim"):
            model_json["hidden_dim"] = temp_params.dim
            model_json["n_layers"] = temp_params.n_layers
            model_json["n_heads"] = temp_params.n_heads
            model_json["post_embed_norm"] = temp_params.post_embed_norm
            model_json["model_norm"] = open_lm_args.model_norm
        else:
            model_json["d_model"] = temp_params.d_model
            model_json["n_layer"] = temp_params.n_layer
            model_json["vocab_size"] = temp_params.vocab_size
            model_json["seq_len"] = temp_params.seq_len
            model_json["rms_norm"] = temp_params.rms_norm
            model_json["residual_in_fp32"] = temp_params.residual_in_fp32
            model_json["fused_add_norm"] = temp_params.fused_add_norm
            model_json["pad_vocab_size_multiple"] = temp_params.pad_vocab_size_multiple
            model_json["weight_tying"] = temp_params.weight_tying

        model_name = open_lm_args.model
        with open(f"{llm_backbone_id}/{model_name}.json", "w") as f:
            json.dump(model_json, f)
        open_lm_args.model = f"{llm_backbone_id}/{model_name}.json"
        ### end of hack

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
                load_model(open_lm_args, self.llm.model, filter_keys=ignore_keys)

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

        if open_lm_args.torchcompile:
            overwatch.info("Compiling llm model with torch.compile", ctx_level=1)
            self.llm.model = torch.compile(self.llm.model)

        # Load Tokenizer
        overwatch.info(f"Loading tokenizer [bold]{self.llm_family} EleutherAI/gpt-neox-20b[/].", ctx_level=1)
        self.tokenizer = CustomTokenizer.from_pretrained(
            "EleutherAI/gpt-neox-20b", model_max_length=self.llm_max_length
        )

        self.bos_exists = (
            self.tokenizer("Testing 123", add_special_tokens=True).input_ids[0] == self.tokenizer.bos_token_id
        ) and (self.tokenizer("Testing 123", add_special_tokens=False).input_ids[0] != self.tokenizer.bos_token_id)

        # Additionally, explicitly verify that Tokenizer padding_side is set to right for training!
        assert self.tokenizer.padding_side == "right", "Tokenizer `padding_side` is not set to `right`!"

        self.llm.config.pad_token_id = self.tokenizer.pad_token_id

    def load_state_dict(self, state_dict, strict: bool = True, assign: bool = False):
        if "state_dict" in state_dict:
            state_dict = state_dict["state_dict"]
        # If the model comes from an OpenLM checkpoint this should be removed
        state_dict = {x.replace("module.", ""): y for x, y in state_dict.items()}
        # If the model comes from a Prismatic checkpoint this should be removed
        state_dict = {x.replace("llm.model.", ""): y for x, y in state_dict.items()}
        state_dict = {x.replace("model.backbone.", ""): y for x, y in state_dict.items()}
        state_dict = {k: v for k, v in state_dict.items() if "inv_freq" not in k}
        # Load the state dict
        self.llm.model.load_state_dict(state_dict, strict=strict, assign=assign)

    @property
    def embed_dim(self) -> int:
        if hasattr(self.llm.config, "dim"):
            return self.llm.config.dim
        elif hasattr(self.llm.config, "d_model"):
            return self.llm.config.d_model
        else:
            raise ValueError("Could not find `dim` or `d_model` in the LLM config.")

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
        # self.strange_input_logger.log(output.loss, input_ids, inputs_embeds=inputs_embeds, labels=labels)
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


@cache
def get_vlm_state_dict(llm_backbone_id: str) -> dict:
    assert llm_backbone_id.startswith("(openvlm)"), "This function is only for OpenLM models"

    # Special Handling for OpenLM Backbones because it contains vision + language components
    pretrained_checkpoint = llm_backbone_id[9:] + "/checkpoints/"
    pretrained_checkpoint = get_latest_checkpoint(pretrained_checkpoint)
    overwatch.info(f"Loading from Provided OpenLM Checkpoint {pretrained_checkpoint}", ctx_level=1)
    full_state_dict = pt_load(pretrained_checkpoint, "cpu")["state_dict"]
    return full_state_dict


def get_vision_state_dict(llm_backbone_id: str) -> dict:
    full_state_dict = get_vlm_state_dict(llm_backbone_id)
    is_extractor = lambda k: k.startswith("image_extractors")
    vision_state_dict = {k: v for k, v in full_state_dict.items() if is_extractor(k)}
    # Replace keys to match the vision backbone naming convention of Prismatic
    vision_state_dict = {k.replace("image_extractors.0.model", "siglip_featurizer"): v for k, v in vision_state_dict.items()}
    vision_state_dict = {k.replace("image_extractors.1.model", "dino_featurizer"): v for k, v in vision_state_dict.items()}
    return vision_state_dict


def get_projector_state_dict(llm_backbone_id: str) -> dict:
    full_state_dict = get_vlm_state_dict(llm_backbone_id)
    is_projector = lambda k: k.startswith("image_projector")
    projector_state_dict = {k: v for k, v in full_state_dict.items() if is_projector(k)}
    projector_state_dict = {k.replace("image_projector.", ""): v for k, v in projector_state_dict.items()}
    return projector_state_dict


class StrangeInputLogger:
    def __init__(self, output_path="strange_input", beta=0.9, rtol=0.5):
        self.mean_loss = 0
        self.beta = beta
        self.rtol = rtol
        self.output_path = output_path
        self.counter = 0

        dist.initialize_dist(get_device(None))
        self.is_master = dist.get_global_rank() == 0

    def log(self, loss, input, **kwargs):
        self.counter += 1
        if self.mean_loss == 0:
            self.mean_loss = loss
        else:
            self.mean_loss = self.beta * self.mean_loss + (1 - self.beta) * loss
        if abs(loss - self.mean_loss) / self.mean_loss > self.rtol:
            self.write(input, loss, log_count=self.counter, **kwargs)

    def write(self, input, loss, **kwargs):
        if self.is_master:
            with open(self.output_path, "a") as f:
                torch.set_printoptions(threshold=sys.maxsize)  # Set print options to display the entire tensor
                f.write(f"Input: {input}\n")
                f.write(f"Loss: {loss}\n")
                for k, v in kwargs.items():
                    f.write(f"{k}: {v}\n")
