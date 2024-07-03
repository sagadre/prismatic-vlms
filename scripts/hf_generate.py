"""
hf_generate.py

Simple CLI script to interactively test generating from a pretrained VLM; provides a minimal REPL for specify image
URLs, prompts, and language generation parameters.

Run with: python scripts/generate.py --model_path <PATH TO LOCAL MODEL OR HF HUB>
"""

import os
from dataclasses import dataclass
from pathlib import Path
from typing import Union

import draccus
import requests
import torch
from PIL import Image

from prismatic import PrismaticForVision2Seq, PrismaticProcessor
from prismatic.overwatch import initialize_overwatch
from prismatic.preprocessing import get_prompt_builder_fn

# Initialize Overwatch =>> Wraps `logging.Logger`
overwatch = initialize_overwatch(__name__)


# Default Image URL (Beignets)
DEFAULT_IMAGE_URL = (
    "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
)


@dataclass
class GenerateConfig:
    # fmt: off
    model_path: Union[str, Path] = (                                    # Path to Pretrained VLM (on disk or HF Hub)
        # "TRI-ML/siglip-224px-7b"
        # "TRI-ML/prism-dinosiglip-224px-7b"

        "TRI-ML/prism-dinosiglip-224px-controlled-7b"
    )

    # HF Hub Credentials (required for Gated Models like Llama-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Path to HF Token (or token as string)

    # Default Generation Parameters
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

    # Debug / Verify
    debug: bool = True

    # fmt: on


@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    overwatch.info(f"Initializing Generation Playground with Prismatic VLM = `{cfg.model_path}`")
    os.environ["HF_TOKEN"] = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else cfg.hf_token
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the Processor & VLM via `.from_pretrained()` =>> by default, we run inference in `bf16`
    overwatch.info(f"Loading Pretrained VLM from Checkpoint at `{cfg.model_path}`")
    processor = PrismaticProcessor.from_pretrained(cfg.model_path)
    vlm = PrismaticForVision2Seq.from_pretrained(
        cfg.model_path,
        attn_implementation="flash_attention_2",
        device_map=device,
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
    ).to(device)

    # Initial Setup
    image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw).convert("RGB")
    prompt_builder_fn = get_prompt_builder_fn(vlm.config.llm_backbone_id)

    # === Verify Batched Generation ===
    if cfg.debug:
        overwatch.info("Debugging / Validating Batched Generation")
        prompt_texts, images = [], []
        for prompt_text in ["What is sitting in the coffee?", "caption.", "Give me a description of the scene"]:
            prompt_builder = prompt_builder_fn()
            prompt_builder.add_turn(role="human", message=prompt_text, add_image_token=True)

            prompt_texts.append(prompt_builder.get_prompt())
            images.append(image)

        # Process =>> Generate!
        inputs = processor(prompt_texts, images, padding=True).to(device, dtype=torch.bfloat16)
        gen_ids = vlm.generate(
            **inputs,
            do_sample=cfg.do_sample,
            temperature=cfg.temperature,
            max_new_tokens=cfg.max_new_tokens,
            min_length=cfg.min_length,
        )
        gen_texts = processor.batch_decode(gen_ids[:, inputs.input_ids.shape[1] :], skip_special_tokens=True)
        for idx, gen_text in enumerate(gen_texts):
            print(f"Generation {idx} :: {gen_text.strip()}")

        return

    # === Interactive Generation CLI ===
    prompt_builder = prompt_builder_fn()
    system_prompt = prompt_builder.system_prompt
    print(
        "[*] Dropping into Prismatic VLM REPL with Default Generation Setup => Initial Conditions:\n"
        f"       => Prompt Template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
        f"       => Default Image URL: `{DEFAULT_IMAGE_URL}`\n===\n"
    )

    # =>> REPL
    repl_prompt = (
        "|=>> Enter (i)mage to fetch image from URL, (p)rompt to update prompt template, (q)uit to exit, or any other"
        " key to enter input questions: "
    )
    while True:
        user_input = input(repl_prompt)

        if user_input.lower().startswith("q"):
            print("\n|=>> Received (q)uit signal => Exiting...")
            return

        elif user_input.lower().startswith("i"):
            # Note => a new image starts a _new_ conversation (for now)
            url = input("\n|=>> Enter Image URL: ")
            image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
            prompt_builder = prompt_builder_fn(system_prompt=system_prompt)

        elif user_input.lower().startswith("p"):
            if system_prompt is None:
                print("\n|=>> Model does not support `system_prompt`!")
                continue

            # Note => a new system prompt starts a _new_ conversation
            system_prompt = input("\n|=>> Enter New System Prompt: ")
            prompt_builder = prompt_builder_fn(system_prompt=system_prompt.strip())
            print(
                "\n[*] Set New System Prompt:\n"
                f"    => Prompt Template:\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
            )

        else:
            print("\n[*] Entering Chat Session - CTRL-C to start afresh!\n===\n")
            try:
                while True:
                    message = input("|=>> Enter Prompt: ")

                    # Build Prompt
                    prompt_builder.add_turn(role="human", message=message)
                    prompt_text = prompt_builder.get_prompt()

                    # Generate from the VLM
                    inputs = processor(prompt_text, image).to(device, dtype=torch.bfloat16)
                    gen_ids = vlm.generate(
                        **inputs,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens,
                        min_length=cfg.min_length,
                    )
                    generated_text = processor.decode(gen_ids[0, inputs.input_ids.shape[1] :], skip_special_tokens=True)
                    prompt_builder.add_turn(role="gpt", message=generated_text)
                    print(f"\t|=>> VLM Response >>> {generated_text}\n")

            except KeyboardInterrupt:
                print("\n===\n")
                continue


if __name__ == "__main__":
    generate()
