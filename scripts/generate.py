"""
generate.py

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

from prismatic import load
from prismatic.overwatch import initialize_overwatch

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
        # "siglip-224px+7b"
        "prism-dinosiglip-224px+7b"
    )

    # HF Hub Credentials (required for Gated Models like LLaMa-2)
    hf_token: Union[str, Path] = Path(".hf_token")                      # Environment variable or Path to HF Token

    # Default Generation Parameters =>> subscribes to HuggingFace's GenerateMixIn API
    do_sample: bool = False
    temperature: float = 1.0
    max_new_tokens: int = 512
    min_length: int = 1

    # fmt: on


@draccus.wrap()
def generate(cfg: GenerateConfig) -> None:
    overwatch.info(f"Initializing Generation Playground with Prismatic Model `{cfg.model_path}`")
    hf_token = cfg.hf_token.read_text().strip() if isinstance(cfg.hf_token, Path) else os.environ[cfg.hf_token]
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

    # Load the pretrained VLM --> uses default `load()` function
    vlm = load(cfg.model_path, hf_token=hf_token)
    vlm.to(device, dtype=torch.bfloat16)

    # Initial Setup
    image = Image.open(requests.get(DEFAULT_IMAGE_URL, stream=True).raw).convert("RGB")
    prompt_builder = vlm.get_prompt_builder()
    system_prompt = prompt_builder.system_prompt

    # === DEBUG ===
    prompt_texts = ["What is sitting in the coffee?", "caption.", "Give me a description of the scene"]

    print("\n")
    print(f"=> [INPUT] {prompt_texts[0]}")
    print(f"=> [INPUT] {prompt_texts[1]}")
    print(f"=> [INPUT] {prompt_texts[2]}")
    print("\n=========\n")

    prompt_builder_0 = vlm.get_prompt_builder()
    prompt_builder_0.add_turn(role="human", message=prompt_texts[0])
    prompt_text_0 = prompt_builder_0.get_prompt()

    prompt_builder_1 = vlm.get_prompt_builder()
    prompt_builder_1.add_turn(role="human", message=prompt_texts[1])
    prompt_text_1 = prompt_builder_1.get_prompt()

    prompt_builder_2 = vlm.get_prompt_builder()
    prompt_builder_2.add_turn(role="human", message=prompt_texts[2])
    prompt_text_2 = prompt_builder_2.get_prompt()

    # Generate from the VLM
    generated_text_0 = vlm.generate(
        image,
        prompt_text_0,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        min_length=cfg.min_length,
    )

    generated_text_1 = vlm.generate(
        image,
        prompt_text_1,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        min_length=cfg.min_length,
    )

    generated_text_2 = vlm.generate(
        image,
        prompt_text_2,
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        min_length=cfg.min_length,
    )

    print(f"[OLD] Generation 0 :: {generated_text_0[0].strip()}")
    print(f"[OLD] Generation 1 :: {generated_text_1[0].strip()}")
    print(f"[OLD] Generation 2 :: {generated_text_2[0].strip()}")
    print("\n---\n")

    # ATTEMPT BATCH GENERATION
    batch_generated = vlm.generate(
        [image, image, image],
        [prompt_text_0, prompt_text_1, prompt_text_2],
        do_sample=cfg.do_sample,
        temperature=cfg.temperature,
        max_new_tokens=cfg.max_new_tokens,
        min_length=cfg.min_length,
    )

    print(f"[NEW] Generation 0 :: {batch_generated[0].strip()}")
    print(f"[NEW] Generation 1 :: {batch_generated[1].strip()}")
    print(f"[NEW] Generation 2 :: {batch_generated[2].strip()}")

    exit(0)

    # === DEBUG ===

    # REPL Welcome Message
    print(
        "[*] Dropping into Prismatic VLM REPL with Default Generation Setup => Initial Conditions:\n"
        f"       => Prompt Template:\n\n{prompt_builder.get_potential_prompt('<INSERT PROMPT HERE>')}\n\n"
        f"       => Default Image URL: `{DEFAULT_IMAGE_URL}`\n===\n"
    )

    # REPL
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
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)

        elif user_input.lower().startswith("p"):
            if system_prompt is None:
                print("\n|=>> Model does not support `system_prompt`!")
                continue

            # Note => a new system prompt starts a _new_ conversation
            system_prompt = input("\n|=>> Enter New System Prompt: ")
            prompt_builder = vlm.get_prompt_builder(system_prompt=system_prompt)
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
                    generated_text = vlm.generate(
                        image,
                        prompt_text,
                        do_sample=cfg.do_sample,
                        temperature=cfg.temperature,
                        max_new_tokens=cfg.max_new_tokens,
                        min_length=cfg.min_length,
                    )
                    prompt_builder.add_turn(role="gpt", message=generated_text)
                    print(f"\t|=>> VLM Response >>> {generated_text}\n")

            except KeyboardInterrupt:
                print("\n===\n")
                continue


if __name__ == "__main__":
    generate()
