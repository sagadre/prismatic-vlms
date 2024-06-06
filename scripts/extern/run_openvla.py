"""
run_openvla.py

Demonstrates how to run inference on an OpenVLA model loaded from the HuggingFace hub.
"""

import time

import numpy as np
import torch
from PIL import Image
from transformers import AutoModelForVision2Seq, AutoProcessor

# VLA Parameters
MODEL_PATH = "openvla/openvla-7b-v01"
SYSTEM_PROMPT = (
    "A chat between a curious user and an artificial intelligence assistant. "
    "The assistant gives helpful, detailed, and polite answers to the user's questions."
)
INSTRUCTION = "put spoon on towel"


def get_openvla_prompt(instruction: str) -> str:
    return f"{SYSTEM_PROMPT} USER: What action should the robot take to {instruction.lower()}? ASSISTANT:"


@torch.inference_mode()
def run_openvla() -> None:
    # Load Processor & VLA
    print("[*] Instantiating Processor and Pretrained OpenVLA")
    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    processor = AutoProcessor.from_pretrained(MODEL_PATH, trust_remote_code=True)
    print("[*] Loading in BF16 with Flash-Attention Enabled")
    vla = AutoModelForVision2Seq.from_pretrained(
        MODEL_PATH,
        attn_implementation="flash_attention_2",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True,
    ).to(device)

    # === 8-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~9GB of VRAM Passive || 10GB of VRAM Active] ===
    # print("[*] Loading in 8-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     load_in_8bit=True,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    # === 4-BIT QUANTIZATION MODE (`pip install bitsandbytes`) :: [~6GB of VRAM Passive || 7GB of VRAM Active] ===
    # print("[*] Loading in 4-Bit Quantization Mode")
    # vla = AutoModelForVision2Seq.from_pretrained(
    #     MODEL_PATH,
    #     attn_implementation="flash_attention_2",
    #     torch_dtype=torch.bfloat16,
    #     load_in_4bit=True,
    #     low_cpu_mem_usage=True,
    #     trust_remote_code=True,
    # )

    for _ in range(100):
        prompt = get_openvla_prompt(INSTRUCTION)
        image = Image.fromarray(np.asarray(np.random.rand(256, 256, 3) * 255, dtype=np.uint8))
        start = time.time()
        inputs = processor(prompt, image).to(device, dtype=torch.bfloat16)

        # Run OpenVLA inference
        action = vla.predict_action(**inputs, unnorm_key="bridge_orig", do_sample=False)
        print(time.time() - start, action)


if __name__ == "__main__":
    run_openvla()
