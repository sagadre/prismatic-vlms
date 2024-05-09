import requests
import torch

from PIL import Image
from pathlib import Path
import random
import requests
from matplotlib import pyplot as plt

from prismatic import load
from prismatic.models import get_llm_backbone_and_tokenizer, get_vision_backbone_and_transform, get_vlm
from prismatic.models.backbones.llm.openlm import get_vision_state_dict, get_projector_state_dict

from open_lm.utils.llm_foundry_wrapper import SimpleComposerOpenLMCausalLM


def get_image_filenames(repo_owner, repo_name):
    # Construct the API URL
    api_url = f"https://api.github.com/repos/{repo_owner}/{repo_name}/contents"
    
    # Send a GET request to the GitHub API
    response = requests.get(api_url)
    
    # Check if the request was successful
    if response.status_code == 200:
        # Extract file information from the response
        files = response.json()
        
        # Filter to get only image files or specific file types if needed
        image_filenames = [file['name'] for file in files if file['type'] == 'file']
        return image_filenames
    else:
        # Handle potential errors (e.g., network issues, access denied)
        print("Failed to retrieve data:", response.status_code)
        return []

def get_random_image_url(files, repo_owner, repo_name):
    # Select a random file from the list
    if files:
        file = random.choice(files)
        # Construct the URL to access the file directly
        return f"https://github.com/{repo_owner}/{repo_name}/blob/master/{file}?raw=true"
    else:
        return "No files available"

# For gated LMs like Llama-2, make sure to request official access, and generate an access token
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

# Load a pretrained VLM (either local path, or ID to auto-download from the HF Hub) 
# model_id = "runs/dinosiglip_openlm_1b+stage-finetune+x7"
# model_id = "(openvlm)llm_checkpoints/openvlm_1b/"
# model_id = "runs/200_steps_dinosiglip_openlm_1b+stage-finetune+x7"
model_id = "(openvlm)llm_checkpoints/openvlm_1b/"

if model_id.startswith("runs"):
    vlm = load(model_id, cache_dir="vlm_checkpoints")
    vlm.to(device, dtype=torch.bfloat16)
    vlm = vlm.to(device)
    tokenizer = vlm.llm_backbone.get_tokenizer()

if model_id.startswith("(openlm)") or model_id.startswith("(openvlm)"):
    # vision_backbone = vision_backbone.to(torch.bfloat16)
    vision_backbone, image_transform = get_vision_backbone_and_transform(
        "dinosiglip-vit-so-384px",
        "resize-naive",
        dino_first=not model_id.startswith("(openvlm)")
    )
    if model_id.startswith("(openvlm)"):
        print("Loading vision state dict")
        vision_state_dict = get_vision_state_dict(model_id)
        vision_backbone.load_state_dict(vision_state_dict)

    llm_backbone, tokenizer = get_llm_backbone_and_tokenizer(
        model_id,
        llm_max_length=1024,
        inference_mode=False,
    )
    # llm_backbone = llm_backbone.to(dtype=torch.bfloat16)
    llm_backbone = llm_backbone.to(device)

    vlm = get_vlm(
        model_id,
        "no-align+fused-gelu-mlp",
        vision_backbone,
        llm_backbone,
        enable_mixed_precision_training=False
    )

    if model_id.startswith("(openvlm)"):
        print("Loading projector state dict")
        projector_state_dict = get_projector_state_dict(model_id)
        vlm.projector.load_state_dict(projector_state_dict)
    
    vlm.to(device, dtype=torch.bfloat16)



# Download an image and specify a prompt
# image_url = "https://huggingface.co/datasets/huggingface/documentation-images/resolve/main/beignets-task-guide.png"
source_image_owner = "EliSchwartz"
source_image_repo_name = "imagenet-sample-images"
image_url = get_random_image_url(get_image_filenames(source_image_owner, source_image_repo_name), source_image_owner, source_image_repo_name)
image = Image.open(requests.get(image_url, stream=True).raw).convert("RGB")
user_prompt = "This is a picture of a  "

# Build prompt
prompt_builder = vlm.get_prompt_builder()
prompt_builder.add_turn(role="human", message=user_prompt)
prompt_text = prompt_builder.get_prompt()

input_ids = tokenizer(prompt_text, return_tensors="pt").input_ids.to(device)
# Test the LLM
composer_model = SimpleComposerOpenLMCausalLM(vlm.llm_backbone.llm, tokenizer)
output = composer_model.generate(input_ids, max_length=512, do_sample=True, temperature=0.8, top_p=0.95)
output_text = tokenizer.decode(output[0, input_ids.shape[1] :])
print("Prompt:", user_prompt)
print("Generated Text (no image input):", output_text)

# Generate!
generated_text = vlm.generate(
    image,
    prompt_text,
    do_sample=True,
    temperature=0.4,
    max_new_tokens=512,
    min_length=1,
)
plt.imshow(image)
plt.show()
print("Prompt:", user_prompt)
print("Generated Text:", generated_text)
