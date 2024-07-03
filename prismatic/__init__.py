from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.models.hf_config import PrismaticConfig
from prismatic.models.hf_vlm import PrismaticForVision2Seq
from prismatic.preprocessing.hf_processor import PrismaticImageProcessor, PrismaticProcessor

from .models import available_model_names, available_models, get_model_description, load

# === Register Models / Processors / Configs to the appropriate HF AutoClasses (required for `.from_pretrained()`)
AutoConfig.register("prismatic", PrismaticConfig)
AutoImageProcessor.register(PrismaticConfig, PrismaticImageProcessor)
AutoProcessor.register(PrismaticConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(PrismaticConfig, PrismaticForVision2Seq)
