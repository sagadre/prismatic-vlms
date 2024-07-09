from transformers import AutoConfig, AutoImageProcessor, AutoModelForVision2Seq, AutoProcessor

from prismatic.models import PrismaticConfig, PrismaticForVision2Seq
from prismatic.preprocessing.processors import PrismaticImageProcessor, PrismaticProcessor

# === Register Models / Processors / Configs to the appropriate HF AutoClasses (required for `.from_pretrained()`)
AutoConfig.register("prismatic", PrismaticConfig)
AutoImageProcessor.register(PrismaticConfig, PrismaticImageProcessor)
AutoProcessor.register(PrismaticConfig, PrismaticProcessor)
AutoModelForVision2Seq.register(PrismaticConfig, PrismaticForVision2Seq)
