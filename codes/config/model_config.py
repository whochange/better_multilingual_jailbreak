from dataclasses import dataclass
from typing import Optional, List, Dict

MULTI_JAIL_LANGUAGES = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]

@dataclass
class ModelConfig:
    name: str
    model_path: str
    device: str = "cuda"
    batch_size: int = 8
    max_length: int = 128
    temperature: float = 0.7
    chat_template: str = None

@dataclass
class GeneratorConfig(ModelConfig):
    languages: List[str] = None
    do_sample: bool = False

@dataclass
class GuardConfig(ModelConfig):
    threshold: float = 0.5

# Default configurations
DEFAULT_MODELS = {
    "aya-101": GeneratorConfig(
        name="aya-101",
        model_path="CohereForAI/aya-101",
        chat_template="aya-101",
        languages=["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"] # this part list the languages in Multijaildataset that the model is specifically trained on (meaning other languages are possible but might lead to bad results)
    ),
    "seallm": GeneratorConfig(
        name="seallm",
        model_path="SeaLLMs/SeaLLM-7B-v2",
        chat_template="seallm",
        languages=["en", "zh", "vi", "th"]
    ),
    "aya-expanse": GeneratorConfig(
        name="aya-expanse",
        model_path="CohereForAI/aya-expanse-8b",
        languages=["en", "zh", "vi", "ar", "it", "ko"] 
        #Arabic, Chinese (simplified & traditional), Czech, Dutch, English, French, German, Greek, Hebrew, Hebrew, Hindi, Indonesian, Italian, Japanese, Korean, Persian, Polish, Portuguese, Romanian, Russian, Spanish, Turkish, Ukrainian, and Vietnamese.
    ),
    "llama-2": GeneratorConfig(
        name="llama-2",
        model_path="meta-llama/Llama-2-7b-chat-hf",
        chat_template="llama-2",
        languages=["en"]
    ),
    "llama-3": GeneratorConfig(
        name="llama-3",
        model_path="meta-llama/Llama-3.1-8B-Instruct",
        chat_template="llama-3",
        languages=["en", "it", "th"]
        # English, German, French, Italian, Portuguese, Hindi, Spanish, and Thai
    ),
    "qwen": GeneratorConfig(
        name="qwen",
        model_path="Qwen/Qwen2.5-7B-Instruct",
        chat_template="qwen",
        languages=["en", "zh", "it", "ko", "vi", "th", "ar"]
        # Chinese, English, French, Spanish, Portuguese, German, Italian, Russian, Japanese, Korean, Vietnamese, Thai, Arabic,
    ),
    
    "sw-llama-2": GeneratorConfig(
        name="sw-llama-2",
        model_path="Jacaranda/UlizaLlama",
        chat_template="sw-llama-2",
        languages=["en", "sw"]
    ),
    
    "sw-gemma": GeneratorConfig(
        name="sw-gemma",
        model_path="Mollel/Swahili_Gemma",
        chat_template="sw-gemma",
        languages=["en", "sw"]
    )
}

if __name__ == "__main__":
    # Test the configurations
    print("Testing model configurations...")
    
    # Test creating a custom config
    custom_config = GeneratorConfig(
        name="test_model",
        model_path="test/path",
        languages=["en", "es"]
    )
    print(f"Custom config: {custom_config}")
    
    # Test default configs
    print("\nDefault configurations:")
    for name, config in DEFAULT_MODELS.items():
        print(f"{name}: {config}")