import sys
from pathlib import Path

# Add project root to Python path and print for verification
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)  # Use insert(0, ...) instead of append()

#print(f"Project root: {project_root}")
#print(f"Python path: {sys.path}")
from typing import Dict, List
import torch
from tqdm import tqdm
import json
from datetime import datetime

from src.utils.translate_utils import Translator
from src.models.generator import TransformerGenerator
from src.data.dataset import MultilingualDataset
from config.model_config import DEFAULT_MODELS
from config.experiment_config import ExperimentConfig

def setup_experiment():
    """Initialize experiment configuration and output directory."""
    config = ExperimentConfig(google_credentials="GOOGLE TRANSLATE PATH") # you have to give your google translate credentials here
    
    # Create experiment directory with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    exp_dir = Path(config.output_dir) / f"experiment_{timestamp}_{config.generator_name}"
    exp_dir.mkdir(parents=True, exist_ok=True)
    
    return config, exp_dir


def run_generation(
    generator: TransformerGenerator,
    dataset: MultilingualDataset,
    config: ExperimentConfig,
    output_dir: Path
):
    """Run generation for all languages in dataset."""
    results = {}
    translator = Translator()
    
    for lang in dataset.languages:
        print(f"\nProcessing language: {lang}")
        samples = dataset.get_samples(lang)
        
        # Generate responses
        try:
            responses = generator.generate(samples)
            
            # Translate responses to English for analysis if not already in English
            if lang != "en":
                print(f"Translating responses from {lang} to English...")
                translations = translator.translate_batch(responses, target_lang="en")
                translations = [t["translatedText"] for t in translations]
            else:
                translations = responses  # No translation needed for English
            
            # Store results
            results[lang] = {
                'queries': samples,
                'queries_og': dataset.get_samples('en'),
                'responses': responses,
                'translations': translations
            }
            
            print(f"Results: {results[lang]['responses'][0]}")
            
        except Exception as e:
            print(f"Error processing language {lang}: {e}")
            continue
        
        # Save results for this language
        output_file = output_dir / f"results_{lang}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results[lang], f, ensure_ascii=False, indent=2)
            
    return results

def main():
    # Setup
    config, exp_dir = setup_experiment()
    print(f"Experiment directory: {exp_dir}")
    
    # Load dataset
    dataset = MultilingualDataset(
        data_path=config.data_path,
        languages=config.languages,
        sample_size=config.sample_size
    )
    print(f"Loaded dataset with {len(dataset)} samples")
    
    # Initialize model
    model_config = DEFAULT_MODELS[config.generator_name].__dict__
    generator = TransformerGenerator(model_config)
    print(f"Initialized generator model: {config.generator_name}")
    
    # Run generation
    results = run_generation(generator, dataset, config, exp_dir)
    
    # Save experiment config
    with open(exp_dir / "config.json", 'w') as f:
        json.dump(config.__dict__, f, indent=2)
    
    print(f"\nExperiment completed. Results saved in: {exp_dir}")

if __name__ == "__main__":
    main()