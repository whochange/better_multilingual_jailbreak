import sys
from pathlib import Path
import logging
#from scripts.stats import analyze_experiment_dir
# Add project root to Python path
project_root = str(Path(__file__).parent.parent.absolute())
sys.path.insert(0, project_root)

from datetime import datetime
import json
import traceback
from typing import List
from src.utils.gpu_utils import clear_gpu_memory
import torch


from src.models.guard import SafetyGuard
from src.utils.translate_utils import Translator
from src.models.generator import TransformerGenerator
from src.data.dataset import MultilingualDataset
from config.model_config import DEFAULT_MODELS
from config.experiment_config import ExperimentConfig
from stats import analyze_experiment_dir
from src.models.guard import SafetyGuard
from generate import run_generation

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def run_full_experiment(
    languages: List[str],
    models: List[str],
    base_dir: str = "experiments/results",
    sample_size: int = 10
) -> None:
    """Run full experiment pipeline for all models."""
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        exp_dir = Path(base_dir) / f"experiment_{timestamp}"
        exp_dir.mkdir(parents=True, exist_ok=True)
        logging.info(f"Created experiment directory: {exp_dir}")

        # Save experiment configuration
        config = {
            "timestamp": timestamp,
            "languages": languages,
            "models": models
        }
        with open(exp_dir / "experiment_config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Run for each model
        for model_name in models:
            try:
                logging.info(f"\nProcessing model: {model_name}")
                
                clear_gpu_memory()
                torch.cuda.empty_cache()
                
                
                # Setup experiment config
                config = ExperimentConfig(
                    google_credentials='YOUR CREDENTIALS',
                    languages=languages,
                    generator_name=model_name,
                    sample_size=sample_size  
                )
                
                # Load dataset
                dataset = MultilingualDataset(
                    data_path=config.data_path,
                    languages=languages,
                    sample_size=config.sample_size
                )
                logging.info(f"Loaded dataset with {len(dataset)} samples")
                
                # Initialize model
                model_config = DEFAULT_MODELS[model_name].__dict__
                generator = TransformerGenerator(model_config)
                logging.info(f"Initialized generator model: {model_name}")
                
                # Create model directory
                model_dir = exp_dir / model_name
                model_dir.mkdir(parents=True, exist_ok=True)
                
                # Run generation
                results = run_generation(generator, dataset, config, model_dir)
                
                # Save model config
                with open(model_dir / "config.json", 'w') as f:
                    json.dump(config.__dict__, f, indent=2)
                
                 # Clean up after model
                del generator
                clear_gpu_memory()
                torch.cuda.empty_cache()
                
                logging.info(f"Completed processing for {model_name}")
                
            except Exception as e:
                logging.error(f"Error processing model {model_name}: {str(e)}")
                logging.error(traceback.format_exc())
                continue
            
            # Evaluate results
            
            logging.info("Evaluating results")
            guard = SafetyGuard()
            try:
                logging.info(f"Evaluating experiment with {model_name}")
                guard.evaluate_experiment(model_dir)
            except Exception as e:
                logging.error(f"Error evaluating experiment: {str(e)}")
                logging.error(traceback.format_exc())
            
            logging.info(f"\nExperiment complete! Results in {exp_dir}")
    
        logging.info("Generating experiment statistics...")
        try:
            stats = analyze_experiment_dir(str(exp_dir))
            logging.info(f"Statistics generated and saved to {exp_dir}/experiment_stats.json")
        except Exception as e:
            logging.error(f"Error generating statistics: {str(e)}")
            logging.error(traceback.format_exc())
        
        logging.info(f"\nExperiment complete! Results in {exp_dir}")
            
    
    except Exception as e:
        logging.error(f"Fatal error in experiment: {str(e)}")
        logging.error(traceback.format_exc())
        sys.exit(1)

if __name__ == "__main__":
    import argparse
    
    # Default languages
    languages = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
    
    parser = argparse.ArgumentParser(description='Run multilingual safety experiments')
    parser.add_argument('--languages', nargs='+', default=languages,
                      help='List of language codes to test')
    parser.add_argument('--models', nargs='+', 
                      default=list(DEFAULT_MODELS.keys()),
                      help='List of models to test')
    parser.add_argument('--output-dir', default="experiments/results",
                      help='Base directory for experiment results')
    parser.add_argument('--sample-size', type=int, default=315,
                      help='Number of samples to use for each language')
    
    args = parser.parse_args()
    
    run_full_experiment(args.languages, args.models, args.output_dir, args.sample_size)