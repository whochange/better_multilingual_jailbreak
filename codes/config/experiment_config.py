from dataclasses import dataclass
from typing import List, Optional
from pathlib import Path
import os

@dataclass
class ExperimentConfig:
    # Data configuration
    data_path: str = "PATH TO MULTIJAIL DATA"
    languages: List[str] = None
    sample_size: int = 10
    google_credentials: Optional[str] = None 
    
    # Output configuration
    output_dir: str = "experiments/results"
    save_responses: bool = True
    
    # Experiment settings
    generator_name: str = "aya-expanse"  # Which model to use
    run_translation: bool = True  # Whether to translate non-English outputs
    
    def __post_init__(self):
        if self.languages is None:
            self.languages = ["en", "zh", "it", "vi", "ar", "ko", "th", "bn", "sw", "jv"]
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
        if self.google_credentials:
            os.environ['GOOGLE_APPLICATION_CREDENTIALS'] = self.google_credentials

if __name__ == "__main__":
    # Test the experiment configuration
    config = ExperimentConfig()
    print("Testing experiment configuration...")
    print(f"Config: {config}")
    print(f"Output directory created: {config.output_dir}")