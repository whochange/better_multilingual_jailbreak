import pandas as pd
from typing import List, Optional
from pathlib import Path

class MultilingualDataset:
    def __init__(
        self,
        data_path: str,
        languages: Optional[List[str]] = None,
        sample_size: Optional[int] = None
    ):
        self.data_path = data_path
        self.languages = languages or ["en", "zh", "it"]
        self.sample_size = sample_size
        self.data = self._load_data()
    
    def _load_data(self) -> pd.DataFrame:
        """Load and preprocess the data."""
        if not Path(self.data_path).exists():
            raise FileNotFoundError(f"Data file not found: {self.data_path}")
            
        df = pd.read_csv(self.data_path)
        
        # Validate languages
        available_langs = [col for col in df.columns if len(col) == 2]
        self.languages = [lang for lang in self.languages if lang in available_langs]
        
        if self.sample_size:
            df = df.sample(n=self.sample_size, random_state=42)
        
        return df
    
    def get_samples(self, language: str) -> List[str]:
        """Get samples for a specific language."""
        if language not in self.languages:
            raise ValueError(f"Language {language} not available")
        return self.data[language].tolist()
    
    def __len__(self):
        return len(self.data)

if __name__ == "__main__":
    # Test the dataset
    print("Testing dataset loading...")
    
    # Test with small sample
    dataset = MultilingualDataset(
        data_path="PATH TO MULTILINGUAL JAILBREAK",
        sample_size=5
    )
    
    print(f"Dataset loaded with {len(dataset)} samples")
    
    # Test language access
    for lang in dataset.languages:
        samples = dataset.get_samples(lang)
        print(f"\nSamples for {lang}:")
        print(samples[:2])  # Print first two samples