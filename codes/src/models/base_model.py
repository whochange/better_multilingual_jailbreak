from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional
import torch
from src.utils.gpu_utils import move_model_to_device, clear_gpu_memory

class BaseModel(ABC):
    def __init__(self, model_config: Dict[str, Any]):
        self.config = model_config
        self.device = model_config.get('device', 'cuda')
        self.model = None
        self.tokenizer = None
    
    @abstractmethod
    def load_model(self):
        """Load model and tokenizer."""
        pass
    
    @abstractmethod
    def generate(self, inputs: List[str], **kwargs) -> List[str]:
        """Generate outputs for given inputs."""
        pass
    
    def to_device(self):
        """Move model to specified device."""
        if self.model is not None:
            self.model = move_model_to_device(self.model, self.device)
    
    def clear_memory(self):
        """Clear GPU memory."""
        clear_gpu_memory()

if __name__ == "__main__":
    print("Testing base model structure...")
    # We don't test instantiation since it's an abstract class
    print("BaseModel is an abstract class and cannot be instantiated directly.")