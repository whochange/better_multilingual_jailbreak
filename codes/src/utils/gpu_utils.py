import torch
import gc
from typing import Optional
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def get_device() -> str:
    """Get the available device (CUDA or CPU)."""
    return "cuda" if torch.cuda.is_available() else "cpu"

def clear_gpu_memory():
    """Clear GPU memory cache and run garbage collection."""
    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()
            logger.info("GPU memory cleared")
    except Exception as e:
        logger.error(f"Error clearing GPU memory: {e}")

def print_gpu_memory():
    """Print current GPU memory usage."""
    if torch.cuda.is_available():
        allocated = torch.cuda.memory_allocated() / 1024**2
        reserved = torch.cuda.memory_reserved() / 1024**2
        logger.info(f"GPU Memory: Allocated={allocated:.2f}MB, Reserved={reserved:.2f}MB")

def move_model_to_device(model: torch.nn.Module, device: Optional[str] = None) -> torch.nn.Module:
    """Safely move model to specified device."""
    if device is None:
        device = get_device()
    try:
        model = model.to(device)
        logger.info(f"Model moved to {device}")
        return model
    except Exception as e:
        logger.error(f"Error moving model to {device}: {e}")
        raise

if __name__ == "__main__":
    # Test GPU utilities
    print("Testing GPU utilities...")
    print(f"Available device: {get_device()}")
    print_gpu_memory()
    clear_gpu_memory()
    print_gpu_memory()