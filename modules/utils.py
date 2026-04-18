import torch
import torch.nn as nn
from pathlib import Path
from datasets import load_dataset
from typing import Optional

def init_weights(module: nn.Module, std: float = 0.02):
    """GPT-2 style weight initialization."""
    if isinstance(module, (nn.Linear, nn.Embedding)):
        nn.init.normal_(module.weight, mean=0.0, std=std)
        if hasattr(module, 'bias') and module.bias is not None:
            nn.init.zeros_(module.bias)

def save_checkpoint(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    step: int,
    checkpoint_path: str,
    **extra,
):
    """Save model, optimizer state, and training step to checkpoint."""
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "step": step,
    }
    if extra:
        checkpoint.update(extra)
    torch.save(checkpoint, checkpoint_path)
    print(f"Checkpoint saved to {checkpoint_path} at step {step}")

def load_checkpoint(
    model: nn.Module,
    optimizer: Optional[torch.optim.Optimizer],
    checkpoint_path: str,
    device: str,
    return_checkpoint: bool = False,
):
    """Load model, optimizer state, and training step from checkpoint."""
    if not Path(checkpoint_path).exists():
        raise FileNotFoundError(f"Checkpoint not found at {checkpoint_path}")
    
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint["model_state_dict"])
    if optimizer is not None and "optimizer_state_dict" in checkpoint:
        optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
    step = checkpoint.get("step", 0)
    print(f"Checkpoint loaded from {checkpoint_path}, resuming from step {step}")
    if return_checkpoint:
        return step, checkpoint
    return step

def load_hf_dataset(dataset_name: str = "wikitext", config_name: str = None, split: str = "train"):
    """Load a dataset from Hugging Face and combine text."""
    print(f"Loading {dataset_name} ({split}) from Hugging Face...")
    dataset = load_dataset(dataset_name, config_name, split=split)
    
    if "text" not in dataset.column_names:
        raise ValueError("Dataset does not contain a 'text' column")

    text = "\n".join(dataset["text"])

    print(f"Loaded {len(text)} characters")
    return text

def pre_chunk_data(data: torch.Tensor, block_size: int):
    """Slice data into non-overlapping chunks of block_size."""
    n_chunks = (len(data) - 1) // block_size
    trimmed_len = n_chunks * block_size + 1
    trimmed = data[:trimmed_len]
    x = trimmed[:-1].view(n_chunks, block_size)
    y = trimmed[1:].view(n_chunks, block_size)
    return x, y
