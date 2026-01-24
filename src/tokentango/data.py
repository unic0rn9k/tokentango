from dataclasses import dataclass
import torch
from typing import Optional, Dict, Any


@dataclass
class CheckpointData:
    epoch: int
    model_state_dict: Dict[str, Any]
    optimizer_state_dict: Dict[str, Any]
    loss: float
    accuracy: float
    train_frac: float = 0.8


@dataclass
class TrainingData:
    train_x: torch.Tensor
    train_y: torch.Tensor
    train_cls: torch.Tensor
    test_x: torch.Tensor
    test_y: torch.Tensor
    test_cls: torch.Tensor
    device: torch.device
    train_frac: float = 0.8
