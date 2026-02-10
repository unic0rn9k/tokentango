"""Dataclasses for training configuration and checkpoints."""

from dataclasses import dataclass, field
import os
import torch
from typing import Optional, Dict, Any, List


@dataclass
class TrainingConfig:
    """Configuration for training with environment variable support."""

    train_frac: float = 0.8
    batch_size: int = 32
    lr: float = 1e-4
    optimizer_type: str = "adamw"  # adam, adamw, sgd
    use_mlm: bool = True
    seed: int = 42
    device: str = "cuda:0"
    run_name: Optional[str] = None  # Cute unique ID, inherited when resuming
    checkpoint_dir: str = "data/checkpoints"

    @classmethod
    def from_env(cls) -> "TrainingConfig":
        """Load configuration from environment variables with TT_ prefix."""
        config = cls()

        if "TT_TRAIN_FRAC" in os.environ:
            config.train_frac = float(os.environ["TT_TRAIN_FRAC"])
        if "TT_BATCH_SIZE" in os.environ:
            config.batch_size = int(os.environ["TT_BATCH_SIZE"])
        if "TT_LR" in os.environ:
            config.lr = float(os.environ["TT_LR"])
        if "TT_OPTIMIZER_TYPE" in os.environ:
            config.optimizer_type = os.environ["TT_OPTIMIZER_TYPE"].lower()
        if "TT_USE_MLM" in os.environ:
            config.use_mlm = os.environ["TT_USE_MLM"].lower() in ("true", "1", "yes")
        if "TT_SEED" in os.environ:
            config.seed = int(os.environ["TT_SEED"])
        if "TT_DEVICE" in os.environ:
            config.device = os.environ["TT_DEVICE"]
        if "TT_RUN_NAME" in os.environ:
            config.run_name = os.environ["TT_RUN_NAME"]
        if "TT_CHECKPOINT_DIR" in os.environ:
            config.checkpoint_dir = os.environ["TT_CHECKPOINT_DIR"]

        return config

    def validate(self) -> None:
        """Validate configuration values."""
        if not 0 < self.train_frac <= 1:
            raise ValueError(f"train_frac must be in (0, 1], got {self.train_frac}")
        if self.batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {self.batch_size}")
        if self.lr <= 0:
            raise ValueError(f"lr must be positive, got {self.lr}")
        if self.optimizer_type not in ("adam", "adamw", "sgd"):
            raise ValueError(
                f"optimizer_type must be adam, adamw, or sgd, got {self.optimizer_type}"
            )


@dataclass
class Checkpoint:
    """Checkpoint containing model state, config, and training metadata."""

    model_state: Dict[str, Any]
    optimizer_state: Dict[str, Any]
    config: TrainingConfig
    epoch: int
    accuracy: float
    timestamp: str
    cls_losses: List[float] = field(default_factory=list)
    mlm_losses: List[float] = field(default_factory=list)
    checkpoint_path: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert checkpoint to dictionary for saving."""
        return {
            "model_state_dict": self.model_state,
            "optimizer_state_dict": self.optimizer_state,
            "config": self.config.__dict__,
            "epoch": self.epoch,
            "accuracy": self.accuracy,
            "timestamp": self.timestamp,
            "cls_losses": self.cls_losses,
            "mlm_losses": self.mlm_losses,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Checkpoint":
        """Create checkpoint from dictionary (handles old format)."""
        # Handle old checkpoint format
        if "config" not in data:
            # Create config from old format fields
            config = TrainingConfig()
            config.train_frac = data.get("train_frac", 0.8)
            # Old checkpoints don't have other config fields, use defaults
        else:
            config = TrainingConfig(**data["config"])

        return cls(
            model_state=data.get("model_state_dict", {}),
            optimizer_state=data.get("optimizer_state_dict", {}),
            config=config,
            epoch=data.get("epoch", 0),
            accuracy=data.get("accuracy", 0.0),
            timestamp=data.get("timestamp", ""),
            cls_losses=data.get("cls_losses", []),
            mlm_losses=data.get("mlm_losses", []),
        )


@dataclass
class EvaluationResult:
    """Result from model evaluation."""

    accuracy: float
    num_samples: int
    confusion_matrix: Optional[Any] = None
