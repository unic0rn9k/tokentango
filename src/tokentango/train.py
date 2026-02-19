import torch
import numpy as np
import random
import datetime as dt
import os
import sys
from torch.amp.autocast_mode import autocast
from torch.amp.grad_scaler import GradScaler
from tokentango.data import BertData
from tokentango.config import TrainingConfig, Checkpoint, EvaluationResult
from typing import Optional
import uuid
import random as rand
import torch.optim as optim


# Cute name generator for run names
ADJECTIVES = [
    "happy",
    "sleepy",
    "grumpy",
    "sneezy",
    "dopey",
    "bashful",
    "doc",
    "merry",
    "jolly",
    "silly",
]
NOUNS = [
    "panda",
    "koala",
    "otter",
    "penguin",
    "llama",
    "alpaca",
    "capybara",
    "quokka",
    "narwhal",
    "axolotl",
]


def generate_run_name() -> str:
    """Generate a cute unique run name."""
    return f"{rand.choice(ADJECTIVES)}-{rand.choice(NOUNS)}-{uuid.uuid4().hex[:6]}"


def list_checkpoints(checkpoints_dir="data/checkpoints") -> list:
    """List all checkpoints in directory, returning list of Checkpoint objects."""
    if not os.path.exists(checkpoints_dir):
        return []

    checkpoints = []
    for f in os.listdir(checkpoints_dir):
        if f.startswith("checkpoint_") and f.endswith(".pth"):
            filepath = os.path.join(checkpoints_dir, f)
            try:
                # Use map_location to handle checkpoints from different devices
                data = torch.load(filepath, weights_only=False, map_location="cpu")
                checkpoint = Checkpoint.from_dict(data)
                checkpoint.checkpoint_path = filepath  # Add path for reference
                checkpoints.append(checkpoint)
            except Exception as e:
                print(f"Warning: Could not load checkpoint {filepath}: {e}")

    # Sort by timestamp (newest first)
    checkpoints.sort(key=lambda x: x.timestamp, reverse=True)
    return checkpoints


def load_checkpoint(model, checkpoint_path) -> Checkpoint:
    """Load checkpoint and return Checkpoint object (model state already loaded into model)."""
    # Use map_location to handle checkpoints from different devices
    data = torch.load(checkpoint_path, weights_only=False, map_location="cpu")
    checkpoint = Checkpoint.from_dict(data)
    checkpoint.checkpoint_path = checkpoint_path
    model.load_state_dict(checkpoint.model_state)
    return checkpoint


def save_checkpoint(checkpoint: Checkpoint, filepath: str) -> str:
    """Save checkpoint to file and return path."""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    torch.save(checkpoint.to_dict(), filepath)
    return filepath


def test_accuracy(
    model, test_data: BertData, device, frac=0.1, use_masked_tokens=False
) -> EvaluationResult:
    """Test model accuracy and return EvaluationResult.

    Args:
        model: The model to test
        test_data: BertData containing source_tokens, masked_tokens, and labels
        device: Device to run on
        frac: Fraction of test data to use
        use_masked_tokens: If True, use masked_tokens instead of source_tokens for input
    """
    device_type = device.type
    model.eval()

    with torch.no_grad():
        with autocast(device_type=device_type):
            sample_size = max(1, int(len(test_data.labels) * frac))
            random_offset = random.randint(0, len(test_data.labels) - sample_size)

            batch_size = 32
            correct = 0
            all_predictions = []
            all_labels = []

            for start_idx in range(
                random_offset, random_offset + sample_size, batch_size
            ):
                end_idx = min(start_idx + batch_size, random_offset + sample_size)
                if use_masked_tokens:
                    x = test_data.masked_tokens[start_idx:end_idx, :]
                else:
                    x = test_data.source_tokens[start_idx:end_idx, :]
                hidden = model.hidden(x)
                output = model.classify(hidden)
                output_sign = np.sign(output.cpu().detach().numpy().flatten())
                true_sign = np.sign(test_data.labels[start_idx:end_idx].cpu().numpy())
                correct += int(np.sum(output_sign == true_sign))
                all_predictions.extend(output_sign)
                all_labels.extend(true_sign)

            accuracy = correct / sample_size * 100

            # Build confusion matrix
            tp = sum(
                1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == 1
            )
            tn = sum(
                1 for p, t in zip(all_predictions, all_labels) if p == -1 and t == -1
            )
            fp = sum(
                1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == -1
            )
            fn = sum(
                1 for p, t in zip(all_predictions, all_labels) if p == -1 and t == 1
            )
            confusion_matrix = np.array([[tn, fp], [fn, tp]])

    model.train()
    return EvaluationResult(
        accuracy=accuracy, num_samples=sample_size, confusion_matrix=confusion_matrix
    )


def train(
    model,
    train_data: BertData,
    test_data: BertData,
    config: Optional[TrainingConfig] = None,
) -> Checkpoint:
    """Train model with given config and return final Checkpoint."""
    if config is None:
        config = TrainingConfig()

    config.validate()

    # Generate run name if not provided (BEFORE setting seeds to ensure randomness)
    if config.run_name is None:
        config.run_name = generate_run_name()

    device = torch.device(config.device)
    model = model.to(device)
    model.train()

    print(f"[TRAIN] Starting training run: {config.run_name}")
    print(f"[TRAIN] Training set fraction: {config.train_frac}")
    print(f"[TRAIN] Optimizer: {config.optimizer_type}, MLM: {config.use_mlm}")

    optimizer_kwargs = {"Adam": {}, "AdamW": {"weight_decay": 0.01}, "SGD": {}}

    OptimizerClass = getattr(optim, config.optimizer_type)
    optimizer = OptimizerClass(
        model.parameters(),
        lr=config.lr,
        **optimizer_kwargs.get(config.optimizer_type, {}),
    )

    mlm_losses = []
    cls_losses = []
    test_accuracies = []

    num_samples = len(train_data.labels)
    num_epochs = 2
    batch_size = config.batch_size
    start_time = dt.datetime.now()
    device_type = device.type
    scaler = GradScaler(device_type)

    total_batches = int(num_samples / batch_size)
    bar_width = 30
    is_tty = sys.stdout.isatty()
    ta = 0
    high_acc_count = 0
    early_stop_threshold = 80.0
    early_stop_iterations = 3

    final_checkpoint = None

    for epoch in range(0, num_epochs):
        for sample, idx in enumerate(range(0, num_samples, batch_size)):
            optimizer.zero_grad()

            x = train_data.source_tokens[idx : idx + batch_size, :]
            masked_tokens = train_data.masked_tokens[idx : idx + batch_size, :]
            cls_class = train_data.labels[idx : idx + batch_size]

            with autocast(device_type=device_type):
                if config.use_mlm:
                    hidden = model.hidden(masked_tokens)
                    loss_cls = model.classify_loss(hidden, cls_class)
                    loss_mlm = model.mlm_loss(hidden, masked_tokens)
                    loss = loss_cls + loss_mlm
                else:
                    hidden = model.hidden(x)
                    loss_cls = model.classify_loss(hidden, cls_class)
                    loss_mlm = torch.tensor(0.0)
                    loss = loss_cls

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            mlm_losses.append(loss_mlm.cpu().item())
            cls_losses.append(loss_cls.cpu().item())

            if sample % 5 == 0:
                progress = sample / total_batches * 100
                filled = int(bar_width * progress / 100)
                bar = "[" + "=" * filled + ">" + " " * (bar_width - filled) + "]"

                elapsed = dt.datetime.now() - start_time
                batches_per_sec = (
                    (sample + 1) / elapsed.total_seconds()
                    if elapsed.total_seconds() > 0
                    else 0
                )
                remaining_batches = total_batches - sample
                eta_seconds = (
                    remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                )
                eta_str = str(dt.timedelta(seconds=int(eta_seconds)))

                pb = f"Epoch {epoch + 1}/{num_epochs} {bar} {progress:5.1f}% | ETA: {eta_str} | TA: {ta:5.2f}%"
                if is_tty:
                    sys.stdout.write("\033[2K\033[1G")
                    sys.stdout.write(pb)
                    sys.stdout.flush()
                else:
                    print(pb)

            if sample % 100 == 0 and sample > 0:
                result = test_accuracy(model, test_data, device)
                ta = result.accuracy
                test_accuracies.append(ta)

                # Check early stopping condition
                if ta > early_stop_threshold:
                    high_acc_count += 1
                    print(
                        f"\n[TRAIN] High accuracy: {ta:.2f}% ({high_acc_count}/{early_stop_iterations} consecutive iterations > {early_stop_threshold}%)"
                    )
                else:
                    high_acc_count = 0

                now = dt.datetime.now()
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                checkpoint_name = f"{config.checkpoint_dir}/checkpoint_{config.run_name}_{timestamp}_{ta:.2f}.pth"

                # Create checkpoint with losses since last save
                checkpoint = Checkpoint(
                    model_state=model.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    config=config,
                    epoch=epoch,
                    accuracy=ta,
                    timestamp=timestamp,
                    cls_losses=cls_losses.copy(),
                    mlm_losses=mlm_losses.copy(),
                )

                save_checkpoint(checkpoint, checkpoint_name)
                final_checkpoint = checkpoint

                elapsed = dt.datetime.now() - start_time
                batches_per_sec = (
                    (sample + 1) / elapsed.total_seconds()
                    if elapsed.total_seconds() > 0
                    else 0
                )
                remaining_batches = total_batches * (num_epochs - epoch) - sample
                eta_seconds = (
                    remaining_batches / batches_per_sec if batches_per_sec > 0 else 0
                )
                eta_str = str(dt.timedelta(seconds=int(eta_seconds)))

                if is_tty:
                    sys.stdout.write("\033[2K\033[1G")
                sys.stdout.write(
                    f"Epoch {epoch + 1}/{num_epochs} [{sample:4d}/{total_batches}] | TA: {ta:5.2f}% | Saved checkpoint | ETA: {eta_str}\n"
                )
                sys.stdout.flush()

                # Clear accumulated losses after saving
                cls_losses = []
                mlm_losses = []

                # Early stopping check
                if high_acc_count >= early_stop_iterations:
                    print(
                        f"\n[TRAIN] Early stopping triggered! {high_acc_count} consecutive iterations with accuracy > {early_stop_threshold}%"
                    )
                    print(
                        f"[TRAIN] Training completed at epoch {epoch + 1}, sample {sample}"
                    )
                    return final_checkpoint

        now = dt.datetime.now()
        eta = (now - start_time) / (epoch + 1) * (num_epochs - epoch - 1)
        loss = np.mean(cls_losses[-100:]) if cls_losses else 0
        mlm_loss = np.mean(mlm_losses[-100:]) if mlm_losses else 0

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {loss:.4f} | MLM Loss: {mlm_loss:.4f}"
        )

    # If no checkpoints were saved during training, create one at the end
    if final_checkpoint is None:
        print(
            "[TRAIN] No checkpoints saved during training, creating final checkpoint..."
        )
        now = dt.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_name = f"{config.checkpoint_dir}/checkpoint_{config.run_name}_{timestamp}_final.pth"

        final_checkpoint = Checkpoint(
            model_state=model.state_dict(),
            optimizer_state=optimizer.state_dict(),
            config=config,
            epoch=num_epochs - 1,
            accuracy=0.0,
            timestamp=timestamp,
            cls_losses=cls_losses.copy(),
            mlm_losses=mlm_losses.copy(),
        )
        save_checkpoint(final_checkpoint, checkpoint_name)

    return final_checkpoint
