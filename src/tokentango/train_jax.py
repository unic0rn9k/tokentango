import jax
import jax.numpy as jnp
import numpy as np
import optax
import equinox as eqx
from jaxtyping import Float, Int
from typing import Optional
import datetime as dt
import os
import sys
import random
import uuid
import pickle

from tokentango.data import BertData
from tokentango.config import (
    TrainingConfig,
    EvaluationResult,
)
from tokentango.model_jax import BertClassifier


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
    return f"{random.choice(ADJECTIVES)}-{random.choice(NOUNS)}-{uuid.uuid4().hex[:6]}"


def create_optimizer(
    optimizer_type: str,
    learning_rate: float,
) -> optax.GradientTransformation:
    optimizer_kwargs = {
        "Adam": lambda lr: optax.adam(learning_rate=lr),
        "AdamW": lambda lr: optax.adamw(learning_rate=lr, weight_decay=0.01),
        "SGD": lambda lr: optax.sgd(learning_rate=lr),
    }

    if optimizer_type not in optimizer_kwargs:
        raise ValueError(f"Unknown optimizer: {optimizer_type}")

    return optimizer_kwargs[optimizer_type](learning_rate)


def train(
    model: BertClassifier,
    train_data: BertData,
    test_data: BertData,
    config: Optional[TrainingConfig] = None,
):
    if config is None:
        config = TrainingConfig()

    config.validate()

    if config.run_name is None:
        config.run_name = generate_run_name()

    print(f"[TRAIN] Starting training run: {config.run_name}")
    print(f"[TRAIN] Training set fraction: {config.train_frac}")
    print(f"[TRAIN] Optimizer: {config.optimizer_type}, MLM: {config.use_mlm}")

    optimizer = create_optimizer(config.optimizer_type, config.lr)

    optimizer_state = optimizer.init(eqx.filter(model, eqx.is_array))

    rng = jax.random.key(42)

    mlm_losses = []
    cls_losses = []
    test_accuracies = []

    num_samples = len(train_data.labels)
    num_epochs = 2
    batch_size = config.batch_size
    start_time = dt.datetime.now()

    total_batches = num_samples // batch_size
    bar_width = 30
    is_tty = sys.stdout.isatty()
    ta = 0
    high_acc_count = 0
    early_stop_threshold = 99.0
    early_stop_iterations = (
        3 if dt.datetime.now() < dt.datetime(2026, 2, 27, 20, 30) else 5
    )

    final_checkpoint = None

    source_tokens = jnp.array(train_data.source_tokens)
    masked_tokens = jnp.array(train_data.masked_tokens)
    labels_arr = jnp.array(train_data.labels)

    def loss_fn(model: BertClassifier, x, masked_x, y):
        if config.use_mlm:
            hidden = model.hidden(masked_x)
            loss_cls = model.classify_loss(hidden, y)
            loss_mlm = model.mlm_loss(hidden, x)
            total_loss = loss_cls + loss_mlm
        else:
            hidden = model.hidden(x)
            loss_cls = model.classify_loss(hidden, y)
            total_loss = loss_cls
            loss_mlm = jnp.array(0.0)
        return total_loss, loss_mlm

    @eqx.filter_jit
    def train_step(model: BertClassifier, optimizer_state, x, masked_x, y):
        (loss, loss_mlm), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
            model, x, masked_x, y
        )
        updates, optimizer_state = optimizer.update(
            grads, optimizer_state, eqx.filter(model, eqx.is_array)
        )
        model = eqx.apply_updates(model, updates)
        return loss, loss_mlm, optimizer_state

    for epoch in range(0, num_epochs):
        rng, perm_key = jax.random.split(rng)
        indices = jax.random.permutation(perm_key, jnp.arange(num_samples))

        for sample, idx in enumerate(range(0, num_samples, batch_size)):
            batch_indices = indices[idx : idx + batch_size]

            x = source_tokens[batch_indices, :]
            masked_x = masked_tokens[batch_indices, :]
            y = labels_arr[batch_indices]

            loss, loss_mlm, optimizer_state = train_step(
                model, optimizer_state, x, masked_x, y
            )

            mlm_losses.append(float(loss_mlm))
            cls_losses.append(float(loss))

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
                result = test_accuracy(model, test_data)
                ta = result.accuracy
                test_accuracies.append(ta)

                if ta > early_stop_threshold:
                    high_acc_count += 1
                    print(
                        f"\n[TRAIN] High accuracy: {ta:.2f}% ({high_acc_count}/{early_stop_iterations} consecutive iterations > {early_stop_threshold}%)"
                    )
                else:
                    high_acc_count = 0

                now = dt.datetime.now()
                timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
                checkpoint_name = f"{config.checkpoint_dir}/checkpoint_{config.run_name}_{timestamp}_{ta:.2f}.jax"

                checkpoint_data = {
                    "model": model,
                    "optimizer_state": optimizer_state,
                    "config": config,
                    "epoch": epoch,
                    "accuracy": ta,
                    "timestamp": timestamp,
                    "cls_losses": cls_losses.copy(),
                    "mlm_losses": mlm_losses.copy(),
                }

                save_checkpoint(checkpoint_data, checkpoint_name)
                final_checkpoint = checkpoint_data

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

                cls_losses = []
                mlm_losses = []

                if high_acc_count >= early_stop_iterations:
                    print(
                        f"\n[TRAIN] Early stopping triggered! {high_acc_count} consecutive iterations with accuracy > {early_stop_threshold}%"
                    )
                    print(
                        f"[TRAIN] Training completed at epoch {epoch + 1}, sample {sample}"
                    )
                    return final_checkpoint

        now = dt.datetime.now()
        loss = np.mean(cls_losses[-100:]) if cls_losses else 0
        mlm_loss = np.mean(mlm_losses[-100:]) if mlm_losses else 0

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {loss:.4f} | MLM Loss: {mlm_loss:.4f}"
        )

    if final_checkpoint is None:
        print(
            "[TRAIN] No checkpoints were saved during training, creating final checkpoint..."
        )
        now = dt.datetime.now()
        timestamp = now.strftime("%Y-%m-%d_%H-%M-%S")
        checkpoint_name = f"{config.checkpoint_dir}/checkpoint_{config.run_name}_{timestamp}_final.jax"

        checkpoint_data = {
            "model": model,
            "optimizer_state": optimizer_state,
            "config": config,
            "epoch": num_epochs - 1,
            "accuracy": 0.0,
            "timestamp": timestamp,
            "cls_losses": cls_losses.copy(),
            "mlm_losses": mlm_losses.copy(),
        }
        save_checkpoint(checkpoint_data, checkpoint_name)

    return final_checkpoint


def test_accuracy(
    model: BertClassifier,
    test_data: BertData,
    frac: float = 0.1,
    use_masked_tokens: bool = False,
) -> EvaluationResult:
    sample_size = max(1, int(len(test_data.labels) * frac))
    random_offset = random.randint(0, len(test_data.labels) - sample_size)

    batch_size = 32
    correct = 0
    all_predictions = []
    all_labels = []

    source_tokens = jnp.array(test_data.source_tokens)
    masked_tokens = jnp.array(test_data.masked_tokens)
    labels_arr = jnp.array(test_data.labels)

    for start_idx in range(random_offset, random_offset + sample_size, batch_size):
        end_idx = min(start_idx + batch_size, random_offset + sample_size)

        if use_masked_tokens:
            x = masked_tokens[start_idx:end_idx, :]
        else:
            x = source_tokens[start_idx:end_idx, :]

        hidden = model.hidden(x)
        output = model.classify(hidden)
        output_sign = jnp.sign(output.flatten())
        true_sign = jnp.sign(labels_arr[start_idx:end_idx])
        correct += int(jnp.sum(output_sign == true_sign))
        all_predictions.extend([float(v) for v in output_sign])
        all_labels.extend([float(v) for v in true_sign])

    accuracy = correct / sample_size * 100

    tp = sum(1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == 1)
    tn = sum(1 for p, t in zip(all_predictions, all_labels) if p == -1 and t == -1)
    fp = sum(1 for p, t in zip(all_predictions, all_labels) if p == 1 and t == -1)
    fn = sum(1 for p, t in zip(all_predictions, all_labels) if p == -1 and t == 1)
    confusion_matrix = np.array([[tn, fp], [fn, tp]])

    return EvaluationResult(
        accuracy=accuracy, num_samples=sample_size, confusion_matrix=confusion_matrix
    )


def save_checkpoint(checkpoint_data: dict, filepath: str) -> str:
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, "wb") as f:
        pickle.dump(checkpoint_data, f)
    return filepath


def load_checkpoint(
    filepath: str,
    model: BertClassifier,
) -> tuple[BertClassifier, dict, TrainingConfig, int, float, str]:
    with open(filepath, "rb") as f:
        checkpoint_data = pickle.load(f)

    model = checkpoint_data["model"]
    optimizer_state = checkpoint_data["optimizer_state"]
    config = checkpoint_data["config"]
    epoch = checkpoint_data["epoch"]
    accuracy = checkpoint_data["accuracy"]
    timestamp = checkpoint_data["timestamp"]

    return model, optimizer_state, config, epoch, accuracy, timestamp
