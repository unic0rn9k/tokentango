import torch
from torch.optim import AdamW
import numpy as np
import random
import datetime as dt
import os
import sys
from torch.amp import autocast, GradScaler
from tokentango.data import BertData


def list_checkpoints(checkpoints_dir="data/checkpoints"):
    if not os.path.exists(checkpoints_dir):
        return []
    checkpoint_files = []
    for f in os.listdir(checkpoints_dir):
        if f.startswith("checkpoint_") and f.endswith(".pth"):
            filepath = os.path.join(checkpoints_dir, f)
            checkpoint_files.append((filepath, os.path.getmtime(filepath)))
    checkpoint_files.sort(key=lambda x: x[1], reverse=True)
    return checkpoint_files


def load_checkpoint(model, checkpoint_path):
    checkpoint = torch.load(checkpoint_path, weights_only=False)
    model.load_state_dict(checkpoint["model_state_dict"])
    return checkpoint


def test_accuracy(model, test_data: BertData, device, frac=0.1):
    device_type = device.type
    with torch.no_grad():
        with autocast(device_type=device_type):
            sample_size = max(1, int(len(test_data.labels) * frac))
            random_offset = random.randint(0, len(test_data.labels) - sample_size)

            batch_size = 32
            correct = 0
            for start_idx in range(
                random_offset, random_offset + sample_size, batch_size
            ):
                end_idx = min(start_idx + batch_size, random_offset + sample_size)
                # x = test_data.masked_tokens[start_idx:end_idx, :]
                x = test_data.source_tokens[start_idx:end_idx, :]
                hidden = model.hidden(x)
                output = model.classify(hidden)
                output_sign = np.sign(output.cpu().detach().numpy().flatten())
                true_sign = np.sign(test_data.labels[start_idx:end_idx].cpu().numpy())
                correct += int(np.sum(output_sign == true_sign))

            return correct / sample_size * 100


def train(
    model,
    train_data: BertData,
    test_data: BertData,
    device,
    train_frac=0.8,
    early_stop_threshold=80.0,
    early_stop_iterations=3,
):
    print("[TRAIN] Starting training loop...")
    print(f"[TRAIN] Training set fraction: {train_frac}")
    print(
        f"[TRAIN] Early stopping: {early_stop_iterations} iterations > {early_stop_threshold}%"
    )
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    mlm_losses = []
    cls_losses = []
    test_accuracies = []

    num_samples = len(train_data.labels)
    num_epochs = 2
    batch_size = 32
    start_time = dt.datetime.now()
    device_type = device.type
    scaler = GradScaler(device_type)

    total_batches = int(num_samples / batch_size)
    bar_width = 30
    total_progress_updates = 0
    is_tty = sys.stdout.isatty()
    ta = 0
    high_acc_count = 0

    for epoch in range(0, num_epochs):
        for sample, idx in enumerate(range(0, num_samples, batch_size)):
            optimizer.zero_grad()

            x = train_data.source_tokens[idx : idx + batch_size, :]
            masked_tokens = train_data.masked_tokens[idx : idx + batch_size, :]
            cls_class = train_data.labels[idx : idx + batch_size]

            with autocast(device_type=device_type):
                hidden = model.hidden(x)
                loss_cls = model.classify_loss(hidden, cls_class)
                loss_mlm = model.mlm_loss(hidden, masked_tokens)
                loss = loss_cls + loss_mlm

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
                ta = test_accuracy(model, test_data, device)
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
                checkpoint_name = (
                    f"data/checkpoints/checkpoint_{timestamp}_{ta:.2f}.pth"
                )

                torch.save(
                    {
                        "epoch": epoch,
                        "model_state_dict": model.state_dict(),
                        "optimizer_state_dict": optimizer.state_dict(),
                        "loss": loss.cpu().item(),
                        "accuracy": ta,
                        "train_frac": train_frac,
                    },
                    checkpoint_name,
                )

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

                # Early stopping check
                if high_acc_count >= early_stop_iterations:
                    print(
                        f"\n[TRAIN] Early stopping triggered! {high_acc_count} consecutive iterations with accuracy > {early_stop_threshold}%"
                    )
                    print(
                        f"[TRAIN] Training completed at epoch {epoch + 1}, sample {sample}"
                    )
                    return cls_losses, mlm_losses

        now = dt.datetime.now()
        eta = (now - start_time) / (epoch + 1) * (num_epochs - epoch - 1)
        loss = np.mean(cls_losses[-100:]) if cls_losses else 0
        mlm_loss = np.mean(mlm_losses[-100:]) if mlm_losses else 0

        print(
            f"Epoch {epoch + 1}/{num_epochs} completed | Loss: {loss:.4f} | MLM Loss: {mlm_loss:.4f}"
        )

    return cls_losses, mlm_losses
