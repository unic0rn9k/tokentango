import jax
import numpy as np
import os
import time

from tokentango.config import TrainingConfig
from tokentango import fake_news
from tokentango.model_jax import BertClassifier
from tokentango import train_jax


def main():
    print(f"JAX devices: {jax.devices()}")

    config = TrainingConfig.from_env()
    print(f"[DATA LOADING] Starting data load...")
    data_start = time.time()
    train_data, test_data = fake_news.load_data(
        config.train_frac, random_state=config.seed
    )
    datatime = time.time() - data_start
    print(f"[DATA LOADING] Completed in {datatime:.2f}s")
    print(
        f"[DATA LOADING] train_data source_tokens shape={train_data.source_tokens.shape}, test_data source_tokens shape={test_data.source_tokens.shape}"
    )

    checkpoint_mode = os.environ.get("MODEL_CHECKPOINT_PATH", "latest").lower()

    key = jax.random.key(config.seed)
    model = BertClassifier(
        vocab_size=40000,
        max_seq_len=300,
        key=key,
    )

    if checkpoint_mode == "train":
        print(
            "[STAGE 1] MODEL_CHECKPOINT_PATH=train: Starting training from scratch..."
        )
        result = train_jax.train(
            model,
            train_data,
            test_data,
            config,
        )
        print(f"[STAGE 2] Training completed! Run: {result['config'].run_name}")
    elif checkpoint_mode == "latest":
        print(
            "[STAGE 1] MODEL_CHECKPOINT_PATH=latest: Looking for latest checkpoint..."
        )
        checkpoints_dir = "data/checkpoints"

        import glob

        jax_checkpoints = glob.glob(f"{checkpoints_dir}/*.jax")

        if jax_checkpoints:
            newest_checkpoint = max(jax_checkpoints, key=os.path.getmtime)
            print(f"[STAGE 2] Found checkpoint: {newest_checkpoint}, loading...")
            load_start = time.time()
            model, optimizer_state, config, epoch, accuracy, timestamp = (
                train_jax.load_checkpoint(newest_checkpoint, model)
            )
            loadtime = time.time() - load_start
            print(f"[STAGE 2] Loaded checkpoint from epoch {epoch} in {loadtime:.2f}s")
            print(f"[STAGE 2] Checkpoint accuracy: {accuracy:.2f}%")
            print(f"[STAGE 2] Training fraction: {config.train_frac}")
            print(f"[STAGE 2] Previous run name: {config.run_name}")
            config.run_name = f"{config.run_name}-{train_jax.generate_run_name()}"
            print(f"[STAGE 2] Continuing training with new run name: {config.run_name}")
            result = train_jax.train(
                model,
                train_data,
                test_data,
                config,
            )
            print(f"[STAGE 2] Training completed! Run: {result['config'].run_name}")
        else:
            print(
                f"[STAGE 2] No checkpoint found in {checkpoints_dir}, starting training..."
            )
            result = train_jax.train(
                model,
                train_data,
                test_data,
                config,
            )
            print(f"[STAGE 2] Training completed! Run: {result['config'].run_name}")
    else:
        checkpoint_path = os.environ.get("MODEL_CHECKPOINT_PATH")
        print(
            f"[STAGE 1] MODEL_CHECKPOINT_PATH={checkpoint_path}: Loading specific checkpoint..."
        )
        if os.path.exists(checkpoint_path):
            load_start = time.time()
            model, optimizer_state, config, epoch, accuracy, timestamp = (
                train_jax.load_checkpoint(checkpoint_path, model)
            )
            loadtime = time.time() - load_start
            print(f"[STAGE 2] Loaded checkpoint from epoch {epoch} in {loadtime:.2f}s")
            print(f"[STAGE 2] Checkpoint accuracy: {accuracy:.2f}%")
            print(f"[STAGE 2] Training fraction: {config.train_frac}")
            print(f"[STAGE 2] Previous run name: {config.run_name}")
            config.run_name = f"{config.run_name}-{train_jax.generate_run_name()}"
            print(f"[STAGE 2] Continuing training with new run name: {config.run_name}")
            result = train_jax.train(
                model,
                train_data,
                test_data,
                config,
            )
            print(f"[STAGE 2] Training completed! Run: {result['config'].run_name}")
        else:
            print(
                f"[STAGE 2] Checkpoint file not found: {checkpoint_path}, starting training..."
            )
            result = train_jax.train(
                model,
                train_data,
                test_data,
                config,
            )
            print(f"[STAGE 2] Training completed! Run: {result['config'].run_name}")

    eval_result = train_jax.test_accuracy(model, test_data, frac=1)
    print(f"Average test accuracy: {eval_result.accuracy:.2f}%")
    print(f"Samples tested: {eval_result.num_samples}")
    if eval_result.confusion_matrix is not None:
        print(f"Confusion matrix:\n{eval_result.confusion_matrix}")


if __name__ == "__main__":
    main()
