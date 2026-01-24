import marimo

__generated_with = "0.19.5"
app = marimo.App(width="columns")


@app.cell
def _():
    import tokentango
    import torch
    import numpy as np
    import os
    import time
    import marimo as mo

    return np, os, time, torch, tokentango, mo


@app.cell
def _(torch):
    print(f"CUDA available: {torch.cuda.is_available()}")
    num_gpus = torch.cuda.device_count()
    print(f"Number of GPUs: {num_gpus}")
    for i in range(num_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")
        print(
            f"  Memory: {torch.cuda.get_device_properties(i).total_memory / 1e9:.2f} GB"
        )
        print(
            f"  CUDA Capability: {torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}"
        )
    return


@app.cell
def _(torch):
    device = torch.device("cuda:0")
    return (device,)


@app.cell
def _(mo):
    checkpoint_selector = mo.ui.dropdown(
        options=["latest", "train", "specific"],
        value="latest",
        label="Checkpoint selection",
    )
    checkpoint_path_input = mo.ui.text(label="Specific checkpoint path")
    return checkpoint_path_input, checkpoint_selector


@app.cell
def _(checkpoint_selector, checkpoint_path_input, mo):
    mode_panel = mo.md(
        f"""
        **Checkpoint Selection: {checkpoint_selector.value}**

        {"**Checkpoint Path:** " + checkpoint_path_input.value if checkpoint_selector.value == "specific" and checkpoint_path_input.value else ""}
        """
    )
    return (mode_panel,)


@app.cell
def _(time, tokentango):
    print("[DATA LOADING] Starting data load...")
    data_start = time.time()
    train_frac = 0.01
    train_data, test_data = tokentango.fake_news.load_data(train_frac)
    datatime = time.time() - data_start
    print(f"[DATA LOADING] Completed in {datatime:.2f}s")
    print(
        f"[DATA LOADING] train_data source_tokens shape={train_data.source_tokens.shape}, test_data source_tokens shape={test_data.source_tokens.shape}"
    )
    return test_data, train_data, train_frac


@app.cell
def _(device, test_data, torch, train_data):
    from tokentango.data import BertData

    train_data_device = BertData(
        train_data.source_tokens.to(device),
        train_data.masked_tokens.to(device),
        train_data.labels.to(device),
    )
    test_data_device = BertData(
        test_data.source_tokens.to(device),
        test_data.masked_tokens.to(device),
        test_data.labels.to(device),
    )
    return test_data_device, train_data_device


@app.cell
def _(device, tokentango):
    model = tokentango.BertClassifier(300, 40000, device).to(device)
    return (model,)


@app.cell
def _(
    device,
    model,
    os,
    test_data_device,
    time,
    tokentango,
    train_data_device,
    train_frac,
):
    checkpoint_mode = os.environ.get("MODEL_CHECKPOINT_PATH", "latest").lower()

    if checkpoint_mode == "train":
        print(
            "[STAGE 1] MODEL_CHECKPOINT_PATH=train: Starting training from scratch..."
        )
        result = tokentango.train.train(
            model,
            train_data_device,
            test_data_device,
            device,
            train_frac,
        )
        print("[STAGE 2] Training completed!")
    elif checkpoint_mode == "latest":
        print(
            "[STAGE 1] MODEL_CHECKPOINT_PATH=latest: Looking for latest checkpoint..."
        )
        checkpoints_dir = "data/checkpoints"
        checkpoint_files = tokentango.train.list_checkpoints(checkpoints_dir)

        if checkpoint_files:
            newest_checkpoint = checkpoint_files[0][0]
            print(f"[STAGE 2] Found checkpoint: {newest_checkpoint}, loading...")
            load_start = time.time()
            checkpoint = tokentango.train.load_checkpoint(model, newest_checkpoint)
            loadtime = time.time() - load_start
            print(
                f"[STAGE 2] Loaded checkpoint from epoch {checkpoint['epoch']} in {loadtime:.2f}s"
            )
            if "accuracy" in checkpoint:
                print(f"[STAGE 2] Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
            checkpoint_train_frac = checkpoint.get("train_frac", 0.8)
            print(f"[STAGE 2] Checkpoint training fraction: {checkpoint_train_frac}")
            print("[STAGE 2] Skipping training - proceeding directly to validation...")
        else:
            print(
                f"[STAGE 2] No checkpoint found in {checkpoints_dir}, starting training..."
            )
            result = tokentango.train.train(
                model,
                train_data_device,
                test_data_device,
                device,
                train_frac,
            )
            print("[STAGE 2] Training completed!")
    else:
        checkpoint_path = os.environ.get("MODEL_CHECKPOINT_PATH")
        print(
            f"[STAGE 1] MODEL_CHECKPOINT_PATH={checkpoint_path}: Loading specific checkpoint..."
        )
        if os.path.exists(checkpoint_path):
            load_start = time.time()
            checkpoint = tokentango.train.load_checkpoint(model, checkpoint_path)
            loadtime = time.time() - load_start
            print(
                f"[STAGE 2] Loaded checkpoint from epoch {checkpoint['epoch']} in {loadtime:.2f}s"
            )
            if "accuracy" in checkpoint:
                print(f"[STAGE 2] Checkpoint accuracy: {checkpoint['accuracy']:.2f}%")
            checkpoint_train_frac = checkpoint.get("train_frac", 0.8)
            print(f"[STAGE 2] Checkpoint training fraction: {checkpoint_train_frac}")
            print("[STAGE 2] Skipping training - proceeding directly to validation...")
        else:
            print(
                f"[STAGE 2] Checkpoint file not found: {checkpoint_path}, starting training..."
            )
            result = tokentango.train.train(
                model,
                train_data_device,
                test_data_device,
                device,
                train_frac,
            )
            print("[STAGE 2] Training completed!")
    return checkpoint_mode


@app.cell
def _(device, model, test_data_device, tokentango):
    acc = tokentango.train.test_accuracy(model, test_data_device, device, frac=1)
    print(f"Average test accuracy: {acc:.2f}%")
    return


@app.cell
def _(test_data_device, train_data_device):
    print(sum((n == 1 for n in train_data_device.labels)))
    print(sum((n == -1 for n in train_data_device.labels)))
    print(sum((n == 1 for n in test_data_device.labels)))
    print(sum((n == -1 for n in test_data_device.labels)))
    return


@app.cell
def _(model, np, test_data_device, torch):
    print("[CONFUSION MATRIX] Computing predictions...")
    with torch.no_grad():
        with torch.amp.autocast("cuda"):
            batch_size = 32
            predictions = []
            true_labels = []

            for start_idx in range(0, len(test_data_device.labels), batch_size):
                end_idx = min(start_idx + batch_size, len(test_data_device.labels))
                x = test_data_device.masked_tokens[start_idx:end_idx, :]
                hidden = model.hidden(x)
                output = model.classify(hidden)
                output_sign = np.sign(output.cpu().detach().numpy().flatten())
                predictions.extend(output_sign)
                true_labels.extend(
                    test_data_device.labels[start_idx:end_idx].cpu().numpy()
                )

    predictions = np.array(predictions)
    true_labels = np.array(true_labels)

    unique_labels = np.unique(np.concatenate([true_labels, predictions]))
    confusion_matrix = np.zeros((len(unique_labels), len(unique_labels)), dtype=int)

    label_to_idx = {label: idx for idx, label in enumerate(unique_labels)}

    for true_label, pred_label in zip(true_labels, predictions):
        true_idx = label_to_idx[true_label]
        pred_idx = label_to_idx[pred_label]
        confusion_matrix[true_idx, pred_idx] += 1

    print("\n[CONFUSION MATRIX]")
    print("                Predicted")
    print("              -1        1")
    header_row = (
        "Actual   -1  "
        + str(confusion_matrix[0, 0])
        + "       "
        + str(confusion_matrix[0, 1])
    )
    print(header_row)
    row_two = (
        "          1   "
        + str(confusion_matrix[1, 0])
        + "       "
        + str(confusion_matrix[1, 1])
    )
    print(row_two)

    correct = np.sum(predictions == true_labels)
    total = len(true_labels)
    accuracy = correct / total * 100
    print(f"\nConfusion matrix accuracy: {accuracy:.2f}% ({correct}/{total})")
    return


if __name__ == "__main__":
    app.run()
