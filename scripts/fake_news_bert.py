import marimo

__generated_with = "0.19.5"
app = marimo.App(width="columns")


@app.cell
def _():
    import marimo as mo

    return (mo,)


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    # Debug Bert notebook
    - [x] trivial case, with predicting shifted tokens, instead of mlm.
    - [x] trivial case, of classifying presence of marker token, at a random position.
    - [x] Add checkpoint selection - headless mode
    - [ ] Normalize frequency of labels across training and test samples
    """)
    return


@app.cell
def _():
    # magic command not supported in marimo; please file an issue to add support
    # %load_ext autoreload
    # '%autoreload 2' command supported automatically in marimo

    import tokentango

    return (tokentango,)


@app.cell
def _():
    import torch
    import numpy as np
    import os
    import time

    return np, os, time, torch


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


@app.cell(hide_code=True)
def _(mo):
    mo.md(r"""
    ```py
    # 1. Embeddings padding_idx = 0, might conflict with cls, or other tokens? (model.py)
    # 2. xs, masks and cls classes are all made with seperate calls to train_test_split (in bert_from_scratch.py)
    #    so maybe the cls' dont match the text
    num_samples = 2000

    xs = torch.randint(low=1, high=10, size=(num_samples,5), dtype=torch.int32)
    xs = torch.cat([torch.zeros(num_samples, 1, dtype=torch.int), xs], dim=1).to(device)

    ys = xs.clone()
    cls_label = [float(any(n == 1 for n in xs[i,:])) for i in range(num_samples)]

    split_at = int(0.2 * num_samples)

    train_x = xs[:split_at, :]
    train_y = ys[:split_at, :]
    train_cls = cls_label[:split_at]

    test_x = xs[split_at:, :]
    test_y = ys[split_at:, :]
    test_cls = cls_label[split_at:]
    ```
    """)
    return


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
    train_x, train_y, train_cls, test_x, test_y, test_cls = (
        tokentango.fake_news.load_data(train_frac)
    )
    datatime = time.time() - data_start
    print(f"[DATA LOADING] Completed in {datatime:.2f}s")
    print(
        f"[DATA LOADING] train_x shape={train_x.shape}, test_x shape={test_x.shape if hasattr(test_x, 'shape') else 'N/A'}"
    )
    return test_cls, test_x, test_y, train_cls, train_frac, train_x, train_y


@app.cell
def _(device, test_cls, test_x, test_y, torch, train_cls, train_x, train_y):
    train_x_1 = train_x.to(device)
    train_y_1 = train_y.to(device)
    train_cls_1 = train_cls.to(device)
    test_x_1 = torch.tensor(test_x).to(device)
    test_y_1 = torch.tensor(test_y).to(device)
    test_cls_1 = torch.tensor(test_cls).to(device)
    return test_cls_1, test_x_1, test_y_1, train_cls_1, train_x_1, train_y_1


@app.cell
def _(device, tokentango):
    model = tokentango.BertClassifier(300, 40000, device).to(device)
    return (model,)


@app.cell
def _(
    device,
    model,
    os,
    test_cls_1,
    test_x_1,
    test_y_1,
    time,
    tokentango,
    torch,
    train_cls_1,
    train_frac,
    train_x_1,
    train_y_1,
):
    checkpoint_mode = os.environ.get("MODEL_CHECKPOINT_PATH", "latest").lower()

    if checkpoint_mode == "train":
        print(
            "[STAGE 1] MODEL_CHECKPOINT_PATH=train: Starting training from scratch..."
        )
        result = tokentango.train.train(
            model,
            train_x_1,
            train_y_1,
            train_cls_1,
            test_x_1,
            test_y_1,
            test_cls_1,
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
                train_x_1,
                train_y_1,
                train_cls_1,
                test_x_1,
                test_y_1,
                test_cls_1,
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
                train_x_1,
                train_y_1,
                train_cls_1,
                test_x_1,
                test_y_1,
                test_cls_1,
                device,
                train_frac,
            )
            print("[STAGE 2] Training completed!")
    return checkpoint_mode


@app.cell
def _(device, model, test_cls_1, test_x_1, test_y_1, tokentango):
    acc = tokentango.train.test_accuracy(
        model, test_x_1, test_y_1, test_cls_1, device, frac=1
    )
    print(f"Average test accuracy: {acc:.2f}%")
    return


@app.cell
def _(test_cls_1, train_cls_1):
    print(sum((n == 1 for n in train_cls_1)))
    print(sum((n == -1 for n in train_cls_1)))
    print(sum((n == 1 for n in test_cls_1)))
    print(sum((n == -1 for n in test_cls_1)))
    return


@app.cell
def _(model, np, test_cls_1, test_y_1, torch):
    print("[CONFUSION MATRIX] Computing predictions...")
    with torch.no_grad():
        with torch.cuda.amp.autocast():
            batch_size = 32
            predictions = []
            true_labels = []

            for start_idx in range(0, len(test_cls_1), batch_size):
                end_idx = min(start_idx + batch_size, len(test_cls_1))
                x = test_y_1[start_idx:end_idx, :]
                hidden = model.hidden(x)
                output = model.classify(hidden)
                output_sign = np.sign(output.cpu().detach().numpy().flatten())
                predictions.extend(output_sign)
                true_labels.extend(test_cls_1[start_idx:end_idx].cpu().numpy())

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
