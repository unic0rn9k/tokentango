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
    import time as _time

    return np, torch, os, _time


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
def _(_time, tokentango):
    print("[DATA LOADING] Starting data load...")
    data_start = _time.time()
    train_x, train_y, train_cls, test_x, test_y, test_cls = (
        tokentango.fake_news.load_data(0.02)
    )
    data_time = _time.time() - data_start
    print(f"[DATA LOADING] Completed in {data_time:.2f}s")
    print(
        f"[DATA LOADING] train_x shape={train_x.shape}, test_x shape={test_x.shape if hasattr(test_x, 'shape') else 'N/A'}"
    )
    return test_cls, test_x, test_y, train_cls, train_x, train_y


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
    _time,
    tokentango,
    torch,
    train_cls_1,
    train_x_1,
    train_y_1,
):
    print("[STAGE 1] Checking for checkpoint.pth...")
    checkpoint_path = "data/checkpoints/checkpoint.pth"

    if os.path.exists(checkpoint_path):
        print(f"[STAGE 2] Checkpoint found at {checkpoint_path}, loading...")
        load_start = _time.time()
        checkpoint = torch.load(checkpoint_path, weights_only=False)
        model.load_state_dict(checkpoint["model_state_dict"])
        load_time = _time.time() - load_start
        print(
            f"[STAGE 2] Loaded checkpoint from epoch {checkpoint['epoch']} in {load_time:.2f}s"
        )
        print("[STAGE 2] Skipping training - proceeding directly to validation...")
    else:
        print(
            f"[STAGE 2] No checkpoint found at {checkpoint_path}, starting training..."
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
        )
        print("[STAGE 2] Training completed!")
    print("[STAGE 3] Ready for model validation...")
    return


@app.cell
def _(device, model, np, test_cls_1, test_x_1, test_y_1, tokentango):
    print("Computing test accuracy (5 iterations)...")
    acc = np.mean(
        [
            tokentango.train.test_accuracy(
                model, test_x_1, test_y_1, test_cls_1, device
            )
            for _ in range(5)
        ]
    )
    print(f"Average test accuracy: {acc:.2f}%")
    return


@app.cell
def _(device, model, test_cls_1, test_x_1, test_y_1, tokentango):
    tokentango.train.test_accuracy(
        model, test_x_1, test_y_1, test_cls_1, device, frac=1
    )
    return


@app.cell
def _(test_cls_1, train_cls_1):
    print(sum((n == 1 for n in train_cls_1)))
    print(sum((n == -1 for n in train_cls_1)))
    print(sum((n == 1 for n in test_cls_1)))
    print(sum((n == -1 for n in test_cls_1)))
    return


if __name__ == "__main__":
    app.run()
