import marimo

__generated_with = "0.19.5"
app = marimo.App()


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
    from torch.optim import Adam, AdamW
    from torch import nn, functional as F
    from plotly import express as px
    import plotly.graph_objects as go
    from sklearn.metrics import classification_report, confusion_matrix
    import numpy as np
    from matplotlib import pyplot as plt
    import itertools
    import math
    import datetime as dt

    return (
        classification_report,
        confusion_matrix,
        itertools,
        np,
        plt,
        px,
        torch,
    )


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
def _(tokentango):
    print("[DATA LOADING] Starting data load...")
    import time as _time

    data_start = _time.time()
    # train_x = tokentango.fake_news.train_mlm
    # train_y = tokentango.fake_news.train_x
    # train_cls = tokentango.fake_news.train_y
    train_x, train_y, train_cls, test_x, test_y, test_cls = (
        tokentango.fake_news.load_data(0.01)
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
def _(test_cls_1, test_x_1, test_y_1, train_cls_1, train_x_1, train_y_1):
    size_in_bytes = (
        train_x_1.element_size() * train_x_1.numel()
        + train_y_1.element_size() * train_y_1.numel()
        + train_cls_1.element_size() * train_cls_1.numel()
    )
    size_in_bytes2 = (
        test_x_1.element_size() * test_x_1.numel()
        + test_y_1.element_size() * test_y_1.numel()
        + test_cls_1.element_size() * test_cls_1.numel()
    )
    size_in_gb = (size_in_bytes + size_in_bytes2) / 1024**3
    print(f"Tensor size: {size_in_gb:.4f} GB")
    return


@app.cell
def _(torch):
    # Current memory usage
    print(torch.cuda.memory_allocated() / 1024**3, "GB allocated")
    print(torch.cuda.memory_reserved() / 1024**3, "GB reserved by caching allocator")

    # Memory summary
    print(torch.cuda.memory_summary(device=None, abbreviated=False))
    return


@app.cell
def _(device, tokentango):
    print("Creating model...")
    model = tokentango.BertClassifier(300, 40000, device).to(device)
    print(f"Model created and moved to device: {device}")
    return (model,)


@app.cell
def _(
    device,
    model,
    test_cls_1,
    test_x_1,
    test_y_1,
    tokentango,
    train_cls_1,
    train_x_1,
    train_y_1,
    torch,
):
    # magic command not supported in marimo; please file an issue to add support
    # %%time
    # 1. embedding padding_idx (from model.py) should probably be set to pad_token_id (from fake_news.py)
    # 3. Maybe change named_parameters to parameters? (also in bert_from_scratch.py)
    # optimizer = AdamW(model.parameters(), lr = 1e-4, eps = 1e-8)

    print("[STAGE 1] Checking for checkpoint.pth...")
    import os
    import time as _time

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
def _(px):
    print("Skipping mlm_losses plot (not available when loading checkpoint)")
    return


@app.cell
def _(px):
    print("Skipping cls_losses plot (not available when loading checkpoint)")
    return


@app.cell
def _(px):
    print("Skipping test_acc plot (not available when loading checkpoint)")
    return


@app.cell
def _(px):
    print("Skipping test_acc plot (not available when loading checkpoint)")
    return


@app.cell
def _(model, np, test_cls_1, train_cls_1, train_y_1):
    correct = 0
    for _idx in range(len(train_cls_1)):
        _x = train_y_1[_idx : _idx + 1, :]
        _hidden = model.hidden(_x)
        _output = model.classify(_hidden)
        correct = correct + int(
            np.sign(_output.cpu().detach().item()) == np.sign(train_cls_1[_idx].cpu())
        )
    accuracy = correct / len(test_cls_1) * 100
    print(f"{accuracy}%")
    return


@app.cell
def _(model, test_cls_1, torch, train_x_1):
    print("Generating model outputs for all test samples...")
    outputs = []
    num_samples = len(test_cls_1)
    model.eval()
    with torch.no_grad():
        for _idx in range(0, num_samples):
            _x = train_x_1[_idx : _idx + 1, :]
            _hidden = model.hidden(_x)
            _output = model.classify(_hidden)
            outputs.append(_output)
    print(f"Generated {len(outputs)} outputs")
    return (outputs,)


@app.cell
def _(classification_report, np, outputs, test_cls_1, torch):
    print("Generating classification report...")
    outputs_1 = torch.sign(torch.cat(outputs))
    predicted_values = torch.round(outputs_1)
    predicted_values = predicted_values.cpu().view(-1).numpy()
    true_values = test_cls_1.cpu().numpy()
    label_values = ["reliable", "fake"]
    test_accuracy = np.sum(predicted_values == true_values) / len(true_values)
    print("Test Accuracy:", test_accuracy)
    print(
        classification_report(
            true_values, predicted_values, target_names=[str(l) for l in label_values]
        )
    )
    print("Classification report complete!")
    return label_values, predicted_values, true_values


@app.cell
def _(
    confusion_matrix,
    itertools,
    label_values,
    np,
    plt,
    predicted_values,
    true_values,
):
    def plot_confusion_matrix(
        cm, classes, normalize=False, title="Confusion matrix", cmap=plt.cm.Blues
    ):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print("Confusion matrix, without normalization")

        print(cm)

        plt.imshow(cm, interpolation="nearest", cmap=cmap)
        plt.title(title)
        plt.colorbar()
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes, rotation=45)
        plt.yticks(tick_marks, classes)

        fmt = ".2f" if normalize else "d"
        thresh = cm.max() / 2.0
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(
                j,
                i,
                format(cm[i, j], fmt),
                horizontalalignment="center",
                color="white" if cm[i, j] > thresh else "black",
            )

        plt.tight_layout()
        plt.ylabel("True label")
        plt.xlabel("Predicted label")

    # In[32]:

    cm_test = confusion_matrix(true_values, predicted_values)

    np.set_printoptions(precision=2)

    # plt.figure(figsize=(6,6))
    # plot_confusion_matrix(cm_test, classes=label_values, title='Confusion Matrix - Test Dataset')
    plt.figure(figsize=(6, 6))
    plot_confusion_matrix(
        cm_test,
        classes=label_values,
        title="Confusion Matrix - Test Dataset",
        normalize=True,
    )
    return


@app.cell
def _(predicted_values):
    print(sum(n == 1 for n in predicted_values))
    print(sum(n == -1 for n in predicted_values))
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
