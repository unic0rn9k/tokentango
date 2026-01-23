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
import random
from torch.cuda.amp import autocast, GradScaler


def test_accuracy(model, test_x, test_y, test_cls, device):
    with torch.no_grad():
        with autocast():
            sample_size = max(1, int(len(test_cls) * 0.1))
            random_offset = random.randint(0, len(test_cls) - sample_size)

            correct = 0
            for idx in range(random_offset, random_offset + sample_size):
                x = test_y[idx : idx + 1, :]
                hidden = model.hidden(x)
                output = model.classify(hidden)
                correct += int(
                    np.sign(output.cpu().detach().item())
                    == np.sign(test_cls[idx].cpu())
                )

            return correct / sample_size * 100


def train(model, train_x, train_y, train_cls, test_x, test_y, test_cls, device):
    model.train()

    optimizer = AdamW(model.parameters(), lr=1e-4, weight_decay=0.01)

    mlm_losses = []
    cls_losses = []
    test_accuracies = []

    num_samples = len(train_cls)
    num_epochs = 40
    batch_size = 32
    start_time = dt.datetime.now()
    scaler = GradScaler()

    for epoch in range(0, num_epochs):
        for sample, idx in enumerate(
            range(0, int(num_samples / batch_size), batch_size)
        ):
            optimizer.zero_grad()

            x = train_x[idx : idx + batch_size, :]
            y = train_y[idx : idx + batch_size, :]
            cls_class = train_cls[idx : idx + batch_size]

            with autocast():
                hidden = model.hidden(x)
                loss_cls = model.classify_loss(hidden, cls_class)
                loss_mlm = model.mlm_loss(hidden, y)
                loss = loss_cls + loss_mlm

            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()

            # loss.backward()
            # optimizer.step()

            mlm_losses.append(loss_mlm.cpu().item())
            cls_losses.append(loss_cls.cpu().item())

            if sample % 100 == 0:
                ta = test_accuracy(model, test_x, test_y, test_cls, device)
                print(f"ta: {ta:.2f}%")
                test_accuracies.append(ta)

        now = dt.datetime.now()
        eta = (now - start_time) / (epoch + 1) * (num_epochs - epoch)
        # loss = np.mean([a + b for a, b in zip(cls_losses, mlm_losses)])
        loss = np.mean(cls_losses)

        print(f"eta: {eta} | loss: {loss:.4f}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "loss": loss,
            },
            "data/checkpoints/checkpoint.pth",
        )

    return cls_losses, mlm_losses
