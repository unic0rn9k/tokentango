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

def train(model, train_x, train_y, train_cls, device):
    model.train()

    optimizer = Adam(model.parameters(), lr = 1e-4)

    num_samples = len(train_cls)
    num_epochs = 3
    mlm_losses = []
    cls_losses = []
    batch_size = 32
    start_time = dt.datetime.now()

    for epoch in range(0, num_epochs):
        for idx in range(0, int(num_samples/batch_size), batch_size):
            optimizer.zero_grad()

            x = train_x[idx:idx+batch_size,:]
            y = train_y[idx:idx+batch_size,:]
            cls_class = train_cls[idx:idx+batch_size]

            hidden = model.hidden(x)
            loss_cls = model.classify_loss(hidden, cls_class)
            loss_mlm = model.mlm_loss(hidden, y)
            loss = loss_cls + loss_mlm * 0.8
            loss.backward()
            optimizer.step()
            
            mlm_losses.append(loss_mlm.cpu().item())
            cls_losses.append(loss_cls.cpu().item())
            now = dt.datetime.now()
            
            if (idx % 100 == 0):
                eta = (num_samples - idx)/(idx+0.0001) * (now - start_time)
                loss = np.mean([a + b for a, b in zip(cls_losses, mlm_losses)])
                print(f"eta: {eta} | loss: {loss:.4f}")

    return cls_losses, mlm_losses
