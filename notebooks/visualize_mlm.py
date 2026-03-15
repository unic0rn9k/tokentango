import marimo

__generated_with = "0.19.5"
app = marimo.App(width="full")


@app.cell
def _():
    import tokentango as tt
    import plotly.graph_objects as go
    import datetime as dt
    import numpy as np
    from copy import copy
    import pandas as pd
    from typing import List
    from tokentango.config import Checkpoint
    return Checkpoint, List, dt, go, np, pd, tt


@app.cell
def _():
    #runs_of_interest = {'silly-llama-03248e', 'jolly-quokka-b92ef4'}
    #runs_of_interest = {'doc-otter-4e6ab5','happy-alpaca-1a941e'}
    #runs_of_interest = {'silly-penguin-eda9ce', 'sleepy-llama-d7b8b2'}
    #runs_of_interest = {'merry-llama-2699f1', 'sleepy-llama-d7b8b2'} # norm_first=True
    #runs_of_interest = {'sneezy-quokka-5290fe', 'sleepy-binturong-7b9979'}
    runs_of_interest = {'sneezy-quokka-5290fe', 'sleepy-binturong-7b9979', 'merry-llama-2699f1', 'sleepy-llama-d7b8b2', 'silly-penguin-eda9ce', 'sleepy-llama-d7b8b2', 'sneezy-capybara-ed50d3', 'merry-llama-2699f1'}
    return (runs_of_interest,)


@app.cell
def _(tt):
    checkpoints = tt.train.list_checkpoints(meta_only=True)
    return (checkpoints,)


@app.cell
def _(checkpoints):
    set([c.config.run_name for c in checkpoints])
    return


@app.cell
def _(Checkpoint, List, dt, np, pd, runs_of_interest):
    def checkpoints_to_df(checkpoints: List[Checkpoint]) -> pd.DataFrame:
        rows = []

        for ckpt in checkpoints:
            if ckpt.config.run_name not in runs_of_interest:
                continue
            mean_cls_loss = np.mean(ckpt.cls_losses)
            mean_mlm_loss = np.mean(ckpt.mlm_losses)

            rows.append({
                "run_name": ckpt.config.run_name,
                "time": dt.datetime.strptime(ckpt.timestamp, '%Y-%m-%d_%H-%M-%S'),
                "accuracy": ckpt.accuracy,
                "mean_cls_loss": mean_cls_loss,
                "mean_mlm_loss": mean_mlm_loss,
                "use_mlm": ckpt.config.use_mlm,
                "epoch": ckpt.epoch,
            })

        return pd.DataFrame(rows).sort_values(by="time")
    return (checkpoints_to_df,)


@app.cell
def _(checkpoints, checkpoints_to_df):
    df = checkpoints_to_df(checkpoints)
    df
    return (df,)


@app.cell
def _(df, go):
    _fig = go.Figure()

    for _run_name in df["run_name"].unique():
        _short_name = " ".join(_run_name.split("-")[:-1])
        _rows = df[df["run_name"] == _run_name].sort_values("time")
        _start = _rows["time"].min()
        _delta = [(t - _start).total_seconds() / 60 for t in _rows["time"]]

        _objective = "MLM + CLS" if all(_rows["use_mlm"]) else "only CLS"
        if any(_rows["use_mlm"]) and not all(_rows["use_mlm"]):
            _objective = "mixed MLM CLS"

        _fig.add_trace(
            go.Scatter(
                x=_delta,
                y=_rows["accuracy"],
                name=f"{_short_name} - {_objective}",
                customdata=_rows[["epoch"]].values,
                hovertemplate=(
                    "<b>%{fullData.name}</b><br>"
                    "Minutes: %{x:.2f}<br>"
                    "Accuracy: %{y:.4f}<br>"
                    "Epoch: %{customdata[0]}<extra></extra>"
                ),
            )
        )

    _fig.update_layout(
        hovermode="x unified",
        showlegend=False,
    )
    return


@app.cell
def _(df, go):
    _fig = go.Figure()

    for _run_name in df["run_name"].unique():
        _rows = df[df["run_name"] == _run_name]
        _start = _rows["time"].min()
        _delta = [(t - _start).total_seconds() / 60 for t in sorted(_rows["time"])]
        if not all(_rows["use_mlm"]):
            continue
        _fig.add_trace(go.Scatter(x=_delta, y=_rows["mean_mlm_loss"], name=f"{_run_name}"))

    _fig
    return


@app.cell
def _(checkpoints):
    mlm_run = [c for c in reversed(checkpoints) if c.config.run_name == 'merry-llama-2699f1']
    cls_run = [c for c in reversed(checkpoints) if c.config.run_name == 'sleepy-llama-d7b8b2']
    return cls_run, mlm_run


@app.cell
def _(cls_run, mlm_run):
    len(cls_run) / len(mlm_run)
    return


@app.cell
def _(cls_run, dt, mlm_run, np):
    mlm_ts = np.array([dt.datetime.strptime(c.timestamp, '%Y-%m-%d_%H-%M-%S') for c in mlm_run])
    cls_ts = np.array([dt.datetime.strptime(c.timestamp, '%Y-%m-%d_%H-%M-%S') for c in cls_run])

    mlm_ts -= mlm_ts[0]
    cls_ts -= cls_ts[0]

    mlm_ts = np.array([t.total_seconds() / 60 for t in mlm_ts], dtype=float)
    cls_ts = np.array([t.total_seconds() / 60 for t in cls_ts], dtype=float)
    return cls_ts, mlm_ts


@app.cell
def _(cls_run, cls_ts, go, mlm_run, mlm_ts):
    go.Figure(data=[
        go.Scatter(x=mlm_ts, y=[c.accuracy for c in mlm_run], name="MLM + CLS"),
        go.Scatter(x=cls_ts, y=[c.accuracy for c in cls_run], name="CLS only"),
    ]).update_layout(
        xaxis_title="Training duration (minutes)",
        yaxis_title="Accuracy on CLS test-set (%)",
    )
    return


@app.cell
def _(cls_run):
    cls_run[13].accuracy
    return


@app.cell
def _(cls_ts, mlm_ts, np):
    def interp_100(arr):
        n = len(arr)
        return np.interp(
            np.linspace(0, n - 1, (n - 1) * 100 + 1),
            np.arange(n),
            arr
        )

    mlm_ts2 = interp_100(mlm_ts)
    cls_ts2 = interp_100(cls_ts)
    return


@app.cell
def _(cls_run, mlm_run, np):
    #mlm_loss = [copy(c.cls_losses) for c in mlm_run]
    #cls_loss = [copy(c.cls_losses) for c in cls_run]

    #mlm_loss[0].pop(0)
    #cls_loss[0].pop(0)

    #mlm_loss = np.array(mlm_loss)
    #cls_loss = np.array(cls_loss)

    mlm_loss = [np.mean(c.cls_losses) for c in mlm_run]
    cls_loss = [np.mean(c.cls_losses) for c in cls_run]
    return cls_loss, mlm_loss


@app.cell
def _(cls_loss, cls_ts, go, mlm_loss, mlm_ts):
    go.Figure(data=[
        #go.Scatter(x=mlm_ts2, y=mlm_loss.flatten(), name="MLM + CLS"),
        #go.Scatter(x=cls_ts2, y=cls_loss.flatten(), name="CLS only"),
        #go.Scatter(x=mlm_ts, y=mlm_loss.mean(axis=1), name="avg MLM + CLS"),
        #go.Scatter(x=cls_ts, y=cls_loss.mean(axis=1), name="avg CLS only"),
        go.Scatter(x=mlm_ts, y=mlm_loss, name="MLM + CLS"),
        go.Scatter(x=cls_ts, y=cls_loss, name="CLS only"),
    ]).update_layout(
        xaxis_title="Training duration (minutes)",
        yaxis_title="CLS loss (smooth L1 loss)",
    )
    return


@app.cell
def _(mlm_run):
    print(mlm_run[0].config)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
