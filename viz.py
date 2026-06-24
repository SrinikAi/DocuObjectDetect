from __future__ import annotations

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import plotly.graph_objects as go
from plotly.subplots import make_subplots


def split_image_counts(data) -> dict[str, list[int]]:
    counts = {}
    for split, idxs in data.split_idx.items():
        per_class = [0] * data.num_classes
        for i in idxs:
            for c in np.unique(data.samples[i].labels):
                per_class[int(c)] += 1
        counts[split] = per_class
    return counts


def split_instance_counts(data) -> dict[str, list[int]]:
    counts = {}
    for split, idxs in data.split_idx.items():
        per_class = [0] * data.num_classes
        for i in idxs:
            for c in data.samples[i].labels:
                per_class[int(c)] += 1
        counts[split] = per_class
    return counts


def plot_split_distributions(data, order=("train", "val", "test"),
                             metric="images"):
    if metric == "images":
        counts = split_image_counts(data)
        y_title, noun = "number of images", "image counts"
        n_of = lambda s: (len(data.split_idx[s]), "imgs")
    elif metric == "instances":
        counts = split_instance_counts(data)
        y_title, noun = "number of instances", "instance counts"
        n_of = lambda s: (sum(counts[s]), "boxes")
    else:
        raise ValueError("metric must be 'images' or 'instances'")

    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}

    fig = make_subplots(
        rows=1, cols=len(order),
        subplot_titles=[f"{s.upper()}  (n={n_of(s)[0]} {n_of(s)[1]})"
                        for s in order],
        shared_yaxes=True, horizontal_spacing=0.04,
    )
    for col, s in enumerate(order, start=1):
        y = counts[s]
        fig.add_trace(
            go.Bar(x=data.classes, y=y, name=s,
                   marker_color=colors.get(s, None),
                   text=y, textposition="outside",
                   showlegend=False),
            row=1, col=col,
        )
        fig.update_xaxes(tickangle=-40, row=1, col=col)

    fig.update_yaxes(title_text=y_title, row=1, col=1)
    fig.update_layout(
        title_text=f"Indoor Object Detection — per-class {noun} by split "
                    "(80 / 10 / 10)",
        height=480, width=1150, bargap=0.25,
        template="plotly_white", margin=dict(t=90, b=120),
    )
    return fig


def plot_class_examples(data, split="train", cols=4, seed=0, figsize=None):
    rng = np.random.default_rng(seed)
    idxs = data.split_idx[split]

    chosen = {}
    for c in range(data.num_classes):
        members = [i for i in idxs if c in data.samples[i].labels]
        if members:
            chosen[c] = int(rng.choice(members))

    cmap = plt.get_cmap("tab10")
    colors = [cmap(i % 10) for i in range(data.num_classes)]

    items = sorted(chosen.items())
    n = len(items)
    cols = min(cols, n) if n else 1
    rows = (n + cols - 1) // cols
    if figsize is None:
        figsize = (cols * 4, rows * 3.4)

    fig, axes = plt.subplots(rows, cols, figsize=figsize, squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    for ax, (c, gidx) in zip(axes, items):
        s = data.samples[gidx]
        ax.imshow(data.get_image(gidx))
        ax.set_title(f"{data.classes[c]}  (img {gidx})", fontsize=11)
        for (x1, y1, x2, y2), lab in zip(s.boxes, s.labels):
            col = colors[int(lab)]
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=col, linewidth=2))
            ax.text(x1, max(0, y1 - 3), data.classes[int(lab)],
                    fontsize=8, color="white",
                    bbox=dict(facecolor=col, edgecolor="none", pad=1))

    fig.suptitle(f"{split} — one example image per class (all boxes drawn)",
                 fontsize=13)
    fig.tight_layout()
    return fig


def plot_ap_vs_instances(data, class_ap, split="train"):
    inst = split_instance_counts(data)[split]
    order = sorted(range(data.num_classes), key=lambda c: inst[c], reverse=True)
    classes = [data.classes[c] for c in order]
    counts = [inst[c] for c in order]
    aps = [class_ap.get(data.classes[c], 0.0) for c in order]

    fig = make_subplots(
        rows=1, cols=2, horizontal_spacing=0.09,
        subplot_titles=(f"{split} instances per class (frequency)",
                        "val AP@0.5 per class"),
    )
    fig.add_trace(
        go.Bar(x=classes, y=counts, marker_color="#4C78A8",
               text=counts, textposition="outside", showlegend=False),
        row=1, col=1)
    fig.add_trace(
        go.Bar(x=classes, y=aps, marker_color="#E45756",
               text=[f"{a:.2f}" for a in aps], textposition="outside",
               showlegend=False),
        row=1, col=2)
    fig.update_xaxes(tickangle=-40)
    fig.update_yaxes(title_text="instances", row=1, col=1)
    fig.update_yaxes(title_text="AP@0.5", range=[0, 1.05], row=1, col=2)
    fig.update_layout(
        title_text="Does class imbalance hurt? — train frequency vs val AP "
                    "(classes sorted most→least frequent)",
        height=470, width=1150, bargap=0.25,
        template="plotly_white", margin=dict(t=90, b=120),
    )
    return fig


def plot_training_curves(csv_path):
    df = pd.read_csv(csv_path)
    df.columns = df.columns.str.strip()
    x = df["epoch"]

    fig = make_subplots(
        rows=2, cols=3,
        subplot_titles=("Box loss", "Cls loss", "DFL loss",
                        "Precision / Recall", "mAP", "Learning rate"),
        horizontal_spacing=0.07, vertical_spacing=0.14,
    )

    def line(col, name, row, c, color, dash=None):
        if col in df.columns:
            fig.add_trace(
                go.Scatter(x=x, y=df[col], name=name, mode="lines",
                           line=dict(color=color, dash=dash)),
                row=row, col=c)

    tr, va = "#4C78A8", "#F58518"
    line("train/box_loss", "box train", 1, 1, tr)
    line("val/box_loss", "box val", 1, 1, va, dash="dash")
    line("train/cls_loss", "cls train", 1, 2, tr)
    line("val/cls_loss", "cls val", 1, 2, va, dash="dash")
    line("train/dfl_loss", "dfl train", 1, 3, tr)
    line("val/dfl_loss", "dfl val", 1, 3, va, dash="dash")
    line("metrics/precision(B)", "precision", 2, 1, "#54A24B")
    line("metrics/recall(B)", "recall", 2, 1, "#E45756")
    line("metrics/mAP50(B)", "mAP50", 2, 2, "#4C78A8")
    line("metrics/mAP50-95(B)", "mAP50-95", 2, 2, "#B279A2")
    line("lr/pg0", "lr", 2, 3, "#79706E")

    fig.update_xaxes(title_text="epoch")
    fig.update_layout(
        title_text="YOLO training curves",
        height=700, width=1150, template="plotly_white",
        margin=dict(t=80), legend=dict(orientation="h", y=-0.08),
    )
    return fig
