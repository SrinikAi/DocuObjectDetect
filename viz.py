from __future__ import annotations

import numpy as np


def split_image_counts(data) -> dict[str, list[int]]:
    counts = {}
    for split, idxs in data.split_idx.items():
        per_class = [0] * data.num_classes
        for i in idxs:
            for c in np.unique(data.samples[i].labels):
                per_class[int(c)] += 1
        counts[split] = per_class
    return counts


def plot_split_distributions(data, order=("train", "val", "test")):
    from plotly.subplots import make_subplots
    import plotly.graph_objects as go

    counts = split_image_counts(data)
    colors = {"train": "#4C78A8", "val": "#F58518", "test": "#54A24B"}

    fig = make_subplots(
        rows=1, cols=len(order),
        subplot_titles=[f"{s.upper()}  (n={len(data.split_idx[s])} imgs)"
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

    fig.update_yaxes(title_text="number of images", row=1, col=1)
    fig.update_layout(
        title_text="Indoor Object Detection — per-class image counts by split "
                    "(80 / 10 / 10)",
        height=480, width=1150, bargap=0.25,
        template="plotly_white", margin=dict(t=90, b=120),
    )
    return fig


def plot_class_examples(data, split="train", cols=4, seed=0, figsize=None):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

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
