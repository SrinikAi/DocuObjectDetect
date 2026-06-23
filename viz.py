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
