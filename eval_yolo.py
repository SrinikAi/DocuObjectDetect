from __future__ import annotations

import numpy as np


def _iou(box, boxes):
    if len(boxes) == 0:
        return np.zeros((0,), dtype=np.float32)
    x1 = np.maximum(box[0], boxes[:, 0])
    y1 = np.maximum(box[1], boxes[:, 1])
    x2 = np.minimum(box[2], boxes[:, 2])
    y2 = np.minimum(box[3], boxes[:, 3])
    iw = np.clip(x2 - x1, 0, None)
    ih = np.clip(y2 - y1, 0, None)
    inter = iw * ih
    area_b = (box[2] - box[0]) * (box[3] - box[1])
    area_s = (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])
    union = area_b + area_s - inter
    return np.where(union > 0, inter / union, 0.0)


def image_mean_iou(gt_boxes, gt_labels, pr_boxes, pr_labels):
    if len(gt_boxes) == 0:
        return 1.0 if len(pr_boxes) == 0 else 0.0
    ious = []
    for gb, gl in zip(gt_boxes, gt_labels):
        same = pr_boxes[pr_labels == gl] if len(pr_boxes) else pr_boxes
        ious.append(float(_iou(gb, same).max()) if len(same) else 0.0)
    return float(np.mean(ious))


def predict_image(model, img_path, conf=0.25):
    r = model(img_path, conf=conf, verbose=False)[0]
    b = r.boxes
    if b is None or len(b) == 0:
        return (np.zeros((0, 4), np.float32),
                np.zeros((0,), int), np.zeros((0,), np.float32))
    return (b.xyxy.cpu().numpy(),
            b.cls.cpu().numpy().astype(int),
            b.conf.cpu().numpy())


def evaluate(weights, data_yaml, split="val"):
    from ultralytics import YOLO
    model = YOLO(weights)
    m = model.val(data=data_yaml, split=split, verbose=False)
    names = getattr(m, "names", None) or model.names

    print(f"\nOverall  mAP50={m.box.map50:.4f}  mAP50-95={m.box.map:.4f}  "
          f"P={m.box.mp:.4f}  R={m.box.mr:.4f}")
    print(f"{'class':<18}{'AP50':>8}{'AP50-95':>10}")
    print("-" * 36)
    class_ap50 = {}
    for i, ci in enumerate(m.box.ap_class_index):
        name = names[int(ci)]
        class_ap50[name] = float(m.box.ap50[i])
        print(f"{name:<18}{m.box.ap50[i]:>8.4f}{m.box.ap[i]:>10.4f}")
    print("-" * 36)
    return model, m, class_ap50


def rank_by_iou(data, model, split="val", conf=0.25):
    out = []
    for gidx in data.split_idx[split]:
        s = data.samples[gidx]
        pb, pl, _ = predict_image(model, s.image_path, conf)
        out.append((gidx, image_mean_iou(s.boxes, s.labels, pb, pl)))
    out.sort(key=lambda t: t[1])
    return out


def plot_predictions(data, model, gidxs, conf=0.25, cols=4, title=""):
    import matplotlib.pyplot as plt
    import matplotlib.patches as patches

    n = len(gidxs)
    cols = min(cols, n) if n else 1
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4, rows * 3.7),
                             squeeze=False)
    axes = axes.ravel()
    for ax in axes:
        ax.axis("off")

    gt_c, pr_c = "#2CA02C", "#D62728"
    for ax, gidx in zip(axes, gidxs):
        s = data.samples[gidx]
        ax.imshow(data.get_image(gidx))
        for (x1, y1, x2, y2) in s.boxes:
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=gt_c, linewidth=2))
        pb, pl, pc = predict_image(model, s.image_path, conf)
        for (x1, y1, x2, y2), lab, cf in zip(pb, pl, pc):
            ax.add_patch(patches.Rectangle(
                (x1, y1), x2 - x1, y2 - y1,
                fill=False, edgecolor=pr_c, linewidth=2, linestyle="--"))
            ax.text(x1, max(0, y1 - 3), f"{data.classes[int(lab)]} {cf:.2f}",
                    fontsize=7, color="white",
                    bbox=dict(facecolor=pr_c, edgecolor="none", pad=1))
        miou = image_mean_iou(s.boxes, s.labels, pb, pl)
        ax.set_title(f"img {gidx}  mIoU={miou:.2f}", fontsize=10)

    handles = [patches.Patch(color=gt_c, label="ground truth"),
               patches.Patch(color=pr_c, label="prediction")]
    fig.legend(handles=handles, loc="upper right", ncol=2)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout()
    return fig


def plot_good_bad(data, model, split="val", k=4, conf=0.25):
    ranked = rank_by_iou(data, model, split, conf)
    worst = [g for g, _ in ranked[:k]]
    best = [g for g, _ in ranked[-k:]][::-1]
    fig_bad = plot_predictions(
        data, model, worst, conf,
        title=f"Worst {k} {split} images (lowest GT-vs-pred IoU)")
    fig_good = plot_predictions(
        data, model, best, conf,
        title=f"Best {k} {split} images (highest GT-vs-pred IoU)")
    return fig_good, fig_bad
