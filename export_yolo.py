from __future__ import annotations

import os
import shutil
from pathlib import Path


def export_yolo(data, out_dir, link=True, splits=("train", "val", "test")):
    out = Path(out_dir)
    for split in splits:
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)

    n_imgs = 0
    for split in splits:
        for gidx in data.split_idx[split]:
            s = data.samples[gidx]
            src = Path(s.image_path)
            dst_img = out / "images" / split / src.name
            _place(src, dst_img, link)

            w, h = float(s.width), float(s.height)
            lines = []
            for (x1, y1, x2, y2), lab in zip(s.boxes, s.labels):
                xc = ((x1 + x2) / 2.0) / w
                yc = ((y1 + y2) / 2.0) / h
                bw = (x2 - x1) / w
                bh = (y2 - y1) / h
                lines.append(f"{int(lab)} {xc:.6f} {yc:.6f} {bw:.6f} {bh:.6f}")
            (out / "labels" / split / f"{src.stem}.txt").write_text("\n".join(lines))
            n_imgs += 1

    yaml_path = out / "data.yaml"
    names = "\n".join(f"  {i}: {c}" for i, c in enumerate(data.classes))
    yaml_path.write_text(
        f"path: {out.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"test: images/test\n"
        f"names:\n{names}\n"
    )
    print(f"Exported {n_imgs} images / {len(data.classes)} classes to {out}")
    print(f"data.yaml -> {yaml_path}")
    return str(yaml_path)


def _place(src: Path, dst: Path, link: bool):
    if dst.exists() or dst.is_symlink():
        return
    if link:
        try:
            os.symlink(src.resolve(), dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)
