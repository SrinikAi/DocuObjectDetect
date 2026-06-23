"""
Indoor Object Detection Dataset - data layer.

Single source of truth for the Zenodo "Indoor Object Detection Dataset"
(record 2654485, ~2213 images, 7 classes, PASCAL VOC XML annotations).

Responsibilities:
  1. Download + extract the dataset once (cached on disk).
  2. Parse every VOC XML into a flat in-memory index (one record per image).
  3. Split 80/10/10 with multi-label iterative stratification + a repair pass
     that GUARANTEES every class appears in train/val/test. The split is
     persisted to disk (keyed by seed+ratios) so it is reproducible and instant
     to reload.
  4. Expose each split as a pure torch Dataset returning (image, target) in
     torchvision detection format, with decoded pixels RAM-cached as uint8.

Usage (Colab):
    !pip -q install scikit-multilearn torch torchvision
    from indoor_od_data import IndoorODData
    data = IndoorODData(root="/content/indoor_od")
    data.summary()                       # per-class counts per split
    train_ds = data.train               # torch Dataset
    val_ds   = data.val
    test_ds  = data.test
    from torch.utils.data import DataLoader
    dl = DataLoader(train_ds, batch_size=4, shuffle=True,
                    collate_fn=IndoorODData.collate_fn)
"""

from __future__ import annotations

import json
import os
import xml.etree.ElementTree as ET
import zipfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

ZENODO_URL = (
    "https://zenodo.org/records/2654485/files/"
    "Indoor%20Object%20Detection%20Dataset.zip?download=1"
)
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".JPG", ".JPEG", ".PNG"}


@dataclass
class Sample:
    """One image and its annotations. Labels are internal 0-indexed class ids."""
    image_path: str
    width: int
    height: int
    boxes: np.ndarray   # (N, 4) float32, xyxy in pixels
    labels: np.ndarray  # (N,)  int64, internal class ids [0 .. num_classes-1]


class IndoorODData:
    def __init__(
        self,
        root: str = "indoor_od",
        ratios: tuple[float, float, float] = (0.8, 0.1, 0.1),
        seed: int = 42,
        cache_images: str = "ram",   # "ram" | "none"
        label_offset: int = 1,        # torchvision reserves 0 for background
        download: bool = True,
    ):
        assert abs(sum(ratios) - 1.0) < 1e-6, "ratios must sum to 1"
        self.root = Path(root)
        self.ratios = tuple(ratios)
        self.seed = seed
        self.cache_images = cache_images
        self.label_offset = label_offset

        self.root.mkdir(parents=True, exist_ok=True)
        self.data_dir = self.root / "dataset"

        if download:
            self._download_and_extract()

        # ----- parse once -> index -----
        self.classes: list[str] = []
        self.class_to_idx: dict[str, int] = {}
        self.samples: list[Sample] = self._parse_voc()
        if not self.samples:
            raise RuntimeError(f"No annotations parsed under {self.data_dir}")

        # ----- split once (persisted) -----
        self.split_idx: dict[str, list[int]] = self._make_or_load_split()

        # ----- shared RAM image cache (uint8 HWC), keyed by global sample index -----
        self._img_cache: dict[int, np.ndarray] = {}

        # ----- build the three Dataset views -----
        self.train = _SplitDataset(self, self.split_idx["train"])
        self.val = _SplitDataset(self, self.split_idx["val"])
        self.test = _SplitDataset(self, self.split_idx["test"])

    # ------------------------------------------------------------------ download
    def _download_and_extract(self):
        marker = self.data_dir / ".extracted"
        if marker.exists():
            return
        zip_path = self.root / "indoor_od.zip"
        if not zip_path.exists():
            print("Downloading dataset (~411 MB) from Zenodo ...")
            self._http_download(ZENODO_URL, zip_path)
        print("Extracting ...")
        self.data_dir.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(zip_path) as zf:
            zf.extractall(self.data_dir)
        marker.write_text("ok")
        print("Extracted to", self.data_dir)

    @staticmethod
    def _http_download(url: str, dst: Path):
        import urllib.request
        req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
        with urllib.request.urlopen(req) as r, open(dst, "wb") as f:
            total = int(r.headers.get("Content-Length", 0))
            done = 0
            chunk = 1 << 20
            while True:
                buf = r.read(chunk)
                if not buf:
                    break
                f.write(buf)
                done += len(buf)
                if total:
                    pct = 100 * done / total
                    print(f"\r  {done/1e6:7.1f} / {total/1e6:7.1f} MB ({pct:4.1f}%)",
                          end="", flush=True)
        print()

    # ------------------------------------------------------------------- parsing
    def _index_images(self) -> dict[str, str]:
        """Map image-stem -> absolute path for every image under data_dir."""
        stem_to_path: dict[str, str] = {}
        for p in self.data_dir.rglob("*"):
            if p.suffix in IMG_EXTS and p.is_file():
                stem_to_path.setdefault(p.stem, str(p))
        return stem_to_path

    def _parse_voc(self) -> list[Sample]:
        stem_to_path = self._index_images()
        xml_files = [p for p in self.data_dir.rglob("*.xml")]
        raw: list[tuple] = []   # (img_path, w, h, boxes_list, names_list)
        class_set: set[str] = set()

        for xml in sorted(xml_files):
            try:
                tree = ET.parse(xml)
            except ET.ParseError:
                continue
            r = tree.getroot()

            # resolve image: prefer <filename>/<path>, fall back to xml stem
            fname = r.findtext("filename") or ""
            stem = Path(fname).stem or xml.stem
            img_path = stem_to_path.get(stem) or stem_to_path.get(xml.stem)
            if img_path is None:
                continue

            size = r.find("size")
            w = int(float(size.findtext("width"))) if size is not None else 0
            h = int(float(size.findtext("height"))) if size is not None else 0
            if w <= 0 or h <= 0:
                with Image.open(img_path) as im:
                    w, h = im.size

            boxes, names = [], []
            for obj in r.findall("object"):
                name = (obj.findtext("name") or "").strip()
                bb = obj.find("bndbox")
                if not name or bb is None:
                    continue
                x1 = float(bb.findtext("xmin")); y1 = float(bb.findtext("ymin"))
                x2 = float(bb.findtext("xmax")); y2 = float(bb.findtext("ymax"))
                x1, x2 = sorted((x1, x2)); y1, y2 = sorted((y1, y2))
                x1 = max(0, min(x1, w)); x2 = max(0, min(x2, w))
                y1 = max(0, min(y1, h)); y2 = max(0, min(y2, h))
                if x2 - x1 < 1 or y2 - y1 < 1:
                    continue
                boxes.append([x1, y1, x2, y2]); names.append(name)
                class_set.add(name)
            if boxes:
                raw.append((img_path, w, h, boxes, names))

        # stable, sorted class vocabulary derived from the data
        self.classes = sorted(class_set)
        self.class_to_idx = {c: i for i, c in enumerate(self.classes)}
        print(f"Parsed {len(raw)} annotated images, "
              f"{len(self.classes)} classes: {self.classes}")

        samples = []
        for img_path, w, h, boxes, names in raw:
            samples.append(Sample(
                image_path=img_path, width=w, height=h,
                boxes=np.asarray(boxes, dtype=np.float32),
                labels=np.asarray([self.class_to_idx[n] for n in names],
                                  dtype=np.int64),
            ))
        return samples

    # --------------------------------------------------------------------- split
    @property
    def num_classes(self) -> int:
        return len(self.classes)

    def _label_matrix(self) -> np.ndarray:
        """Binary (n_samples, n_classes) presence matrix for stratification."""
        Y = np.zeros((len(self.samples), self.num_classes), dtype=np.int8)
        for i, s in enumerate(self.samples):
            Y[i, np.unique(s.labels)] = 1
        return Y

    def _split_cache_path(self) -> Path:
        tag = f"split_seed{self.seed}_{int(self.ratios[0]*100)}-" \
              f"{int(self.ratios[1]*100)}-{int(self.ratios[2]*100)}.json"
        return self.root / tag

    def _make_or_load_split(self) -> dict[str, list[int]]:
        cache = self._split_cache_path()
        if cache.exists():
            split = json.loads(cache.read_text())
            # guard against a stale cache from a different parse
            if sum(len(v) for v in split.values()) == len(self.samples):
                print(f"Loaded split from {cache.name}")
                return split

        split = self._stratified_split()
        cache.write_text(json.dumps(split))
        print(f"Saved split to {cache.name}")
        return split

    def _stratified_split(self) -> dict[str, list[int]]:
        Y = self._label_matrix()
        n = Y.shape[0]
        idx = np.arange(n)

        try:
            from skmultilearn.model_selection import IterativeStratification
            # 80 / 20
            strat1 = IterativeStratification(
                n_splits=2, order=2,
                sample_distribution_per_fold=[self.ratios[0], 1 - self.ratios[0]],
                random_state=self.seed)
            train_i, rest_i = next(strat1.split(idx.reshape(-1, 1), Y))
            # split the 20% into val/test (half/half by default)
            rest = idx[rest_i]
            val_frac = self.ratios[1] / (self.ratios[1] + self.ratios[2])
            strat2 = IterativeStratification(
                n_splits=2, order=2,
                sample_distribution_per_fold=[val_frac, 1 - val_frac],
                random_state=self.seed)
            v_i, t_i = next(strat2.split(rest.reshape(-1, 1), Y[rest]))
            split = {
                "train": idx[train_i].tolist(),
                "val": rest[v_i].tolist(),
                "test": rest[t_i].tolist(),
            }
            print("Split via iterative stratification.")
        except ImportError:
            print("scikit-multilearn not found -> greedy rarest-first split.")
            split = self._greedy_split(Y)

        self._repair_split(split, Y)
        return split

    def _greedy_split(self, Y: np.ndarray) -> dict[str, list[int]]:
        rng = np.random.default_rng(self.seed)
        assigned = -np.ones(Y.shape[0], dtype=int)  # 0=train,1=val,2=test
        order = np.argsort(Y.sum(axis=0))            # rarest class first
        for c in order:
            members = np.where((Y[:, c] == 1) & (assigned < 0))[0]
            rng.shuffle(members)
            n = len(members)
            n_tr = int(round(n * self.ratios[0]))
            n_va = int(round(n * self.ratios[1]))
            assigned[members[:n_tr]] = 0
            assigned[members[n_tr:n_tr + n_va]] = 1
            assigned[members[n_tr + n_va:]] = 2
        leftover = np.where(assigned < 0)[0]
        for i in leftover:
            assigned[i] = rng.choice(3, p=self.ratios)
        return {
            "train": np.where(assigned == 0)[0].tolist(),
            "val": np.where(assigned == 1)[0].tolist(),
            "test": np.where(assigned == 2)[0].tolist(),
        }

    def _repair_split(self, split: dict[str, list[int]], Y: np.ndarray):
        """Guarantee every class is present in val and test (pull from train)."""
        sets = {k: set(v) for k, v in split.items()}
        for target in ("val", "test"):
            for c in range(self.num_classes):
                present = any(Y[i, c] for i in sets[target])
                if present:
                    continue
                donors = [i for i in sets["train"] if Y[i, c] == 1]
                if not donors:  # fall back to the other small split
                    other = "test" if target == "val" else "val"
                    donors = [i for i in sets[other] if Y[i, c] == 1]
                    src = other
                else:
                    src = "train"
                if donors:
                    mv = donors[0]
                    sets[src].discard(mv)
                    sets[target].add(mv)
                    print(f"  repair: moved img {mv} -> {target} to cover "
                          f"'{self.classes[c]}'")
        for k in split:
            split[k] = sorted(sets[k])

    # ----------------------------------------------------------------- utilities
    def get_image(self, gidx: int) -> np.ndarray:
        """Decoded uint8 HWC image for global sample index, RAM-cached."""
        if self.cache_images == "ram" and gidx in self._img_cache:
            return self._img_cache[gidx]
        with Image.open(self.samples[gidx].image_path) as im:
            arr = np.asarray(im.convert("RGB"), dtype=np.uint8)
        if self.cache_images == "ram":
            self._img_cache[gidx] = arr
        return arr

    def summary(self):
        Y = self._label_matrix()
        header = f"{'class':<18}" + "".join(f"{s:>8}" for s in
                                            ("train", "val", "test", "imgs", "inst"))
        print(header); print("-" * len(header))
        for c in range(self.num_classes):
            row = {}
            for s in ("train", "val", "test"):
                row[s] = sum(int(Y[i, c]) for i in self.split_idx[s])
            inst = sum(int((self.samples[i].labels == c).sum())
                       for i in range(len(self.samples)))
            imgs = int(Y[:, c].sum())
            print(f"{self.classes[c]:<18}"
                  f"{row['train']:>8}{row['val']:>8}{row['test']:>8}"
                  f"{imgs:>8}{inst:>8}")
        print("-" * len(header))
        print(f"{'TOTAL images':<18}"
              f"{len(self.split_idx['train']):>8}"
              f"{len(self.split_idx['val']):>8}"
              f"{len(self.split_idx['test']):>8}"
              f"{len(self.samples):>8}")

    @staticmethod
    def collate_fn(batch):
        return tuple(zip(*batch))


class _SplitDataset(Dataset):
    """A single split as a torchvision-style detection Dataset."""

    def __init__(self, parent: IndoorODData, indices: list[int],
                 transforms: Optional[Callable] = None):
        self.parent = parent
        self.indices = list(indices)
        self.transforms = transforms

    def __len__(self):
        return len(self.indices)

    def set_transforms(self, t: Callable):
        self.transforms = t
        return self

    def __getitem__(self, i):
        gidx = self.indices[i]
        s = self.parent.samples[gidx]
        arr = self.parent.get_image(gidx)                       # uint8 HWC
        img = torch.from_numpy(arr).permute(2, 0, 1).float() / 255.0
        target = {
            "boxes": torch.as_tensor(s.boxes, dtype=torch.float32),
            "labels": torch.as_tensor(s.labels + self.parent.label_offset,
                                      dtype=torch.int64),
            "image_id": torch.tensor([gidx]),
            "area": torch.as_tensor(
                (s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1]),
                dtype=torch.float32),
            "iscrowd": torch.zeros((len(s.labels),), dtype=torch.int64),
        }
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target
