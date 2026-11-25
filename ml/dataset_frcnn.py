"""
Faster R-CNN Dataset — Croppie Coffee (YOLO-format → Faster R-CNN format)
===========================================================================
Reads YOLO-format annotations (class cx cy w h, all normalized) from the
extracted dataset and converts them to Faster R-CNN targets:
  boxes  : FloatTensor[N, 4]  — absolute pixel coords [x1, y1, x2, y2]
  labels : Int64Tensor[N]     — 1-indexed class ids  (0 = background)

The combined train+val pool (5836 + 2497 = 8333 images) is re-split 80:20
with a fixed random seed so every run uses the same partition.
"""

import os
import random
from pathlib import Path
from typing import List, Tuple, Dict, Any, Optional

import torch
from torch.utils.data import Dataset
from PIL import Image
import torchvision.transforms.functional as F


# ──────────────────────────────────────────────────────────────────────────────
# Class mapping (0-indexed YOLO → 1-indexed Faster R-CNN; 0 reserved for bg)
# ──────────────────────────────────────────────────────────────────────────────
CLASS_NAMES = {
    0: "dark_brown_cherry",
    1: "green_cherry",
    2: "low_visibility_unsure",
    3: "red_cherry",
    4: "yellow_cherry",
}
NUM_CLASSES = len(CLASS_NAMES) + 1  # 6 (5 classes + background)

LABEL_MAP_FRCNN = {yolo_id: yolo_id + 1 for yolo_id in CLASS_NAMES}  # 0→1 … 4→5


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────

def _yolo_to_xyxy(cx: float, cy: float, w: float, h: float,
                  img_w: int, img_h: int) -> List[float]:
    """Convert YOLO normalised [cx, cy, w, h] → absolute pixel [x1, y1, x2, y2]."""
    x1 = (cx - w / 2) * img_w
    y1 = (cy - h / 2) * img_h
    x2 = (cx + w / 2) * img_w
    y2 = (cy + h / 2) * img_h
    # Clamp to image bounds
    x1, y1 = max(0.0, x1), max(0.0, y1)
    x2, y2 = min(float(img_w), x2), min(float(img_h), y2)
    return [x1, y1, x2, y2]


def _parse_label_file(label_path: Path, img_w: int, img_h: int
                      ) -> Tuple[List[List[float]], List[int]]:
    """
    Parse a YOLO label file and return (boxes_xyxy, frcnn_labels).
    Skips degenerate boxes (w=0 or h=0 after conversion).
    """
    boxes, labels = [], []
    if not label_path.exists():
        return boxes, labels

    with open(label_path, "r") as fh:
        for line in fh:
            parts = line.strip().split()
            if len(parts) < 5:
                continue
            cls_id = int(parts[0])
            cx, cy, bw, bh = float(parts[1]), float(parts[2]), float(parts[3]), float(parts[4])
            box = _yolo_to_xyxy(cx, cy, bw, bh, img_w, img_h)
            # Skip degenerate boxes
            if (box[2] - box[0]) < 1.0 or (box[3] - box[1]) < 1.0:
                continue
            boxes.append(box)
            labels.append(LABEL_MAP_FRCNN.get(cls_id, cls_id + 1))
    return boxes, labels


# ──────────────────────────────────────────────────────────────────────────────
# Dataset class
# ──────────────────────────────────────────────────────────────────────────────

class CoffeeCherryFRCNNDataset(Dataset):
    """
    Parameters
    ----------
    image_paths : list of Path  — absolute paths to .jpg / .png images
    label_paths : list of Path  — corresponding YOLO .txt label files
    transforms  : optional callable applied to (image, target) pair;
                  defaults to ToTensor() if None
    augment     : bool — apply training-time augmentations
    """

    def __init__(
        self,
        image_paths: List[Path],
        label_paths: List[Path],
        transforms=None,
        augment: bool = False,
    ):
        assert len(image_paths) == len(label_paths), "image/label count mismatch"
        self.image_paths = image_paths
        self.label_paths = label_paths
        self.transforms = transforms
        self.augment = augment

    def __len__(self) -> int:
        return len(self.image_paths)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, Dict[str, Any]]:
        img_path = self.image_paths[idx]
        lbl_path = self.label_paths[idx]

        img = Image.open(img_path).convert("RGB")
        img_w, img_h = img.size

        boxes_list, labels_list = _parse_label_file(lbl_path, img_w, img_h)

        if boxes_list:
            boxes = torch.as_tensor(boxes_list, dtype=torch.float32)
            labels = torch.as_tensor(labels_list, dtype=torch.int64)
        else:
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)

        # Optional horizontal flip augmentation
        if self.augment and random.random() > 0.5:
            img = F.hflip(img)
            if boxes.numel() > 0:
                # flip x coords: x1_new = W - x2_old, x2_new = W - x1_old
                boxes[:, [0, 2]] = img_w - boxes[:, [2, 0]]

        image_id = torch.tensor([idx])
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        iscrowd = torch.zeros((len(labels),), dtype=torch.int64)

        target = {
            "boxes": boxes,
            "labels": labels,
            "image_id": image_id,
            "area": area,
            "iscrowd": iscrowd,
        }

        if self.transforms is not None:
            img, target = self.transforms(img, target)
        else:
            img = F.to_tensor(img)

        return img, target


# ──────────────────────────────────────────────────────────────────────────────
# Split builder
# ──────────────────────────────────────────────────────────────────────────────

def build_datasets(
    dataset_root: str = None,
    train_ratio: float = 0.80,
    seed: int = 42,
    augment_train: bool = True,
) -> Tuple[CoffeeCherryFRCNNDataset, CoffeeCherryFRCNNDataset]:
    """
    Collect ALL images from extracted/train and extracted/val, then re-split
    80:20 with a fixed random seed for reproducibility.

    Returns (train_dataset, val_dataset).
    """
    if dataset_root is None:
        base = Path(__file__).resolve().parent.parent
        dataset_root = base / "datasets" / "croppie_coffee" / "extracted"
    else:
        dataset_root = Path(dataset_root)

    all_image_paths: List[Path] = []
    all_label_paths: List[Path] = []

    for split in ("train", "val"):
        img_dir = dataset_root / split / "images"
        lbl_dir = dataset_root / split / "labels"
        if not img_dir.exists():
            print(f"[Dataset] Warning: {img_dir} does not exist — skipping.")
            continue
        for img_file in sorted(img_dir.iterdir()):
            if img_file.suffix.lower() not in (".jpg", ".jpeg", ".png", ".bmp"):
                continue
            lbl_file = lbl_dir / (img_file.stem + ".txt")
            all_image_paths.append(img_file)
            all_label_paths.append(lbl_file)

    total = len(all_image_paths)
    print(f"[Dataset] Total images found: {total}")

    # Shuffle with fixed seed
    indices = list(range(total))
    rng = random.Random(seed)
    rng.shuffle(indices)

    split_idx = int(total * train_ratio)
    train_idx = indices[:split_idx]
    val_idx = indices[split_idx:]

    print(f"[Dataset] 80/20 split — Train: {len(train_idx)}, Val: {len(val_idx)}")

    train_imgs = [all_image_paths[i] for i in train_idx]
    train_lbls = [all_label_paths[i] for i in train_idx]
    val_imgs   = [all_image_paths[i] for i in val_idx]
    val_lbls   = [all_label_paths[i] for i in val_idx]

    train_ds = CoffeeCherryFRCNNDataset(train_imgs, train_lbls, augment=augment_train)
    val_ds   = CoffeeCherryFRCNNDataset(val_imgs,   val_lbls,   augment=False)

    return train_ds, val_ds


# ──────────────────────────────────────────────────────────────────────────────
# Collate function (required because targets are variable-length dicts)
# ──────────────────────────────────────────────────────────────────────────────

def collate_fn(batch):
    return tuple(zip(*batch))


if __name__ == "__main__":
    train_ds, val_ds = build_datasets()
    img, tgt = train_ds[0]
    print(f"Sample image tensor: {img.shape}")
    print(f"Sample target boxes: {tgt['boxes'][:3]}")
    print(f"Sample target labels: {tgt['labels'][:3]}")
