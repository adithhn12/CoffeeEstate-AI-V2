"""
Faster R-CNN Evaluation Script — Croppie Coffee Dataset
========================================================
Evaluates a trained Faster R-CNN checkpoint against the validation split and
produces a full metrics report:
  • mAP@50, mAP@50:95 (via torchmetrics)
  • Per-class: Precision, Recall, F1-score (at IoU=0.50, conf=0.25)
  • Summary table printed to stdout and saved to ml/logs/frcnn_eval_report.json

Usage (from project root):
    python ml/evaluate_frcnn.py
    python ml/evaluate_frcnn.py --weights ml/weights/frcnn_best.pth --conf 0.25
"""

import sys
import json
import time
import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset_frcnn import build_datasets, collate_fn, NUM_CLASSES, CLASS_NAMES
from ml.model_frcnn import build_fasterrcnn, load_checkpoint
from ml.train_frcnn import validate, compute_prf1

# Color definitions matching vision_service
PALETTE = {
    1: "#78350F",  # dark brown
    2: "#10B981",  # green
    3: "#06B6D4",  # low visibility
    4: "#EF4444",  # red
    5: "#F59E0B",  # yellow
}


def generate_training_curves(history: list, output_path: Path):
    """Plot Loss, mAP@50, and mAP@50:95 across training epochs."""
    if not history:
        return
    epochs = [h["epoch"] for h in history]
    losses = [h["loss"] for h in history]
    map50s = [h["map50"] for h in history]
    map95s = [h["map95"] for h in history]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
    ax1.plot(epochs, losses, "o-", color="#DC2626", linewidth=2, label="Training Loss")
    ax1.set_xlabel("Epoch", fontsize=12)
    ax1.set_ylabel("Loss", fontsize=12)
    ax1.set_title("Training Loss vs Epoch", fontsize=14, fontweight="bold")
    ax1.grid(True, linestyle="--", alpha=0.6)
    ax1.legend(fontsize=11)

    ax2.plot(epochs, map50s, "s-", color="#10B981", linewidth=2, label="mAP@50")
    ax2.plot(epochs, map95s, "^-", color="#3B82F6", linewidth=2, label="mAP@50:95")
    ax2.set_xlabel("Epoch", fontsize=12)
    ax2.set_ylabel("mAP", fontsize=12)
    ax2.set_title("Validation mAP vs Epoch", fontsize=14, fontweight="bold")
    ax2.grid(True, linestyle="--", alpha=0.6)
    ax2.legend(fontsize=11)

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"[Eval] Training curves saved to: {output_path}")


def generate_prediction_plot(model, val_ds, device, output_path: Path, num_samples: int = 4, conf_thresh: float = 0.25):
    """Generate visual prediction grid on validation samples."""
    model.eval()
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    axes = axes.flatten()

    indices = [int(len(val_ds) * (k + 1) / (num_samples + 1)) for k in range(num_samples)]

    with torch.no_grad():
        for ax_idx, ds_idx in enumerate(indices):
            ax = axes[ax_idx]
            img_tensor, tgt = val_ds[ds_idx]
            c, h, w = img_tensor.shape

            # Convert tensor to displayable image
            img_np = img_tensor.permute(1, 2, 0).cpu().numpy()
            img_np = np.clip(img_np, 0, 1)
            ax.imshow(img_np)

            # Inference
            outputs = model([img_tensor.to(device)])
            pred = outputs[0]
            keep = pred["scores"] >= conf_thresh
            boxes = pred["boxes"][keep].cpu().tolist()
            scores = pred["scores"][keep].cpu().tolist()
            labels = pred["labels"][keep].cpu().tolist()

            det_count = len(boxes)
            for box, score, label in zip(boxes, scores, labels):
                x1, y1, x2, y2 = box
                color = PALETTE.get(label, "#FFFFFF")
                c_name = CLASS_NAMES.get(label - 1, f"c{label}").replace("_cherry", "")
                rect = patches.Rectangle(
                    (x1, y1), x2 - x1, y2 - y1,
                    linewidth=1.8, edgecolor=color, facecolor="none"
                )
                ax.add_patch(rect)
                ax.text(
                    x1, max(0, y1 - 4),
                    f"{c_name} {int(score*100)}%",
                    color="white", fontsize=7, fontweight="bold",
                    bbox=dict(boxstyle="square,pad=0.2", facecolor=color, alpha=0.85, edgecolor="none")
                )

            ax.set_title(f"Val Sample #{ds_idx} — {det_count} detections", fontsize=11, fontweight="bold")
            ax.axis("off")

    plt.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(str(output_path), dpi=200)
    plt.close()
    print(f"[Eval] Sample predictions saved to: {output_path}")


# ─────────────────────────────────────────────────────────────────────────────

def evaluate_frcnn(
    weights_path: str = None,
    device_str: str = "0",
    conf_thresh: float = 0.25,
    iou_thresh: float = 0.50,
    batch_size: int = 4,
    num_workers: int = 0,
    dataset_root: str = None,
    seed: int = 42,
):
    # ── Weights path ───────────────────────────────────────────────────────
    if weights_path is None:
        candidates = [
            ROOT / "ml" / "weights" / "frcnn_best.pth",
            ROOT / "ml" / "weights" / "frcnn_last.pth",
        ]
        for c in candidates:
            if c.exists():
                weights_path = str(c)
                break
    if weights_path is None or not Path(weights_path).exists():
        print(f"[Eval] ERROR: No checkpoint found. Train first or specify --weights")
        sys.exit(1)

    # ── Device ─────────────────────────────────────────────────────────────
    if device_str == "cpu" or not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{device_str}")
        print(f"[Eval] GPU: {torch.cuda.get_device_name(int(device_str))}")

    # ── Dataset (val split only) ───────────────────────────────────────────
    print("[Eval] Building validation dataset…")
    _, val_ds = build_datasets(dataset_root=dataset_root, seed=seed, augment_train=False)
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
    )
    print(f"[Eval] Validation images: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────
    model = build_fasterrcnn(num_classes=NUM_CLASSES, pretrained=False)
    load_checkpoint(model, weights_path, device)
    model.to(device)
    model.eval()

    # ── Evaluation ─────────────────────────────────────────────────────────
    print("[Eval] Running inference and calculating metrics (mAP@50, mAP@50:95, P, R, F1)...")
    t0 = time.time()
    map_result = validate(model, val_loader, device, conf_thresh=conf_thresh)
    elapsed = time.time() - t0

    prf1 = map_result.get("per_class", {})
    macro_p = map_result["macro_precision"]
    macro_r = map_result["macro_recall"]
    macro_f1 = map_result["macro_f1"]

    # ── Print report ───────────────────────────────────────────────────────
    SEP = "=" * 70
    print(f"\n{SEP}")
    print("         FASTER R-CNN — COFFEE CHERRY EVALUATION REPORT")
    print(SEP)
    print(f"  Checkpoint : {weights_path}")
    print(f"  Val images : {len(val_ds)}")
    print(f"  Conf thresh: {conf_thresh}   IoU thresh: {iou_thresh}")
    print(f"  Elapsed    : {elapsed:.1f}s")
    print(SEP)
    print(f"  mAP@50     : {map_result['map_50']:.4f}  ({map_result['map_50']*100:.1f}%)")
    print(f"  mAP@50:95  : {map_result['map']:.4f}     ({map_result['map']*100:.1f}%)")
    print(f"  mAP@75     : {map_result['map_75']:.4f}   ({map_result['map_75']*100:.1f}%)")
    print(SEP)
    print(f"  {'Cls':>3}  {'Class Name':<26}  {'Precision':>9}  {'Recall':>7}  {'F1':>7}  {'TP':>5}  {'FP':>5}  {'FN':>5}")
    print("-" * 70)
    for cls_id, m in prf1.items():
        name = CLASS_NAMES.get(cls_id - 1, f"class_{cls_id}")
        print(f"  {cls_id:>3}  {name:<26}  {m['precision']:9.4f}  {m['recall']:7.4f}  "
              f"{m['f1']:7.4f}  {m['tp']:5d}  {m['fp']:5d}  {m['fn']:5d}")
    print("-" * 70)
    print(f"  {'ALL':>3}  {'MACRO AVERAGE':<26}  {macro_p:9.4f}  {macro_r:7.4f}  {macro_f1:7.4f}")
    print(SEP)

    # ── Save JSON report ───────────────────────────────────────────────────
    report = {
        "checkpoint": str(weights_path),
        "val_images": len(val_ds),
        "conf_threshold": conf_thresh,
        "iou_threshold": iou_thresh,
        "map_50":  map_result["map_50"],
        "map_75":  map_result["map_75"],
        "map_5095": map_result["map"],
        "macro_precision": macro_p,
        "macro_recall":    macro_r,
        "macro_f1":        macro_f1,
        "per_class": {
            CLASS_NAMES.get(cls_id - 1, f"cls_{cls_id}"): m
            for cls_id, m in prf1.items()
        },
    }
    log_dir = ROOT / "ml" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    report_path = log_dir / "frcnn_eval_report.json"
    with open(report_path, "w") as fh:
        json.dump(report, fh, indent=2)
    print(f"\n  Full report saved to: {report_path}")

    # ── Generate visual prediction plot ────────────────────────────────────
    sample_plot_path = log_dir / "sample_predictions.png"
    try:
        generate_prediction_plot(model, val_ds, device, sample_plot_path, num_samples=4, conf_thresh=conf_thresh)
    except Exception as e:
        print(f"[Eval] Warning: Failed to generate sample predictions plot: {e}")

    # ── Check for training history to plot curves ──────────────────────────
    hist_path = log_dir / "frcnn_training_history.json"
    if hist_path.exists():
        try:
            with open(hist_path, "r") as fh:
                history = json.load(fh)
            generate_training_curves(history, log_dir / "training_curves.png")
        except Exception as e:
            print(f"[Eval] Warning: Failed to plot training curves: {e}")

    return report


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Faster R-CNN on coffee cherry val set")
    parser.add_argument("--weights", default=None,
                        help="Path to .pth checkpoint (auto-detects frcnn_best.pth if omitted)")
    parser.add_argument("--device",  default="0",
                        help="CUDA device index or 'cpu'")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold")
    parser.add_argument("--iou",     type=float, default=0.50,
                        help="IoU threshold for TP/FP matching")
    parser.add_argument("--batch",   type=int,   default=1)
    parser.add_argument("--workers", type=int,   default=0)
    parser.add_argument("--dataset_root", default=None)
    args = parser.parse_args()

    evaluate_frcnn(
        weights_path=args.weights,
        device_str=args.device,
        conf_thresh=args.conf,
        iou_thresh=args.iou,
        batch_size=args.batch,
        num_workers=args.workers,
        dataset_root=args.dataset_root,
    )
