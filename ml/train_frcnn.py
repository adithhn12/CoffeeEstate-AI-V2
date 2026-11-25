"""
Faster R-CNN Training Script — Croppie Coffee Dataset
======================================================
Implements:
  • Transfer learning from COCO pretrained ResNet-50-FPN weights
  • 80/20 dataset split with fixed seed (seed=42)
  • SGD + cosine LR scheduling optimised for RTX 4060 8 GB
  • Per-epoch validation with mAP@50 and mAP@50:95 via torchmetrics
  • Checkpoint saving (every epoch) + best-model selection based on mAP@50
  • Detailed epoch logs written to ml/logs/

Usage (from project root):
    python ml/train_frcnn.py
    python ml/train_frcnn.py --epochs 30 --batch 2 --device 0

Requirements: torch, torchvision, torchmetrics
  pip install torchmetrics
"""

import os
import sys
import time
import json
import shutil
import argparse
import math
from pathlib import Path

import torch
import torch.optim as optim
from torch.utils.data import DataLoader

# ── allow imports from project root ──────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ml.dataset_frcnn import build_datasets, collate_fn, NUM_CLASSES
from ml.dataset_frcnn import build_datasets, collate_fn, NUM_CLASSES, CLASS_NAMES
from ml.model_frcnn import build_fasterrcnn, load_checkpoint
from ml.metrics import evaluate_detections


# ─────────────────────────────────────────────────────────────────────────────
# Validation loop
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def validate(model, data_loader, device, conf_thresh: float = 0.25, max_samples: int = None):
    """
    Run inference on validation set and compute full detection metrics (mAP@50, mAP@50:95, P, R, F1).
    """
    model.eval()

    all_preds = []
    all_targets = []
    sample_count = 0

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, tgt in zip(outputs, targets):
            all_preds.append({
                "boxes": pred["boxes"].cpu(),
                "scores": pred["scores"].cpu(),
                "labels": pred["labels"].cpu(),
            })
            all_targets.append({
                "boxes": tgt["boxes"].cpu(),
                "labels": tgt["labels"].cpu(),
            })
            sample_count += 1

        if max_samples is not None and sample_count >= max_samples:
            break

    eval_result = evaluate_detections(
        all_preds, all_targets, num_classes=NUM_CLASSES, conf_thresh=conf_thresh
    )
    return eval_result


# ─────────────────────────────────────────────────────────────────────────────
# Per-class P/R/F1 (simple IoU=0.5 TP/FP/FN tally, same as COCO 11-pt)
# ─────────────────────────────────────────────────────────────────────────────

def _iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    aB = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


@torch.no_grad()
def compute_prf1(model, data_loader, device, conf_thresh: float = 0.25,
                 iou_thresh: float = 0.50, num_classes: int = NUM_CLASSES):
    """
    Compute per-class precision, recall, F1 at a fixed IoU threshold.
    Returns dict: class_id → {precision, recall, f1, tp, fp, fn}
    """
    model.eval()
    # 1-indexed classes (0 = background, skip)
    tp_c = {c: 0 for c in range(1, num_classes)}
    fp_c = {c: 0 for c in range(1, num_classes)}
    fn_c = {c: 0 for c in range(1, num_classes)}

    for images, targets in data_loader:
        images = [img.to(device) for img in images]
        outputs = model(images)

        for pred, tgt in zip(outputs, targets):
            keep = pred["scores"] >= conf_thresh
            pred_boxes  = pred["boxes"][keep].cpu().tolist()
            pred_labels = pred["labels"][keep].cpu().tolist()
            gt_boxes    = tgt["boxes"].cpu().tolist()
            gt_labels   = tgt["labels"].cpu().tolist()

            matched_gt = set()

            for pb, pl in zip(pred_boxes, pred_labels):
                if pl not in tp_c:
                    continue
                best_iou, best_j = 0.0, -1
                for j, (gb, gl) in enumerate(zip(gt_boxes, gt_labels)):
                    if gl != pl or j in matched_gt:
                        continue
                    iou = _iou(pb, gb)
                    if iou > best_iou:
                        best_iou, best_j = iou, j
                if best_iou >= iou_thresh:
                    tp_c[pl] += 1
                    matched_gt.add(best_j)
                else:
                    fp_c[pl] += 1

            for j, gl in enumerate(gt_labels):
                if gl not in fn_c:
                    continue
                if j not in matched_gt:
                    fn_c[gl] += 1

    result = {}
    for c in range(1, num_classes):
        tp, fp, fn = tp_c[c], fp_c[c], fn_c[c]
        prec = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        rec  = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1   = 2 * prec * rec / (prec + rec) if (prec + rec) > 0 else 0.0
        result[c] = {"precision": prec, "recall": rec, "f1": f1,
                     "tp": tp, "fp": fp, "fn": fn}
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Training loop
# ─────────────────────────────────────────────────────────────────────────────

def train_one_epoch(model, optimizer, data_loader, device, epoch: int,
                    print_freq: int = 50, scaler=None, start_batch: int = 0,
                    checkpoint_callback=None):
    model.train()
    total_loss = 0.0
    n_batches  = 0
    t0 = time.time()

    for i, (images, targets) in enumerate(data_loader):
        if i < start_batch:
            continue

        images  = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        optimizer.zero_grad()

        if scaler is not None:
            with torch.amp.autocast('cuda'):
                loss_dict = model(images, targets)
                losses = sum(loss_dict.values())
            scaler.scale(losses).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            loss_dict = model(images, targets)
            losses = sum(loss_dict.values())
            losses.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=10.0)
            optimizer.step()

        total_loss += losses.item()
        n_batches  += 1

        if (i + 1) % print_freq == 0 or (i + 1) == len(data_loader):
            avg = total_loss / n_batches
            elapsed = time.time() - t0
            loss_detail = " | ".join(f"{k}: {v.item():.4f}" for k, v in loss_dict.items())
            gpu_mem = f"{torch.cuda.memory_allocated() / (1024**2):.0f}MB / {torch.cuda.max_memory_allocated() / (1024**2):.0f}MB" if torch.cuda.is_available() else "N/A"
            print(f"  Epoch {epoch} [{i+1}/{len(data_loader)}]  "
                  f"avg_loss={avg:.4f}  VRAM={gpu_mem}  [{loss_detail}]  "
                  f"elapsed={elapsed:.1f}s")

        # Periodic mid-epoch checkpointing for crash resilience
        if checkpoint_callback is not None and ((i + 1) % 500 == 0 or (i + 1) == len(data_loader)):
            checkpoint_callback(epoch, i + 1, avg if n_batches > 0 else 0.0)

    return total_loss / max(1, n_batches)


# ─────────────────────────────────────────────────────────────────────────────
# Main training entry-point
# ─────────────────────────────────────────────────────────────────────────────

def train_frcnn(
    dataset_root: str = None,
    epochs: int = 5,
    batch_size: int = 2,
    device_str: str = "0",
    lr: float = 0.005,
    weight_decay: float = 1e-4,
    momentum: float = 0.9,
    warmup_epochs: int = 2,
    trainable_backbone_layers: int = 3,
    conf_thresh: float = 0.25,
    resume: str = None,
    output_dir: str = None,
    num_workers: int = 0,       # 0 = safer on Windows
    seed: int = 42,
    use_amp: bool = True,       # mixed precision (RTX 4060)
):
    # ── Paths ──────────────────────────────────────────────────────────────
    if output_dir is None:
        output_dir = str(ROOT / "ml" / "runs" / "frcnn_coffee")
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    weights_dir = ROOT / "ml" / "weights"
    weights_dir.mkdir(parents=True, exist_ok=True)
    log_dir = ROOT / "ml" / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)

    # ── Device ─────────────────────────────────────────────────────────────
    if device_str == "cpu":
        device = torch.device("cpu")
        use_amp = False
    elif torch.cuda.is_available():
        device = torch.device(f"cuda:{device_str}")
        gpu_name = torch.cuda.get_device_name(int(device_str))
        vram_mb  = torch.cuda.get_device_properties(int(device_str)).total_memory / (1024**2)
        print(f"[Train] GPU: {gpu_name}  ({vram_mb:.0f} MB VRAM)")
    else:
        print("[Train] CUDA not available — using CPU (training will be slow)")
        device = torch.device("cpu")
        use_amp = False

    # ── Dataset ────────────────────────────────────────────────────────────
    print("[Train] Building datasets (80/20 split, seed=42)…")
    train_ds, val_ds = build_datasets(dataset_root=dataset_root, seed=seed, augment_train=True)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )
    val_loader = DataLoader(
        val_ds, batch_size=4, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn,
        pin_memory=(device.type == "cuda"),
    )

    print(f"[Train] Train batches: {len(train_loader)}, Val images: {len(val_ds)}")

    # ── Model ──────────────────────────────────────────────────────────────
    print("[Train] Building Faster R-CNN (ResNet-50-FPN, COCO pretrained)…")
    model = build_fasterrcnn(
        num_classes=NUM_CLASSES,
        pretrained=True,
        trainable_backbone_layers=trainable_backbone_layers,
    )
    model.to(device)

    # ── Optimiser & LR schedule ────────────────────────────────────────────
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)

    # Cosine annealing after warm-up
    def lr_lambda(epoch_idx):
        if epoch_idx < warmup_epochs:
            return float(epoch_idx + 1) / float(warmup_epochs)
        progress = (epoch_idx - warmup_epochs) / max(1, epochs - warmup_epochs)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda)
    scaler = torch.amp.GradScaler('cuda') if use_amp and device.type == 'cuda' else None

    # ── Resume ─────────────────────────────────────────────────────────────
    start_epoch = 0
    start_batch = 0
    best_map50  = 0.0
    history     = []

    # Check for auto-resume candidates
    if not resume:
        latest_step = out_path / "latest_step.pth"
        if latest_step.exists():
            resume = str(latest_step)
        else:
            # Check for latest epoch checkpoint
            epoch_ckpts = sorted(list(out_path.glob("epoch_*.pth")))
            if epoch_ckpts:
                resume = str(epoch_ckpts[-1])

    if resume and Path(resume).exists():
        print(f"[Train] Resuming from checkpoint: {resume}")
        ckpt = torch.load(resume, map_location=device)
        model.load_state_dict(ckpt["model_state_dict"])
        if "optimizer_state_dict" in ckpt:
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])
        if scaler and "scaler_state_dict" in ckpt:
            scaler.load_state_dict(ckpt["scaler_state_dict"])
        start_epoch = ckpt.get("epoch", 0)
        start_batch = ckpt.get("batch_idx", 0)
        best_map50  = ckpt.get("best_map50", 0.0)
        history     = ckpt.get("history", [])

        # If it was a completed epoch checkpoint, move to next epoch
        if start_batch >= len(train_loader) or start_batch == 0:
            start_epoch = start_epoch + 1 if "batch_idx" not in ckpt else start_epoch
            start_batch = 0

        for _ in range(start_epoch):
            scheduler.step()
        print(f"[Train] Resumed at Epoch {start_epoch + 1}, Batch {start_batch}, Best mAP@50: {best_map50:.4f}")

    # Mid-epoch checkpoint callback
    def save_step_checkpoint(ep, b_idx, curr_loss):
        step_ckpt = out_path / "latest_step.pth"
        torch.save({
            "epoch": ep - 1,
            "batch_idx": b_idx,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict": scaler.state_dict() if scaler else None,
            "best_map50": best_map50,
            "loss": curr_loss,
            "history": history,
        }, str(step_ckpt))

    # ── Training loop ──────────────────────────────────────────────────────
    print("=" * 70)
    print(f"  Faster R-CNN Training — {epochs} epochs, batch={batch_size}, lr={lr}")
    print(f"  Output → {out_path}")
    print("=" * 70)

    for epoch in range(start_epoch, epochs):
        t_epoch = time.time()
        current_lr = optimizer.param_groups[0]["lr"]
        print(f"\n[Epoch {epoch+1}/{epochs}]  LR={current_lr:.6f}")

        b_start = start_batch if epoch == start_epoch else 0
        avg_loss = train_one_epoch(
            model, optimizer, train_loader, device,
            epoch=epoch + 1, print_freq=100, scaler=scaler,
            start_batch=b_start, checkpoint_callback=save_step_checkpoint
        )
        scheduler.step()

        # Validate every epoch (evaluate 400 validation samples for intermediate epochs, full 1667 set on final epoch)
        print(f"  → Running validation…")
        max_v = None if (epoch + 1 == epochs) else 400
        map_result = validate(model, val_loader, device, conf_thresh=conf_thresh, max_samples=max_v)
        map50  = map_result["map_50"]
        map95  = map_result["map"]
        epoch_time = time.time() - t_epoch

        print(f"  → mAP@50={map50:.4f}  mAP@50:95={map95:.4f}  "
              f"P={map_result['macro_precision']:.4f}  R={map_result['macro_recall']:.4f}  "
              f"F1={map_result['macro_f1']:.4f}  loss={avg_loss:.4f}  time={epoch_time:.1f}s")

        for cls_id, m in map_result.get("per_class", {}).items():
            name = CLASS_NAMES.get(cls_id - 1, f"class_{cls_id}")
            print(f"     [{cls_id}] {name:<24} AP50={m['ap50']:.3f} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

        # Save checkpoint every epoch
        ckpt_path = out_path / f"epoch_{epoch+1:03d}.pth"
        torch.save({
            "epoch": epoch,
            "model_state_dict":     model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "scaler_state_dict":    scaler.state_dict() if scaler else None,
            "best_map50":           best_map50,
            "map50":  map50,
            "map95":  map95,
            "loss":   avg_loss,
            "history": history,
        }, str(ckpt_path))

        # Best model
        if map50 > best_map50:
            best_map50 = map50
            best_path  = weights_dir / "frcnn_best.pth"
            shutil.copy(str(ckpt_path), str(best_path))
            print(f"  ★ New best  mAP@50={best_map50:.4f} → saved to {best_path}")

        # Always keep last
        last_path = weights_dir / "frcnn_last.pth"
        shutil.copy(str(ckpt_path), str(last_path))

        # History
        entry = {
            "epoch": epoch + 1,
            "lr":    current_lr,
            "loss":  round(avg_loss, 6),
            "map50": round(map50, 6),
            "map95": round(map95, 6),
            "macro_precision": round(map_result["macro_precision"], 6),
            "macro_recall": round(map_result["macro_recall"], 6),
            "macro_f1": round(map_result["macro_f1"], 6),
            "per_class": {str(k): v for k, v in map_result.get("per_class", {}).items()},
        }
        history.append(entry)

        # Save history JSON
        with open(log_dir / "frcnn_training_history.json", "w") as fh:
            json.dump(history, fh, indent=2)

    print("\n" + "=" * 70)
    print(f"  Training complete!  Best mAP@50 = {best_map50:.4f}")
    print(f"  Best weights: {weights_dir / 'frcnn_best.pth'}")
    print("=" * 70)

    # ── Generate training curves & sample predictions ──────────────────────
    try:
        from ml.evaluate_frcnn import generate_training_curves, generate_prediction_plot
        generate_training_curves(history, log_dir / "training_curves.png")
        generate_prediction_plot(model, val_ds, device, log_dir / "sample_predictions.png", num_samples=4)
    except Exception as e:
        print(f"[Train] Plot generation note: {e}")

    return history


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Faster R-CNN on Croppie Coffee Dataset")
    parser.add_argument("--dataset_root", default=None,
                        help="Path to datasets/croppie_coffee/extracted (auto-detected if omitted)")
    parser.add_argument("--epochs",  type=int,   default=5)
    parser.add_argument("--batch",   type=int,   default=2,
                        help="Batch size (2 recommended for RTX 4060 8 GB)")
    parser.add_argument("--device",  default="0",
                        help="CUDA device index (e.g. '0') or 'cpu'")
    parser.add_argument("--lr",      type=float, default=0.005)
    parser.add_argument("--backbone_layers", type=int, default=3,
                        help="Number of trainable ResNet backbone layers (0-5)")
    parser.add_argument("--conf",    type=float, default=0.25,
                        help="Confidence threshold for validation filtering")
    parser.add_argument("--resume",  default=None,
                        help="Path to a .pth checkpoint to resume from")
    parser.add_argument("--output_dir", default=None,
                        help="Directory to save per-epoch checkpoints")
    parser.add_argument("--workers", type=int, default=0,
                        help="DataLoader workers (0 = main thread, safer on Windows)")
    parser.add_argument("--no_amp",  action="store_true",
                        help="Disable mixed-precision training")
    args = parser.parse_args()

    train_frcnn(
        dataset_root=args.dataset_root,
        epochs=args.epochs,
        batch_size=args.batch,
        device_str=args.device,
        lr=args.lr,
        trainable_backbone_layers=args.backbone_layers,
        conf_thresh=args.conf,
        resume=args.resume,
        output_dir=args.output_dir,
        num_workers=args.workers,
        use_amp=not args.no_amp,
    )
