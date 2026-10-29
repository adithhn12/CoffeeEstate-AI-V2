"""
YOLOv8 Training & Fine-tuning Script for Croppie Coffee Dataset
Optimized for NVIDIA GeForce RTX 4060 Laptop GPU (8GB VRAM) with tailored augmentations for coffee clusters.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path

def train_yolo(
    data_yaml: str = "coffee_cherry.yaml",
    model_name: str = "yolov8n.pt",
    epochs: int = 3,
    batch_size: int = 8,
    imgsz: int = 640,
    device: str = "0",
    resume: bool = False,
    project_dir: str = "runs",
    name: str = "coffee_yolo",
    patience: int = 5
):
    # 1. Path verification
    yaml_path = Path(data_yaml).resolve()
    if not yaml_path.exists():
        print(f"ERROR: Dataset YAML not found at: {yaml_path}")
        sys.exit(1)

    print(f"Using dataset config: {yaml_path}")

    # 2. Disk space pre-check
    total, used, free = shutil.disk_usage(".")
    free_gb = free / (1024 ** 3)
    print(f"Available free disk space: {free_gb:.2f} GB")
    if free_gb < 2.0:
        print("WARNING: Less than 2 GB of free disk space available. Training may halt prematurely.")

    # 3. Import PyTorch and Ultralytics
    try:
        import torch
        from ultralytics import YOLO
    except ImportError as e:
        print(f"ERROR: Missing dependencies: {e}. Please run 'pip install ultralytics torch'")
        sys.exit(1)

    # 4. Device verification
    if device == "0" and not torch.cuda.is_available():
        print("CUDA requested but not available. Falling back to CPU mode.")
        device = "cpu"
        batch_size = min(batch_size, 4)
    elif torch.cuda.is_available() and device != "cpu":
        gpu_name = torch.cuda.get_device_name(0)
        vram_mb = torch.cuda.get_device_properties(0).total_memory / (1024 ** 2)
        print(f"CUDA GPU Active: {gpu_name} ({vram_mb:.0f} MB VRAM)")

    print("=" * 60)
    print(f"Starting YOLO Fine-Tuning for Coffee Cherry Detection:")
    print(f"  Base Model: {model_name}")
    print(f"  Epochs:     {epochs}")
    print(f"  Batch Size: {batch_size}")
    print(f"  Img Size:   {imgsz}")
    print(f"  Device:     {device}")
    print(f"  Patience:   {patience}")
    print(f"  Resume:     {resume}")
    print("=" * 60)

    # 5. Initialize Model
    model = YOLO(model_name)

    # 6. Start Training with tailored cherry augmentations
    try:
        results = model.train(
            data=str(yaml_path),
            epochs=epochs,
            batch=batch_size,
            imgsz=imgsz,
            device=device,
            resume=resume,
            project=project_dir,
            name=name,
            workers=0,      # Windows DataLoader stability
            optimizer="AdamW",
            lr0=0.001,
            lrf=0.01,
            patience=patience,
            # Augmentation parameters tuned for shade-grown coffee cherries
            hsv_h=0.015,   # Subtle hue jitter (preserves distinct red/yellow/green separation)
            hsv_s=0.7,     # High saturation jitter (morning mist vs direct sun)
            hsv_v=0.4,     # Brightness variation (simulates shade tree canopy shadow)
            degrees=10.0,  # Slight rotation for branch orientations
            translate=0.1, # Translation
            scale=0.5,     # Scale jitter (crucial for near vs far clusters)
            mosaic=1.0,    # Mosaic augmentation (combines 4 scenes, boosts small cherry detection)
            mixup=0.1,     # Slight mixup for cluster boundary regularization
            fliplr=0.5,    # Horizontal flip
            flipud=0.0,    # No vertical flip (gravity orientation)
            save=True,
            val=True,
            plots=True,
            exist_ok=True
        )

        print("\nTraining completed successfully!")

        # Copy best.pt / last.pt to ml/weights/best.pt
        target_dir = Path("ml") / "weights"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_pt = target_dir / "best.pt"

        # Check trainer save_dir first
        candidate_paths = []
        if hasattr(model, 'trainer') and model.trainer is not None:
            if hasattr(model.trainer, 'best') and model.trainer.best:
                candidate_paths.append(Path(model.trainer.best))
            if hasattr(model.trainer, 'save_dir') and model.trainer.save_dir:
                candidate_paths.append(Path(model.trainer.save_dir) / "weights" / "best.pt")
                candidate_paths.append(Path(model.trainer.save_dir) / "weights" / "last.pt")

        candidate_paths.extend([
            Path(project_dir) / name / "weights" / "best.pt",
            Path(project_dir) / name / "weights" / "last.pt",
            Path("runs/detect") / name / "weights" / "best.pt",
            Path("runs/detect/runs/detect") / name / "weights" / "best.pt",
        ])

        copied = False
        for c in candidate_paths:
            if c.exists() and c.is_file():
                shutil.copy(c, target_pt)
                print(f"Copied trained weights from {c} to: {target_pt}")
                copied = True
                break

        if not copied:
            # Fallback search anywhere under project_dir
            found_weights = list(Path(project_dir).glob("**/weights/*.pt"))
            if found_weights:
                best_match = next((w for w in found_weights if "best" in w.name), found_weights[0])
                shutil.copy(best_match, target_pt)
                print(f"Found and copied weights from {best_match} to: {target_pt}")
            else:
                print("WARNING: Could not find trained weights file to copy.")

        return results

    except Exception as e:
        print(f"\nERROR during training: {e}")
        # Search for any generated weights
        target_dir = Path("ml") / "weights"
        target_dir.mkdir(parents=True, exist_ok=True)
        target_pt = target_dir / "best.pt"
        found_weights = list(Path(project_dir).glob("**/weights/*.pt"))
        if found_weights:
            best_match = next((w for w in found_weights if "best" in w.name), found_weights[0])
            shutil.copy(best_match, target_pt)
            print(f"Rescued weights from {best_match} to: {target_pt}")
            return
        sys.exit(1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train YOLO on Croppie Coffee Dataset")
    parser.add_argument("--data", default="coffee_cherry.yaml", help="Path to coffee_cherry.yaml")
    parser.add_argument("--model", default="yolov8n.pt", help="Pretrained model base")
    parser.add_argument("--epochs", type=int, default=3, help="Number of training epochs")
    parser.add_argument("--batch", type=int, default=8, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--patience", type=int, default=5, help="Early stopping patience")
    parser.add_argument("--resume", action="store_true", help="Resume interrupted training")
    args = parser.parse_args()

    train_yolo(
        data_yaml=args.data,
        model_name=args.model,
        epochs=args.epochs,
        batch_size=args.batch,
        imgsz=args.imgsz,
        device=args.device,
        patience=args.patience,
        resume=args.resume
    )
