"""
Comprehensive Model Evaluation & Audit Script for Coffee Cherry Detection
Evaluates YOLO weights against the Croppie validation split (2,497 images).
Computes Precision, Recall, mAP@50, mAP@50-95 overall and per-class,
and generates confusion matrix summaries and sample prediction plots.
"""

import os
import sys
import argparse
from pathlib import Path

def evaluate_model(
    weights_path: str = "ml/weights/best.pt",
    data_yaml: str = "coffee_cherry.yaml",
    imgsz: int = 640,
    conf_threshold: float = 0.25,
    iou_threshold: float = 0.55,
    device: str = "0",
    max_det: int = 500
):
    path = Path(weights_path)
    if not path.exists():
        # Check standard Ultralytics run paths
        candidates = [
            Path("runs/detect/runs/detect/train/weights/best.pt"),
            Path("runs/detect/train/weights/best.pt"),
            Path("ml/weights/best.pt"),
            Path("yolov8s.pt"),
            Path("yolov8n.pt")
        ]
        for c in candidates:
            if c.exists():
                path = c
                break

    print(f"Loading weights for evaluation: {path}")
    from ultralytics import YOLO

    model = YOLO(str(path))
    class_names = model.names
    print(f"Model classes ({len(class_names)}): {class_names}")

    # Run validation
    metrics = model.val(
        data=data_yaml,
        imgsz=imgsz,
        conf=conf_threshold,
        iou=iou_threshold,
        device=device,
        max_det=max_det,
        split="val",
        plots=True,
        save_json=False,
        verbose=True
    )

    print("\n" + "=" * 65)
    print("           OVERALL VALIDATION EVALUATION METRICS")
    print("=" * 65)
    print(f"  Precision:   {metrics.box.mp:.4f} ({metrics.box.mp*100:.1f}%)")
    print(f"  Recall:      {metrics.box.mr:.4f} ({metrics.box.mr*100:.1f}%)")
    print(f"  mAP@50:      {metrics.box.map50:.4f} ({metrics.box.map50*100:.1f}%)")
    print(f"  mAP@50-95:   {metrics.box.map:.4f} ({metrics.box.map*100:.1f}%)")
    print("=" * 65)

    print("\n" + "-" * 65)
    print("                 PER-CLASS PERFORMANCE")
    print("-" * 65)
    print(f"  {'Class ID':<10} {'Class Name':<22} {'Precision':<10} {'Recall':<10} {'mAP50':<10} {'mAP50-95':<10}")
    print("-" * 65)

    for i, c_name in enumerate(class_names.values() if isinstance(class_names, dict) else class_names):
        p = metrics.box.p[i] if i < len(metrics.box.p) else 0.0
        r = metrics.box.r[i] if i < len(metrics.box.r) else 0.0
        map50 = metrics.box.ap50[i] if i < len(metrics.box.ap50) else 0.0
        map95 = metrics.box.ap[i] if i < len(metrics.box.ap) else 0.0
        print(f"  {i:<10} {c_name:<22} {p:<10.4f} {r:<10.4f} {map50:<10.4f} {map95:<10.4f}")

    print("-" * 65)

    # Check for confusion matrix and plot files
    save_dir = Path(metrics.save_dir)
    print(f"\nEvaluation artifacts and plots saved to: {save_dir}")
    for plot_file in ["confusion_matrix.png", "confusion_matrix_normalized.png", "val_batch0_labels.jpg", "val_batch0_pred.jpg"]:
        p = save_dir / plot_file
        if p.exists():
            print(f"  ✓ {plot_file} ({p.stat().st_size / 1024:.1f} KB)")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate YOLO on coffee cherry validation dataset")
    parser.add_argument("--weights", default="ml/weights/best.pt", help="Path to trained weights")
    parser.add_argument("--data", default="coffee_cherry.yaml", help="Dataset YAML")
    parser.add_argument("--conf", type=float, default=0.25, help="Confidence threshold")
    parser.add_argument("--iou", type=float, default=0.55, help="NMS IoU threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image resolution")
    parser.add_argument("--device", default="0", help="CUDA device index or 'cpu'")
    parser.add_argument("--max_det", type=int, default=500, help="Max detections per image")
    args = parser.parse_args()

    evaluate_model(
        weights_path=args.weights,
        data_yaml=args.data,
        imgsz=args.imgsz,
        conf_threshold=args.conf,
        iou_threshold=args.iou,
        device=args.device,
        max_det=args.max_det
    )
