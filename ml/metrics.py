"""
Metrics calculation for Object Detection (COCO-style mAP@50, mAP@50:95, P, R, F1)
Pure PyTorch/NumPy implementation with 0 external dependency conflicts.
"""

from typing import Dict, List, Any
import numpy as np
import torch


def box_iou_numpy(boxA: np.ndarray, boxB: np.ndarray) -> float:
    """Calculate IoU between two [x1, y1, x2, y2] boxes."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    inter = max(0.0, xB - xA) * max(0.0, yB - yA)
    aA = max(0.0, boxA[2] - boxA[0]) * max(0.0, boxA[3] - boxA[1])
    aB = max(0.0, boxB[2] - boxB[0]) * max(0.0, boxB[3] - boxB[1])
    union = aA + aB - inter
    return inter / union if union > 0 else 0.0


def compute_ap_101(recalls: np.ndarray, precisions: np.ndarray) -> float:
    """Compute COCO 101-point interpolated Average Precision."""
    if len(recalls) == 0:
        return 0.0
    rec_points = np.linspace(0, 1.0, 101)
    p_interp = []
    for r in rec_points:
        p_candidates = precisions[recalls >= r]
        p_interp.append(np.max(p_candidates) if len(p_candidates) > 0 else 0.0)
    return float(np.mean(p_interp))


def evaluate_detections(
    predictions: List[Dict[str, Any]],
    targets: List[Dict[str, Any]],
    num_classes: int = 6,
    conf_thresh: float = 0.25,
) -> Dict[str, Any]:
    """
    Compute mAP@50, mAP@50:95, macro Precision, Recall, and F1 across all evaluation images.

    predictions: list of dict with 'boxes' (N, 4), 'scores' (N,), 'labels' (N,)
    targets: list of dict with 'boxes' (M, 4), 'labels' (M,)
    num_classes: 6 (1 to 5 are foreground coffee cherry classes, 0 is background)
    """
    iou_thresholds = np.linspace(0.50, 0.95, 10)  # 0.50, 0.55, ..., 0.95

    # Organize predictions and ground truths by class
    # For classes 1..num_classes-1
    fg_classes = list(range(1, num_classes))

    # Store results per IoU threshold and per class
    # aps_grid[iou_idx][cls_id] = AP
    aps_grid = {iou_idx: {} for iou_idx in range(len(iou_thresholds))}

    # Also compute overall P, R, F1 at conf_thresh and IoU=0.50
    tp_c = {c: 0 for c in fg_classes}
    fp_c = {c: 0 for c in fg_classes}
    fn_c = {c: 0 for c in fg_classes}

    # Evaluate each foreground class
    for cls_id in fg_classes:
        # Collect all predictions for this class across all images
        cls_preds = []  # list of (score, box, img_idx)
        total_gt = 0
        img_gts = {}  # img_idx -> list of boxes

        for img_idx, (pred, tgt) in enumerate(zip(predictions, targets)):
            p_boxes = pred["boxes"].cpu().numpy() if isinstance(pred["boxes"], torch.Tensor) else np.array(pred["boxes"])
            p_scores = pred["scores"].cpu().numpy() if isinstance(pred["scores"], torch.Tensor) else np.array(pred["scores"])
            p_labels = pred["labels"].cpu().numpy() if isinstance(pred["labels"], torch.Tensor) else np.array(pred["labels"])

            t_boxes = tgt["boxes"].cpu().numpy() if isinstance(tgt["boxes"], torch.Tensor) else np.array(tgt["boxes"])
            t_labels = tgt["labels"].cpu().numpy() if isinstance(tgt["labels"], torch.Tensor) else np.array(tgt["labels"])

            # Filter for this class
            gt_mask = (t_labels == cls_id)
            img_gts[img_idx] = t_boxes[gt_mask]
            total_gt += len(img_gts[img_idx])

            pred_mask = (p_labels == cls_id)
            for box, score in zip(p_boxes[pred_mask], p_scores[pred_mask]):
                cls_preds.append((float(score), box, img_idx))

        if total_gt == 0:
            for iou_idx in range(len(iou_thresholds)):
                aps_grid[iou_idx][cls_id] = 0.0
            continue

        if len(cls_preds) == 0:
            for iou_idx in range(len(iou_thresholds)):
                aps_grid[iou_idx][cls_id] = 0.0
            fn_c[cls_id] = total_gt
            continue

        # Sort predictions descending by score
        cls_preds.sort(key=lambda x: x[0], reverse=True)

        # Compute AP for each IoU threshold
        for iou_idx, iou_thresh in enumerate(iou_thresholds):
            # Track matched ground truths per image
            matched_gt = {img_idx: set() for img_idx in img_gts}
            tp = np.zeros(len(cls_preds))
            fp = np.zeros(len(cls_preds))

            for p_idx, (score, p_box, img_idx) in enumerate(cls_preds):
                gt_list = img_gts.get(img_idx, [])
                best_iou = 0.0
                best_gt_idx = -1

                for g_idx, g_box in enumerate(gt_list):
                    if g_idx in matched_gt[img_idx]:
                        continue
                    iou = box_iou_numpy(p_box, g_box)
                    if iou > best_iou:
                        best_iou = iou
                        best_gt_idx = g_idx

                if best_iou >= iou_thresh and best_gt_idx != -1:
                    tp[p_idx] = 1.0
                    matched_gt[img_idx].add(best_gt_idx)
                else:
                    fp[p_idx] = 1.0

            # Cumulative TP and FP
            cum_tp = np.cumsum(tp)
            cum_fp = np.cumsum(fp)
            precisions = cum_tp / (cum_tp + cum_fp + 1e-10)
            recalls = cum_tp / (total_gt + 1e-10)

            aps_grid[iou_idx][cls_id] = compute_ap_101(recalls, precisions)

            # At iou_thresh=0.50, compute P, R, F1 for detections >= conf_thresh
            if abs(iou_thresh - 0.50) < 1e-4:
                conf_mask = np.array([p[0] >= conf_thresh for p in cls_preds])
                c_tp = int(np.sum(tp[conf_mask])) if len(conf_mask) > 0 else 0
                c_fp = int(np.sum(fp[conf_mask])) if len(conf_mask) > 0 else 0
                c_fn = max(0, total_gt - c_tp)
                tp_c[cls_id] = c_tp
                fp_c[cls_id] = c_fp
                fn_c[cls_id] = c_fn

    # Aggregate mAP
    # mAP@50
    map_50 = float(np.mean(list(aps_grid[0].values()))) if aps_grid[0] else 0.0

    # mAP@75 (index 5 corresponds to 0.75)
    map_75 = float(np.mean(list(aps_grid[5].values()))) if aps_grid[5] else 0.0

    # mAP@50:95 (average over all 10 thresholds)
    all_maps = [float(np.mean(list(aps_grid[idx].values()))) for idx in range(len(iou_thresholds))]
    map_5095 = float(np.mean(all_maps)) if all_maps else 0.0

    # Per-class metrics
    per_class = {}
    for cls_id in fg_classes:
        tp, fp, fn = tp_c[cls_id], fp_c[cls_id], fn_c[cls_id]
        p = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        r = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0
        per_class[cls_id] = {
            "ap50": float(aps_grid[0].get(cls_id, 0.0)),
            "precision": float(p),
            "recall": float(r),
            "f1": float(f1),
            "tp": int(tp),
            "fp": int(fp),
            "fn": int(fn),
        }

    macro_p = float(np.mean([m["precision"] for m in per_class.values()]))
    macro_r = float(np.mean([m["recall"] for m in per_class.values()]))
    macro_f1 = float(np.mean([m["f1"] for m in per_class.values()]))

    return {
        "map_50": map_50,
        "map_75": map_75,
        "map": map_5095,
        "macro_precision": macro_p,
        "macro_recall": macro_r,
        "macro_f1": macro_f1,
        "per_class": per_class,
    }
