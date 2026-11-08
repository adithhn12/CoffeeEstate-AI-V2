import os
import cv2
import json
import time
import numpy as np
from pathlib import Path
from typing import Dict, Any, List, Tuple, Optional
from PIL import Image

from app.config import settings

# Hex and BGR color definitions for visual annotations
CLASS_COLORS = {
    0: {"name": "dark_brown_cherry", "bgr": (15, 53, 120), "hex": "#78350F"},    # Dark brown (Overripe / Dried)
    1: {"name": "green_cherry", "bgr": (50, 180, 50), "hex": "#10B981"},         # Green (Unripe)
    2: {"name": "low_visibility_unsure", "bgr": (212, 182, 6), "hex": "#06B6D4"},# Cyan (Uncertain / Occluded)
    3: {"name": "red_cherry", "bgr": (30, 30, 220), "hex": "#EF4444"},            # Red (Ripe)
    4: {"name": "yellow_cherry", "bgr": (20, 180, 240), "hex": "#F59E0B"}        # Yellow (Semi-Ripe / Turning)
}

class VisionService:
    def __init__(self):
        self.model = None
        self.model_status = "Model not trained"
        self.model_version = "YOLOv8-Croppie-Public-v1.0"
        self.active_weights_path = None
        self._load_model()

    def _load_model(self):
        """
        Loads custom fine-tuned YOLO weights if available;
        checks candidate paths across runs and fallback directories.
        """
        candidate_paths = [
            Path(settings.DEFAULT_MODEL_PATH),
            Path(settings.BASE_DIR) / "ml" / "weights" / "best.pt",
            Path(settings.BASE_DIR) / "runs" / "detect" / "runs" / "coffee_yolo" / "weights" / "best.pt",
            Path(settings.BASE_DIR) / "runs" / "detect" / "coffee_yolo" / "weights" / "best.pt",
            Path(settings.BASE_DIR) / "runs" / "coffee_yolo" / "weights" / "best.pt",
            Path(settings.BASE_DIR) / "runs" / "detect" / "runs" / "detect" / "train" / "weights" / "best.pt",
        ]

        target_path = None
        for cand in candidate_paths:
            if cand.exists() and cand.is_file():
                target_path = str(cand)
                break

        # If not found yet, search recursively under runs for any best.pt
        if not target_path:
            found = list((Path(settings.BASE_DIR) / "runs").glob("**/weights/best.pt"))
            if found:
                target_path = str(found[0])

        try:
            from ultralytics import YOLO
            if target_path:
                self.model = YOLO(target_path)
                self.active_weights_path = target_path
                # Verify that it has the 5 coffee cherry classes
                if hasattr(self.model, "names") and len(self.model.names) == 5:
                    self.model_status = "Trained on Croppie Coffee dataset (Uganda Public v1.0 - Awaiting Coorg Local Validation)"
                    self.model_version = "YOLOv8-Croppie-Finetuned-v1.1"
                    print(f"[VisionService] Loaded trained coffee cherry YOLO model ({len(self.model.names)} classes) from: {target_path}")
                else:
                    self.model_status = "Generic YOLO weights (Heuristic fallback active)"
                    print(f"[VisionService] Warning: Loaded model at {target_path} has {len(getattr(self.model, 'names', []))} classes; expected 5.")
            else:
                self.model = YOLO(settings.PRETRAINED_FALLBACK)
                self.active_weights_path = settings.PRETRAINED_FALLBACK
                self.model_status = "Preliminary model (Pretrained base / CV Heuristic fallback)"
                print(f"[VisionService] Loaded base model: {settings.PRETRAINED_FALLBACK}")
        except Exception as e:
            print(f"[VisionService] Ultralytics initialization note: {e}")
            self.model = None

    def reload_model(self) -> Dict[str, Any]:
        """Dynamically re-scans disk for newest trained weights and reloads model into memory."""
        self._load_model()
        return self.get_model_status_info()

    def get_model_status_info(self) -> Dict[str, Any]:
        """Returns the current scientific & validation status of the AI model."""
        has_trained_weights = (
            self.model is not None 
            and hasattr(self.model, "names") 
            and len(self.model.names) == 5 
            and "red_cherry" in str(self.model.names)
        )

        status_label = (
            "Trained on Croppie Coffee dataset (Awaiting Coorg local validation)"
            if has_trained_weights
            else "Preliminary model (Baseline / Ready to train)"
        )
        status_color = "emerald" if has_trained_weights else "amber"

        cuda_avail = False
        device_name = "CPU"
        try:
            import torch
            cuda_avail = torch.cuda.is_available()
            if cuda_avail:
                device_name = torch.cuda.get_device_name(0)
        except Exception:
            pass

        return {
            "model_name": "YOLOv8 Coffee Cherry Detector",
            "model_version": self.model_version,
            "weights_path": self.active_weights_path or settings.DEFAULT_MODEL_PATH,
            "weights_exist": has_trained_weights,
            "device": device_name,
            "cuda_available": cuda_avail,
            "status_label": status_label,
            "status_badge_color": status_color,
            "class_mappings": settings.CLASS_NAMES,
            "dataset_info": {
                "source": "Croppie Coffee Uganda (Alliance Bioversity & CIAT / Producers Direct)",
                "train_images": 5836,
                "val_images": 2497,
                "classes": 5
            },
            "detection_parameters": {
                "confidence_threshold": settings.CONFIDENCE_THRESHOLD,
                "nms_iou_threshold": settings.IOU_THRESHOLD,
                "max_detections": settings.MAX_DETECTIONS,
                "input_resolution": 640
            },
            "scientific_disclaimer": (
                "IMPORTANT SCIENTIFIC NOTICE: The public dataset consists of smartphone images collected in Uganda. "
                "Cherry color correlates with, but does not independently establish, biological maturity or cup quality. "
                "Coorg (Kodagu) estates feature unique microclimates, two-tier shade canopies (Silver Oak, Jackfruit), "
                "and distinct cultivars (Selection 795, Chandragiri, Robusta CxR). Local validation and fine-tuning "
                "using December-January Coorg harvest imagery is required before treating predictions as ground truth."
            )
        }

    def analyze_image(
        self,
        image_path: Path,
        confidence_threshold: float = 0.25,
        save_annotated: bool = True,
        variety: str = "Arabica (Selection 795)",
        mode: str = "branch"
    ) -> Dict[str, Any]:
        """
        Runs cherry analysis on the provided image:
        - mode='branch': Standard close-up branch inference with individual bounding box annotations.
        - mode='whole_plant': High-resolution tiled canopy inference across the visible plant without bounding boxes.
        """
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found at {image_path}")

        img_bgr = cv2.imread(str(image_path))
        if img_bgr is None:
            raise ValueError(f"Could not decode image at {image_path}. File may be corrupted or invalid format.")

        # Digital slide / dark screen guard
        gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
        mean_brightness = np.mean(gray)
        if mean_brightness < 45:
            raise ValueError(
                "Invalid image: This appears to be a digital presentation slide or dark screenshot, "
                "not a photograph of coffee cherries."
            )

        if mode == "whole_plant":
            return self._analyze_whole_plant(
                img_bgr=img_bgr,
                image_path=image_path,
                confidence_threshold=confidence_threshold,
                variety=variety
            )

        # ==========================================
        # BRANCH-LEVEL CLOSE-UP INFERENCE (DEFAULT)
        # ==========================================
        height, width = img_bgr.shape[:2]
        t0 = time.time()

        detections = []
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}

        # Check if we have an active custom-trained YOLO model with 5 cherry classes
        has_custom_yolo = False
        if self.model is not None:
            try:
                if hasattr(self.model, "names") and len(self.model.names) == 5 and "red_cherry" in str(self.model.names):
                    has_custom_yolo = True
            except Exception:
                pass

        if has_custom_yolo:
            results = self.model.predict(
                source=str(image_path),
                conf=confidence_threshold,
                iou=settings.IOU_THRESHOLD,
                max_det=settings.MAX_DETECTIONS,
                verbose=False
            )
            for r in results:
                boxes = r.boxes
                for box in boxes:
                    cls_id = int(box.cls[0].item())
                    conf = float(box.conf[0].item())
                    xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]
                    
                    x1, y1, x2, y2 = xyxy
                    w_box = (x2 - x1) / width
                    h_box = (y2 - y1) / height
                    xc = (x1 + x2) / (2.0 * width)
                    yc = (y1 + y2) / (2.0 * height)

                    if cls_id in counts:
                        counts[cls_id] += 1
                        detections.append({
                            "class_id": cls_id,
                            "class_name": settings.CLASS_NAMES[cls_id],
                            "confidence": round(conf, 4),
                            "bbox": [round(xc, 5), round(yc, 5), round(w_box, 5), round(h_box, 5)],
                            "pixel_coords": [int(x1), int(y1), int(x2), int(y2)]
                        })
        else:
            # CV-Heuristic fallback for preliminary mode
            detections, counts = self._detect_cherries_heuristic(img_bgr, confidence_threshold)

        # ==========================================
        # STRICT IMAGE CONTENT VALIDATION GUARD
        # ==========================================
        total_detected = sum(counts.values())
        evaluable_check = counts[0] + counts[1] + counts[3] + counts[4]
        mean_conf_check = float(np.mean([d["confidence"] for d in detections])) if detections else 0.0

        if total_detected < 8 or evaluable_check == 0 or mean_conf_check < 0.55:
            raise ValueError(
                "Invalid image: This does not look like a correct image of coffee cherries. "
                "Please upload a clear close-up photograph of a coffee branch with visible cherries."
            )
        # ==========================================

        elapsed_ms = round((time.time() - t0) * 1000, 2)
        
        evaluable_cherries = evaluable_check
        uncertain_cherries = counts[2]

        if total_detected > 0:
            red_pct = round((counts[3] / total_detected) * 100, 2)
            yellow_pct = round((counts[4] / total_detected) * 100, 2)
            green_pct = round((counts[1] / total_detected) * 100, 2)
            dark_brown_pct = round((counts[0] / total_detected) * 100, 2)
            low_vis_pct = round((counts[2] / total_detected) * 100, 2)
            uncertain_pct = low_vis_pct
        else:
            red_pct = yellow_pct = green_pct = dark_brown_pct = low_vis_pct = uncertain_pct = 0.0

        if evaluable_cherries > 0:
            evaluable_red_pct = round((counts[3] / evaluable_cherries) * 100, 2)
            evaluable_yellow_pct = round((counts[4] / evaluable_cherries) * 100, 2)
            evaluable_green_pct = round((counts[1] / evaluable_cherries) * 100, 2)
            evaluable_dark_brown_pct = round((counts[0] / evaluable_cherries) * 100, 2)
        else:
            evaluable_red_pct = evaluable_yellow_pct = evaluable_green_pct = evaluable_dark_brown_pct = 0.0

        mean_conf = round(mean_conf_check, 4)

        maturity_stage, maturity_notes = self._interpret_maturity(
            red_pct=red_pct,
            yellow_pct=yellow_pct,
            green_pct=green_pct,
            dark_brown_pct=dark_brown_pct,
            low_vis_pct=low_vis_pct,
            evaluable_red_pct=evaluable_red_pct,
            evaluable_yellow_pct=evaluable_yellow_pct,
            evaluable_green_pct=evaluable_green_pct,
            evaluable_dark_brown_pct=evaluable_dark_brown_pct,
            evaluable_cherries=evaluable_cherries,
            total_cherries=total_detected,
            variety=variety
        )

        annotated_path = None
        if save_annotated and len(detections) > 0:
            annotated_img = img_bgr.copy()
            for det in detections:
                cid = det["class_id"]
                coords = det.get("pixel_coords")
                if coords:
                    x1, y1, x2, y2 = coords
                else:
                    xc, yc, bw, bh = det["bbox"]
                    x1 = int((xc - bw / 2) * width)
                    x2 = int((xc + bw / 2) * width)
                    y1 = int((yc - bh / 2) * height)
                    y2 = int((yc + bh / 2) * height)

                color_bgr = CLASS_COLORS.get(cid, {}).get("bgr", (0, 255, 0))
                cv2.rectangle(annotated_img, (x1, y1), (x2, y2), color_bgr, 2)
                
                label = f"{CLASS_COLORS[cid]['name'].replace('_cherry', '')} {int(det['confidence']*100)}%"
                font_scale = 0.4
                font_thickness = 1
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)
                cv2.rectangle(annotated_img, (x1, max(0, y1 - 16)), (x1 + tw + 4, max(16, y1)), color_bgr, -1)
                cv2.putText(annotated_img, label, (x1 + 2, max(12, y1 - 4)), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), font_thickness, cv2.LINE_AA)

            out_filename = f"{image_path.stem}_annotated.jpg"
            annotated_full_path = settings.UPLOAD_DIR / "annotated" / out_filename
            cv2.imwrite(str(annotated_full_path), annotated_img)
            annotated_path = f"uploads/annotated/{out_filename}"

        return {
            "model_version": self.model_version,
            "model_status_at_inference": self.model_status,
            "inference_time_ms": elapsed_ms,
            "confidence_threshold": confidence_threshold,
            "analysis_mode": "branch",
            "total_cherries": total_detected,
            "dark_brown_count": counts[0],
            "green_count": counts[1],
            "low_visibility_count": counts[2],
            "red_count": counts[3],
            "yellow_count": counts[4],
            "dark_brown_pct": dark_brown_pct,
            "green_pct": green_pct,
            "low_visibility_pct": low_vis_pct,
            "red_pct": red_pct,
            "yellow_pct": yellow_pct,
            "evaluable_cherries": evaluable_cherries,
            "evaluable_red_pct": evaluable_red_pct,
            "evaluable_yellow_pct": evaluable_yellow_pct,
            "evaluable_green_pct": evaluable_green_pct,
            "evaluable_dark_brown_pct": evaluable_dark_brown_pct,
            "uncertain_cherries": uncertain_cherries,
            "uncertain_pct": uncertain_pct,
            "mean_confidence": mean_conf,
            "maturity_stage": maturity_stage,
            "maturity_notes": maturity_notes,
            "detections": detections,
            "annotated_storage_path": annotated_path,
            "plant_ripeness_score": min(100, round((evaluable_red_pct * 1.0) + (evaluable_yellow_pct * 0.45))),
            "tiles_processed": 1,
            "quality_assessment": "High (Close-up Branch Analysis)",
            "denominator_note": (
                "Percentages breakdown: visual shares (red_pct, etc.) use total_cherries denominator. "
                "Derived maturity shares use evaluable_cherries denominator (excluding low_visibility_unsure)."
            ),
            "yield_disclaimer": (
                "PHOTOGRAPHIC SAMPLING NOTICE: Cherry counts are detected from 2D photographs of canopy branches. "
                "They indicate sampling density and maturity distribution, not total plantation yield per acre."
            )
        }

    def _analyze_whole_plant(
        self,
        img_bgr: np.ndarray,
        image_path: Path,
        confidence_threshold: float = 0.20,
        variety: str = "Arabica (Selection 795)"
    ) -> Dict[str, Any]:
        """
        Whole-Plant / Canopy Analysis using high-resolution tiled YOLO inference.
        Divides high-resolution canopy photos into overlapping 640x640 patches,
        reconciles duplicate detections via Non-Maximum Suppression (NMS),
        estimates whole-bush ripeness distribution, and suppresses bounding boxes.
        """
        orig_h, orig_w = img_bgr.shape[:2]
        t0 = time.time()

        # Step 1: Safe Resolution Handling
        # If image exceeds 2560px on either side, downscale smoothly to prevent GPU OOM
        max_dim = max(orig_h, orig_w)
        if max_dim > 2560:
            scale = 2560.0 / max_dim
            proc_w = int(orig_w * scale)
            proc_h = int(orig_h * scale)
            proc_img = cv2.resize(img_bgr, (proc_w, proc_h), interpolation=cv2.INTER_AREA)
        else:
            scale = 1.0
            proc_w, proc_h = orig_w, orig_h
            proc_img = img_bgr

        # Step 2: High-Resolution Tiling (640x640 with 128px overlap)
        tile_size = 640
        stride = 512

        if proc_w <= tile_size and proc_h <= tile_size:
            padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
            padded[:proc_h, :proc_w] = proc_img
            tiles = [(0, 0, padded)]
        else:
            xs = list(range(0, max(1, proc_w - tile_size + 1), stride))
            if xs[-1] != max(0, proc_w - tile_size):
                xs.append(max(0, proc_w - tile_size))
            xs = sorted(list(set(xs)))

            ys = list(range(0, max(1, proc_h - tile_size + 1), stride))
            if ys[-1] != max(0, proc_h - tile_size):
                ys.append(max(0, proc_h - tile_size))
            ys = sorted(list(set(ys)))

            tiles = []
            for y in ys:
                for x in xs:
                    x_end = min(proc_w, x + tile_size)
                    y_end = min(proc_h, y + tile_size)
                    crop = proc_img[y:y_end, x:x_end]
                    if crop.shape[0] < tile_size or crop.shape[1] < tile_size:
                        padded = np.zeros((tile_size, tile_size, 3), dtype=np.uint8)
                        padded[:crop.shape[0], :crop.shape[1]] = crop
                        tiles.append((x, y, padded))
                    else:
                        tiles.append((x, y, crop))

        # Step 3: Batched YOLO Inference across Tiles
        all_boxes = []  # [x1, y1, x2, y2, conf, cls_id]
        batch_size = 8

        has_custom_yolo = (
            self.model is not None 
            and hasattr(self.model, "names") 
            and len(self.model.names) == 5 
            and "red_cherry" in str(self.model.names)
        )

        if has_custom_yolo:
            for i in range(0, len(tiles), batch_size):
                batch_subset = tiles[i:i + batch_size]
                batch_imgs = [t[2] for t in batch_subset]

                results = self.model.predict(
                    source=batch_imgs,
                    conf=max(0.15, confidence_threshold * 0.8),
                    iou=0.50,
                    max_det=300,
                    verbose=False
                )

                for (x_offset, y_offset, _), res in zip(batch_subset, results):
                    if res.boxes is not None and len(res.boxes) > 0:
                        for box in res.boxes:
                            cls_id = int(box.cls[0].item())
                            conf = float(box.conf[0].item())
                            b_xyxy = box.xyxy[0].tolist()  # [x1, y1, x2, y2]

                            gx1 = b_xyxy[0] + x_offset
                            gy1 = b_xyxy[1] + y_offset
                            gx2 = b_xyxy[2] + x_offset
                            gy2 = b_xyxy[3] + y_offset

                            gx1 = max(0.0, min(float(proc_w), gx1))
                            gy1 = max(0.0, min(float(proc_h), gy1))
                            gx2 = max(0.0, min(float(proc_w), gx2))
                            gy2 = max(0.0, min(float(proc_h), gy2))

                            if (gx2 - gx1) > 2 and (gy2 - gy1) > 2:
                                all_boxes.append([gx1, gy1, gx2, gy2, conf, cls_id])
        else:
            for x_offset, y_offset, tile_img in tiles:
                dets, _ = self._detect_cherries_heuristic(tile_img, confidence_threshold)
                for d in dets:
                    coords = d.get("pixel_coords", [0, 0, 0, 0])
                    all_boxes.append([
                        coords[0] + x_offset,
                        coords[1] + y_offset,
                        coords[2] + x_offset,
                        coords[3] + y_offset,
                        d["confidence"],
                        d["class_id"]
                    ])

        # Step 4: Duplicate Reconciliation via Non-Maximum Suppression (NMS)
        reconciled_boxes = self._reconcile_tile_duplicates(all_boxes, proc_w, proc_h)

        # Step 5: Mode-Aware Validation Guards
        total_detected = len(reconciled_boxes)
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        for b in reconciled_boxes:
            cid = int(b[5])
            if cid in counts:
                counts[cid] += 1

        evaluable_count = counts[0] + counts[1] + counts[3] + counts[4]
        mean_conf = float(np.mean([b[4] for b in reconciled_boxes])) if reconciled_boxes else 0.0

        if total_detected < 8 or evaluable_count == 0:
            raise ValueError(
                "Insufficient visual detail: Unable to detect sufficient coffee cherries on this plant. "
                "Please upload a clearer photograph captured from 1.5–2.5 meters away with visible fruit clusters."
            )

        # Foliage context check (HSV green/earthy presence)
        hsv = cv2.cvtColor(proc_img, cv2.COLOR_BGR2HSV)
        canopy_mask = cv2.inRange(hsv, np.array([20, 20, 20]), np.array([100, 255, 255]))
        canopy_coverage = float(np.count_nonzero(canopy_mask)) / (proc_w * proc_h)
        if canopy_coverage < 0.08 and total_detected < 20:
            raise ValueError(
                "Invalid image: The uploaded photograph does not appear to show a coffee plant canopy or foliage."
            )

        elapsed_ms = round((time.time() - t0) * 1000, 2)

        # Step 6: Percentage Breakdown
        evaluable_cherries = evaluable_count
        uncertain_cherries = counts[2]

        red_pct = round((counts[3] / total_detected) * 100, 1)
        yellow_pct = round((counts[4] / total_detected) * 100, 1)
        green_pct = round((counts[1] / total_detected) * 100, 1)
        dark_brown_pct = round((counts[0] / total_detected) * 100, 1)
        low_vis_pct = round((counts[2] / total_detected) * 100, 1)

        evaluable_red_pct = round((counts[3] / evaluable_cherries) * 100, 1) if evaluable_cherries > 0 else 0.0
        evaluable_yellow_pct = round((counts[4] / evaluable_cherries) * 100, 1) if evaluable_cherries > 0 else 0.0
        evaluable_green_pct = round((counts[1] / evaluable_cherries) * 100, 1) if evaluable_cherries > 0 else 0.0
        evaluable_dark_brown_pct = round((counts[0] / evaluable_cherries) * 100, 1) if evaluable_cherries > 0 else 0.0

        plant_ripeness_score = min(100, round((evaluable_red_pct * 1.0) + (evaluable_yellow_pct * 0.45)))

        if total_detected >= 50 and mean_conf >= 0.45:
            quality_assessment = "High Sampling Confidence (Dense Fruit Sampling Across Canopy)"
        elif total_detected >= 25:
            quality_assessment = "Moderate Sampling Confidence (Medium Cluster Density)"
        else:
            quality_assessment = "Lower Confidence (Sparse / Partially Shaded Clusters)"

        maturity_stage, maturity_notes = self._interpret_whole_plant_maturity(
            evaluable_red_pct=evaluable_red_pct,
            evaluable_yellow_pct=evaluable_yellow_pct,
            evaluable_green_pct=evaluable_green_pct,
            evaluable_dark_brown_pct=evaluable_dark_brown_pct,
            evaluable_cherries=evaluable_cherries,
            total_cherries=total_detected,
            variety=variety
        )

        return {
            "model_version": self.model_version,
            "model_status_at_inference": self.model_status,
            "inference_time_ms": elapsed_ms,
            "confidence_threshold": confidence_threshold,
            "analysis_mode": "whole_plant",
            "total_cherries": total_detected,
            "dark_brown_count": counts[0],
            "green_count": counts[1],
            "low_visibility_count": counts[2],
            "red_count": counts[3],
            "yellow_count": counts[4],
            "dark_brown_pct": dark_brown_pct,
            "green_pct": green_pct,
            "low_visibility_pct": low_vis_pct,
            "red_pct": red_pct,
            "yellow_pct": yellow_pct,
            "evaluable_cherries": evaluable_cherries,
            "evaluable_red_pct": evaluable_red_pct,
            "evaluable_yellow_pct": evaluable_yellow_pct,
            "evaluable_green_pct": evaluable_green_pct,
            "evaluable_dark_brown_pct": evaluable_dark_brown_pct,
            "uncertain_cherries": uncertain_cherries,
            "uncertain_pct": low_vis_pct,
            "mean_confidence": round(mean_conf, 4),
            "maturity_stage": maturity_stage,
            "maturity_notes": maturity_notes,
            "detections": [],  # Clean image: individual bounding boxes suppressed
            "annotated_storage_path": None,  # Original photo displayed cleanly
            "plant_ripeness_score": plant_ripeness_score,
            "tiles_processed": len(tiles),
            "quality_assessment": quality_assessment,
            "denominator_note": (
                "Whole-Plant Distribution: Ripeness percentages reflect collective fruit maturity across visible foliage. "
                "Derived maturity shares use evaluable cherries (excluding occluded fruit)."
            ),
            "yield_disclaimer": (
                "WHOLE-CANOPY SAMPLING NOTICE: Percentages represent visible canopy clusters. "
                "Interior cherries obscured by dense foliage are not included in surface optical estimation."
            )
        }

    def _reconcile_tile_duplicates(
        self,
        boxes: List[List[float]],
        width: int,
        height: int,
        iou_threshold: float = 0.45
    ) -> List[List[float]]:
        if not boxes:
            return []

        try:
            import torch
            import torchvision.ops

            b_t = torch.tensor([[b[0], b[1], b[2], b[3]] for b in boxes], dtype=torch.float32)
            s_t = torch.tensor([b[4] for b in boxes], dtype=torch.float32)
            c_t = torch.tensor([b[5] for b in boxes], dtype=torch.float32)

            max_coord = float(max(width, height) + 1000)
            shifted = b_t + c_t.unsqueeze(1) * max_coord
            keep = torchvision.ops.nms(shifted, s_t, iou_threshold=iou_threshold)
            return [boxes[i] for i in keep.tolist()]
        except Exception:
            boxes_arr = np.array(boxes)
            x1 = boxes_arr[:, 0]
            y1 = boxes_arr[:, 1]
            x2 = boxes_arr[:, 2]
            y2 = boxes_arr[:, 3]
            scores = boxes_arr[:, 4]
            classes = boxes_arr[:, 5]

            areas = (x2 - x1) * (y2 - y1)
            order = scores.argsort()[::-1]

            keep = []
            while order.size > 0:
                i = order[0]
                keep.append(i)
                if order.size == 1:
                    break

                xx1 = np.maximum(x1[i], x1[order[1:]])
                yy1 = np.maximum(y1[i], y1[order[1:]])
                xx2 = np.minimum(x2[i], x2[order[1:]])
                yy2 = np.minimum(y2[i], y2[order[1:]])

                w = np.maximum(0.0, xx2 - xx1)
                h = np.maximum(0.0, yy2 - yy1)
                inter = w * h
                ovr = inter / (areas[i] + areas[order[1:]] - inter)

                same_cls = (classes[order[1:]] == classes[i])
                suppress = np.where(same_cls & (ovr > iou_threshold))[0]
                order = np.delete(order, np.concatenate(([0], suppress + 1)))

            return [boxes[k] for k in keep]

    def _interpret_whole_plant_maturity(
        self,
        evaluable_red_pct: float,
        evaluable_yellow_pct: float,
        evaluable_green_pct: float,
        evaluable_dark_brown_pct: float,
        evaluable_cherries: int,
        total_cherries: int,
        variety: str = "Arabica (Selection 795)"
    ) -> Tuple[str, str]:
        variety_lower = variety.lower()

        if total_cherries == 0 or evaluable_cherries == 0:
            return (
                "Insufficient Fruit Sampling",
                "No discernible ripe, turning, or green cherries detected across the visible plant foliage. "
                "Recapture in diffused lighting with the camera positioned 1.5–2 meters from the coffee canopy."
            )

        estate_prep = "washed Arabica (Plantation Grade A)" if "arabica" in variety_lower else "premium natural Robusta"

        # Tier 1: Significant Overripe Percentage -> Prioritise Overripe Cherry Picking
        if evaluable_dark_brown_pct >= 15.0:
            return (
                "Overripe Canopy Alert — Prioritise Overripe Picking",
                f"Plant-wide analysis shows {evaluable_dark_brown_pct:.1f}% overripe or dried cherries across the canopy. "
                "Prioritise immediate selective picking of overripe cherries and sanitation plucking across this section "
                "to suppress coffee berry borer (Hypothenemus hampei) colonization and avoid fungal souring. "
                "Float harvested cherries in water tanks to separate dried/damaged fruit before processing.\n"
                "[Ripeness-based estimate: Immediate sanitation picking recommended; actual fruit degradation rate depends on ambient humidity and sun exposure.]"
            )

        # Cultivar check for Yellow Bourbon
        if "yellow bourbon" in variety_lower:
            if evaluable_yellow_pct >= 70.0 and evaluable_green_pct <= 18.0:
                return (
                    "Optimal Whole-Plant Harvest Ready (Yellow Bourbon)",
                    f"Canopy displays uniform golden maturity with {evaluable_yellow_pct:.1f}% yellow cherries and minimal unripe green fruit ({evaluable_green_pct:.1f}%). "
                    "In Yellow Bourbon cultivars, uniform golden coloration indicates peak Brix sugar development. Broad harvesting readiness: bush is ready for full canopy picking.\n"
                    "[Ripeness-based estimate: Optimal broad harvesting window is estimated within the next 3–6 days based on visual ripeness; actual timeline will vary with local weather and temperature.]"
                )
            elif (evaluable_yellow_pct >= 35.0 or (evaluable_yellow_pct + evaluable_red_pct >= 50.0)) and evaluable_green_pct > 18.0:
                return (
                    "Selective Harvesting Advised (Yellow Bourbon)",
                    f"Plant canopy exhibits mixed maturation with {evaluable_yellow_pct:.1f}% ripe yellow cherries alongside {evaluable_green_pct:.1f}% unripe green cherries. "
                    "Do not perform whole-plant strip plucking, as stripping immature green cherries will cause astringent cup defects and reduce estate grade. "
                    "Selective hand-picking is advised: harvest only ripe golden cherries while leaving unripe green fruit on the bush to continue ripening.\n"
                    "[Ripeness-based estimate: Conduct a selective picking pass now; reassess remaining green fruit in 10–14 days based on visual progress, subject to local weather conditions.]"
                )

        # Tier 2: High Ripe Percentage with Very Few Unripe Cherries -> Broad Harvesting Readiness
        if evaluable_red_pct >= 70.0 and evaluable_green_pct <= 18.0:
            return (
                "Optimal Whole-Plant Harvest Ready (Broad Harvesting)",
                f"Canopy has reached uniform maturity with {evaluable_red_pct:.1f}% ripe red cherries and very low unripe fruit ({evaluable_green_pct:.1f}%). "
                f"Broad harvesting readiness: bush is ready for full canopy picking. Prime window for {estate_prep}. "
                "Depulp harvested cherries within 6 to 8 hours to maintain specialty washed parchment quality.\n"
                "[Ripeness-based estimate: Optimal broad harvesting window is estimated within the next 3–6 days based on visual maturity; actual timeline will vary with local temperature and rainfall.]"
            )

        # Tier 3: Moderate Ripe Percentage with Significant Unripe Cherries -> Selective Harvesting
        # (Specifically handles mixed maturation such as 55.9% ripe and 42.1% unripe)
        if (evaluable_red_pct >= 35.0 or (evaluable_red_pct + evaluable_yellow_pct >= 50.0)) and evaluable_green_pct > 18.0:
            return (
                "Selective Harvesting Advised (Moderate Ripe, High Unripe)",
                f"Plant canopy exhibits staggered ripening with {evaluable_red_pct:.1f}% ripe red cherries alongside a significant proportion of immature green cherries ({evaluable_green_pct:.1f}%). "
                "Do not perform broad or whole-plant strip plucking, as stripping immature green cherries will cause astringent cup defects and reduce estate grade. "
                "Selective hand-picking is strongly recommended: harvest only mature crimson cherries from exposed branches while leaving green clusters on the bush to continue ripening.\n"
                "[Ripeness-based estimate: Conduct a selective picking pass now; reassess remaining green fruit in 10–14 days based on visual progress, subject to local weather conditions.]"
            )

        # Tier 4: Low Ripe Percentage -> Wait and Reassess
        if evaluable_green_pct >= 50.0 or (evaluable_red_pct < 35.0 and evaluable_red_pct + evaluable_yellow_pct < 45.0):
            return (
                "Immature Canopy — Wait and Reassess",
                f"Canopy is predominantly immature ({evaluable_green_pct:.1f}% green cherries, only {evaluable_red_pct:.1f}% ripe red). "
                "Immature cherries have not developed sufficient mucilage sugars or Brix content. Hold harvesting operations to prevent astringent, sour liquor. "
                "Allow clusters to mature on the plant.\n"
                "[Ripeness-based estimate: Defer harvest and reassess canopy in 2 to 4 weeks; maturation timing will depend on rainfall patterns and sunlight exposure.]"
            )

        # Fallback: Turning Phase / Mixed Maturation
        return (
            "Turning Phase — Selective Monitoring",
            f"Canopy is in an active turning phase ({evaluable_red_pct:.1f}% red, {evaluable_yellow_pct:.1f}% semi-ripe yellow, {evaluable_green_pct:.1f}% green). "
            "Staggered flowering flushes detected. Selective fly-picking of early ripe cherries may be performed on sun-exposed branches, while deferring the primary harvest.\n"
            "[Ripeness-based estimate: Monitor canopy development over the next 7–10 days; estimate is based on visual ripeness without microclimate integration.]"
        )

    def _interpret_maturity(
        self,
        red_pct: float,
        yellow_pct: float,
        green_pct: float,
        dark_brown_pct: float,
        low_vis_pct: float,
        evaluable_red_pct: float,
        evaluable_yellow_pct: float,
        evaluable_green_pct: float,
        evaluable_dark_brown_pct: float,
        evaluable_cherries: int,
        total_cherries: int,
        variety: str = "Arabica (Selection 795)"
    ) -> Tuple[str, str]:
        if total_cherries == 0:
            return "No Cherries Detected", "No coffee cherries recognized. Verify camera focus and lighting."

        if evaluable_cherries == 0:
            return (
                "High Uncertainty Only",
                f"All {total_cherries} detections are low visibility / occluded. "
                "Unable to assess maturity. Please recapture branch photo under diffused daylight."
            )

        variety_lower = variety.lower()
        uncertainty_warning = ""
        if low_vis_pct > 30.0:
            uncertainty_warning = (
                f"\n[Uncertainty Caution: {low_vis_pct:.1f}% of detections are occluded or low visibility. "
                "Resample branch under diffused sunlight for higher precision.]"
            )

        if "yellow bourbon" in variety_lower:
            if evaluable_yellow_pct >= 55.0:
                return (
                    "Optimal Harvest Window (Yellow Bourbon)",
                    f"{evaluable_yellow_pct:.1f}% yellow cherries among discernible fruit. "
                    "In Yellow Bourbon cultivars, yellow signifies peak Brix ripeness.{uncertainty_warning}"
                )

        if evaluable_dark_brown_pct > 25.0:
            return (
                "Overripe / Dried Risk",
                f"{evaluable_dark_brown_pct:.1f}% overripe/dark brown cherries among discernible fruit. "
                "Immediate strip picking recommended to prevent berry borer (Hypothenemus hampei) infestation and fungal spoilage.{uncertainty_warning}"
            )
        elif evaluable_red_pct >= 55.0:
            estate_prep = "washed Arabica (Plantation Grade A)" if "arabica" in variety_lower else "premium natural Robusta"
            return (
                "Optimal Harvest Window",
                f"{evaluable_red_pct:.1f}% ripe red cherries among discernible fruit. "
                f"Peak maturity for {estate_prep}. High mucilage sugar Brix accumulation.{uncertainty_warning}"
            )
        elif (evaluable_red_pct + evaluable_yellow_pct >= 60.0) and evaluable_red_pct >= 35.0:
            return (
                "Semi-Ripe / Fly Picking",
                f"{evaluable_red_pct:.1f}% red and {evaluable_yellow_pct:.1f}% yellow cherries. "
                "Selective 'fly picking' recommended: harvest crimson ripe berries first, "
                f"leaving turning cherries for secondary harvest in 7-10 days.{uncertainty_warning}"
            )
        elif evaluable_green_pct >= 50.0:
            return (
                "Early Unripe Stage",
                f"{evaluable_green_pct:.1f}% green cherries. Immature development. "
                f"Allow 2 to 4 weeks of maturation before harvesting.{uncertainty_warning}"
            )
        else:
            return (
                "Mixed Maturation",
                f"Heterogeneous maturation ({evaluable_red_pct:.1f}% red, {evaluable_yellow_pct:.1f}% yellow, {evaluable_green_pct:.1f}% green). "
                f"Selective manual picking advised to prevent astringent unripe beans.{uncertainty_warning}"
            )

    def _detect_cherries_heuristic(self, img_bgr: np.ndarray, conf_thresh: float) -> Tuple[List[Dict[str, Any]], Dict[int, int]]:
        height, width = img_bgr.shape[:2]
        hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)

        red_lower1 = np.array([0, 70, 50])
        red_upper1 = np.array([10, 255, 255])
        red_lower2 = np.array([165, 70, 50])
        red_upper2 = np.array([180, 255, 255])
        mask_red = cv2.bitwise_or(
            cv2.inRange(hsv, red_lower1, red_upper1),
            cv2.inRange(hsv, red_lower2, red_upper2)
        )

        mask_yellow = cv2.inRange(hsv, np.array([15, 80, 70]), np.array([35, 255, 255]))
        mask_green = cv2.inRange(hsv, np.array([36, 60, 50]), np.array([85, 255, 220]))
        mask_brown = cv2.inRange(hsv, np.array([8, 60, 20]), np.array([22, 220, 80]))

        class_masks = [
            (3, "red_cherry", mask_red),
            (4, "yellow_cherry", mask_yellow),
            (1, "green_cherry", mask_green),
            (0, "dark_brown_cherry", mask_brown)
        ]

        detections = []
        counts = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

        for cid, cname, mask in class_masks:
            cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
            contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                area = cv2.contourArea(cnt)
                if 200 < area < 15000:
                    x, y, w, h = cv2.boundingRect(cnt)
                    aspect = float(w) / h
                    if 0.5 <= aspect <= 2.0:
                        perimeter = cv2.arcLength(cnt, True)
                        circularity = 4 * np.pi * (area / (perimeter * perimeter)) if perimeter > 0 else 0
                        conf = float(np.clip(0.45 + (circularity * 0.4), 0.3, 0.95))

                        if conf >= conf_thresh:
                            xc = (x + w / 2) / width
                            yc = (y + h / 2) / height
                            bw = w / width
                            bh = h / height

                            counts[cid] += 1
                            detections.append({
                                "class_id": cid,
                                "class_name": cname,
                                "confidence": round(conf, 3),
                                "bbox": [round(xc, 5), round(yc, 5), round(bw, 5), round(bh, 5)],
                                "pixel_coords": [x, y, x + w, y + h]
                            })

        return detections, counts

vision_service = VisionService()