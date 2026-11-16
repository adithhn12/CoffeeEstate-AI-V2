import os
import sys
from pathlib import Path
from fastapi.testclient import TestClient

# Ensure backend directory is in path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from app.main import app
from app.database import Base, engine, SessionLocal
from app.models.analysis import ImageAnalysis
from app.services.vision_service import vision_service

client = TestClient(app)

def test_health_check_endpoint():
    """Verifies that /api/health returns 200 with database and model status."""
    response = client.get("/api/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    assert data["database_connected"] is True
    assert "model_status" in data
    assert "device" in data

def test_model_status_loads_trained_weights():
    """Verifies that vision service loads the fine-tuned 5-class YOLO checkpoint."""
    info = vision_service.get_model_status_info()
    assert info["weights_exist"] is True
    assert len(info["class_mappings"]) == 5
    assert "best.pt" in info["weights_path"]
    assert info["detection_parameters"]["max_detections"] == 500
    assert info["detection_parameters"]["nms_iou_threshold"] == 0.55

def test_dataset_yaml_and_class_definitions():
    """Verifies that coffee_cherry.yaml exists, has 5 classes, and correct paths."""
    yaml_file = Path("coffee_cherry.yaml")
    assert yaml_file.exists(), "coffee_cherry.yaml missing from project root"
    content = yaml_file.read_text(encoding="utf-8")
    assert "dark_brown_cherry" in content
    assert "green_cherry" in content
    assert "low_visibility_unsure" in content
    assert "red_cherry" in content
    assert "yellow_cherry" in content

def test_invalid_image_upload_handling():
    """Verifies that non-image or invalid uploads are rejected with 400 Bad Request."""
    fake_file = ("document.pdf", b"this is not an image", "application/pdf")
    res = client.post(
        "/api/analysis/upload",
        data={"confidence_threshold": 0.25},
        files={"files": fake_file}
    )
    assert res.status_code == 400
    assert "Unsupported file" in res.json()["detail"]

def test_maturity_interpretation_logic():
    """Verifies Phase 4 maturity denominator separation and variety-specific interpretation."""
    stage, notes = vision_service._interpret_maturity(
        red_pct=40.0,
        yellow_pct=13.3,
        green_pct=13.3,
        dark_brown_pct=0.0,
        low_vis_pct=33.3,
        evaluable_red_pct=60.0,
        evaluable_yellow_pct=20.0,
        evaluable_green_pct=20.0,
        evaluable_dark_brown_pct=0.0,
        evaluable_cherries=100,
        total_cherries=150,
        variety="Arabica (Selection 795)"
    )
    assert stage == "Optimal Harvest Window"
    assert "60.0% ripe red cherries among discernible fruit" in notes
    assert "Uncertainty Caution" in notes

def test_analysis_history_and_deletion():
    """Verifies that analysis history is retrievable and individual records can be safely deleted."""
    res = client.get("/api/analysis/history")
    assert res.status_code == 200
    items = res.json()
    assert isinstance(items, list)
    assert len(items) > 0

    # Test deleting a sample record
    sample_item = next((item for item in items if item.get("is_sample_data")), items[0])
    target_id = sample_item["id"]

    del_res = client.delete(f"/api/analysis/history/{target_id}")
    assert del_res.status_code == 200
    assert del_res.json()["message"] == "Analysis deleted successfully"

def test_whole_plant_mode_and_tile_duplicate_reconciliation():
    """Verifies that whole-plant mode processes tiled patches, reconciles duplicates, and suppresses bounding boxes."""
    test_img = Path("datasets/croppie_coffee/extracted/val/images/1687438155343.jpg")
    assert test_img.exists()

    with open(test_img, "rb") as f:
        res = client.post(
            "/api/analysis/upload",
            data={"confidence_threshold": 0.25, "analysis_mode": "whole_plant"},
            files={"files": ("canopy_test.jpg", f, "image/jpeg")}
        )

    assert res.status_code == 200
    data = res.json()
    assert data["analysis_mode"] == "whole_plant"
    assert "plant_ripeness_score" in data
    assert 0 <= data["plant_ripeness_score"] <= 100

    item = data["individual_results"][0]
    assert item["annotated_storage_path"] is None
    assert len(item["detections"]) == 0
    assert item["total_cherries"] >= 10
    assert item["tiles_processed"] >= 1
    assert item["quality_assessment"] is not None

def test_reconcile_tile_duplicates_direct():
    """Verifies that duplicate bounding boxes across overlapping tiles are reconciled via NMS."""
    boxes = [
        [100.0, 100.0, 150.0, 150.0, 0.85, 3],
        [102.0, 101.0, 151.0, 152.0, 0.72, 3],
        [300.0, 300.0, 350.0, 350.0, 0.91, 1],
    ]
    reconciled = vision_service._reconcile_tile_duplicates(boxes, width=640, height=640, iou_threshold=0.45)
    assert len(reconciled) == 2
    confs = [b[4] for b in reconciled]
    assert 0.85 in confs
    assert 0.91 in confs
    assert 0.72 not in confs

def test_whole_plant_insufficient_visual_detail_rejection():
    """Verifies that an image with insufficient visual detail in whole plant mode returns 400 Bad Request."""
    import cv2
    import numpy as np
    blank_img = np.full((640, 640, 3), (40, 120, 40), dtype=np.uint8)
    _, enc = cv2.imencode(".jpg", blank_img)
    blank_bytes = enc.tobytes()

    res = client.post(
        "/api/analysis/upload",
        data={"confidence_threshold": 0.25, "analysis_mode": "whole_plant"},
        files={"files": ("blank_plant.jpg", blank_bytes, "image/jpeg")}
    )
    assert res.status_code == 400
    assert "Insufficient visual detail" in res.json()["detail"]
