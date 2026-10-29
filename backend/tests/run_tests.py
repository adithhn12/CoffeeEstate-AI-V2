"""
Automated Test Runner for Coffee Estate AI Backend
Runs with Python standard library unittest.
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from fastapi.testclient import TestClient
from app.main import app
from app.database import SessionLocal
from app.services.vision_service import vision_service
from app.models.analysis import ImageAnalysis

class TestCoffeeEstateBackend(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.client = TestClient(app)

    def test_01_health_check_endpoint(self):
        """Verifies that /api/health returns 200 with database and model status."""
        response = self.client.get("/api/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
        self.assertTrue(data["database_connected"])
        self.assertIn("model_status", data)
        self.assertIn("device", data)

    def test_02_model_status_loads_trained_weights(self):
        """Verifies that vision service loads the fine-tuned 5-class YOLO checkpoint."""
        info = vision_service.get_model_status_info()
        self.assertTrue(info["weights_exist"], "Trained weights not detected by vision_service")
        self.assertEqual(len(info["class_mappings"]), 5)
        self.assertIn("best.pt", info["weights_path"])
        self.assertEqual(info["detection_parameters"]["max_detections"], 500)
        self.assertEqual(info["detection_parameters"]["nms_iou_threshold"], 0.55)

    def test_03_dataset_yaml_and_class_definitions(self):
        """Verifies that coffee_cherry.yaml exists, has 5 classes, and correct paths."""
        yaml_file = Path("coffee_cherry.yaml")
        self.assertTrue(yaml_file.exists(), "coffee_cherry.yaml missing from project root")
        content = yaml_file.read_text(encoding="utf-8")
        self.assertIn("dark_brown_cherry", content)
        self.assertIn("green_cherry", content)
        self.assertIn("low_visibility_unsure", content)
        self.assertIn("red_cherry", content)
        self.assertIn("yellow_cherry", content)

    def test_04_invalid_image_upload_handling(self):
        """Verifies that non-image or invalid uploads are rejected with 400 Bad Request."""
        fake_file = ("document.pdf", b"this is not an image", "application/pdf")
        res = self.client.post(
            "/api/analysis/upload",
            data={"confidence_threshold": 0.25},
            files={"files": fake_file}
        )
        self.assertEqual(res.status_code, 400)
        self.assertIn("Unsupported file", res.json()["detail"])

    def test_05_maturity_interpretation_logic(self):
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
        self.assertEqual(stage, "Optimal Harvest Window")
        self.assertIn("60.0% ripe red cherries among discernible fruit", notes)
        self.assertIn("Uncertainty Caution", notes)

    def test_06_analysis_history_and_deletion(self):
        """Verifies that analysis history is retrievable and individual records can be safely deleted."""
        res = self.client.get("/api/analysis/history")
        self.assertEqual(res.status_code, 200)
        items = res.json()
        self.assertIsInstance(items, list)
        self.assertGreater(len(items), 0)

        sample_item = next((item for item in items if item.get("is_sample_data")), items[0])
        target_id = sample_item["id"]

        del_res = self.client.delete(f"/api/analysis/history/{target_id}")
        self.assertEqual(del_res.status_code, 200)
        self.assertEqual(del_res.json()["message"], "Analysis deleted successfully")

if __name__ == "__main__":
    suite = unittest.TestLoader().loadTestsFromTestCase(TestCoffeeEstateBackend)
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(suite)
    sys.exit(not result.wasSuccessful())
