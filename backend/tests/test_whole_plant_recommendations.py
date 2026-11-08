"""
Unit Tests for Whole-Plant Ripeness Recommendation Logic
Verifies that:
1. Moderate ripe percentage with significant unripe cherries (e.g. 55.9% ripe, 42.1% unripe)
   triggers Selective Harvesting Advised rather than broad harvest readiness.
2. High ripe percentage with very few unripe cherries triggers Broad Harvesting Readiness.
3. Low ripe percentage triggers Wait and Reassess.
4. Significant overripe percentage triggers Prioritise Overripe Picking.
5. All recommendations explicitly identify timing as ripeness-based estimates.
6. Branch mode logic remains intact.
"""

import sys
import unittest
from pathlib import Path

# Add backend directory to path
backend_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(backend_dir))

from app.services.vision_service import vision_service

class TestWholePlantRecommendations(unittest.TestCase):

    def test_01_moderate_ripe_high_unripe_triggers_selective_harvesting(self):
        """User Issue 1 & 2: 55.9% ripe and 42.1% unripe must advise selective harvesting, not optimal broad harvest."""
        stage, notes = vision_service._interpret_whole_plant_maturity(
            evaluable_red_pct=55.9,
            evaluable_yellow_pct=2.0,
            evaluable_green_pct=42.1,
            evaluable_dark_brown_pct=0.0,
            evaluable_cherries=100,
            total_cherries=105,
            variety="Arabica (Selection 795)"
        )
        self.assertEqual(stage, "Selective Harvesting Advised (Moderate Ripe, High Unripe)")
        self.assertNotIn("Optimal Whole-Plant Harvest Ready", stage)
        self.assertIn("Selective hand-picking is strongly recommended", notes)
        self.assertIn("leaving green clusters on the bush", notes)
        self.assertIn("Ripeness-based estimate", notes)

    def test_02_high_ripe_few_unripe_triggers_broad_harvesting(self):
        """User Issue 2: High ripe percentage (e.g. 82%) with few unripe (8%) triggers broad harvest readiness."""
        stage, notes = vision_service._interpret_whole_plant_maturity(
            evaluable_red_pct=82.0,
            evaluable_yellow_pct=6.0,
            evaluable_green_pct=8.0,
            evaluable_dark_brown_pct=4.0,
            evaluable_cherries=200,
            total_cherries=210,
            variety="Arabica (Selection 795)"
        )
        self.assertEqual(stage, "Optimal Whole-Plant Harvest Ready (Broad Harvesting)")
        self.assertIn("Broad harvesting readiness", notes)
        self.assertIn("Ripeness-based estimate", notes)

    def test_03_low_ripe_triggers_wait_and_reassess(self):
        """User Issue 2: Low ripe percentage / high green percentage triggers Wait and Reassess."""
        stage, notes = vision_service._interpret_whole_plant_maturity(
            evaluable_red_pct=15.0,
            evaluable_yellow_pct=10.0,
            evaluable_green_pct=75.0,
            evaluable_dark_brown_pct=0.0,
            evaluable_cherries=150,
            total_cherries=160,
            variety="Arabica (Selection 795)"
        )
        self.assertEqual(stage, "Immature Canopy — Wait and Reassess")
        self.assertIn("Hold harvesting operations", notes)
        self.assertIn("Ripeness-based estimate", notes)

    def test_04_significant_overripe_triggers_prioritise_overripe_picking(self):
        """User Issue 2: Significant overripe percentage triggers Prioritise Overripe Picking."""
        stage, notes = vision_service._interpret_whole_plant_maturity(
            evaluable_red_pct=45.0,
            evaluable_yellow_pct=10.0,
            evaluable_green_pct=25.0,
            evaluable_dark_brown_pct=20.0,
            evaluable_cherries=120,
            total_cherries=130,
            variety="Arabica (Selection 795)"
        )
        self.assertEqual(stage, "Overripe Canopy Alert — Prioritise Overripe Picking")
        self.assertIn("Prioritise immediate selective picking of overripe cherries", notes)
        self.assertIn("Ripeness-based estimate", notes)

    def test_05_branch_mode_interpretation_preserved(self):
        """User Issue 5: Branch-mode interpretation and regression test remain intact."""
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

if __name__ == "__main__":
    unittest.main()

