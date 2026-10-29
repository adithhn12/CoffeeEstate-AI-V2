import json
from datetime import datetime, timedelta
from sqlalchemy.orm import Session
from app.models.estate import Estate, Section
from app.models.analysis import ImageAnalysis

def seed_sample_data_if_empty(db: Session):
    """
    Seeds initial estate section and demonstration analyses if database is empty.
    """
    existing_estate = db.query(Estate).first()
    if not existing_estate:
        estate = Estate(
            name="Glenlora Heritage Estate",
            location="Madikeri, Kodagu (Coorg), Karnataka, India",
            total_acres=50.0,
            total_hectares=20.23,
            area_unit="acres",
            description="Historic shade-grown Arabica & Robusta plantation established in 1928 at 1,140m elevation in central Kodagu.",
            boundary_geojson=None,
            manager_name="H R Nagesh",
            contact_phone="+91 98450 12345",
            latitude=12.4244,
            longitude=75.7382,
            elevation_meters=1140.0,
            is_sample_data=True
        )
        db.add(estate)
        db.commit()
        db.refresh(estate)

        section = Section(
            estate_id=estate.id,
            name="Valley Block A (Arabica Selection 795)",
            section_code="SEC-A1",
            area_acres=8.5,
            area_hectares=3.44,
            color="#15803d",
            coffee_variety="Arabica (Selection 795)",
            elevation_meters=1140.0,
            slope_aspect="North-East Facing Gentle Slope",
            soil_type="Rich Red Clay Loam with Deep Leaf Humus",
            planting_year=2014,
            shade_tree_species="Silver Oak, Jackfruit, Fig, Rosewood",
            target_plants_per_acre=480,
            available_workers=6,
            min_workers_needed=3,
            max_workers_allowed=10,
            status="harvest_ready",
            notes="Optimal shade cover. High cherry density on outer canopy. Cherry color transition to deep crimson.",
            is_sample_data=True
        )
        db.add(section)
        db.commit()

    # Seed sample analyses if none exist
    analysis_count = db.query(ImageAnalysis).count()
    if analysis_count == 0:
        first_section = db.query(Section).first()
        sec_id = first_section.id if first_section else None

        now = datetime.utcnow()
        sample_analyses = [
            ImageAnalysis(
                section_id=sec_id,
                original_filename="arabica_peak_ripe_sample.jpg",
                storage_path="uploads/raw/sample_arabica_peak.jpg",
                annotated_storage_path="uploads/annotated/sample_arabica_peak.jpg",
                uploaded_at=now - timedelta(days=2),
                analyzed_at=now - timedelta(days=2),
                model_version="YOLOv8-Croppie-Finetuned-v1.1",
                model_status_at_inference="Trained on Croppie Coffee dataset",
                inference_time_ms=18.4,
                confidence_threshold=0.25,
                total_cherries=72,
                dark_brown_count=1,
                green_count=5,
                low_visibility_count=2,
                red_count=58,
                yellow_count=6,
                dark_brown_pct=1.4,
                green_pct=6.9,
                low_visibility_pct=2.8,
                red_pct=80.6,
                yellow_pct=8.3,
                maturity_stage="Optimal Harvest Window",
                maturity_notes="82.9% ripe red cherries among discernible fruit. Optimal harvest window reached.",
                detections_json="[]",
                is_sample_data=True
            ),
            ImageAnalysis(
                section_id=sec_id,
                original_filename="canopy_turning_sample.jpg",
                storage_path="uploads/raw/sample_canopy_turning.jpg",
                annotated_storage_path="uploads/annotated/sample_canopy_turning.jpg",
                uploaded_at=now - timedelta(days=5),
                analyzed_at=now - timedelta(days=5),
                model_version="YOLOv8-Croppie-Finetuned-v1.1",
                model_status_at_inference="Trained on Croppie Coffee dataset",
                inference_time_ms=19.1,
                confidence_threshold=0.25,
                total_cherries=64,
                dark_brown_count=0,
                green_count=22,
                low_visibility_count=3,
                red_count=18,
                yellow_count=21,
                dark_brown_pct=0.0,
                green_pct=34.4,
                low_visibility_pct=4.7,
                red_pct=28.1,
                yellow_pct=32.8,
                maturity_stage="Turning Phase (Semi-Ripe)",
                maturity_notes="63.9% combined red and yellow cherries. Selective fly picking recommended.",
                detections_json="[]",
                is_sample_data=True
            )
        ]
        db.add_all(sample_analyses)
        db.commit()

def clear_all_sample_data(db: Session):
    """
    Removes demonstration sample analyses.
    """
    db.query(ImageAnalysis).filter(ImageAnalysis.is_sample_data == True).delete()
    db.commit()
