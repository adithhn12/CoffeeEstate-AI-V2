import os
import shutil
import uuid
import json
from pathlib import Path
from typing import List, Optional
from datetime import datetime

from fastapi import APIRouter, Depends, UploadFile, File, Form, HTTPException, status
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db
from app.models.estate import Section
from app.models.analysis import ImageAnalysis
from app.models.user import User
from app.schemas.analysis import ImageAnalysisResponse, BatchAnalysisResponse, ModelStatusResponse
from app.services.vision_service import vision_service
from app.services.sample_data import seed_sample_data_if_empty, clear_all_sample_data
from app.auth import get_optional_current_user, get_current_user

router = APIRouter(prefix="/api/analysis", tags=["Ripeness Analysis & CV"])
sample_data_router = APIRouter(prefix="/api/sample-data", tags=["Sample Data"])

ALLOWED_EXTENSIONS = {".jpg", ".jpeg", ".png"}
MAX_FILE_SIZE = 15 * 1024 * 1024  # 15 MB

@router.get("/model-status", response_model=ModelStatusResponse)
def get_model_status():
    """Returns AI model version, training weights status and model specifications."""
    return vision_service.get_model_status_info()

@router.post("/upload", response_model=BatchAnalysisResponse)
async def upload_and_analyze_cherries(
    section_id: Optional[int] = Form(default=None),
    confidence_threshold: float = Form(default=0.25),
    analysis_mode: str = Form(default="branch"),
    files: List[UploadFile] = File(...),
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """
    Accepts one or more coffee cherry photos, verifies image validity,
    runs the YOLO cherry detection pipeline (Branch close-up or Whole-Plant canopy),
    calculates class counts, ripeness percentages, and maturity stage.

    Rule 1: Anyone can access and analyze photos without login.
    Rule 2: If the user is logged in, the analysis record is saved to the SQLite database.

    """
    section: Optional[Section] = None
    if section_id is not None:
        section = db.query(Section).filter(Section.id == section_id).first()

    inferred_variety = section.coffee_variety if section else "Arabica (Selection 795)"

    effective_section_id: Optional[int] = section_id
    if effective_section_id is None:
        first_section = db.query(Section).order_by(Section.id).first()
        if first_section:
            effective_section_id = first_section.id

    if not files:
        raise HTTPException(status_code=400, detail="No image files uploaded")

    individual_results = []
    total_cherries_accum = 0
    red_counts_accum = 0
    yellow_counts_accum = 0
    green_counts_accum = 0
    dark_brown_counts_accum = 0
    low_vis_counts_accum = 0

    for file in files:
        # 1. Validate file extension
        ext = Path(file.filename).suffix.lower()
        if ext not in ALLOWED_EXTENSIONS:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported file '{file.filename}'. Allowed formats: {', '.join(ALLOWED_EXTENSIONS)}"
            )

        # 2. Save file safely
        unique_id = uuid.uuid4().hex[:10]
        safe_filename = f"{datetime.now().strftime('%Y%m%d_%H%M%S')}_{unique_id}{ext}"
        raw_path = settings.UPLOAD_DIR / "raw" / safe_filename

        try:
            with open(raw_path, "wb") as buffer:
                content = await file.read()
                if len(content) > MAX_FILE_SIZE:
                    raise HTTPException(
                        status_code=400,
                        detail=f"File '{file.filename}' exceeds maximum allowed size of 15MB"
                    )
                buffer.write(content)
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(status_code=500, detail=f"Failed to save upload '{file.filename}': {str(e)}")

        # 3. Run Vision Pipeline & Validate Image Content
        try:
            analysis_dict = vision_service.analyze_image(
                image_path=raw_path,
                confidence_threshold=confidence_threshold,
                save_annotated=(analysis_mode == "branch"),
                variety=inferred_variety,
                mode=analysis_mode
            )
            
            # Additional direct validation check inside the route loop for branch mode
            evaluable_count = (
                analysis_dict.get("red_count", 0) +
                analysis_dict.get("yellow_count", 0) +
                analysis_dict.get("green_count", 0) +
                analysis_dict.get("dark_brown_count", 0)
            )
            if analysis_mode == "branch" and (analysis_dict.get("total_cherries", 0) < 3 or evaluable_count == 0):
                raise ValueError("This does not look like a correct image of coffee cherries. Please upload a valid coffee branch image.")

        except ValueError as ve:
            # Clean up the raw file if it fails validation
            if os.path.exists(raw_path):
                try:
                    os.remove(raw_path)
                except Exception:
                    pass
            raise HTTPException(
                status_code=400,
                detail=str(ve)
            )
        except Exception as e:
            if isinstance(e, HTTPException):
                raise e
            raise HTTPException(
                status_code=422,
                detail=f"Error analyzing image '{file.filename}': {str(e)}"
            )

        # 4. Save ImageAnalysis record to Database ONLY if user is logged in
        saved_id = 0
        is_sample = False
        if current_user:
            db_analysis = ImageAnalysis(
                user_id=current_user.id,
                section_id=effective_section_id,
                original_filename=file.filename,
                storage_path=f"uploads/raw/{safe_filename}",
                annotated_storage_path=analysis_dict.get("annotated_storage_path"),
                uploaded_at=datetime.utcnow(),
                analyzed_at=datetime.utcnow(),
                analysis_mode=analysis_mode,
                model_version=analysis_dict["model_version"],
                model_status_at_inference=analysis_dict["model_status_at_inference"],
                inference_time_ms=analysis_dict["inference_time_ms"],
                confidence_threshold=confidence_threshold,
                total_cherries=analysis_dict["total_cherries"],
                dark_brown_count=analysis_dict["dark_brown_count"],
                green_count=analysis_dict["green_count"],
                low_visibility_count=analysis_dict["low_visibility_count"],
                red_count=analysis_dict["red_count"],
                yellow_count=analysis_dict["yellow_count"],
                dark_brown_pct=analysis_dict["dark_brown_pct"],
                green_pct=analysis_dict["green_pct"],
                low_visibility_pct=analysis_dict["low_visibility_pct"],
                red_pct=analysis_dict["red_pct"],
                yellow_pct=analysis_dict["yellow_pct"],
                maturity_stage=analysis_dict["maturity_stage"],
                maturity_notes=analysis_dict["maturity_notes"],
                detections_json=json.dumps(analysis_dict.get("detections", [])),
                is_sample_data=False
            )
            db.add(db_analysis)
            db.commit()
            db.refresh(db_analysis)
            saved_id = db_analysis.id
            is_sample = db_analysis.is_sample_data


        # Aggregate counts
        total_cherries_accum += analysis_dict["total_cherries"]
        red_counts_accum += analysis_dict["red_count"]
        yellow_counts_accum += analysis_dict["yellow_count"]
        green_counts_accum += analysis_dict["green_count"]
        dark_brown_counts_accum += analysis_dict["dark_brown_count"]
        low_vis_counts_accum += analysis_dict["low_visibility_count"]

        # Parse response item
        individual_results.append(
            ImageAnalysisResponse(
                id=saved_id,
                user_id=current_user.id if current_user else None,
                section_id=effective_section_id,
                original_filename=file.filename,
                storage_path=f"uploads/raw/{safe_filename}",
                annotated_storage_path=analysis_dict.get("annotated_storage_path"),
                uploaded_at=datetime.utcnow(),
                analyzed_at=datetime.utcnow(),
                analysis_mode=analysis_mode,
                model_version=analysis_dict["model_version"],
                model_status_at_inference=analysis_dict["model_status_at_inference"],
                inference_time_ms=analysis_dict["inference_time_ms"],
                confidence_threshold=confidence_threshold,
                total_cherries=analysis_dict["total_cherries"],
                dark_brown_count=analysis_dict["dark_brown_count"],
                green_count=analysis_dict["green_count"],
                low_visibility_count=analysis_dict["low_visibility_count"],
                red_count=analysis_dict["red_count"],
                yellow_count=analysis_dict["yellow_count"],
                dark_brown_pct=analysis_dict["dark_brown_pct"],
                green_pct=analysis_dict["green_pct"],
                low_visibility_pct=analysis_dict["low_visibility_pct"],
                red_pct=analysis_dict["red_pct"],
                yellow_pct=analysis_dict["yellow_pct"],
                maturity_stage=analysis_dict["maturity_stage"],
                maturity_notes=analysis_dict["maturity_notes"],

                detections=analysis_dict.get("detections"),
                is_sample_data=is_sample,
                evaluable_cherries=analysis_dict.get("evaluable_cherries"),
                evaluable_red_pct=analysis_dict.get("evaluable_red_pct"),
                evaluable_yellow_pct=analysis_dict.get("evaluable_yellow_pct"),
                evaluable_green_pct=analysis_dict.get("evaluable_green_pct"),
                evaluable_dark_brown_pct=analysis_dict.get("evaluable_dark_brown_pct"),
                uncertain_cherries=analysis_dict.get("uncertain_cherries"),
                uncertain_pct=analysis_dict.get("uncertain_pct"),
                mean_confidence=analysis_dict.get("mean_confidence"),
                denominator_note=analysis_dict.get("denominator_note"),
                yield_disclaimer=analysis_dict.get("yield_disclaimer"),
                plant_ripeness_score=analysis_dict.get("plant_ripeness_score"),
                tiles_processed=analysis_dict.get("tiles_processed"),
                quality_assessment=analysis_dict.get("quality_assessment")
            )
        )

    # Aggregate batch percentages
    if total_cherries_accum > 0:
        avg_red = round((red_counts_accum / total_cherries_accum) * 100, 1)
        avg_yellow = round((yellow_counts_accum / total_cherries_accum) * 100, 1)
        avg_green = round((green_counts_accum / total_cherries_accum) * 100, 1)
        avg_brown = round((dark_brown_counts_accum / total_cherries_accum) * 100, 1)
        avg_lowvis = round((low_vis_counts_accum / total_cherries_accum) * 100, 1)
    else:
        avg_red = avg_yellow = avg_green = avg_brown = avg_lowvis = 0.0

    aggregated_evaluable = red_counts_accum + yellow_counts_accum + green_counts_accum + dark_brown_counts_accum
    if aggregated_evaluable > 0:
        avg_eval_red = round((red_counts_accum / aggregated_evaluable) * 100, 1)
        avg_eval_yellow = round((yellow_counts_accum / aggregated_evaluable) * 100, 1)
        avg_eval_green = round((green_counts_accum / aggregated_evaluable) * 100, 1)
        avg_eval_brown = round((dark_brown_counts_accum / aggregated_evaluable) * 100, 1)
    else:
        avg_eval_red = avg_eval_yellow = avg_eval_green = avg_eval_brown = 0.0

    avg_uncertain_pct = avg_lowvis

    if analysis_mode == "whole_plant":
        stage, note = vision_service._interpret_whole_plant_maturity(
            evaluable_red_pct=avg_eval_red,
            evaluable_yellow_pct=avg_eval_yellow,
            evaluable_green_pct=avg_eval_green,
            evaluable_dark_brown_pct=avg_eval_brown,
            evaluable_cherries=aggregated_evaluable,
            total_cherries=total_cherries_accum,
            variety=inferred_variety
        )
    else:
        stage, note = vision_service._interpret_maturity(
            red_pct=avg_red,
            yellow_pct=avg_yellow,
            green_pct=avg_green,
            dark_brown_pct=avg_brown,
            low_vis_pct=avg_lowvis,
            evaluable_red_pct=avg_eval_red,
            evaluable_yellow_pct=avg_eval_yellow,
            evaluable_green_pct=avg_eval_green,
            evaluable_dark_brown_pct=avg_eval_brown,
            evaluable_cherries=aggregated_evaluable,
            total_cherries=total_cherries_accum,
            variety=inferred_variety
        )

    plant_ripeness_calc = min(100, round((avg_eval_red * 1.0) + (avg_eval_yellow * 0.45)))

    return BatchAnalysisResponse(
        section_id=effective_section_id,
        section_name=section.name if section else None,

        analysis_mode=analysis_mode,
        total_images_analyzed=len(files),
        aggregated_total_cherries=total_cherries_accum,
        avg_red_pct=avg_red,
        avg_yellow_pct=avg_yellow,
        avg_green_pct=avg_green,
        avg_dark_brown_pct=avg_brown,
        avg_low_visibility_pct=avg_lowvis,
        overall_maturity_stage=stage,
        maturity_recommendation=note,
        individual_results=individual_results,
        aggregated_evaluable_cherries=aggregated_evaluable,
        avg_evaluable_red_pct=avg_eval_red,
        avg_evaluable_yellow_pct=avg_eval_yellow,
        avg_evaluable_green_pct=avg_eval_green,
        avg_evaluable_dark_brown_pct=avg_eval_brown,
        avg_uncertain_pct=avg_uncertain_pct,
        plant_ripeness_score=plant_ripeness_calc,
        photographic_yield_disclaimer=(
            "WHOLE-CANOPY SAMPLING NOTICE: Percentages represent visible canopy clusters. "
            "Interior cherries obscured by dense foliage are not included in surface optical estimation."
            if analysis_mode == "whole_plant" else
            "PHOTOGRAPHIC SAMPLING NOTICE: Cherry counts are detected from 2D photographs of canopy branches. "
            "They indicate sampling density and maturity distribution, not total plantation yield per acre."
        )
    )

@router.get("/history", response_model=List[ImageAnalysisResponse])
def get_analysis_history(
    section_id: Optional[int] = None,
    limit: int = 50,
    current_user: Optional[User] = Depends(get_optional_current_user),
    db: Session = Depends(get_db)
):
    """
    Fetches previously analyzed images and detection breakdowns.
    - If user is logged in: returns only that user's saved scans.
    - If user is guest (not logged in): returns empty list.
    """
    if not current_user:
        return []

    query = db.query(ImageAnalysis).filter(ImageAnalysis.user_id == current_user.id)
    if section_id:
        query = query.filter(ImageAnalysis.section_id == section_id)
    analyses = query.order_by(ImageAnalysis.analyzed_at.desc()).limit(limit).all()

    results = []
    for a in analyses:
        detections = []
        if a.detections_json:
            try:
                detections = json.loads(a.detections_json)
            except Exception:
                pass

        eval_count = a.red_count + a.yellow_count + a.green_count + a.dark_brown_count
        eval_red = round((a.red_count / eval_count) * 100, 2) if eval_count > 0 else 0.0
        eval_yellow = round((a.yellow_count / eval_count) * 100, 2) if eval_count > 0 else 0.0
        eval_green = round((a.green_count / eval_count) * 100, 2) if eval_count > 0 else 0.0
        eval_brown = round((a.dark_brown_count / eval_count) * 100, 2) if eval_count > 0 else 0.0
        p_score = min(100, round((eval_red * 1.0) + (eval_yellow * 0.45)))
        a_mode = getattr(a, "analysis_mode", "branch") or "branch"

        results.append(
            ImageAnalysisResponse(
                id=a.id,
                user_id=a.user_id,
                section_id=a.section_id,
                original_filename=a.original_filename,
                storage_path=a.storage_path,
                annotated_storage_path=a.annotated_storage_path,
                uploaded_at=a.uploaded_at,
                analyzed_at=a.analyzed_at,
                analysis_mode=a_mode,
                model_version=a.model_version,
                model_status_at_inference=a.model_status_at_inference,
                inference_time_ms=a.inference_time_ms,
                confidence_threshold=a.confidence_threshold,
                total_cherries=a.total_cherries,
                dark_brown_count=a.dark_brown_count,
                green_count=a.green_count,
                low_visibility_count=a.low_visibility_count,
                red_count=a.red_count,
                yellow_count=a.yellow_count,
                dark_brown_pct=a.dark_brown_pct,
                green_pct=a.green_pct,
                low_visibility_pct=a.low_visibility_pct,
                red_pct=a.red_pct,
                yellow_pct=a.yellow_pct,
                maturity_stage=a.maturity_stage,
                maturity_notes=a.maturity_notes,
                detections=detections,
                is_sample_data=a.is_sample_data,
                evaluable_cherries=eval_count,
                evaluable_red_pct=eval_red,
                evaluable_yellow_pct=eval_yellow,
                evaluable_green_pct=eval_green,
                evaluable_dark_brown_pct=eval_brown,
                uncertain_cherries=a.low_visibility_count,
                uncertain_pct=a.low_visibility_pct,
                plant_ripeness_score=p_score
            )
        )
    return results

@router.delete("/history/{analysis_id}")
def delete_analysis(
    analysis_id: int,
    current_user: User = Depends(get_current_user),
    db: Session = Depends(get_db)
):
    """Deletes an analysis record belonging to the logged-in user and its corresponding images."""
    record = db.query(ImageAnalysis).filter(
        ImageAnalysis.id == analysis_id,
        ImageAnalysis.user_id == current_user.id
    ).first()
    if not record:
        raise HTTPException(status_code=404, detail="Analysis record not found or unauthorized")

    try:
        if record.storage_path and os.path.exists(record.storage_path):
            os.remove(record.storage_path)
        if record.annotated_storage_path and os.path.exists(record.annotated_storage_path):
            os.remove(record.annotated_storage_path)
    except Exception as e:
        print(f"File cleanup note: {e}")

    db.delete(record)
    db.commit()
    return {"message": "Analysis deleted successfully"}

# Sample data management endpoints
@sample_data_router.post("/seed")
def seed_sample_data(db: Session = Depends(get_db)):
    """Seeds demonstration sample analyses if database is empty."""
    seed_sample_data_if_empty(db)
    return {"message": "Sample demonstration data loaded successfully."}

@sample_data_router.post("/clear")
def clear_sample_data(db: Session = Depends(get_db)):
    """Removes demonstration sample analyses."""
    clear_all_sample_data(db)
    return {"message": "All sample demonstration data removed successfully."}
