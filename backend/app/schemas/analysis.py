from typing import Optional, List, Dict, Any
from datetime import datetime
from pydantic import BaseModel, Field

class DetectionBox(BaseModel):
    class_id: int
    class_name: str
    confidence: float
    bbox: List[float]  # [x_center, y_center, width, height] normalized
    pixel_coords: Optional[List[int]] = None  # [x1, y1, x2, y2] in pixels

class ImageAnalysisResponse(BaseModel):
    id: Optional[int] = 0
    user_id: Optional[int] = None
    section_id: Optional[int] = None
    original_filename: str
    storage_path: str
    annotated_storage_path: Optional[str] = None
    uploaded_at: datetime
    analyzed_at: datetime
    analysis_mode: str = "branch"
    model_version: str
    model_status_at_inference: str
    inference_time_ms: float
    confidence_threshold: float
    
    total_cherries: int
    dark_brown_count: int
    green_count: int
    low_visibility_count: int
    red_count: int
    yellow_count: int
    
    dark_brown_pct: float
    green_pct: float
    low_visibility_pct: float
    red_pct: float
    yellow_pct: float
    
    maturity_stage: str
    maturity_notes: Optional[str] = None
    detections: Optional[List[DetectionBox]] = None
    is_sample_data: bool

    # Phase 4 Enriched Maturity Fields
    evaluable_cherries: Optional[int] = None
    evaluable_red_pct: Optional[float] = None
    evaluable_yellow_pct: Optional[float] = None
    evaluable_green_pct: Optional[float] = None
    evaluable_dark_brown_pct: Optional[float] = None
    uncertain_cherries: Optional[int] = None
    uncertain_pct: Optional[float] = None
    mean_confidence: Optional[float] = None
    denominator_note: Optional[str] = None
    yield_disclaimer: Optional[str] = None
    plant_ripeness_score: Optional[int] = None
    tiles_processed: Optional[int] = None
    quality_assessment: Optional[str] = None

    class Config:
        from_attributes = True

class BatchAnalysisResponse(BaseModel):
    section_id: Optional[int] = None
    section_name: Optional[str] = None
    analysis_mode: str = "branch"
    total_images_analyzed: int
    aggregated_total_cherries: int
    avg_red_pct: float
    avg_yellow_pct: float
    avg_green_pct: float
    avg_dark_brown_pct: float
    avg_low_visibility_pct: float
    overall_maturity_stage: str
    maturity_recommendation: str
    individual_results: List[ImageAnalysisResponse]

    # Phase 4 Enriched Batch Fields
    aggregated_evaluable_cherries: Optional[int] = None
    avg_evaluable_red_pct: Optional[float] = None
    avg_evaluable_yellow_pct: Optional[float] = None
    avg_evaluable_green_pct: Optional[float] = None
    avg_evaluable_dark_brown_pct: Optional[float] = None
    avg_uncertain_pct: Optional[float] = None
    photographic_yield_disclaimer: Optional[str] = None
    plant_ripeness_score: Optional[int] = None

class ModelStatusResponse(BaseModel):
    model_name: str
    model_version: str
    weights_path: str
    weights_exist: bool
    device: str
    cuda_available: bool
    status_label: str
    status_badge_color: str
    class_mappings: Dict[int, str]
    dataset_info: Dict[str, Any]
    scientific_disclaimer: str
    detection_parameters: Optional[Dict[str, Any]] = None
