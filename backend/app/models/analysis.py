from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from app.database import Base

class ImageAnalysis(Base):
    __tablename__ = "image_analyses"

    id = Column(Integer, primary_key=True, index=True)
    user_id = Column(Integer, ForeignKey("users.id", ondelete="CASCADE"), nullable=True, index=True)
    section_id = Column(Integer, ForeignKey("sections.id", ondelete="CASCADE"), nullable=True)
    original_filename = Column(String(255), nullable=False)
    storage_path = Column(String(500), nullable=False)
    annotated_storage_path = Column(String(500), nullable=True)
    
    # Metadata
    uploaded_at = Column(DateTime, default=datetime.utcnow)
    analyzed_at = Column(DateTime, default=datetime.utcnow)
    analysis_mode = Column(String(50), default="branch")  # 'branch' or 'whole_plant'
    model_version = Column(String(100), default="YOLOv8s-Croppie-v1.0")
    model_status_at_inference = Column(String(100), default="Trained on public dataset")
    inference_time_ms = Column(Float, default=0.0)
    confidence_threshold = Column(Float, default=0.25)
    
    # Counts
    total_cherries = Column(Integer, default=0)
    dark_brown_count = Column(Integer, default=0)
    green_count = Column(Integer, default=0)
    low_visibility_count = Column(Integer, default=0)
    red_count = Column(Integer, default=0)
    yellow_count = Column(Integer, default=0)
    
    # Percentages
    dark_brown_pct = Column(Float, default=0.0)
    green_pct = Column(Float, default=0.0)
    low_visibility_pct = Column(Float, default=0.0)
    red_pct = Column(Float, default=0.0)
    yellow_pct = Column(Float, default=0.0)
    
    # Maturity interpretation
    maturity_stage = Column(String(50), default="Optimal Harvest")
    maturity_notes = Column(Text, nullable=True)
    
    # Detections list JSON
    detections_json = Column(Text, nullable=True)
    is_sample_data = Column(Boolean, default=False)

    # Relationships
    section = relationship("Section", back_populates="image_analyses")
    user = relationship("User", back_populates="image_analyses")
