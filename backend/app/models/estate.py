from datetime import datetime
from sqlalchemy import (
    Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
)
from sqlalchemy.orm import relationship
from app.database import Base

class Estate(Base):
    __tablename__ = "estates"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    location = Column(String(200), default="Madikeri, Kodagu, Karnataka")
    total_acres = Column(Float, default=50.0)
    total_hectares = Column(Float, nullable=True)
    area_unit = Column(String(20), default="acres")
    description = Column(Text, nullable=True)
    boundary_geojson = Column(Text, nullable=True)
    manager_name = Column(String(100), default="H R Nagesh")
    contact_phone = Column(String(20), default="+91 98450 12345")
    latitude = Column(Float, default=12.4244)
    longitude = Column(Float, default=75.7382)
    elevation_meters = Column(Float, default=1150.0)
    is_sample_data = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    sections = relationship("Section", back_populates="estate", cascade="all, delete-orphan")


class Section(Base):
    __tablename__ = "sections"

    id = Column(Integer, primary_key=True, index=True)
    estate_id = Column(Integer, ForeignKey("estates.id", ondelete="CASCADE"), nullable=False)
    name = Column(String(100), nullable=False)
    section_code = Column(String(20), nullable=False, index=True)
    area_acres = Column(Float, nullable=False, default=5.0)
    area_hectares = Column(Float, nullable=True)
    color = Column(String(20), default="#15803d")
    boundary_geojson = Column(Text, nullable=True)
    coffee_variety = Column(String(100), default="Arabica (Selection 795)")
    elevation_meters = Column(Float, default=1120.0)
    slope_aspect = Column(String(50), default="North-East Facing")
    soil_type = Column(String(100), default="Red Clay Loam with Humus")
    planting_year = Column(Integer, default=2015)
    shade_tree_species = Column(String(150), default="Silver Oak, Fig, Rosewood, Albizia")
    target_plants_per_acre = Column(Integer, default=450)
    available_workers = Column(Integer, default=4)
    min_workers_needed = Column(Integer, default=2)
    max_workers_allowed = Column(Integer, default=12)
    status = Column(String(50), default="active")
    notes = Column(Text, nullable=True)
    is_sample_data = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    # Relationships
    estate = relationship("Estate", back_populates="sections")
    image_analyses = relationship("ImageAnalysis", back_populates="section", cascade="all, delete-orphan")
