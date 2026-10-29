from fastapi import APIRouter, Depends
from sqlalchemy.orm import Session
from sqlalchemy import text
from app.database import get_db
from app.services.vision_service import vision_service

router = APIRouter(prefix="/api/health", tags=["Health"])

@router.get("")
def health_check(db: Session = Depends(get_db)):
    """Health check endpoint for frontend status indicator & monitoring."""
    db_ok = False
    try:
        db.execute(text("SELECT 1"))
        db_ok = True
    except Exception as e:
        db_error = str(e)

    model_info = vision_service.get_model_status_info()

    return {
        "status": "healthy" if db_ok else "degraded",
        "database_connected": db_ok,
        "model_status": model_info["status_label"],
        "device": model_info["device"],
        "cuda_available": model_info["cuda_available"],
        "app_name": "Coffee Estate AI",
        "version": "1.0.0"
    }
