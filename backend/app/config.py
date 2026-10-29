import os
from pathlib import Path
from typing import Dict

class Settings:
    APP_NAME: str = os.getenv("APP_NAME", "Coffee Estate AI")
    APP_VERSION: str = "1.0.0"
    ENVIRONMENT: str = os.getenv("ENVIRONMENT", "development")
    DEBUG: bool = os.getenv("DEBUG", "True").lower() == "true"
    
    # Base directories
    BASE_DIR: Path = Path(__file__).resolve().parent.parent.parent
    UPLOAD_DIR: Path = BASE_DIR / "uploads"
    DATASET_DIR: Path = BASE_DIR / "datasets" / "croppie_coffee"
    MODEL_WEIGHTS_DIR: Path = BASE_DIR / "ml" / "weights"
    
    # Database
    DATABASE_URL: str = os.getenv("DATABASE_URL", f"sqlite:///{BASE_DIR}/coffee_estate.db")
    
    # Model configuration
    DEFAULT_MODEL_PATH: str = str(MODEL_WEIGHTS_DIR / "best.pt")
    PRETRAINED_FALLBACK: str = "yolov8s.pt"
    CONFIDENCE_THRESHOLD: float = float(os.getenv("CONFIDENCE_THRESHOLD", "0.25"))
    IOU_THRESHOLD: float = float(os.getenv("IOU_THRESHOLD", "0.55"))  # 0.55 tuned for tightly packed coffee clusters
    MAX_DETECTIONS: int = int(os.getenv("MAX_DETECTIONS", "500"))      # 500 accommodates dense branch clusters
    
    # Coorg coordinates (Madikeri / Kodagu, Karnataka)
    DEFAULT_LATITUDE: float = float(os.getenv("DEFAULT_LATITUDE", "12.4244"))
    DEFAULT_LONGITUDE: float = float(os.getenv("DEFAULT_LONGITUDE", "75.7382"))
    DEFAULT_LOCATION_NAME: str = os.getenv("DEFAULT_LOCATION_NAME", "Madikeri, Kodagu, Karnataka")
    
    # Class mappings
    CLASS_NAMES: Dict[int, str] = {
        0: "dark_brown_cherry",
        1: "green_cherry",
        2: "low_visibility_unsure",
        3: "red_cherry",
        4: "yellow_cherry"
    }

settings = Settings()

# Ensure required directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.MODEL_WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
(settings.UPLOAD_DIR / "annotated").mkdir(parents=True, exist_ok=True)
(settings.UPLOAD_DIR / "raw").mkdir(parents=True, exist_ok=True)
(settings.UPLOAD_DIR / "coorg_local").mkdir(parents=True, exist_ok=True)
