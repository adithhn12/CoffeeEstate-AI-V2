import os
from pathlib import Path
from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings
from app.database import engine, Base, SessionLocal
from app.database_migrations import run_database_migrations
from app.services.sample_data import seed_sample_data_if_empty

# Import routers
from app.routes.health import router as health_router
from app.routes.analysis import router as analysis_router, sample_data_router
from app.routes.auth import router as auth_router

# Initialize database schema
Base.metadata.create_all(bind=engine)

# Safely run SQLite column migrations
try:
    run_database_migrations()
except Exception as e:
    print(f"Migration notice: {e}")

# Seed initial data on startup if database is empty
with SessionLocal() as db:
    try:
        seed_sample_data_if_empty(db)
    except Exception as e:
        print(f"Startup seed notice: {e}")

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Coffee Estate AI — Cherry Ripeness Analysis and AI Model Status"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount uploads directory for static image serving
app.mount("/uploads", StaticFiles(directory=str(settings.UPLOAD_DIR)), name="uploads")

# Include API Routers
app.include_router(health_router)
app.include_router(analysis_router)
app.include_router(sample_data_router)
app.include_router(auth_router)

@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    print(f"Unhandled error processing {request.url}: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "An internal server error occurred while processing the request."}
    )

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)
