"""
Main FastAPI Application Entry Point for Deep Polyp Segmentation System.
"""

import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from backend.app.api.endpoints import router as api_router

app = FastAPI(
    title="Deep Polyp Segmentation & Comparative Analysis API",
    description=(
        "Academic research prototype for automated gastrointestinal lesion segmentation, "
        "comparative evaluation (U-Net, LinkNet, SegFormer), and model benchmarking."
    ),
    version="2.0.0"
)

# CORS Configuration - permit local Next.js frontend on standard and fallback dev ports
origins_env = os.environ.get("CORS_ORIGINS", "")
default_origins = [
    "http://localhost:3000",
    "http://127.0.0.1:3000",
    "http://localhost:3001",
    "http://127.0.0.1:3001",
    "http://localhost:8000",
    "http://127.0.0.1:8000",
]
origins = [o.strip() for o in origins_env.split(",") if o.strip()] if origins_env else default_origins

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_origin_regex=r"^https?://(localhost|127\.0\.0\.1)(:\d+)?$",
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Register API Router
app.include_router(api_router)


@app.get("/")
def read_root():
    return {
        "title": "Deep Polyp Segmentation & Comparative Analysis API",
        "version": "2.0.0",
        "docs_url": "/docs",
        "health_check": "/api/health",
        "training_status": "/api/train/status",
        "disclaimer": "Academic research prototype only. Not for clinical diagnostic use."
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    if isinstance(exc, HTTPException):
        return JSONResponse(
            status_code=exc.status_code,
            content={"detail": exc.detail}
        )
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during processing. Please verify input data and retry."
        }
    )
