"""
Main FastAPI Application Entry Point for Deep Polyp Segmentation System.
"""

import os
from fastapi import FastAPI, Request
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

# CORS Configuration
origins = os.environ.get(
    "CORS_ORIGINS",
    "http://localhost:3000,http://127.0.0.1:3000,http://localhost:8000"
).split(",")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[origin.strip() for origin in origins if origin.strip()],
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
        "disclaimer": "Academic research prototype only. Not for clinical diagnostic use."
    }


@app.exception_handler(Exception)
async def global_exception_handler(request: Request, exc: Exception):
    # Log internally and return clean, safe error message to client
    return JSONResponse(
        status_code=500,
        content={
            "error": "InternalServerError",
            "message": "An unexpected error occurred during processing. Please verify input data and retry."
        }
    )
