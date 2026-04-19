"""
FastAPI main application entry point for Danfoss RAG Chatbot.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from contextlib import asynccontextmanager
import os

from .config import get_settings
from .routers import chat, auth, ingest


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan events."""
    # Startup
    settings = get_settings()
    print(f"Starting {settings.app_name} v{settings.app_version}")
    yield
    # Shutdown
    print("Shutting down application")


def create_app() -> FastAPI:
    """Create and configure the FastAPI application."""
    settings = get_settings()

    app = FastAPI(
        title=settings.app_name,
        version=settings.app_version,
        description="RAG-powered chatbot for Danfoss distributor product lookups",
        lifespan=lifespan,
    )

    # Configure CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins + ["*"],  # Allow all for widget embedding
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Include routers
    app.include_router(chat.router, prefix="/api", tags=["Chat"])
    app.include_router(auth.router, prefix="/api/auth", tags=["Authentication"])
    app.include_router(ingest.router, prefix="/api", tags=["Document Ingestion"])

    # Mount static files for widget serving (if directory exists)
    # Check Docker path first (/app/frontend), then local dev path (../../frontend)
    frontend_candidates = [
        os.path.join(os.path.dirname(__file__), "..", "frontend"),      # Docker: /app/frontend
        os.path.join(os.path.dirname(__file__), "..", "..", "frontend"),  # Local dev
    ]
    for frontend_path in frontend_candidates:
        if os.path.exists(frontend_path):
            app.mount("/static", StaticFiles(directory=frontend_path), name="static")
            break

    @app.get("/health", tags=["Health"])
    async def health_check():
        """Health check endpoint."""
        return {
            "status": "healthy",
            "app": settings.app_name,
            "version": settings.app_version
        }

    @app.get("/", tags=["Root"])
    async def root():
        """Root endpoint with API information."""
        return {
            "message": f"Welcome to {settings.app_name}",
            "version": settings.app_version,
            "docs": "/docs",
            "health": "/health"
        }

    return app


# Create the app instance
app = create_app()
