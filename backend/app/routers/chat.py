"""
Chat API endpoint for RAG queries.
"""

import logging
from fastapi import APIRouter, HTTPException, Depends, File, Form, UploadFile
from typing import Optional

from ..config import get_settings
from ..models.schemas import (
    ChatRequest, ChatResponse, ConfidenceLevel, NameplateData, SourceDocument,
)
from ..services.rag_service import RAGService
from ..services.vision_service import VisionService

logger = logging.getLogger(__name__)

router = APIRouter()

# Service instances (in production, use dependency injection)
_rag_service: Optional[RAGService] = None
_vision_service: Optional[VisionService] = None


def get_rag_service() -> RAGService:
    """Get or create the RAG service instance."""
    global _rag_service
    if _rag_service is None:
        _rag_service = RAGService()
    return _rag_service


def get_vision_service() -> VisionService:
    """Get or create the VisionService instance."""
    global _vision_service
    if _vision_service is None:
        _vision_service = VisionService()
    return _vision_service


@router.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    rag_service: RAGService = Depends(get_rag_service)
):
    """
    Process a chat message and return a RAG-powered response.

    - Accepts user message with optional session_id for conversation continuity
    - Returns response with confidence score and sources
    - Automatically generates session_id if not provided
    """
    try:
        result = await rag_service.query(
            message=request.message,
            session_id=request.session_id,
            distributor_id=request.distributor_id
        )

        # Convert sources to response model
        sources = [
            SourceDocument(
                file=s.get("file", "Unknown"),
                type=s.get("type", "Unknown"),
                chunk_id=s.get("chunk_id")
            )
            for s in result.get("sources", [])
        ]

        # Map confidence level
        level_str = result.get("confidence_level", "low")
        confidence_level = ConfidenceLevel(level_str)

        return ChatResponse(
            response=result["response"],
            confidence=result["confidence"],
            confidence_level=confidence_level,
            session_id=result["session_id"],
            sources=sources if sources else None,
            disclaimer=result.get("disclaimer")
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing chat request: {str(e)}"
        )


@router.post("/chat/image", response_model=ChatResponse)
async def chat_with_image(
    image: UploadFile = File(...),
    session_id: Optional[str] = Form(None),
    distributor_id: Optional[str] = Form(None),
    rag_service: RAGService = Depends(get_rag_service),
    vision_service: VisionService = Depends(get_vision_service),
):
    """
    Process a competitor nameplate image: extract data via GPT-4o vision,
    then search for Danfoss equivalents via the RAG pipeline.

    Returns the RAG response with parsed_nameplate populated for
    frontend confirmation flow.
    """
    settings = get_settings()

    # Validate image type
    if image.content_type not in settings.allowed_image_types:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported image type '{image.content_type}'. "
                   f"Allowed: {', '.join(settings.allowed_image_types)}",
        )

    # Read and validate image size
    image_bytes = await image.read()
    max_bytes = settings.max_image_size_mb * 1024 * 1024
    if len(image_bytes) > max_bytes:
        raise HTTPException(
            status_code=400,
            detail=f"Image too large. Maximum size is {settings.max_image_size_mb}MB.",
        )

    # Extract nameplate data via vision
    try:
        nameplate = await vision_service.extract_nameplate_data(
            image_bytes, image.content_type
        )
    except Exception as e:
        logger.error(f"Nameplate extraction failed: {e}")
        raise HTTPException(
            status_code=422,
            detail="Could not read the nameplate image. Please try a clearer photo.",
        )

    # If confidence is too low, return early with suggestion
    if nameplate.confidence < settings.vision_min_confidence:
        return ChatResponse(
            response=(
                "I couldn't clearly read the nameplate in this image. "
                "Please try taking a clearer, well-lit photo of the nameplate."
            ),
            confidence=nameplate.confidence,
            confidence_level=ConfidenceLevel.LOW,
            session_id=session_id or "",
            parsed_nameplate=nameplate,
        )

    # Build search message from extracted data
    parts = []
    if nameplate.manufacturer:
        parts.append(nameplate.manufacturer)
    if nameplate.model_number:
        parts.append(nameplate.model_number)

    if parts:
        search_message = f"Find Danfoss equivalent for {' '.join(parts)}"
    else:
        search_message = f"Find Danfoss equivalent for: {nameplate.raw_text[:200]}"

    # Run through existing RAG pipeline
    try:
        result = await rag_service.query(
            message=search_message,
            session_id=session_id,
            distributor_id=distributor_id,
        )

        sources = [
            SourceDocument(
                file=s.get("file", "Unknown"),
                type=s.get("type", "Unknown"),
                chunk_id=s.get("chunk_id"),
            )
            for s in result.get("sources", [])
        ]

        level_str = result.get("confidence_level", "low")
        confidence_level = ConfidenceLevel(level_str)

        return ChatResponse(
            response=result["response"],
            confidence=result["confidence"],
            confidence_level=confidence_level,
            session_id=result["session_id"],
            sources=sources if sources else None,
            disclaimer=result.get("disclaimer"),
            parsed_nameplate=nameplate,
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing image chat request: {str(e)}",
        )


@router.delete("/chat/session/{session_id}")
async def clear_session(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Clear a conversation session."""
    rag_service.clear_session(session_id)
    return {"status": "cleared", "session_id": session_id}


@router.get("/chat/session/{session_id}/history")
async def get_session_history(
    session_id: str,
    rag_service: RAGService = Depends(get_rag_service)
):
    """Get conversation history for a session."""
    history = rag_service.get_session_history(session_id)
    return {"session_id": session_id, "messages": history}
