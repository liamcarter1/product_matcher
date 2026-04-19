"""
Document ingestion API endpoint.
Handles file uploads and processing into the vector database.
"""

import os
import time
import tempfile
from typing import Optional

from fastapi import APIRouter, HTTPException, UploadFile, File, Depends
from fastapi.responses import JSONResponse

from ..config import get_settings
from ..models.schemas import IngestResponse, DeleteResponse
from ..services.document_loader import DocumentLoader
from ..services.pinecone_service import PineconeService

router = APIRouter()

# Allowed file extensions
ALLOWED_EXTENSIONS = {".pdf", ".xlsx", ".xls", ".csv"}

# Service instances
_document_loader: Optional[DocumentLoader] = None
_pinecone_service: Optional[PineconeService] = None


def get_document_loader() -> DocumentLoader:
    """Get or create the document loader instance."""
    global _document_loader
    if _document_loader is None:
        _document_loader = DocumentLoader()
    return _document_loader


def get_pinecone_service() -> PineconeService:
    """Get or create the Pinecone service instance."""
    global _pinecone_service
    if _pinecone_service is None:
        _pinecone_service = PineconeService()
    return _pinecone_service


def validate_file_extension(filename: str) -> str:
    """Validate file extension and return it."""
    ext = os.path.splitext(filename)[1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=f"File type '{ext}' not supported. Allowed: {', '.join(ALLOWED_EXTENSIONS)}"
        )
    return ext


@router.post("/ingest", response_model=IngestResponse)
async def ingest_document(
    file: UploadFile = File(...),
    document_loader: DocumentLoader = Depends(get_document_loader),
    pinecone_service: PineconeService = Depends(get_pinecone_service)
):
    """
    Upload and ingest a document into the vector database.

    Supported file types:
    - PDF (.pdf) - Product guides, manuals, documentation
    - Excel (.xlsx, .xls) - Parts cross-reference, specifications
    - CSV (.csv) - Parts data, specifications

    The system automatically detects the document type and extracts
    appropriate metadata for optimal retrieval.
    """
    start_time = time.time()

    # Validate file extension
    ext = validate_file_extension(file.filename)

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        # Load and process the document
        documents = document_loader.load_file(tmp_path)

        if not documents:
            raise HTTPException(
                status_code=400,
                detail="No content could be extracted from the file"
            )

        # Update metadata with original filename
        for doc in documents:
            doc.metadata["source_file"] = file.filename
            doc.metadata["original_filename"] = file.filename

        # Upsert to Pinecone
        docs_added = pinecone_service.upsert_documents(documents)

        processing_time = time.time() - start_time

        return IngestResponse(
            status="success",
            documents_added=docs_added,
            file=file.filename,
            file_type=ext[1:],  # Remove the dot
            processing_time_seconds=round(processing_time, 2)
        )

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing file: {str(e)}"
        )
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)


@router.delete("/ingest/{filename}", response_model=DeleteResponse)
async def delete_document(
    filename: str,
    pinecone_service: PineconeService = Depends(get_pinecone_service)
):
    """
    Delete all documents from a specific source file.

    This removes all vector embeddings associated with the given filename
    from the database.
    """
    try:
        docs_removed = pinecone_service.delete_by_source(filename)

        return DeleteResponse(
            status="deleted",
            documents_removed=docs_removed,
            file=filename
        )

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error deleting documents: {str(e)}"
        )


@router.get("/ingest/stats")
async def get_ingestion_stats(
    pinecone_service: PineconeService = Depends(get_pinecone_service)
):
    """
    Get statistics about ingested documents.

    Returns the total number of vectors and breakdown by namespace.
    """
    try:
        stats = pinecone_service.get_stats()
        return {"status": "success", "stats": stats}
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error fetching stats: {str(e)}"
        )


@router.get("/ingest/info")
async def get_file_info(
    file: UploadFile = File(...),
    document_loader: DocumentLoader = Depends(get_document_loader)
):
    """
    Get information about a file without ingesting it.

    Returns file metadata like page count, row count, columns, etc.
    """
    ext = validate_file_extension(file.filename)

    # Save file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=ext) as tmp:
        content = await file.read()
        tmp.write(content)
        tmp_path = tmp.name

    try:
        info = document_loader.get_file_info(tmp_path)
        info["filename"] = file.filename
        return {"status": "success", "info": info}

    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error analyzing file: {str(e)}"
        )
    finally:
        if os.path.exists(tmp_path):
            os.unlink(tmp_path)
