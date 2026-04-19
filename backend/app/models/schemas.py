"""
Pydantic schemas for API requests and responses.
"""

from typing import Dict, List, Optional
from enum import Enum
from pydantic import BaseModel, Field
from datetime import datetime


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


class SourceDocument(BaseModel):
    """Source document reference."""
    file: str
    type: str
    chunk_id: Optional[str] = None


# Nameplate Extraction
class NameplateData(BaseModel):
    """Extracted data from a competitor nameplate image."""
    model_number: Optional[str] = None
    manufacturer: Optional[str] = None
    specifications: Dict[str, str] = {}
    raw_text: str
    confidence: float = Field(..., ge=0, le=100)


# Chat Models
class ChatRequest(BaseModel):
    """Chat request payload."""
    message: str = Field(..., min_length=1, max_length=2000)
    session_id: Optional[str] = None
    distributor_id: Optional[str] = None

    class Config:
        json_schema_extra = {
            "example": {
                "message": "What Danfoss part replaces CompetitorA XYZ-456?",
                "session_id": "abc123",
                "distributor_id": "dist001"
            }
        }


class ChatResponse(BaseModel):
    """Chat response payload."""
    response: str
    confidence: float = Field(..., ge=0, le=100)
    confidence_level: ConfidenceLevel
    session_id: str
    sources: Optional[List[SourceDocument]] = None
    disclaimer: Optional[str] = None
    parsed_nameplate: Optional[NameplateData] = None

    class Config:
        json_schema_extra = {
            "example": {
                "response": "Danfoss part ABC-123 is equivalent to CompetitorA XYZ-456...",
                "confidence": 85.5,
                "confidence_level": "high",
                "session_id": "abc123",
                "sources": [{"file": "parts_crossref.xlsx", "type": "excel"}]
            }
        }


# Auth Models
class LoginRequest(BaseModel):
    """Login request payload."""
    username: str
    password: str


class LoginResponse(BaseModel):
    """Login response payload."""
    access_token: str
    token_type: str = "bearer"


class UserInfo(BaseModel):
    """User information payload."""
    distributor_id: str
    name: str
    region: Optional[str] = None


# Ingest Models
class IngestResponse(BaseModel):
    """Document ingest response payload."""
    status: str
    documents_added: int
    file: str
    file_type: str
    processing_time_seconds: float


class DeleteResponse(BaseModel):
    """Document delete response payload."""
    status: str
    documents_removed: int
    file: str


# Health Models
class HealthResponse(BaseModel):
    """Health check response payload."""
    status: str
    app: str
    version: str


# Message History
class Message(BaseModel):
    """Chat message."""
    role: str  # "user" or "assistant"
    content: str
    timestamp: datetime = Field(default_factory=datetime.utcnow)
    confidence: Optional[float] = None


class ConversationSession(BaseModel):
    """Conversation session with history."""
    session_id: str
    distributor_id: Optional[str] = None
    messages: List[Message] = []
    created_at: datetime = Field(default_factory=datetime.utcnow)
    last_activity: datetime = Field(default_factory=datetime.utcnow)
