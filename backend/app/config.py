"""
Configuration settings for the Danfoss RAG Chatbot.
"""

from functools import lru_cache
from typing import List
from pydantic_settings import BaseSettings
from pydantic import Field


class Settings(BaseSettings):
    """Application settings loaded from environment variables."""

    # OpenAI Configuration
    openai_api_key: str = Field(..., env="OPENAI_API_KEY")
    openai_model: str = Field(default="gpt-4o", env="OPENAI_MODEL")
    openai_embedding_model: str = Field(
        default="text-embedding-3-small",
        env="OPENAI_EMBEDDING_MODEL"
    )

    # Pinecone Configuration
    pinecone_api_key: str = Field(..., env="PINECONE_API_KEY")
    pinecone_index: str = Field(default="danfoss-products", env="PINECONE_INDEX")
    pinecone_environment: str = Field(default="us-east-1", env="PINECONE_ENVIRONMENT")

    # JWT Authentication
    jwt_secret: str = Field(default="change-me-in-production", env="JWT_SECRET")
    jwt_algorithm: str = Field(default="HS256", env="JWT_ALGORITHM")
    jwt_expiration_hours: int = Field(default=24, env="JWT_EXPIRATION_HOURS")

    # CORS Configuration
    cors_origins: List[str] = Field(
        default=["http://localhost:3000", "http://localhost:8000"],
        env="CORS_ORIGINS"
    )

    # Application Settings
    app_name: str = "Danfoss RAG Chatbot"
    app_version: str = "1.0.0"
    debug: bool = Field(default=False, env="DEBUG")

    # Vector DB Settings
    embedding_dimension: int = 1536  # text-embedding-3-small dimension
    default_top_k: int = 5

    # Confidence Thresholds
    high_confidence_threshold: float = 80.0
    medium_confidence_threshold: float = 50.0

    # Fuzzy Matching
    fuzzy_match_threshold: float = 85.0

    # Image Processing
    max_image_size_mb: int = 10
    allowed_image_types: List[str] = [
        "image/jpeg", "image/png", "image/webp"
    ]
    vision_min_confidence: float = 50.0

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"
        extra = "ignore"


@lru_cache()
def get_settings() -> Settings:
    """Get cached settings instance."""
    return Settings()
