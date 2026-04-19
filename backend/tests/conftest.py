"""
Pytest configuration and shared fixtures for RAG application tests.
"""

import os
import sys
import pytest
from unittest.mock import MagicMock, AsyncMock, patch
from pathlib import Path
import tempfile
import pandas as pd

# Add backend to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Set test environment variables before importing app modules
os.environ["OPENAI_API_KEY"] = "sk-test-key-not-real"
os.environ["PINECONE_API_KEY"] = "pc-test-key-not-real"
os.environ["PINECONE_INDEX"] = "test-index"
os.environ["JWT_SECRET"] = "test-secret"


@pytest.fixture
def mock_settings():
    """Create mock settings for testing."""
    from app.config import Settings

    return Settings(
        openai_api_key="sk-test-key",
        pinecone_api_key="pc-test-key",
        pinecone_index="test-index",
        jwt_secret="test-secret",
        high_confidence_threshold=80.0,
        medium_confidence_threshold=50.0,
        fuzzy_match_threshold=85.0,
        default_top_k=5,
        embedding_dimension=1536,
    )


@pytest.fixture
def sample_pdf_content():
    """Sample PDF-like content for testing."""
    return """
    Danfoss Product Manual

    Chapter 1: Introduction
    This manual covers the installation and operation of Danfoss industrial components.

    Chapter 2: Specifications
    Voltage Rating: 24V DC
    Current Rating: 5A
    Operating Temperature: -40 to 85 degrees Celsius

    Chapter 3: Installation
    Follow these steps for proper installation of the component.
    """


@pytest.fixture
def sample_parts_dataframe():
    """Sample parts cross-reference DataFrame."""
    return pd.DataFrame({
        "danfoss_part": ["ABC-123", "DEF-456", "GHI-789"],
        "competitor_brand": ["BrandA", "BrandB", "BrandC"],
        "competitor_part": ["XYZ-111", "XYZ-222", "XYZ-333"],
        "description": ["Valve Assembly", "Pressure Sensor", "Control Module"],
        "voltage": ["24V", "12V", "48V"],
    })


@pytest.fixture
def sample_general_dataframe():
    """Sample general data DataFrame (non-parts)."""
    return pd.DataFrame({
        "product_name": ["Widget A", "Widget B"],
        "category": ["Electronics", "Mechanical"],
        "price": [99.99, 149.99],
    })


@pytest.fixture
def temp_csv_file(sample_parts_dataframe):
    """Create a temporary CSV file for testing."""
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        sample_parts_dataframe.to_csv(f, index=False)
        temp_path = f.name
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def temp_excel_file(sample_parts_dataframe):
    """Create a temporary Excel file for testing."""
    with tempfile.NamedTemporaryFile(suffix='.xlsx', delete=False) as f:
        temp_path = f.name
    sample_parts_dataframe.to_excel(temp_path, index=False, engine='openpyxl')
    yield temp_path
    os.unlink(temp_path)


@pytest.fixture
def mock_openai_embeddings():
    """Mock OpenAI embeddings."""
    mock = MagicMock()
    # Return a fake embedding vector
    mock.embed_query.return_value = [0.1] * 1536
    mock.embed_documents.return_value = [[0.1] * 1536]
    return mock


@pytest.fixture
def mock_pinecone_index():
    """Mock Pinecone index."""
    mock_index = MagicMock()

    # Mock query response
    mock_match = MagicMock()
    mock_match.id = "test-id-1"
    mock_match.score = 0.85
    mock_match.metadata = {
        "text": "Danfoss part ABC-123 replaces competitor part XYZ-111.",
        "source_file": "parts.csv",
        "file_type": "csv",
        "danfoss_part": "ABC-123",
        "competitor_part": "XYZ-111",
    }

    mock_response = MagicMock()
    mock_response.matches = [mock_match]
    mock_index.query.return_value = mock_response

    # Mock upsert
    mock_index.upsert.return_value = None

    # Mock stats
    mock_stats = MagicMock()
    mock_stats.total_vector_count = 100
    mock_stats.namespaces = {}
    mock_index.describe_index_stats.return_value = mock_stats

    return mock_index


@pytest.fixture
def mock_pinecone_client(mock_pinecone_index):
    """Mock Pinecone client."""
    mock_client = MagicMock()
    mock_client.list_indexes.return_value = [MagicMock(name="test-index")]
    mock_client.Index.return_value = mock_pinecone_index
    return mock_client


@pytest.fixture
def mock_llm_response():
    """Mock LLM response."""
    mock_response = MagicMock()
    mock_response.content = "Based on the documentation, Danfoss part ABC-123 is the equivalent replacement for competitor part XYZ-111."
    return mock_response


@pytest.fixture
def sample_retrieved_docs():
    """Sample retrieved documents for confidence testing."""
    return [
        {
            "id": "doc1",
            "score": 0.92,
            "content": "Danfoss part ABC-123 replaces competitor part XYZ-111. Product: Valve Assembly.",
            "metadata": {"source_file": "parts.csv", "file_type": "csv"}
        },
        {
            "id": "doc2",
            "score": 0.78,
            "content": "The ABC-123 valve is rated for 24V DC operation.",
            "metadata": {"source_file": "manual.pdf", "file_type": "pdf"}
        },
        {
            "id": "doc3",
            "score": 0.65,
            "content": "Installation instructions for valve assemblies.",
            "metadata": {"source_file": "guide.pdf", "file_type": "pdf"}
        },
    ]


@pytest.fixture
def app_client():
    """Create test client for FastAPI app."""
    from fastapi.testclient import TestClient
    from app.main import create_app

    # Patch external services
    with patch('app.services.pinecone_service.Pinecone'), \
         patch('app.services.pinecone_service.OpenAIEmbeddings'), \
         patch('app.routers.chat.get_rag_service') as mock_rag:

        # Setup mock RAG service
        mock_service = AsyncMock()
        mock_service.query.return_value = {
            "response": "Test response",
            "confidence": 85.0,
            "confidence_level": "high",
            "session_id": "test-session",
            "sources": [{"file": "test.csv", "type": "csv", "chunk_id": "123"}],
            "query_type": "general_question",
            "disclaimer": None
        }
        mock_rag.return_value = mock_service

        app = create_app()
        client = TestClient(app)
        yield client
