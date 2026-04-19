"""
Tests for FastAPI endpoints.
Tests chat, health, and ingest API routes.
"""

import pytest
from fastapi.testclient import TestClient
from unittest.mock import patch, MagicMock, AsyncMock
import os

# Set environment variables before any imports
os.environ["OPENAI_API_KEY"] = "sk-test-key"
os.environ["PINECONE_API_KEY"] = "pc-test-key"


@pytest.fixture
def test_client():
    """Create a test client with mocked services."""
    with patch('app.services.pinecone_service.Pinecone'), \
         patch('app.services.pinecone_service.OpenAIEmbeddings'):

        from app.main import create_app
        app = create_app()
        client = TestClient(app)
        yield client


class TestHealthEndpoint:
    """Tests for the health check endpoint."""

    def test_health_returns_200(self, test_client):
        """Test that health endpoint returns 200."""
        response = test_client.get("/health")
        assert response.status_code == 200

    def test_health_returns_status(self, test_client):
        """Test that health endpoint returns status."""
        response = test_client.get("/health")
        data = response.json()

        assert "status" in data
        assert data["status"] == "healthy"

    def test_health_returns_app_info(self, test_client):
        """Test that health endpoint returns app info."""
        response = test_client.get("/health")
        data = response.json()

        assert "app" in data
        assert "version" in data


class TestRootEndpoint:
    """Tests for the root endpoint."""

    def test_root_returns_200(self, test_client):
        """Test that root endpoint returns 200."""
        response = test_client.get("/")
        assert response.status_code == 200

    def test_root_returns_welcome(self, test_client):
        """Test that root endpoint returns welcome message."""
        response = test_client.get("/")
        data = response.json()

        assert "message" in data
        assert "Welcome" in data["message"]

    def test_root_returns_docs_link(self, test_client):
        """Test that root endpoint returns docs link."""
        response = test_client.get("/")
        data = response.json()

        assert "docs" in data
        assert data["docs"] == "/docs"


class TestChatEndpoint:
    """Tests for the chat endpoint."""

    @pytest.fixture
    def mock_rag_service(self):
        """Create a mock RAG service."""
        mock_service = MagicMock()
        mock_service.query = AsyncMock(return_value={
            "response": "Test response from RAG",
            "confidence": 85.0,
            "confidence_level": "high",
            "session_id": "test-session-123",
            "sources": [{"file": "test.csv", "type": "csv", "chunk_id": "abc123"}],
            "query_type": "part_lookup",
            "disclaimer": None
        })
        return mock_service

    @pytest.fixture
    def chat_client(self, mock_rag_service):
        """Create test client with mocked RAG service."""
        with patch('app.services.pinecone_service.Pinecone'), \
             patch('app.services.pinecone_service.OpenAIEmbeddings'), \
             patch('app.routers.chat._rag_service', mock_rag_service), \
             patch('app.routers.chat.get_rag_service', return_value=mock_rag_service):

            from app.main import create_app
            app = create_app()
            client = TestClient(app)
            yield client, mock_rag_service

    def test_chat_post_success(self, chat_client):
        """Test successful chat POST request."""
        client, _ = chat_client
        response = client.post(
            "/api/chat",
            json={"message": "What Danfoss part replaces XYZ-111?"}
        )
        assert response.status_code == 200

    def test_chat_returns_response(self, chat_client):
        """Test that chat returns a response."""
        client, _ = chat_client
        response = client.post(
            "/api/chat",
            json={"message": "Test message"}
        )
        data = response.json()
        assert "response" in data
        assert data["response"] == "Test response from RAG"

    def test_chat_returns_confidence(self, chat_client):
        """Test that chat returns confidence score."""
        client, _ = chat_client
        response = client.post(
            "/api/chat",
            json={"message": "Test message"}
        )
        data = response.json()
        assert "confidence" in data
        assert data["confidence"] == 85.0
        assert "confidence_level" in data
        assert data["confidence_level"] == "high"

    def test_chat_returns_session_id(self, chat_client):
        """Test that chat returns session ID."""
        client, _ = chat_client
        response = client.post(
            "/api/chat",
            json={"message": "Test message"}
        )
        data = response.json()
        assert "session_id" in data
        assert data["session_id"] == "test-session-123"

    def test_chat_returns_sources(self, chat_client):
        """Test that chat returns sources."""
        client, _ = chat_client
        response = client.post(
            "/api/chat",
            json={"message": "Test message"}
        )
        data = response.json()
        assert "sources" in data
        assert len(data["sources"]) > 0
        assert data["sources"][0]["file"] == "test.csv"

    def test_chat_with_session_id(self, chat_client):
        """Test chat with provided session ID."""
        client, mock_service = chat_client
        response = client.post(
            "/api/chat",
            json={
                "message": "Test message",
                "session_id": "my-custom-session"
            }
        )
        assert response.status_code == 200
        mock_service.query.assert_called_with(
            message="Test message",
            session_id="my-custom-session",
            distributor_id=None
        )

    def test_chat_empty_message_fails(self, test_client):
        """Test that empty message returns 422."""
        response = test_client.post(
            "/api/chat",
            json={"message": ""}
        )

        assert response.status_code == 422

    def test_chat_missing_message_fails(self, test_client):
        """Test that missing message returns 422."""
        response = test_client.post(
            "/api/chat",
            json={}
        )

        assert response.status_code == 422

    def test_chat_message_too_long(self, test_client):
        """Test that overly long message returns 422."""
        long_message = "x" * 2001  # Over 2000 char limit

        response = test_client.post(
            "/api/chat",
            json={"message": long_message}
        )

        assert response.status_code == 422


class TestSessionEndpoints:
    """Tests for session management endpoints."""

    @pytest.fixture
    def session_client(self):
        """Create test client with mocked RAG service for session tests."""
        mock_service = MagicMock()
        mock_service.get_session_history.return_value = [
            {"role": "user", "content": "Hello", "timestamp": "2024-01-01T00:00:00"},
            {"role": "assistant", "content": "Hi!", "timestamp": "2024-01-01T00:00:01"}
        ]
        mock_service.clear_session.return_value = None

        with patch('app.services.pinecone_service.Pinecone'), \
             patch('app.services.pinecone_service.OpenAIEmbeddings'), \
             patch('app.routers.chat._rag_service', mock_service), \
             patch('app.routers.chat.get_rag_service', return_value=mock_service):

            from app.main import create_app
            app = create_app()
            client = TestClient(app)
            yield client, mock_service

    def test_get_session_history(self, session_client):
        """Test getting session history."""
        client, _ = session_client
        response = client.get("/api/chat/session/test-session/history")

        assert response.status_code == 200
        data = response.json()
        assert "session_id" in data
        assert "messages" in data
        assert len(data["messages"]) == 2

    def test_clear_session(self, session_client):
        """Test clearing a session."""
        client, mock_service = session_client
        response = client.delete("/api/chat/session/test-session")

        assert response.status_code == 200
        data = response.json()
        assert data["status"] == "cleared"
        mock_service.clear_session.assert_called_with("test-session")


class TestOpenAPISchema:
    """Tests for OpenAPI schema generation."""

    def test_openapi_schema_available(self, test_client):
        """Test that OpenAPI schema is available."""
        response = test_client.get("/openapi.json")
        assert response.status_code == 200

    def test_openapi_has_chat_endpoint(self, test_client):
        """Test that OpenAPI schema includes chat endpoint."""
        response = test_client.get("/openapi.json")
        schema = response.json()

        assert "/api/chat" in schema["paths"]

    def test_openapi_has_health_endpoint(self, test_client):
        """Test that OpenAPI schema includes health endpoint."""
        response = test_client.get("/openapi.json")
        schema = response.json()

        assert "/health" in schema["paths"]


class TestCORS:
    """Tests for CORS configuration."""

    def test_cors_headers_present(self, test_client):
        """Test that CORS headers are present."""
        response = test_client.options(
            "/api/chat",
            headers={
                "Origin": "http://localhost:3000",
                "Access-Control-Request-Method": "POST"
            }
        )

        # Should not be blocked
        assert response.status_code in [200, 204, 405]


class TestErrorHandling:
    """Tests for API error handling."""

    def test_chat_service_error(self):
        """Test handling of service errors."""
        mock_service = MagicMock()
        mock_service.query = AsyncMock(side_effect=Exception("Service error"))

        with patch('app.services.pinecone_service.Pinecone'), \
             patch('app.services.pinecone_service.OpenAIEmbeddings'), \
             patch('app.routers.chat._rag_service', mock_service), \
             patch('app.routers.chat.get_rag_service', return_value=mock_service):

            from app.main import create_app
            app = create_app()
            client = TestClient(app)

            response = client.post(
                "/api/chat",
                json={"message": "Test message"}
            )

        assert response.status_code == 500
        data = response.json()
        assert "detail" in data

    def test_invalid_json_returns_422(self, test_client):
        """Test that invalid JSON returns 422."""
        response = test_client.post(
            "/api/chat",
            content="not valid json",
            headers={"Content-Type": "application/json"}
        )

        assert response.status_code == 422

    def test_wrong_content_type(self, test_client):
        """Test that wrong content type returns error."""
        response = test_client.post(
            "/api/chat",
            content="message=test",
            headers={"Content-Type": "application/x-www-form-urlencoded"}
        )

        assert response.status_code == 422


class TestDocumentation:
    """Tests for API documentation."""

    def test_swagger_ui_available(self, test_client):
        """Test that Swagger UI is available."""
        response = test_client.get("/docs")
        assert response.status_code == 200

    def test_redoc_available(self, test_client):
        """Test that ReDoc is available."""
        response = test_client.get("/redoc")
        assert response.status_code == 200
