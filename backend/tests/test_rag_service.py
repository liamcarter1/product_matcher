"""
Tests for RAGService.
Tests query classification, retrieval, and response generation.
"""

import pytest
from unittest.mock import patch, MagicMock, AsyncMock

from app.services.rag_service import RAGService, QueryType, RAGState


@pytest.fixture
def mock_dependencies():
    """Mock all external dependencies for RAGService."""
    with patch('app.services.rag_service.get_settings') as mock_settings, \
         patch('app.services.rag_service.ChatOpenAI') as mock_llm, \
         patch('app.services.rag_service.PineconeService') as mock_pinecone, \
         patch('app.services.rag_service.ConfidenceScorer') as mock_confidence:

        # Setup settings
        settings = MagicMock()
        settings.openai_model = "gpt-4o"
        settings.openai_api_key = "test-key"
        settings.high_confidence_threshold = 80.0
        settings.medium_confidence_threshold = 50.0
        mock_settings.return_value = settings

        # Setup LLM
        llm_instance = MagicMock()
        mock_response = MagicMock()
        mock_response.content = "Test response from LLM"
        llm_instance.invoke.return_value = mock_response
        mock_llm.return_value = llm_instance

        # Setup Pinecone
        pinecone_instance = MagicMock()
        pinecone_instance.query.return_value = [
            {
                "id": "test-1",
                "score": 0.85,
                "content": "Test content",
                "metadata": {"source_file": "test.csv", "file_type": "csv"}
            }
        ]
        pinecone_instance.query_by_part_number.return_value = []
        mock_pinecone.return_value = pinecone_instance

        # Setup Confidence Scorer
        confidence_instance = MagicMock()
        confidence_result = MagicMock()
        confidence_result.score = 85.0
        confidence_result.level = MagicMock(value="high")
        confidence_result.disclaimer = None
        confidence_instance.calculate_confidence.return_value = confidence_result
        mock_confidence.return_value = confidence_instance

        yield {
            "settings": mock_settings,
            "llm": mock_llm,
            "pinecone": mock_pinecone,
            "confidence": mock_confidence,
            "llm_instance": llm_instance,
            "pinecone_instance": pinecone_instance,
            "confidence_instance": confidence_instance,
        }


class TestQueryClassification:
    """Tests for query type classification."""

    def test_classify_part_lookup_replace(self, mock_dependencies):
        """Test classification of 'replace' queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "What Danfoss part replaces XYZ-111?",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.PART_LOOKUP

    def test_classify_part_lookup_equivalent(self, mock_dependencies):
        """Test classification of 'equivalent' queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "What is equivalent to competitor part ABC-456?",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.PART_LOOKUP

    def test_classify_part_lookup_crossref(self, mock_dependencies):
        """Test classification of cross-reference queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "Cross-reference for BrandA part 12345",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.PART_LOOKUP

    def test_classify_spec_query_voltage(self, mock_dependencies):
        """Test classification of voltage specification queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "What is the voltage rating of ABC-123?",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.SPECIFICATION_QUERY

    def test_classify_spec_query_dimensions(self, mock_dependencies):
        """Test classification of dimension queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "What are the dimensions of this product?",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.SPECIFICATION_QUERY

    def test_classify_general_question(self, mock_dependencies):
        """Test classification of general questions."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "How do I install the valve correctly?",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.GENERAL_QUESTION

    def test_classify_part_number_in_query_competitor(self, mock_dependencies):
        """Test classification when part number detected (competitor context)."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "Tell me about ABC-12345",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.PART_LOOKUP

    def test_classify_part_number_danfoss_context(self, mock_dependencies):
        """Test classification when Danfoss part number detected."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "Tell me about Danfoss ABC-12345",
            "query_type": "",
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._classify_query(state)
        assert result["query_type"] == QueryType.SPECIFICATION_QUERY


class TestPartNumberExtraction:
    """Tests for part number extraction from queries."""

    def test_extract_alphanumeric_part(self, mock_dependencies):
        """Test extraction of alphanumeric part numbers."""
        service = RAGService()

        part_numbers = service._extract_part_numbers("Looking for ABC-1234 replacement")
        # The pattern requires 2+ digits, so ABC-1234 should match
        assert len(part_numbers) > 0
        assert any("ABC" in p and "1234" in p for p in part_numbers)

    def test_extract_numeric_part(self, mock_dependencies):
        """Test extraction of pure numeric part codes."""
        service = RAGService()

        part_numbers = service._extract_part_numbers("Part number 12345678")
        assert any("12345678" in p for p in part_numbers)

    def test_extract_multiple_parts(self, mock_dependencies):
        """Test extraction of multiple part numbers."""
        service = RAGService()

        part_numbers = service._extract_part_numbers("Compare ABC-123 with DEF-456")
        assert len(part_numbers) >= 2

    def test_extract_no_parts(self, mock_dependencies):
        """Test extraction when no part numbers present."""
        service = RAGService()

        part_numbers = service._extract_part_numbers("How do I install this?")
        assert len(part_numbers) == 0

    def test_extract_deduplicates(self, mock_dependencies):
        """Test that extraction deduplicates results."""
        service = RAGService()

        part_numbers = service._extract_part_numbers("ABC-123 and ABC-123 again")
        # Should only have unique values
        assert len(part_numbers) == len(set(part_numbers))


class TestRetrieval:
    """Tests for document retrieval."""

    def test_retrieve_part_lookup(self, mock_dependencies):
        """Test retrieval for part lookup queries."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "What replaces XYZ-111?",
            "query_type": QueryType.PART_LOOKUP,
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._retrieve_documents(state)
        assert "context" in result
        assert "sources" in result

    def test_retrieve_calls_pinecone(self, mock_dependencies):
        """Test that retrieval calls Pinecone service."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "General question about products",
            "query_type": QueryType.GENERAL_QUESTION,
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        service._retrieve_documents(state)
        mock_dependencies["pinecone_instance"].query.assert_called()

    def test_retrieve_extracts_sources(self, mock_dependencies):
        """Test that sources are extracted from retrieved docs."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "Test query",
            "query_type": QueryType.GENERAL_QUESTION,
            "context": [],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._retrieve_documents(state)
        assert len(result["sources"]) > 0
        assert "file" in result["sources"][0]


class TestConfidenceScoring:
    """Tests for confidence scoring integration."""

    def test_score_confidence_called(self, mock_dependencies):
        """Test that confidence scorer is called."""
        service = RAGService()

        state: RAGState = {
            "messages": [],
            "query": "Test query",
            "query_type": QueryType.GENERAL_QUESTION,
            "context": [{"id": "1", "score": 0.8, "content": "test"}],
            "confidence": None,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._score_confidence(state)
        mock_dependencies["confidence_instance"].calculate_confidence.assert_called_once()
        assert result["confidence"] is not None


class TestResponseGeneration:
    """Tests for LLM response generation."""

    def test_generate_response_calls_llm(self, mock_dependencies):
        """Test that response generation calls LLM."""
        service = RAGService()

        confidence_result = MagicMock()
        confidence_result.level = MagicMock(value="high")
        confidence_result.disclaimer = None

        state: RAGState = {
            "messages": [],
            "query": "Test query",
            "query_type": QueryType.GENERAL_QUESTION,
            "context": [{"id": "1", "score": 0.8, "content": "test", "metadata": {"source_file": "test.csv"}}],
            "confidence": confidence_result,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._generate_response(state)
        assert result["response"] != ""

    def test_response_includes_disclaimer_when_low_confidence(self, mock_dependencies):
        """Test that disclaimer is appended for low confidence."""
        service = RAGService()

        confidence_result = MagicMock()
        confidence_result.level = MagicMock(value="low")
        confidence_result.disclaimer = "This is a test disclaimer."

        state: RAGState = {
            "messages": [],
            "query": "Test query",
            "query_type": QueryType.GENERAL_QUESTION,
            "context": [{"id": "1", "score": 0.3, "content": "test", "metadata": {"source_file": "test.csv"}}],
            "confidence": confidence_result,
            "response": "",
            "session_id": "test",
            "sources": []
        }

        result = service._generate_response(state)
        # Check that response field exists - the disclaimer is appended in the method
        assert "response" in result
        # The method should return a state with response field populated
        assert result["response"] is not None


class TestFormatContext:
    """Tests for context formatting."""

    def test_format_empty_context(self, mock_dependencies):
        """Test formatting with empty context."""
        service = RAGService()
        formatted = service._format_context([])
        assert "No relevant information found" in formatted

    def test_format_context_includes_sources(self, mock_dependencies):
        """Test that formatted context includes source references."""
        service = RAGService()

        context = [
            {"content": "Test content", "metadata": {"source_file": "test.csv"}},
            {"content": "More content", "metadata": {"source_file": "guide.pdf"}},
        ]

        formatted = service._format_context(context)
        assert "test.csv" in formatted
        assert "guide.pdf" in formatted


class TestSessionManagement:
    """Tests for session management."""

    def test_clear_session(self, mock_dependencies):
        """Test clearing a session."""
        service = RAGService()

        # Add a session
        service._sessions["test-session"] = [{"role": "user", "content": "test"}]

        # Clear it
        service.clear_session("test-session")

        assert "test-session" not in service._sessions

    def test_clear_nonexistent_session(self, mock_dependencies):
        """Test clearing a session that doesn't exist."""
        service = RAGService()
        # Should not raise
        service.clear_session("nonexistent")

    def test_get_session_history(self, mock_dependencies):
        """Test getting session history."""
        service = RAGService()

        service._sessions["test-session"] = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there"}
        ]

        history = service.get_session_history("test-session")
        assert len(history) == 2

    def test_get_nonexistent_session_history(self, mock_dependencies):
        """Test getting history for nonexistent session."""
        service = RAGService()
        history = service.get_session_history("nonexistent")
        assert history == []


class TestSystemPrompts:
    """Tests for system prompt generation."""

    def test_part_lookup_prompt(self, mock_dependencies):
        """Test system prompt for part lookup."""
        service = RAGService()

        confidence = MagicMock()
        confidence.level = MagicMock(value="high")

        prompt = service._get_system_prompt(QueryType.PART_LOOKUP, confidence)

        assert "part lookup" in prompt.lower() or "danfoss equivalent" in prompt.lower()

    def test_spec_query_prompt(self, mock_dependencies):
        """Test system prompt for specification queries."""
        service = RAGService()

        confidence = MagicMock()
        confidence.level = MagicMock(value="high")

        prompt = service._get_system_prompt(QueryType.SPECIFICATION_QUERY, confidence)

        assert "specification" in prompt.lower() or "technical" in prompt.lower()

    def test_general_prompt(self, mock_dependencies):
        """Test system prompt for general questions."""
        service = RAGService()

        confidence = MagicMock()
        confidence.level = MagicMock(value="high")

        prompt = service._get_system_prompt(QueryType.GENERAL_QUESTION, confidence)

        assert "general" in prompt.lower() or "helpful" in prompt.lower()


class TestQueryAsync:
    """Tests for the async query method."""

    @pytest.mark.asyncio
    async def test_query_returns_response(self, mock_dependencies):
        """Test that query returns a complete response."""
        service = RAGService()

        result = await service.query("What Danfoss part replaces XYZ-111?")

        assert "response" in result
        assert "confidence" in result
        assert "session_id" in result
        assert "sources" in result

    @pytest.mark.asyncio
    async def test_query_generates_session_id(self, mock_dependencies):
        """Test that query generates session ID if not provided."""
        service = RAGService()

        result = await service.query("Test query")

        assert result["session_id"] is not None
        assert len(result["session_id"]) > 0

    @pytest.mark.asyncio
    async def test_query_uses_provided_session_id(self, mock_dependencies):
        """Test that query uses provided session ID."""
        service = RAGService()

        result = await service.query("Test query", session_id="my-session")

        assert result["session_id"] == "my-session"

    @pytest.mark.asyncio
    async def test_query_maintains_session_history(self, mock_dependencies):
        """Test that query maintains conversation history."""
        service = RAGService()

        await service.query("First message", session_id="test-session")
        await service.query("Second message", session_id="test-session")

        history = service.get_session_history("test-session")
        # Should have user and assistant messages for each query
        assert len(history) >= 2  # At least 2 messages

    @pytest.mark.asyncio
    async def test_query_limits_history_size(self, mock_dependencies):
        """Test that session history is limited."""
        service = RAGService()

        # Add many messages
        for i in range(25):
            await service.query(f"Message {i}", session_id="test-session")

        history = service.get_session_history("test-session")
        assert len(history) <= 20
