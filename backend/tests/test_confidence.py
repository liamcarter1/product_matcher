"""
Tests for ConfidenceScorer service.
Tests confidence calculation and level determination.
"""

import pytest
from unittest.mock import patch, MagicMock

from app.services.confidence import ConfidenceScorer, ConfidenceLevel, ConfidenceResult


@pytest.fixture
def mock_settings_for_confidence():
    """Mock settings for confidence scorer."""
    mock = MagicMock()
    mock.high_confidence_threshold = 80.0
    mock.medium_confidence_threshold = 50.0
    return mock


@pytest.fixture
def scorer(mock_settings_for_confidence):
    """Create a ConfidenceScorer with mocked settings."""
    with patch('app.services.confidence.get_settings', return_value=mock_settings_for_confidence):
        return ConfidenceScorer()


class TestConfidenceCalculation:
    """Tests for confidence score calculation."""

    def test_empty_docs_returns_zero(self, scorer):
        """Test that empty document list returns zero confidence."""
        result = scorer.calculate_confidence("test query", [])

        assert result.score == 0.0
        assert result.level == ConfidenceLevel.LOW
        assert "no_documents" in result.factors

    def test_high_score_docs_high_confidence(self, scorer):
        """Test high confidence with high-scoring documents."""
        docs = [
            {"score": 0.95, "content": "test query relevant content"},
            {"score": 0.90, "content": "more relevant content"},
            {"score": 0.85, "content": "additional relevant info"},
        ]

        result = scorer.calculate_confidence("test query", docs)

        assert result.score >= 80.0
        assert result.level == ConfidenceLevel.HIGH

    def test_low_score_docs_low_confidence(self, scorer):
        """Test low confidence with low-scoring documents."""
        docs = [
            {"score": 0.3, "content": "unrelated content"},
            {"score": 0.2, "content": "more unrelated stuff"},
        ]

        result = scorer.calculate_confidence("test query", docs)

        assert result.score < 50.0
        assert result.level == ConfidenceLevel.LOW

    def test_medium_confidence_range(self, scorer):
        """Test medium confidence with moderate scores."""
        docs = [
            {"score": 0.6, "content": "somewhat relevant test query"},
            {"score": 0.55, "content": "partial match content"},
        ]

        result = scorer.calculate_confidence("test query", docs)

        assert 50.0 <= result.score < 80.0
        assert result.level == ConfidenceLevel.MEDIUM


class TestConfidenceFactors:
    """Tests for individual confidence factors."""

    def test_top_doc_score_factor(self, scorer):
        """Test top document score calculation."""
        docs = [{"score": 0.85}, {"score": 0.7}]
        score = scorer._calculate_top_doc_score(docs)
        assert score == 0.85

    def test_top_doc_score_empty(self, scorer):
        """Test top document score with empty list."""
        score = scorer._calculate_top_doc_score([])
        assert score == 0.0

    def test_top_doc_score_clamped(self, scorer):
        """Test that top document score is clamped to 1.0."""
        docs = [{"score": 1.5}]  # Invalid score
        score = scorer._calculate_top_doc_score(docs)
        assert score == 1.0

    def test_avg_doc_score_calculation(self, scorer):
        """Test average document score calculation."""
        docs = [{"score": 0.8}, {"score": 0.6}, {"score": 0.4}]
        score = scorer._calculate_avg_doc_score(docs)
        assert score == pytest.approx(0.6, rel=0.01)

    def test_avg_doc_score_empty(self, scorer):
        """Test average document score with empty list."""
        score = scorer._calculate_avg_doc_score([])
        assert score == 0.0

    def test_keyword_overlap_full_match(self, scorer):
        """Test keyword overlap with full match."""
        query = "danfoss valve assembly"
        docs = [{"content": "This danfoss valve assembly is compatible."}]
        score = scorer._calculate_keyword_overlap(query, docs)
        assert score == 1.0

    def test_keyword_overlap_partial_match(self, scorer):
        """Test keyword overlap with partial match."""
        query = "danfoss valve pressure sensor"
        docs = [{"content": "The danfoss valve is working."}]
        score = scorer._calculate_keyword_overlap(query, docs)
        assert 0.0 < score < 1.0

    def test_keyword_overlap_no_match(self, scorer):
        """Test keyword overlap with no matching keywords."""
        query = "specific product name"
        docs = [{"content": "completely unrelated text here"}]
        score = scorer._calculate_keyword_overlap(query, docs)
        assert score == 0.0

    def test_keyword_overlap_short_words_ignored(self, scorer):
        """Test that short words (<=2 chars) are ignored."""
        query = "a is the an"  # All short words except "the" which is 3 chars
        docs = [{"content": "some other content"}]
        score = scorer._calculate_keyword_overlap(query, docs)
        # "the" has 3 chars so it's included, returns 0 if no match
        # or 0.5 if no valid keywords are extracted
        assert 0.0 <= score <= 0.5

    def test_doc_count_score_many_relevant(self, scorer):
        """Test document count score with many relevant docs."""
        docs = [
            {"score": 0.9},
            {"score": 0.8},
            {"score": 0.7},
            {"score": 0.6},
        ]
        score = scorer._calculate_doc_count_score(docs)
        assert score == 1.0  # 4 docs > 0.5 threshold

    def test_doc_count_score_few_relevant(self, scorer):
        """Test document count score with few relevant docs."""
        docs = [
            {"score": 0.9},
            {"score": 0.3},  # Below threshold
            {"score": 0.2},  # Below threshold
        ]
        score = scorer._calculate_doc_count_score(docs)
        assert score == pytest.approx(1 / 3, rel=0.01)


class TestConfidenceLevel:
    """Tests for confidence level determination."""

    def test_high_confidence_level(self, scorer):
        """Test high confidence level threshold."""
        level = scorer._get_confidence_level(85.0)
        assert level == ConfidenceLevel.HIGH

    def test_medium_confidence_level(self, scorer):
        """Test medium confidence level threshold."""
        level = scorer._get_confidence_level(65.0)
        assert level == ConfidenceLevel.MEDIUM

    def test_low_confidence_level(self, scorer):
        """Test low confidence level threshold."""
        level = scorer._get_confidence_level(40.0)
        assert level == ConfidenceLevel.LOW

    def test_boundary_high_medium(self, scorer):
        """Test boundary between high and medium."""
        # Exactly at threshold should be high
        level = scorer._get_confidence_level(80.0)
        assert level == ConfidenceLevel.HIGH

        # Just below should be medium
        level = scorer._get_confidence_level(79.9)
        assert level == ConfidenceLevel.MEDIUM

    def test_boundary_medium_low(self, scorer):
        """Test boundary between medium and low."""
        # Exactly at threshold should be medium
        level = scorer._get_confidence_level(50.0)
        assert level == ConfidenceLevel.MEDIUM

        # Just below should be low
        level = scorer._get_confidence_level(49.9)
        assert level == ConfidenceLevel.LOW


class TestDisclaimer:
    """Tests for disclaimer generation."""

    def test_low_confidence_disclaimer(self, scorer):
        """Test disclaimer for low confidence."""
        disclaimer = scorer._get_disclaimer(30.0)
        assert disclaimer is not None
        assert "low confidence" in disclaimer.lower()
        assert "30%" in disclaimer

    def test_medium_confidence_disclaimer(self, scorer):
        """Test disclaimer for medium confidence."""
        disclaimer = scorer._get_disclaimer(65.0)
        assert disclaimer is not None
        assert "medium confidence" in disclaimer.lower()

    def test_high_confidence_no_disclaimer(self, scorer):
        """Test no disclaimer for high confidence."""
        disclaimer = scorer._get_disclaimer(85.0)
        assert disclaimer is None


class TestConfidenceResult:
    """Tests for ConfidenceResult dataclass."""

    def test_result_contains_all_factors(self, scorer, sample_retrieved_docs):
        """Test that result contains all calculated factors."""
        result = scorer.calculate_confidence("test query", sample_retrieved_docs)

        assert "top_document_similarity" in result.factors
        assert "average_similarity" in result.factors
        assert "keyword_overlap" in result.factors
        assert "document_coverage" in result.factors

    def test_result_score_in_range(self, scorer, sample_retrieved_docs):
        """Test that result score is in valid range."""
        result = scorer.calculate_confidence("test query", sample_retrieved_docs)
        assert 0.0 <= result.score <= 100.0

    def test_result_score_rounded(self, scorer, sample_retrieved_docs):
        """Test that result score is rounded to 1 decimal place."""
        result = scorer.calculate_confidence("test query", sample_retrieved_docs)
        # Check that the score has at most 1 decimal place
        assert result.score == round(result.score, 1)


class TestFormatConfidenceBadge:
    """Tests for confidence badge formatting."""

    def test_high_confidence_badge(self, scorer):
        """Test badge format for high confidence."""
        result = ConfidenceResult(
            score=90.0,
            level=ConfidenceLevel.HIGH,
            factors={}
        )
        badge = scorer.format_confidence_badge(result)
        assert "90%" in badge
        assert "high" in badge.lower()

    def test_medium_confidence_badge(self, scorer):
        """Test badge format for medium confidence."""
        result = ConfidenceResult(
            score=65.0,
            level=ConfidenceLevel.MEDIUM,
            factors={}
        )
        badge = scorer.format_confidence_badge(result)
        assert "65%" in badge
        assert "medium" in badge.lower()

    def test_low_confidence_badge(self, scorer):
        """Test badge format for low confidence."""
        result = ConfidenceResult(
            score=30.0,
            level=ConfidenceLevel.LOW,
            factors={}
        )
        badge = scorer.format_confidence_badge(result)
        assert "30%" in badge
        assert "low" in badge.lower()


class TestConfidenceWithRealQuery:
    """Integration-style tests with realistic query scenarios."""

    def test_part_lookup_high_confidence(self, scorer):
        """Test confidence for a part lookup with good matches."""
        query = "What Danfoss part replaces XYZ-111?"
        docs = [
            {
                "score": 0.92,
                "content": "Danfoss part ABC-123 replaces competitor part XYZ-111. Product: Valve Assembly."
            },
            {
                "score": 0.85,
                "content": "XYZ-111 from BrandA is equivalent to Danfoss ABC-123."
            },
            {
                "score": 0.78,
                "content": "The ABC-123 valve replaces multiple competitor parts including XYZ-111."
            },
        ]

        result = scorer.calculate_confidence(query, docs)
        assert result.level == ConfidenceLevel.HIGH

    def test_general_query_medium_confidence(self, scorer):
        """Test confidence for a general query with moderate matches."""
        query = "How do I install a pressure sensor?"
        docs = [
            {
                "score": 0.65,
                "content": "Installation guide for industrial sensors."
            },
            {
                "score": 0.55,
                "content": "Sensor mounting instructions."
            },
        ]

        result = scorer.calculate_confidence(query, docs)
        assert result.level == ConfidenceLevel.MEDIUM

    def test_off_topic_query_low_confidence(self, scorer):
        """Test confidence for an off-topic query."""
        query = "What is the weather today?"
        docs = [
            {
                "score": 0.25,
                "content": "Danfoss temperature sensors for industrial use."
            },
        ]

        result = scorer.calculate_confidence(query, docs)
        assert result.level == ConfidenceLevel.LOW
