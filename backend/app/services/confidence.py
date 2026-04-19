"""
Confidence scoring service for RAG responses.
Calculates relevance confidence based on multiple factors.
"""

import re
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass
from enum import Enum

from ..config import get_settings


class ConfidenceLevel(str, Enum):
    """Confidence level categories."""
    HIGH = "high"
    MEDIUM = "medium"
    LOW = "low"


@dataclass
class ConfidenceResult:
    """Result of confidence calculation."""
    score: float  # 0-100
    level: ConfidenceLevel
    factors: Dict[str, float]
    disclaimer: str = None


class ConfidenceScorer:
    """
    Calculates confidence scores for RAG responses.

    Factors considered:
    - Top document similarity score
    - Average document similarity
    - Keyword overlap between query and retrieved docs
    - Number of relevant documents found
    """

    def __init__(self):
        """Initialize the confidence scorer."""
        self.settings = get_settings()

    def calculate_confidence(
        self,
        query: str,
        retrieved_docs: List[Dict[str, Any]],
        response: str = None
    ) -> ConfidenceResult:
        """
        Calculate confidence score for a query-response pair.

        Args:
            query: User's query
            retrieved_docs: List of retrieved documents with scores
            response: Generated response (optional, for additional analysis)

        Returns:
            ConfidenceResult with score, level, and factors
        """
        if not retrieved_docs:
            return ConfidenceResult(
                score=0.0,
                level=ConfidenceLevel.LOW,
                factors={"no_documents": 0.0},
                disclaimer=self._get_disclaimer(0.0)
            )

        # Calculate individual factors
        top_doc_score = self._calculate_top_doc_score(retrieved_docs)
        avg_doc_score = self._calculate_avg_doc_score(retrieved_docs)
        keyword_score = self._calculate_keyword_overlap(query, retrieved_docs)
        doc_count_score = self._calculate_doc_count_score(retrieved_docs)

        # Weighted combination
        # Formula: 0.4 * top_doc + 0.3 * avg_doc + 0.2 * keyword + 0.1 * doc_count
        confidence = (
            0.4 * top_doc_score +
            0.3 * avg_doc_score +
            0.2 * keyword_score +
            0.1 * doc_count_score
        ) * 100

        # Clamp to 0-100
        confidence = max(0.0, min(100.0, confidence))

        # Determine level
        level = self._get_confidence_level(confidence)

        # Get disclaimer if needed
        disclaimer = self._get_disclaimer(confidence) if confidence < self.settings.high_confidence_threshold else None

        factors = {
            "top_document_similarity": round(top_doc_score * 100, 2),
            "average_similarity": round(avg_doc_score * 100, 2),
            "keyword_overlap": round(keyword_score * 100, 2),
            "document_coverage": round(doc_count_score * 100, 2)
        }

        return ConfidenceResult(
            score=round(confidence, 1),
            level=level,
            factors=factors,
            disclaimer=disclaimer
        )

    def _calculate_top_doc_score(self, docs: List[Dict[str, Any]]) -> float:
        """Get the similarity score of the top document."""
        if not docs:
            return 0.0

        # Pinecone returns cosine similarity as score (0-1)
        top_score = docs[0].get("score", 0.0)
        return min(1.0, max(0.0, top_score))

    def _calculate_avg_doc_score(self, docs: List[Dict[str, Any]]) -> float:
        """Calculate average similarity across retrieved documents."""
        if not docs:
            return 0.0

        scores = [doc.get("score", 0.0) for doc in docs]
        return sum(scores) / len(scores)

    def _calculate_keyword_overlap(
        self,
        query: str,
        docs: List[Dict[str, Any]]
    ) -> float:
        """
        Calculate keyword overlap between query and retrieved documents.

        Args:
            query: User's query
            docs: Retrieved documents

        Returns:
            Overlap ratio (0-1)
        """
        # Extract meaningful keywords from query (length > 2)
        query_words = set(
            word.lower() for word in re.findall(r'\b\w+\b', query)
            if len(word) > 2
        )

        if not query_words:
            return 0.5  # Neutral score if no keywords

        # Collect words from all documents
        doc_words = set()
        for doc in docs:
            content = doc.get("content", "")
            words = set(
                word.lower() for word in re.findall(r'\b\w+\b', content)
                if len(word) > 2
            )
            doc_words.update(words)

        # Calculate overlap
        if not doc_words:
            return 0.0

        overlap = len(query_words.intersection(doc_words))
        overlap_ratio = overlap / len(query_words)

        return min(1.0, overlap_ratio)

    def _calculate_doc_count_score(self, docs: List[Dict[str, Any]]) -> float:
        """
        Score based on number of relevant documents found.
        More documents generally means higher confidence.

        Args:
            docs: Retrieved documents

        Returns:
            Score (0-1)
        """
        # Consider docs with score > 0.5 as "relevant"
        relevant_count = sum(
            1 for doc in docs
            if doc.get("score", 0) > 0.5
        )

        # 3+ relevant docs = full score
        if relevant_count >= 3:
            return 1.0
        else:
            return relevant_count / 3.0

    def _get_confidence_level(self, score: float) -> ConfidenceLevel:
        """Determine confidence level from score."""
        if score >= self.settings.high_confidence_threshold:
            return ConfidenceLevel.HIGH
        elif score >= self.settings.medium_confidence_threshold:
            return ConfidenceLevel.MEDIUM
        else:
            return ConfidenceLevel.LOW

    def _get_disclaimer(self, score: float) -> str:
        """Get appropriate disclaimer for confidence level."""
        if score < self.settings.medium_confidence_threshold:
            return (
                f"Note: This answer has low confidence ({score:.0f}%). "
                "The information may be incomplete or require verification. "
                "Please contact Danfoss technical support for confirmation."
            )
        elif score < self.settings.high_confidence_threshold:
            return (
                f"Note: This answer has medium confidence ({score:.0f}%). "
                "Please verify critical specifications with Danfoss documentation."
            )
        return None

    def format_confidence_badge(self, result: ConfidenceResult) -> str:
        """
        Format confidence for display.

        Args:
            result: ConfidenceResult object

        Returns:
            Formatted string for display
        """
        emoji = {
            ConfidenceLevel.HIGH: "🟢",
            ConfidenceLevel.MEDIUM: "🟡",
            ConfidenceLevel.LOW: "🔴"
        }

        return f"{emoji[result.level]} Confidence: {result.score:.0f}% ({result.level.value})"
