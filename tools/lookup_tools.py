"""
ProductMatchPro - Lookup & Matching Tools
Tools for the LangGraph workflow to find and compare products.
"""

from typing import Optional
from models import HydraulicProduct, MatchResult, ScoreBreakdown, CONFIDENCE_THRESHOLD
from storage.product_db import ProductDB
from storage.vector_store import VectorStore


class LookupTools:
    """Provides all lookup and matching operations for the LangGraph workflow."""

    def __init__(self, db: ProductDB, vector_store: VectorStore):
        self.db = db
        self.vs = vector_store

    def identify_competitor_product(
        self,
        partial_code: str,
        competitor_name: Optional[str] = None,
        category: Optional[str] = None,
    ) -> dict:
        """Identify a competitor product from a partial model code.

        Returns:
            {
                "status": "found" | "ambiguous" | "not_found",
                "product": HydraulicProduct or None,
                "options": list of (HydraulicProduct, score) for ambiguous,
                "match_score": float (fuzzy match score)
            }
        """
        # Resolve brand synonyms
        if competitor_name:
            competitor_name = self.db.resolve_synonym(competitor_name)

        # Check confirmed equivalents first
        confirmed = self.db.get_confirmed_equivalent(partial_code)
        if confirmed:
            # Find the competitor product too
            comp_results = self.db.fuzzy_lookup_model(
                partial_code, company=competitor_name, threshold=80
            )
            comp_product = comp_results[0][0] if comp_results else None
            return {
                "status": "confirmed",
                "product": comp_product,
                "confirmed_equivalent": confirmed,
                "match_score": 0.95,
            }

        # Fuzzy lookup in competitor products
        results = self.db.fuzzy_lookup_model(
            partial_code, company=competitor_name, limit=5, threshold=60
        )

        # If competitor_name not specified, filter to non-my_company products
        if not competitor_name:
            results = [
                (p, s) for p, s in results
                if p.company.lower() not in ("my_company", "{my_company}")
            ]

        if not results:
            # Try semantic search as fallback
            semantic_results = self.vs.search_competitor(
                query=partial_code, company=competitor_name, n_results=3
            )
            if semantic_results:
                fallback_products = []
                for prod_id, score in semantic_results:
                    product = self.db.get_product(prod_id)
                    if product:
                        fallback_products.append((product, score))
                if fallback_products:
                    if len(fallback_products) == 1 or fallback_products[0][1] > 0.8:
                        return {
                            "status": "found",
                            "product": fallback_products[0][0],
                            "match_score": fallback_products[0][1],
                        }
                    return {
                        "status": "ambiguous",
                        "product": None,
                        "options": fallback_products,
                        "match_score": 0.0,
                    }
            return {"status": "not_found", "product": None, "match_score": 0.0}

        # Single strong match
        if len(results) == 1 or results[0][1] > 0.85:
            return {
                "status": "found",
                "product": results[0][0],
                "match_score": results[0][1],
            }

        # Multiple ambiguous matches
        if results[0][1] - results[1][1] < 0.1:
            return {
                "status": "ambiguous",
                "product": None,
                "options": results[:5],
                "match_score": results[0][1],
            }

        # First match is clearly better
        return {
            "status": "found",
            "product": results[0][0],
            "match_score": results[0][1],
        }

    def find_my_company_equivalents(
        self,
        competitor: HydraulicProduct,
        my_company_name: str = "my_company",
        top_k: int = 5,
    ) -> list[MatchResult]:
        """Find {my_company} equivalents for a competitor product.
        Uses semantic search + spec comparison + reranking."""

        # Build query from competitor product
        query = self.vs._build_indexable_text(competitor)

        # Semantic search in {my_company} collection
        semantic_results = self.vs.search_my_company(
            query=query,
            category=competitor.category or None,
            n_results=20,
            rerank_top_k=10,
        )

        if not semantic_results:
            # Try without category filter
            semantic_results = self.vs.search_my_company(
                query=query,
                category=None,
                n_results=20,
                rerank_top_k=10,
            )

        if not semantic_results:
            return []

        # For each semantic match, compute full spec comparison
        match_results = []
        for product_id, semantic_score in semantic_results:
            candidate = self.db.get_product(product_id)
            if not candidate:
                continue

            confidence, breakdown = self.db.spec_comparison(
                competitor, candidate, semantic_score
            )

            match = MatchResult(
                my_company_product=candidate,
                competitor_product=competitor,
                confidence_score=round(confidence, 3),
                score_breakdown=breakdown,
                meets_threshold=confidence >= CONFIDENCE_THRESHOLD,
            )
            match_results.append(match)

        # Sort by confidence descending
        match_results.sort(key=lambda m: m.confidence_score, reverse=True)
        return match_results[:top_k]

    def get_product_details(self, model_code: str) -> Optional[HydraulicProduct]:
        """Get full details for a product by model code."""
        results = self.db.fuzzy_lookup_model(model_code, threshold=90, limit=1)
        if results:
            return results[0][0]
        return None

    def get_guide_context(
        self,
        model_code: str,
        company: Optional[str] = None,
        query: Optional[str] = None,
    ) -> str:
        """Get relevant user guide context for a product."""
        search_query = query or f"specifications for {model_code}"
        results = self.vs.search_guides(
            query=search_query,
            company=company,
            model_code=model_code,
            n_results=3,
        )
        if results:
            return "\n\n".join(text for _, text, _ in results)
        return ""

    def store_feedback(
        self, query: str, competitor_code: str, my_code: str,
        confidence: float, thumbs_up: bool,
    ):
        """Store distributor feedback on a match result."""
        self.db.store_feedback(query, competitor_code, my_code, confidence, thumbs_up)

    def get_typeahead_suggestions(
        self, partial: str, limit: int = 10
    ) -> list[str]:
        """Get model code suggestions for typeahead as the user types."""
        all_codes = self.db.get_all_model_codes()
        partial_upper = partial.upper()
        # Prefix match first
        prefix_matches = [c for c in all_codes if c.upper().startswith(partial_upper)]
        if len(prefix_matches) >= limit:
            return sorted(prefix_matches)[:limit]
        # Then fuzzy
        from fuzzywuzzy import process, fuzz
        fuzzy_matches = process.extract(
            partial_upper,
            [c.upper() for c in all_codes],
            scorer=fuzz.partial_ratio,
            limit=limit,
        )
        seen = set(m.upper() for m in prefix_matches)
        result = list(prefix_matches)
        for match_tuple in fuzzy_matches:
            matched = match_tuple[0]
            score = match_tuple[1]
            if matched not in seen and score > 50:
                # Find original case
                for c in all_codes:
                    if c.upper() == matched:
                        result.append(c)
                        seen.add(matched)
                        break
            if len(result) >= limit:
                break
        return result[:limit]
