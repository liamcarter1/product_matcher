"""
ProductMatchPro - Lookup & Matching Tools
Tools for the LangGraph workflow to find and compare products.
"""

import logging
from typing import Optional
from models import HydraulicProduct, MatchResult, ScoreBreakdown, CONFIDENCE_THRESHOLD
from storage.product_db import ProductDB
from storage.vector_store import VectorStore
from tools.parse_tools import compute_canonical_pattern

logger = logging.getLogger(__name__)


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

        # If competitor_name not specified, filter to non-Danfoss products
        if not competitor_name:
            results = [
                (p, s) for p, s in results
                if p.company.lower() not in ("danfoss",)
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

    def enrich_competitor_spool(self, product: HydraulicProduct) -> HydraulicProduct:
        """Enrich a competitor product with spool function data for cross-manufacturer matching.

        When a competitor product is looked up, it typically only has a raw spool_type
        code (e.g. "D" for Bosch). This method:
        1. Decodes the model code to extract/confirm the spool segment
        2. Looks up the spool reference table for that manufacturer+series+code
        3. Computes the canonical spool pattern for cross-manufacturer comparison

        Without this, spool matching falls back to exact string comparison
        ("D" vs "2A" = 0.0) instead of canonical pattern matching
        ("BLOCKED|PA-BT|PB-AT" == "BLOCKED|PA-BT|PB-AT" = 1.0).
        """
        if not product.spool_type:
            # Try to decode the model code to find the spool type
            decoded = self.db.decode_model_code(product.model_code, product.company)
            if decoded:
                spool_code = decoded.get("_raw_spool_type") or decoded.get("spool_type")
                if spool_code:
                    product.spool_type = spool_code
                    logger.info("Decoded spool_type '%s' from model code %s",
                                spool_code, product.model_code)

        if not product.spool_type:
            return product

        # Already has canonical pattern — no enrichment needed
        extra = product.extra_specs or {}
        if extra.get("canonical_spool_pattern"):
            return product

        spool_code = product.spool_type.strip()

        # Look up spool reference table for this manufacturer
        # Try with series prefix extracted from model code
        refs = []
        decoded = self.db.decode_model_code(product.model_code, product.company)
        series = decoded.get("series", "") if decoded else ""
        if series:
            refs = self.db.get_spool_type_references(series, product.company)
        if not refs:
            # Broader search: all references for this manufacturer
            refs = self.db.get_spool_type_references(manufacturer=product.company)

        # Find the matching spool code in references
        matched_ref = None
        spool_upper = spool_code.upper()
        for ref in refs:
            if ref.get("spool_code", "").upper() == spool_upper:
                matched_ref = ref
                break

        if matched_ref:
            if product.extra_specs is None:
                product.extra_specs = {}
            # Use stored canonical pattern or compute one
            canonical = matched_ref.get("canonical_pattern", "")
            if not canonical:
                canonical = compute_canonical_pattern(
                    matched_ref.get("center_condition", ""),
                    matched_ref.get("solenoid_a_function", ""),
                    matched_ref.get("solenoid_b_function", ""),
                )
            if canonical:
                product.extra_specs["canonical_spool_pattern"] = canonical
            if matched_ref.get("center_condition"):
                product.extra_specs.setdefault("center_condition",
                                               matched_ref["center_condition"])
            if matched_ref.get("solenoid_a_function"):
                product.extra_specs.setdefault("solenoid_a_energised",
                                               matched_ref["solenoid_a_function"])
            if matched_ref.get("solenoid_b_function"):
                product.extra_specs.setdefault("solenoid_b_energised",
                                               matched_ref["solenoid_b_function"])
            logger.info("Enriched competitor spool: %s %s -> canonical '%s'",
                        product.company, spool_code, canonical)
        else:
            logger.info("No spool reference found for %s %s code '%s'",
                        product.company, series, spool_code)

        return product

    def find_my_company_equivalents(
        self,
        competitor: HydraulicProduct,
        my_company_name: str = "Danfoss",
        top_k: int = 5,
    ) -> list[MatchResult]:
        """Find Danfoss equivalents for a competitor product.
        Uses semantic search + spec comparison + reranking.
        Falls back to category-based DB search if semantic search yields nothing."""

        # Build query from competitor product
        query = self.vs._build_indexable_text(competitor)

        # Enrich query with cross-reference series hint (additive, not replacing)
        xref_hints = self.db.lookup_series_by_competitor_prefix(
            competitor.model_code, competitor.company
        )
        if xref_hints:
            # Use the most specific match (longest competitor_series prefix)
            best_hint = xref_hints[0]
            hint_series = best_hint["my_company_series"]
            hint_type = best_hint.get("product_type", "")
            hint_text = f" {my_company_name} equivalent series: {hint_series}"
            if hint_type:
                hint_text += f" ({hint_type})"
            query = query + hint_text
            logger.info("Cross-reference hint: %s %s -> %s %s",
                        competitor.company, competitor.model_code,
                        my_company_name, hint_series)

        # Semantic search in Danfoss collection with category
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

        # Fallback: if vector store has no Danfoss products, pull from DB directly
        if not semantic_results:
            db_candidates = []
            if competitor.category:
                db_candidates = self.db.get_products_by_category(
                    competitor.category, company=my_company_name
                )
            if not db_candidates:
                db_candidates = self.db.get_all_products(company=my_company_name)

            # Score each DB candidate directly (no semantic score available)
            match_results = []
            for candidate in db_candidates[:50]:
                confidence, breakdown = self.db.spec_comparison(
                    competitor, candidate, semantic_score=0.0
                )
                match = MatchResult(
                    my_company_product=candidate,
                    competitor_product=competitor,
                    confidence_score=round(confidence, 3),
                    score_breakdown=breakdown,
                    meets_threshold=confidence >= CONFIDENCE_THRESHOLD,
                )
                match_results.append(match)

            match_results.sort(key=lambda m: m.confidence_score, reverse=True)
            return match_results[:top_k]

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
