"""
ProductMatchPro - Chat Agent
Replaces the LangGraph-based MatchGraph with a simple stateful class.
Query parsing uses TIER_LOW (Haiku), response generation uses TIER_MID (Sonnet).
"""

import json
import logging
import re
import uuid
from typing import Optional

from models import HydraulicProduct, MatchResult, CONFIDENCE_THRESHOLD
from storage.product_db import ProductDB
from storage.vector_store import VectorStore
from tools.lookup_tools import LookupTools
from tools.llm_client import call_llm, call_llm_json, TIER_LOW, TIER_MID
from prompts import (
    QUERY_PARSER_PROMPT,
    COMPARISON_NARRATIVE_PROMPT,
    BELOW_THRESHOLD_PROMPT,
    CLARIFICATION_PROMPT,
    NO_MATCH_PROMPT,
    KB_QA_PROMPT,
)

logger = logging.getLogger(__name__)

# Configurable
MY_COMPANY_NAME = "Danfoss"
SALES_CONTACT = "sales@danfoss.com | +44 (0)XXX XXX XXXX"


class ChatAgent:
    """Simple state-machine chat agent for distributor product matching.

    Replaces the LangGraph-based MatchGraph with direct function calls.
    Same public interface: search_sync(message, thread_id) -> str.
    """

    def __init__(
        self,
        db: Optional[ProductDB] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.db = db or ProductDB()
        self.vs = vector_store or VectorStore()
        self.lookup = LookupTools(self.db, self.vs)
        self.conversations: dict[str, list[dict]] = {}

    # ── Public Interface ──────────────────────────────────────────────

    def search_sync(
        self, message: str, thread_id: Optional[str] = None
    ) -> str:
        """Process a user message and return a response string.

        This is the main entry point, matching the MatchGraph.search_sync() interface.
        """
        if not thread_id:
            thread_id = str(uuid.uuid4())

        # Parse the query
        parsed = self._parse_query(message)
        intent = parsed.get("intent", "lookup")

        # Route: info queries go to KB, everything else to product matching
        if intent == "info":
            return self._handle_kb_query(message, parsed)
        return self._handle_product_matching(message, parsed)

    # ── Query Parsing (TIER_LOW — fast/cheap) ─────────────────────────

    def _parse_query(self, message: str) -> dict:
        """Parse user query to extract model code, competitor, and intent."""
        trimmed = message[:500]

        try:
            result = call_llm_json(
                TIER_LOW,
                QUERY_PARSER_PROMPT,
                f"User query: {trimmed}",
                max_tokens=512,
            )
            return result if isinstance(result, dict) else self._regex_parse_fallback(trimmed)
        except Exception as e:
            logger.warning("LLM parse_query failed, using regex fallback: %s", e)
            return self._regex_parse_fallback(trimmed)

    @staticmethod
    def _regex_parse_fallback(user_message: str) -> dict:
        """Extract model code and competitor from user message using regex."""
        text = user_message.strip()

        known_brands = [
            "bosch rexroth", "rexroth", "parker", "atos",
            "moog", "hydac", "bucher", "casappa", "denison",
        ]
        competitor_name = None
        for brand in known_brands:
            if brand in text.lower():
                competitor_name = brand.title()
                break

        category_hints = {
            "proportional directional": "proportional_directional_valves",
            "proportional valve": "proportional_directional_valves",
            "directional": "directional_valves",
            "pressure": "pressure_valves",
            "flow": "flow_valves",
            "pump": "pumps",
            "motor": "motors",
            "cylinder": "cylinders",
            "filter": "filters",
            "accumulator": "accumulators",
        }
        category = None
        for hint, cat in category_hints.items():
            if hint in text.lower():
                category = cat
                break

        model_match = re.search(r'[A-Z0-9][A-Z0-9\-/\.]{3,}[A-Z0-9]', text, re.IGNORECASE)
        model_code = model_match.group(0) if model_match else text.split()[0] if text else ""

        return {
            "model_code": model_code,
            "competitor_name": competitor_name,
            "category": category,
            "specs": {},
            "is_followup": False,
            "intent": "lookup",
        }

    # ── Product Matching Flow ─────────────────────────────────────────

    def _handle_product_matching(self, message: str, parsed: dict) -> str:
        """Full product matching pipeline: lookup → find equivalents → respond."""

        # Step 1: Lookup competitor product
        model_code = parsed.get("model_code", message)
        competitor_name = parsed.get("competitor_name")
        category = parsed.get("category")

        if not model_code:
            return NO_MATCH_PROMPT.format(query=message, contact_info=SALES_CONTACT)

        result = self.lookup.identify_competitor_product(
            partial_code=model_code,
            competitor_name=competitor_name,
            category=category,
        )

        # Step 2: Handle different statuses
        status = result.get("status", "not_found")

        if status == "confirmed":
            confirmed_product = result.get("confirmed_equivalent")
            comp_product = result.get("product")
            if confirmed_product:
                return self._format_confirmed_match(confirmed_product, comp_product)

        if status == "ambiguous":
            options = result.get("options", [])
            option_texts = []
            for i, (product, score) in enumerate(options[:5]):
                option_texts.append(
                    f"{i+1}. {product.model_code} - {product.product_name} "
                    f"({product.company}) [{score:.0%} match]"
                )
            return CLARIFICATION_PROMPT.format(
                query=message,
                options="\n".join(option_texts),
            )

        if status == "not_found":
            return NO_MATCH_PROMPT.format(query=message, contact_info=SALES_CONTACT)

        # status == "found" — find equivalents
        competitor = result["product"]

        # Enrich competitor spool data
        competitor = self.lookup.enrich_competitor_spool(competitor)

        matches = self.lookup.find_my_company_equivalents(
            competitor=competitor,
            my_company_name=MY_COMPANY_NAME,
            top_k=5,
        )

        # Step 3: Generate response
        return self._generate_match_response(message, competitor, matches)

    def _generate_match_response(
        self,
        query: str,
        competitor: HydraulicProduct,
        matches: list[MatchResult],
    ) -> str:
        """Generate the final match response."""
        if not matches:
            return self._no_candidates_response(competitor)

        best_match = matches[0]

        if best_match.meets_threshold:
            message = self._format_match_above_threshold(best_match, matches[1:3])
        else:
            message = self._format_match_below_threshold(matches[:3])

        # Generate narrative
        comp_product = best_match.competitor_product
        my_product = best_match.my_company_product

        narrative_prompt = COMPARISON_NARRATIVE_PROMPT.format(
            my_company=MY_COMPANY_NAME,
            competitor_code=comp_product.model_code,
            competitor_company=comp_product.company,
            my_company_code=my_product.model_code,
            confidence=f"{best_match.confidence_score:.0%}",
            spec_table=self._build_spec_table(comp_product, my_product),
            score_breakdown=self._format_breakdown(best_match.score_breakdown),
        )

        try:
            narrative = call_llm(
                TIER_MID,
                "You are a hydraulic product expert helping a distributor.",
                narrative_prompt,
                max_tokens=1024,
            )
            message += f"\n\n**Analysis:**\n{narrative}"
        except Exception:
            pass

        return message

    def _no_candidates_response(self, competitor: HydraulicProduct) -> str:
        """Build detailed explanation when no candidates found."""
        my_count = len(self.db.get_all_products(company=MY_COMPANY_NAME))
        vs_counts = self.vs.get_collection_counts()
        vs_count = vs_counts.get("my_company", 0)

        lines = [
            "I found the competitor product, but couldn't find a matching "
            f"{MY_COMPANY_NAME} equivalent. Here's why:\n",
        ]

        if my_count == 0:
            lines.append(
                f"• **No {MY_COMPANY_NAME} products** have been uploaded to the "
                f"database yet. Please ask your administrator to upload {MY_COMPANY_NAME} "
                f"catalogues or datasheets first."
            )
        elif vs_count == 0:
            lines.append(
                f"• There are {my_count} {MY_COMPANY_NAME} product(s) in the database, "
                f"but the search index appears to be empty."
            )
        else:
            lines.append(
                f"• There are {my_count} {MY_COMPANY_NAME} product(s) in the database "
                f"and {vs_count} indexed for search, but none were close enough."
            )
            if competitor.category:
                lines.append(
                    f"• The competitor product's category is **{competitor.category}**. "
                    f"Make sure {MY_COMPANY_NAME} products in the same category have been uploaded."
                )

        lines.append(
            f"\nPlease contact your local sales representative for assistance:\n{SALES_CONTACT}"
        )
        return "\n".join(lines)

    # ── KB Q&A Flow ───────────────────────────────────────────────────

    def _handle_kb_query(self, message: str, parsed: dict) -> str:
        """Handle information/knowledge-base queries."""
        model_code = parsed.get("model_code")
        competitor_name = parsed.get("competitor_name")

        # Multi-pass retrieval
        results = self.vs.search_guides_with_metadata(
            query=message,
            company=competitor_name,
            model_code=model_code if model_code else None,
            n_results=15,
        )

        if not results and model_code:
            results = self.vs.search_guides_with_metadata(
                query=message, company=competitor_name, n_results=15,
            )
        if not results and competitor_name:
            results = self.vs.search_guides_with_metadata(
                query=message, n_results=15,
            )

        # Filter and deduplicate
        kb_chunks = []
        seen_texts = set()
        for chunk_id, text, metadata, score in results:
            if score < 0.15:
                continue
            text_key = text[:100]
            if text_key in seen_texts:
                continue
            seen_texts.add(text_key)
            kb_chunks.append({
                "text": text,
                "source": metadata.get("source_document", "unknown"),
                "company": metadata.get("company", ""),
                "score": round(float(score), 3),
            })

        if not kb_chunks:
            return (
                "I couldn't find specific information about that in our product documentation.\n\n"
                f"Please contact your local sales representative for assistance:\n{SALES_CONTACT}"
            )

        # Build context
        context_parts = []
        source_set = set()
        for i, chunk in enumerate(kb_chunks, 1):
            context_parts.append(f"[{i}] {chunk['text']}")
            if chunk["source"] and chunk["source"] != "unknown":
                source_set.add(chunk["source"])

        context_block = "\n\n".join(context_parts)
        sources_text = ", ".join(sorted(source_set)) if source_set else "product documentation"

        prompt = KB_QA_PROMPT.format(
            my_company=MY_COMPANY_NAME,
            question=message,
            context=context_block,
            sources=sources_text,
        )

        try:
            answer = call_llm(
                TIER_MID,
                "You are a knowledgeable hydraulic product expert.",
                prompt,
                max_tokens=1024,
            )
            return answer
        except Exception as e:
            logger.warning("KB answer generation failed: %s", e)
            return (
                "I found some relevant information but encountered an error. "
                f"Please try again or contact:\n{SALES_CONTACT}"
            )

    # ── Formatting Helpers ────────────────────────────────────────────

    def _format_match_above_threshold(
        self, best: MatchResult, alternatives: list[MatchResult]
    ) -> str:
        comp = best.competitor_product
        my = best.my_company_product
        confidence_pct = f"{best.confidence_score:.0%}"
        bar = self._confidence_bar(best.confidence_score)

        lines = [
            f"**Match Found! Confidence: {confidence_pct}** {bar}",
            "",
            f"**Competitor:** {comp.model_code} ({comp.company})"
            + (f" - {comp.product_name}" if comp.product_name else ""),
            f"**{MY_COMPANY_NAME} Equivalent:** {my.model_code}"
            + (f" - {my.product_name}" if my.product_name else ""),
            "",
            self._build_spec_table(comp, my),
        ]

        if alternatives:
            lines.append("\n**Other options:**")
            for alt in alternatives:
                alt_pct = f"{alt.confidence_score:.0%}"
                lines.append(
                    f"- {alt.my_company_product.model_code} ({alt_pct} confidence)"
                )

        return "\n".join(lines)

    def _format_match_below_threshold(self, matches: list[MatchResult]) -> str:
        partial_lines = []
        for m in matches:
            pct = f"{m.confidence_score:.0%}"
            partial_lines.append(
                f"- {m.my_company_product.model_code} ({pct} confidence)"
            )
        partial_text = "\n".join(partial_lines)
        best_pct = f"{matches[0].confidence_score:.0%}" if matches else "0%"

        return BELOW_THRESHOLD_PROMPT.format(
            confidence=best_pct,
            threshold=f"{CONFIDENCE_THRESHOLD:.0%}",
            partial_matches=partial_text,
            my_company=MY_COMPANY_NAME,
            contact_info=SALES_CONTACT,
        )

    def _format_confirmed_match(
        self, confirmed: HydraulicProduct, competitor: Optional[HydraulicProduct]
    ) -> str:
        lines = [
            "**Confirmed Equivalent** (verified by our team)",
            "",
        ]
        if competitor:
            lines.append(f"**Competitor:** {competitor.model_code} ({competitor.company})")
        lines.append(f"**{MY_COMPANY_NAME} Equivalent:** {confirmed.model_code}")
        if confirmed.product_name:
            lines.append(f"**Product:** {confirmed.product_name}")
        if competitor:
            lines.append("")
            lines.append(self._build_spec_table(competitor, confirmed))
        return "\n".join(lines)

    @staticmethod
    def _build_spec_table(comp: HydraulicProduct, my: HydraulicProduct) -> str:
        """Build a markdown spec comparison table."""
        rows = []
        spec_display = [
            ("Category", comp.category, my.category),
            ("Max Pressure", f"{comp.max_pressure_bar} bar" if comp.max_pressure_bar else "-",
             f"{my.max_pressure_bar} bar" if my.max_pressure_bar else "-"),
            ("Max Flow", f"{comp.max_flow_lpm} lpm" if comp.max_flow_lpm else "-",
             f"{my.max_flow_lpm} lpm" if my.max_flow_lpm else "-"),
            ("Valve Size", comp.valve_size or "-", my.valve_size or "-"),
            ("Spool Type", comp.spool_type or "-", my.spool_type or "-"),
            ("Actuation", comp.actuator_type or "-", my.actuator_type or "-"),
            ("Coil Voltage", comp.coil_voltage or "-", my.coil_voltage or "-"),
            ("Port Size", comp.port_size or "-", my.port_size or "-"),
            ("Mounting", comp.mounting or "-", my.mounting or "-"),
            ("Mounting Pattern", comp.mounting_pattern or "-", my.mounting_pattern or "-"),
            ("Seal Material", comp.seal_material or "-", my.seal_material or "-"),
            ("Body Material", comp.body_material or "-", my.body_material or "-"),
        ]

        for label, comp_val, my_val in spec_display:
            if comp_val != "-" or my_val != "-":
                match_indicator = ""
                if comp_val != "-" and my_val != "-":
                    if str(comp_val).lower() == str(my_val).lower():
                        match_indicator = " ✓"
                    else:
                        match_indicator = " ✗"
                rows.append(f"| {label} | {comp_val} | {my_val}{match_indicator} |")

        if not rows:
            return "*No detailed specifications available for comparison*"

        header = "| Specification | Competitor | Ours |\n|---|---|---|"
        return header + "\n" + "\n".join(rows)

    @staticmethod
    def _format_breakdown(breakdown) -> str:
        lines = []
        for field in [
            "category_match", "pressure_match", "flow_match", "valve_size_match",
            "actuator_type_match", "coil_voltage_match", "spool_function_match",
            "port_match", "mounting_match", "seal_material_match",
            "temp_range_match", "semantic_similarity",
        ]:
            val = getattr(breakdown, field, 0.0)
            label = field.replace("_", " ").title()
            lines.append(f"  {label}: {val:.0%}")
        lines.append(f"  Spec Coverage: {breakdown.spec_coverage:.0%}")
        return "\n".join(lines)

    @staticmethod
    def _confidence_bar(score: float, width: int = 10) -> str:
        filled = int(score * width)
        empty = width - filled
        return "█" * filled + "░" * empty
