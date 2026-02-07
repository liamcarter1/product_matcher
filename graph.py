"""
ProductMatchPro - LangGraph Matching Workflow
Stateful graph for competitor product lookup, matching, and conversation.
"""

from typing import Annotated, Any, Optional
from typing_extensions import TypedDict
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
from langchain_openai import ChatOpenAI
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from dotenv import load_dotenv
import json
import re
import logging
import uuid

logger = logging.getLogger(__name__)

from models import HydraulicProduct, MatchResult, CONFIDENCE_THRESHOLD
from storage.product_db import ProductDB
from storage.vector_store import VectorStore
from tools.lookup_tools import LookupTools
from prompts import (
    QUERY_PARSER_PROMPT,
    COMPARISON_NARRATIVE_PROMPT,
    BELOW_THRESHOLD_PROMPT,
    CLARIFICATION_PROMPT,
    NO_MATCH_PROMPT,
)

load_dotenv(override=True)

# Configurable
MY_COMPANY_NAME = "Danfoss"
SALES_CONTACT = "sales@danfoss.com | +44 (0)XXX XXX XXXX"


class MatchState(TypedDict):
    messages: Annotated[list[Any], add_messages]
    query: str
    parsed_query: Optional[dict]
    identified_competitor: Optional[dict]
    candidate_matches: Optional[list[dict]]
    needs_clarification: bool
    clarification_options: Optional[list[str]]


class MatchGraph:
    """LangGraph-based matching workflow for the distributor app."""

    def __init__(
        self,
        db: Optional[ProductDB] = None,
        vector_store: Optional[VectorStore] = None,
    ):
        self.db = db or ProductDB()
        self.vs = vector_store or VectorStore()
        self.lookup = LookupTools(self.db, self.vs)
        self.llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.1)
        self.memory = MemorySaver()
        self.graph = self._build_graph()

    def _build_graph(self) -> StateGraph:
        graph = StateGraph(MatchState)

        # Add nodes
        graph.add_node("parse_query", self.parse_query)
        graph.add_node("lookup_competitor", self.lookup_competitor)
        graph.add_node("clarify", self.clarify)
        graph.add_node("find_equivalents", self.find_equivalents)
        graph.add_node("generate_response", self.generate_response)

        # Edges
        graph.add_edge(START, "parse_query")
        graph.add_edge("parse_query", "lookup_competitor")
        graph.add_conditional_edges(
            "lookup_competitor",
            self._route_after_lookup,
            {
                "clarify": "clarify",
                "find_equivalents": "find_equivalents",
                "no_match": "generate_response",
                "confirmed": "generate_response",
            },
        )
        graph.add_edge("clarify", END)
        graph.add_edge("find_equivalents", "generate_response")
        graph.add_edge("generate_response", END)

        return graph.compile(checkpointer=self.memory)

    # ── Graph Nodes ───────────────────────────────────────────────────

    def parse_query(self, state: MatchState) -> dict:
        """Parse the user's query to extract model code, competitor, and intent.
        Falls back to regex extraction if the LLM call fails (e.g. API down)."""
        messages = state["messages"]
        user_message = ""
        for msg in reversed(messages):
            if isinstance(msg, HumanMessage) or (isinstance(msg, dict) and msg.get("role") == "user"):
                user_message = msg.content if isinstance(msg, HumanMessage) else msg.get("content", "")
                break

        # Truncate extremely long input to prevent API cost abuse
        user_message_trimmed = user_message[:500]

        try:
            response = self.llm.invoke([
                SystemMessage(content=QUERY_PARSER_PROMPT),
                HumanMessage(content=f"User query: {user_message_trimmed}"),
            ])
            parsed = json.loads(response.content)
        except json.JSONDecodeError:
            parsed = self._regex_parse_fallback(user_message_trimmed)
        except Exception as e:
            logger.warning(f"LLM parse_query failed, using regex fallback: {e}")
            parsed = self._regex_parse_fallback(user_message_trimmed)

        return {
            "query": user_message,
            "parsed_query": parsed,
        }

    @staticmethod
    def _regex_parse_fallback(user_message: str) -> dict:
        """Extract model code and competitor from user message using regex.
        Used when the LLM is unavailable."""
        text = user_message.strip()

        # Known competitor names to detect
        known_brands = [
            "bosch rexroth", "rexroth", "parker", "atos",
            "moog", "hydac", "bucher", "casappa", "denison",
        ]
        competitor_name = None
        for brand in known_brands:
            if brand in text.lower():
                competitor_name = brand.title()
                break

        # Extract category hints
        category_hints = {
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

        # Try to extract a model code (alphanumeric with dashes/slashes)
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

    def lookup_competitor(self, state: MatchState) -> dict:
        """Look up the competitor product from the parsed query."""
        parsed = state.get("parsed_query", {})
        model_code = parsed.get("model_code", state.get("query", ""))
        competitor_name = parsed.get("competitor_name")

        if not model_code:
            return {
                "identified_competitor": {"status": "not_found"},
                "needs_clarification": False,
            }

        result = self.lookup.identify_competitor_product(
            partial_code=model_code,
            competitor_name=competitor_name,
            category=parsed.get("category"),
        )

        if result["status"] == "confirmed":
            # We have a confirmed equivalent - skip to response
            confirmed_product = result.get("confirmed_equivalent")
            comp_product = result.get("product")
            return {
                "identified_competitor": {
                    "status": "confirmed",
                    "product": comp_product.model_dump() if comp_product else None,
                    "confirmed_equivalent": confirmed_product.model_dump() if confirmed_product else None,
                },
                "needs_clarification": False,
            }

        if result["status"] == "ambiguous":
            options = result.get("options", [])
            option_texts = []
            option_data = []
            for i, (product, score) in enumerate(options[:5]):
                option_texts.append(
                    f"{i+1}. {product.model_code} - {product.product_name} "
                    f"({product.company}) [{score:.0%} match]"
                )
                option_data.append(product.model_dump())
            return {
                "identified_competitor": {
                    "status": "ambiguous",
                    "options": option_data,
                },
                "needs_clarification": True,
                "clarification_options": option_texts,
            }

        if result["status"] == "found":
            product = result["product"]
            return {
                "identified_competitor": {
                    "status": "found",
                    "product": product.model_dump(),
                    "match_score": result.get("match_score", 0.0),
                },
                "needs_clarification": False,
            }

        return {
            "identified_competitor": {"status": "not_found"},
            "needs_clarification": False,
        }

    def clarify(self, state: MatchState) -> dict:
        """Ask the user to clarify which product they meant."""
        options = state.get("clarification_options", [])
        query = state.get("query", "")
        options_text = "\n".join(options)

        message = CLARIFICATION_PROMPT.format(
            query=query, options=options_text
        )

        return {
            "messages": [AIMessage(content=message)],
        }

    def find_equivalents(self, state: MatchState) -> dict:
        """Find {my_company} equivalents for the identified competitor product."""
        comp_data = state.get("identified_competitor", {})
        product_dict = comp_data.get("product")

        if not product_dict:
            return {"candidate_matches": []}

        competitor = HydraulicProduct(**product_dict)
        matches = self.lookup.find_my_company_equivalents(
            competitor=competitor,
            my_company_name=MY_COMPANY_NAME,
            top_k=5,
        )

        return {
            "candidate_matches": [m.model_dump() for m in matches],
        }

    def generate_response(self, state: MatchState) -> dict:
        """Generate the final response for the distributor."""
        comp_data = state.get("identified_competitor", {})
        status = comp_data.get("status", "not_found")

        # Handle confirmed equivalent
        if status == "confirmed":
            confirmed = comp_data.get("confirmed_equivalent")
            comp = comp_data.get("product")
            if confirmed:
                confirmed_prod = HydraulicProduct(**confirmed)
                message = self._format_confirmed_match(
                    confirmed_prod, HydraulicProduct(**comp) if comp else None
                )
                return {"messages": [AIMessage(content=message)]}

        # Handle no match
        if status == "not_found":
            query = state.get("query", "")
            message = NO_MATCH_PROMPT.format(
                query=query, contact_info=SALES_CONTACT
            )
            return {"messages": [AIMessage(content=message)]}

        # Handle found matches
        matches_data = state.get("candidate_matches", [])
        if not matches_data:
            query = state.get("query", "")
            message = (
                f"I found the competitor product, but couldn't find a matching "
                f"{MY_COMPANY_NAME} equivalent in our database.\n\n"
                f"Please contact your local sales representative:\n{SALES_CONTACT}"
            )
            return {"messages": [AIMessage(content=message)]}

        matches = [MatchResult(**m) for m in matches_data]
        best_match = matches[0]

        if best_match.meets_threshold:
            message = self._format_match_above_threshold(best_match, matches[1:3])
        else:
            message = self._format_match_below_threshold(matches[:3])

        # Generate narrative from LLM
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
            narrative_response = self.llm.invoke([
                HumanMessage(content=narrative_prompt),
            ])
            message += f"\n\n**Analysis:**\n{narrative_response.content}"
        except Exception:
            pass

        return {"messages": [AIMessage(content=message)]}

    # ── Routing ───────────────────────────────────────────────────────

    def _route_after_lookup(self, state: MatchState) -> str:
        comp_data = state.get("identified_competitor", {})
        status = comp_data.get("status", "not_found")

        if status == "confirmed":
            return "confirmed"
        if state.get("needs_clarification"):
            return "clarify"
        if status == "found":
            return "find_equivalents"
        return "no_match"

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

        # Only include rows where at least one side has data
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

    # ── Public Interface ──────────────────────────────────────────────

    async def search(
        self, message: str, thread_id: Optional[str] = None
    ) -> str:
        """Run a search query through the matching graph.
        Returns the assistant's response text."""
        if not thread_id:
            thread_id = str(uuid.uuid4())

        config = {"configurable": {"thread_id": thread_id}}
        state = {
            "messages": [HumanMessage(content=message)],
            "query": message,
            "needs_clarification": False,
        }

        result = await self.graph.ainvoke(state, config=config)

        # Extract the last AI message
        for msg in reversed(result.get("messages", [])):
            if isinstance(msg, AIMessage):
                return msg.content

        return "I encountered an error processing your request. Please try again."

    def search_sync(
        self, message: str, thread_id: Optional[str] = None
    ) -> str:
        """Synchronous version of search for Gradio integration."""
        import asyncio
        try:
            loop = asyncio.get_event_loop()
            if loop.is_running():
                import concurrent.futures
                with concurrent.futures.ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        asyncio.run, self.search(message, thread_id)
                    )
                    return future.result()
            else:
                return loop.run_until_complete(self.search(message, thread_id))
        except RuntimeError:
            return asyncio.run(self.search(message, thread_id))
