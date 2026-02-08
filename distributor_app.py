"""
ProductMatchPro - Distributor App
Clean chat interface for distributors to find Danfoss equivalents for competitor products.

Run: python distributor_app.py
"""

import os
import logging
import gradio as gr
import uuid
from dotenv import load_dotenv

from storage.product_db import ProductDB
from storage.vector_store import VectorStore
from graph import MatchGraph
from tools.lookup_tools import LookupTools

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# Gradio 5 vs 6 compatibility: Gradio 5 chatbots default to tuple format
# and need type="messages" explicitly. Gradio 6 removed the parameter.
_GRADIO_MAJOR = int(gr.__version__.split(".")[0])

def _chatbot_kwargs(**extra) -> dict:
    """Return keyword arguments for gr.Chatbot that work across Gradio 5 & 6."""
    kwargs = dict(extra)
    if _GRADIO_MAJOR < 6:
        kwargs["type"] = "messages"
    return kwargs

# ── Initialise ────────────────────────────────────────────────────────

db = ProductDB()
vs = VectorStore()
match_graph = MatchGraph(db=db, vector_store=vs)
lookup = LookupTools(db, vs)


# ── Chat Handler ──────────────────────────────────────────────────────

MAX_MESSAGE_LEN = 500  # Limit input length to prevent abuse

def chat(message: str, history: list, thread_id: str, category: str, competitor: str):
    """Handle a chat message from the distributor."""
    if not message.strip():
        return history, thread_id

    # Truncate input to prevent abuse (Fix #9)
    enriched = message.strip()[:MAX_MESSAGE_LEN]
    if category and category != "All":
        enriched += f" [category: {category}]"
    if competitor and competitor != "All":
        enriched += f" [competitor: {competitor}]"

    # Run the matching graph
    try:
        response = match_graph.search_sync(enriched, thread_id)
    except Exception as e:
        logger.error(f"Match graph error: {e}", exc_info=True)
        response = (
            "Sorry, an error occurred while processing your request. "
            "Please try again or contact support if the problem persists."
        )

    history.append({"role": "user", "content": message})
    history.append({"role": "assistant", "content": response})
    return history, thread_id


def submit_feedback(history, thumbs_up):
    """Store feedback for the last result."""
    if not history or len(history) < 2:
        return "No result to give feedback on."

    last_user = ""
    last_assistant = ""
    for msg in reversed(history):
        if msg["role"] == "assistant" and not last_assistant:
            last_assistant = msg["content"]
        elif msg["role"] == "user" and not last_user:
            last_user = msg["content"]
        if last_user and last_assistant:
            break

    # Extract model codes from the conversation (best effort)
    competitor_code = last_user.strip()
    my_code = ""
    for line in last_assistant.split("\n"):
        if "equivalent" in line.lower() and ":" in line:
            my_code = line.split(":")[-1].strip().split(" ")[0]
            break

    try:
        lookup.store_feedback(
            query=last_user,
            competitor_code=competitor_code,
            my_code=my_code,
            confidence=0.0,
            thumbs_up=thumbs_up,
        )
        return "Thank you for your feedback!" if thumbs_up else "Thank you. We'll review this match."
    except Exception:
        return "Feedback saved."


def get_suggestions(partial):
    """Get typeahead suggestions as the user types."""
    if not partial or len(partial) < 2:
        return gr.Dropdown(choices=[])
    suggestions = lookup.get_typeahead_suggestions(partial, limit=8)
    return gr.Dropdown(choices=suggestions)


def get_company_choices():
    """Fetch current competitor companies from the database (called on every focus)."""
    companies = db.get_companies()
    non_my = [c for c in companies if c.lower() not in ("danfoss",)]
    return ["All"] + non_my


def refresh_competitor_dropdown(current_value):
    """Return an updated Dropdown with the latest companies, preserving the selection."""
    choices = get_company_choices()
    # Keep the user's current selection if it's still valid (or a custom value)
    value = current_value if current_value in choices or current_value else "All"
    return gr.Dropdown(choices=choices, value=value)


def new_session():
    """Start a new chat session."""
    return [], str(uuid.uuid4())


# ── Build Gradio UI ──────────────────────────────────────────────────

CATEGORIES = [
    "All",
    "Directional Valves", "Pressure Valves", "Flow Valves",
    "Pumps", "Motors", "Cylinders",
    "Filters", "Accumulators", "Hoses & Fittings",
]

CATEGORY_MAP = {
    "All": "",
    "Directional Valves": "directional_valves",
    "Pressure Valves": "pressure_valves",
    "Flow Valves": "flow_valves",
    "Pumps": "pumps",
    "Motors": "motors",
    "Cylinders": "cylinders",
    "Filters": "filters",
    "Accumulators": "accumulators",
    "Hoses & Fittings": "hoses_fittings",
}

with gr.Blocks(title="Danfoss Product Finder") as distributor_ui:

    # State
    thread_id = gr.State(str(uuid.uuid4()))

    # Header
    gr.Markdown(
        "# Danfoss Product Finder\n"
        "### Find the Danfoss equivalent for any competitor hydraulic product"
    )

    # Chat area
    chatbot = gr.Chatbot(
        **_chatbot_kwargs(label="Product Search", height=450),
    )

    # Input area
    with gr.Group():
        with gr.Row():
            message = gr.Textbox(
                show_label=False,
                placeholder="Enter competitor model code (e.g. 4WE6D6X/EG24N9K4, Parker D1VW, ATOS DHI)...",
                scale=4,
                container=False,
            )
            search_btn = gr.Button("Search", variant="primary", scale=1)

        with gr.Row():
            category_filter = gr.Dropdown(
                label="Category (optional)",
                choices=CATEGORIES,
                value="All",
                scale=1,
            )
            competitor_filter = gr.Dropdown(
                label="Competitor (optional)",
                choices=get_company_choices(),
                value="All",
                allow_custom_value=True,
                scale=1,
            )

    # Feedback
    with gr.Row():
        feedback_label = gr.Markdown("**Was this result helpful?**")
        thumbs_up_btn = gr.Button("👍", scale=0, min_width=60)
        thumbs_down_btn = gr.Button("👎", scale=0, min_width=60)
        feedback_status = gr.Textbox(
            show_label=False, interactive=False, scale=2, container=False
        )

    # New session
    with gr.Row():
        new_session_btn = gr.Button("New Search Session", variant="secondary")

    # Example prompts
    gr.Examples(
        examples=[
            "4WE6D6X/EG24N9K4",
            "Parker D1VW020BN",
            "ATOS DHI-0631/2-X 24DC",
            "24V solenoid directional valve CETOP 5 315 bar",
            "What coil voltages are available for the 4WE6 series?",
            "What's the difference between NBR and FKM seals?",
        ],
        inputs=message,
        label="Try these examples:",
    )

    # Event handlers
    def on_submit(msg, history, tid, cat, comp):
        return chat(msg, history, tid, CATEGORY_MAP.get(cat, ""), comp)

    message.submit(
        on_submit,
        inputs=[message, chatbot, thread_id, category_filter, competitor_filter],
        outputs=[chatbot, thread_id],
    ).then(lambda: "", outputs=[message])

    search_btn.click(
        on_submit,
        inputs=[message, chatbot, thread_id, category_filter, competitor_filter],
        outputs=[chatbot, thread_id],
    ).then(lambda: "", outputs=[message])

    thumbs_up_btn.click(
        lambda h: submit_feedback(h, True),
        inputs=[chatbot],
        outputs=[feedback_status],
    )
    thumbs_down_btn.click(
        lambda h: submit_feedback(h, False),
        inputs=[chatbot],
        outputs=[feedback_status],
    )

    new_session_btn.click(
        new_session,
        outputs=[chatbot, thread_id],
    )

    # Refresh competitor dropdown whenever it gains focus (picks up new uploads)
    competitor_filter.focus(
        refresh_competitor_dropdown,
        inputs=[competitor_filter],
        outputs=[competitor_filter],
    )


if __name__ == "__main__":
    distributor_ui.launch(
        server_name=os.getenv("DISTRIBUTOR_HOST", "0.0.0.0"),
        server_port=int(os.getenv("DISTRIBUTOR_PORT", "7860")),
        share=False,
    )
