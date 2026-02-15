"""
ProductMatchPro - Admin Console (Back-Office)
Used by Danfoss staff to upload PDFs, manage products, review feedback, and configure settings.
NOT visible to distributors.

Run: python admin_app.py
"""

import os
import logging
import gradio as gr
import pandas as pd
import json
from pathlib import Path
from dotenv import load_dotenv

from models import UploadMetadata, DocumentType, ExtractedProduct
from storage.product_db import ProductDB
from storage.vector_store import VectorStore
from ingest import IngestionPipeline

load_dotenv(override=True)

logger = logging.getLogger(__name__)

# ── Configuration from environment ───────────────────────────────────
# Auth: set ADMIN_USERNAME and ADMIN_PASSWORD in .env to enable authentication
ADMIN_USERNAME = os.getenv("ADMIN_USERNAME")
ADMIN_PASSWORD = os.getenv("ADMIN_PASSWORD")
# Max PDF upload size in MB (default 50MB)
MAX_UPLOAD_SIZE_MB = int(os.getenv("MAX_UPLOAD_SIZE_MB", "50"))
# Max file size in bytes for server-side validation
MAX_UPLOAD_SIZE_BYTES = MAX_UPLOAD_SIZE_MB * 1024 * 1024
# PDF magic bytes
PDF_MAGIC = b"%PDF"

# ── Initialise storage ────────────────────────────────────────────────

db = ProductDB()
vs = VectorStore()
pipeline = IngestionPipeline(db, vs)

# Known companies for the dropdown (Danfoss always first)
KNOWN_COMPANIES = ["Danfoss", "Bosch Rexroth", "Parker", "ATOS"]

# Input length limits
MAX_MODEL_CODE_LEN = 200
MAX_COMPANY_NAME_LEN = 100
MAX_SEARCH_LEN = 200


def get_company_dropdown_choices():
    """Build company dropdown: Danfoss first, then DB companies, then known competitors."""
    db_companies = db.get_companies()
    # Start with Danfoss, then merge in DB companies and known list (no duplicates)
    seen = set()
    choices = []
    for c in ["Danfoss"] + db_companies + KNOWN_COMPANIES:
        if c.lower() not in seen:
            seen.add(c.lower())
            choices.append(c)
    return choices


def _validate_pdf(file_path: str) -> str | None:
    """Validate that a file is a real PDF and within size limits.
    Returns an error message string, or None if valid."""
    path = Path(file_path)
    if not path.exists():
        return f"File not found: {path.name}"
    # Check file size
    size = path.stat().st_size
    if size > MAX_UPLOAD_SIZE_BYTES:
        return f"{path.name} is too large ({size / 1024 / 1024:.1f}MB, max {MAX_UPLOAD_SIZE_MB}MB)"
    if size == 0:
        return f"{path.name} is empty"
    # Check PDF magic bytes
    try:
        with open(file_path, "rb") as f:
            header = f.read(8)
        if not header.startswith(PDF_MAGIC):
            return f"{path.name} is not a valid PDF file"
    except Exception:
        return f"{path.name} could not be read"
    return None


def _sanitize_error(e: Exception) -> str:
    """Return a safe error message without leaking internals."""
    error_str = str(e)
    # Strip file paths and sensitive info
    if "OPENAI_API_KEY" in error_str or "api_key" in error_str.lower():
        return "API configuration error. Check your OpenAI API key."
    if len(error_str) > 200:
        error_str = error_str[:200] + "..."
    return error_str


# ── Upload Tab ────────────────────────────────────────────────────────

def process_upload(files, company, doc_type, category, pending_state):
    """Process one or more uploaded PDFs and return extracted products for review.
    All files in a single batch share the same company, doc type, and category.
    Uses gr.State for pending extractions to isolate concurrent sessions."""

    if files is None:
        return "Please upload one or more PDF files.", None, pending_state

    # Gradio file_count="multiple" with type="filepath" passes a list of string paths
    if isinstance(files, str):
        file_paths = [files]
    elif isinstance(files, list):
        file_paths = [str(f) for f in files]
    else:
        file_paths = [str(files)]

    company_name = (company.strip() if company else "Danfoss")[:MAX_COMPANY_NAME_LEN]

    all_extracted = []
    all_chunk_counts = 0
    file_summaries = []

    for file_path in file_paths:
        # Server-side PDF validation (Fix #6)
        validation_error = _validate_pdf(file_path)
        if validation_error:
            file_summaries.append(f"  {Path(file_path).name}: SKIPPED - {validation_error}")
            continue

        metadata = UploadMetadata(
            company=company_name,
            document_type=DocumentType(doc_type),
            category=category if category != "All / Auto-detect" else "",
            filename=Path(file_path).name,
        )

        try:
            extracted = pipeline.process_pdf(file_path, metadata)
            chunk_count = pipeline.index_guide_text(file_path, metadata)
            all_chunk_counts += chunk_count

            if extracted:
                all_extracted.extend(extracted)
                file_summaries.append(
                    f"  {metadata.filename}: {len(extracted)} products, {chunk_count} text chunks"
                )
            else:
                file_summaries.append(
                    f"  {metadata.filename}: no products extracted"
                )
        except Exception as e:
            logger.error(f"Error processing {Path(file_path).name}: {e}", exc_info=True)
            file_summaries.append(
                f"  {Path(file_path).name}: ERROR - {_sanitize_error(e)}"
            )

    if not all_extracted:
        summary = "No products could be extracted from any of the uploaded files.\n\n"
        summary += "File results:\n" + "\n".join(file_summaries)
        summary += "\n\nTry a different document type setting."
        return summary, None, pending_state

    # Store pending data in gr.State (Fix #2 — session-isolated, not global)
    pending_state = {
        "extractions": [ep.model_dump() for ep in all_extracted],
        "metadata": {
            "company": company_name,
            "document_type": doc_type,
            "category": category if category != "All / Auto-detect" else "",
            "filename": f"{len(file_paths)} files",
        },
    }

    # Build a review dataframe
    rows = []
    for ep in all_extracted:
        row = {
            "Source": ep.source if hasattr(ep, "source") else "llm",
            "Model Code": ep.model_code,
            "Product Name": ep.product_name,
            "Category": ep.category,
            "Confidence": f"{ep.confidence:.0%}",
        }
        # Add key specs
        for key in ["max_pressure_bar", "max_flow_lpm", "coil_voltage",
                    "valve_size", "spool_type", "seal_material", "actuator_type",
                    "port_size", "mounting", "num_ports"]:
            row[key] = ep.specs.get(key, "")
        rows.append(row)

    df = pd.DataFrame(rows)

    # Build source-aware status message
    source_counts = {}
    for ep in all_extracted:
        src = ep.source if hasattr(ep, "source") else "llm"
        source_counts[src] = source_counts.get(src, 0) + 1

    source_details = []
    if source_counts.get("table"):
        source_details.append(f"{source_counts['table']} from table extraction")
    if source_counts.get("llm"):
        source_details.append(f"{source_counts['llm']} from LLM text extraction")
    if source_counts.get("ordering_code"):
        source_details.append(f"{source_counts['ordering_code']} generated from ordering code table(s)")

    status = (
        f"Extracted {len(all_extracted)} products from {len(file_paths)} file(s). "
        f"Also indexed {all_chunk_counts} text chunks.\n"
        f"Sources: {', '.join(source_details)}.\n\n"
        f"File results:\n" + "\n".join(file_summaries) + "\n\n"
        f"Review the products below and click 'Confirm & Index' to add them to the database."
    )
    return status, df, pending_state


def confirm_extraction(pending_state):
    """Confirm the extracted products and store them.
    Uses gr.State for session isolation (Fix #2).
    Wraps in try/except for robustness (Fix #3)."""

    if not pending_state or not pending_state.get("extractions"):
        return "No pending extractions to confirm. Upload a PDF first.", pending_state

    try:
        # Reconstruct from state
        extractions = [ExtractedProduct(**ep) for ep in pending_state["extractions"]]
        meta_dict = pending_state["metadata"]
        metadata = UploadMetadata(
            company=meta_dict["company"],
            document_type=DocumentType(meta_dict["document_type"]),
            category=meta_dict.get("category", ""),
            filename=meta_dict.get("filename", ""),
        )

        count = pipeline.confirm_and_store(extractions, metadata)

        # Clear pending state after successful confirmation
        pending_state = {}
        return f"Successfully indexed {count} products into the database and vector store.", pending_state

    except Exception as e:
        logger.error(f"Error during confirm_and_store: {e}", exc_info=True)
        # Don't clear pending state on error — admin can retry
        return (
            f"Error during indexing: {_sanitize_error(e)}. "
            f"Your extracted data is preserved — you can retry."
        ), pending_state


# ── Product Database Tab ──────────────────────────────────────────────

def get_product_counts():
    """Get product counts by company and category."""
    counts = db.get_product_counts()
    if not counts:
        return pd.DataFrame(columns=["Company", "Category", "Count"])
    return pd.DataFrame(counts).rename(columns={
        "company": "Company", "category": "Category", "count": "Count"
    })


def search_products(search_term, company_filter):
    """Search products in the database."""
    # Truncate input (Fix #9)
    search_term = (search_term or "")[:MAX_SEARCH_LEN]

    if company_filter and company_filter != "All":
        products = db.get_all_products(company=company_filter)
    else:
        products = db.get_all_products()

    if search_term:
        search_upper = search_term.upper()
        products = [
            p for p in products
            if search_upper in p.model_code.upper()
            or search_upper in p.product_name.upper()
        ]

    rows = []
    for p in products[:100]:
        rows.append({
            "ID": p.id[:8],
            "Company": p.company,
            "Model Code": p.model_code,
            "Name": p.product_name,
            "Category": p.category,
            "Pressure (bar)": p.max_pressure_bar or "",
            "Flow (lpm)": p.max_flow_lpm or "",
            "Coil Voltage": p.coil_voltage or "",
            "Valve Size": p.valve_size or "",
            "Spool Type": p.spool_type or "",
            "Seal": p.seal_material or "",
            "Ports": p.num_ports or "",
            "Mounting": p.mounting or "",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def export_csv():
    """Export all products as CSV."""
    import tempfile
    try:
        products = db.get_all_products()
        rows = []
        for p in products:
            row = p.model_dump()
            row.pop("model_code_decoded", None)
            row.pop("raw_text", None)
            rows.append(row)
        df = pd.DataFrame(rows)
        csv_path = str(Path(tempfile.gettempdir()) / "products_export.csv")
        df.to_csv(csv_path, index=False)
        return csv_path
    except Exception as e:
        logger.error(f"CSV export error: {e}", exc_info=True)
        return None


def reindex_vectors():
    """Rebuild the vector store from the SQLite database."""
    try:
        products = db.get_all_products()
        vs.rebuild_from_products(products)
        return f"Re-indexed {len(products)} products in the vector store."
    except Exception as e:
        logger.error(f"Reindex error: {e}", exc_info=True)
        return f"Error during re-indexing: {_sanitize_error(e)}"


def delete_product(product_id):
    """Delete a product by ID prefix."""
    if not product_id:
        return "Enter a product ID to delete."
    product_id = product_id.strip()[:64]  # Limit input length
    try:
        products = db.get_all_products()
        for p in products:
            if p.id.startswith(product_id):
                db.delete_product(p.id)
                try:
                    vs.delete_product(p.id, p.company)
                except Exception as ve:
                    logger.warning(f"Vector delete failed for {p.id}: {ve}")
                    # DB delete succeeded, vector will be cleaned up on next reindex
                return f"Deleted product {p.model_code} ({p.id[:8]})"
        return f"No product found with ID starting with '{product_id}'"
    except Exception as e:
        logger.error(f"Delete error: {e}", exc_info=True)
        return f"Error deleting product: {_sanitize_error(e)}"


# ── Feedback Review Tab ───────────────────────────────────────────────

def get_feedback():
    """Get distributor feedback."""
    feedback = db.get_feedback(limit=100)
    if not feedback:
        return pd.DataFrame(columns=[
            "Date", "Query", "Competitor", "Our Product", "Confidence", "Thumbs Up"
        ])
    rows = []
    for f in feedback:
        rows.append({
            "Date": f.get("created_at", ""),
            "Query": f.get("query", ""),
            "Competitor": f.get("competitor_model_code", ""),
            "Our Product": f.get("my_company_model_code", ""),
            "Confidence": f"{f.get('confidence_score', 0):.0%}",
            "Thumbs Up": "Yes" if f.get("thumbs_up") else "No",
        })
    return pd.DataFrame(rows)


def confirm_equivalent(competitor_code, my_company_code, competitor_company):
    """Manually confirm an equivalent pairing."""
    if not competitor_code or not my_company_code:
        return "Both competitor and Danfoss model codes are required."
    # Truncate inputs (Fix #9)
    db.insert_confirmed_equivalent(
        competitor_code=competitor_code.strip()[:MAX_MODEL_CODE_LEN],
        competitor_company=(competitor_company.strip() if competitor_company else "Unknown")[:MAX_COMPANY_NAME_LEN],
        my_company_code=my_company_code.strip()[:MAX_MODEL_CODE_LEN],
        confirmed_by="admin",
    )
    return f"Confirmed: {competitor_code.strip()} -> {my_company_code.strip()}"


# ── Settings Tab ──────────────────────────────────────────────────────

def add_synonym(term, canonical):
    """Add a brand synonym."""
    if not term or not canonical:
        return "Both term and canonical name are required."
    db.insert_synonym(term.strip()[:MAX_COMPANY_NAME_LEN], canonical.strip()[:MAX_COMPANY_NAME_LEN])
    return f"Added synonym: '{term.strip()}' -> '{canonical.strip()}'"


def get_vector_counts():
    """Get vector store collection counts."""
    counts = vs.get_collection_counts()
    return (
        f"Danfoss Products: {counts['my_company']}\n"
        f"Competitor Products: {counts['competitor']}\n"
        f"Guide Chunks: {counts['guides']}"
    )


# ── Build Gradio UI ──────────────────────────────────────────────────

def get_company_list():
    companies = db.get_companies()
    return ["All"] + companies


CATEGORIES = [
    "All / Auto-detect",
    "directional_valves", "proportional_directional_valves",
    "pressure_valves", "flow_valves",
    "pumps", "motors", "cylinders",
    "filters", "accumulators", "hoses_fittings", "other",
]

with gr.Blocks(title="ProductMatchPro - Admin Console") as admin_ui:

    # Session-scoped state for pending extractions (Fix #2)
    pending_state = gr.State({})

    gr.Markdown("# ProductMatchPro - Admin Console")
    gr.Markdown("Manage product database, upload catalogues, and review distributor feedback.")

    with gr.Tabs():
        # ── Upload Tab ────────────────────────────────────────────
        with gr.Tab("Upload Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file = gr.File(
                        label="Upload PDF(s)",
                        file_types=[".pdf"],
                        file_count="multiple",
                        type="filepath",
                    )
                    upload_company = gr.Dropdown(
                        label="Company Name",
                        choices=get_company_dropdown_choices(),
                        value="Danfoss",
                        allow_custom_value=True,
                    )
                    upload_doc_type = gr.Dropdown(
                        label="Document Type",
                        choices=["catalogue", "user_guide", "datasheet"],
                        value="catalogue",
                    )
                    upload_category = gr.Dropdown(
                        label="Product Category",
                        choices=CATEGORIES,
                        value="All / Auto-detect",
                    )
                    upload_btn = gr.Button("Upload & Process", variant="primary")

                with gr.Column(scale=2):
                    upload_status = gr.Textbox(label="Status", lines=3)
                    review_table = gr.Dataframe(
                        label="Extracted Products (review before confirming)",
                        interactive=True,
                    )
                    confirm_btn = gr.Button("Confirm & Index", variant="secondary")
                    confirm_status = gr.Textbox(label="Confirmation Status")

            upload_btn.click(
                process_upload,
                inputs=[upload_file, upload_company, upload_doc_type, upload_category, pending_state],
                outputs=[upload_status, review_table, pending_state],
            )
            confirm_btn.click(
                confirm_extraction,
                inputs=[pending_state],
                outputs=[confirm_status, pending_state],
            )

        # ── Product Database Tab ──────────────────────────────────
        with gr.Tab("Product Database"):
            with gr.Row():
                db_search = gr.Textbox(
                    label="Search",
                    placeholder="Search by model code or name...",
                    max_lines=1,
                )
                db_company = gr.Dropdown(
                    label="Company",
                    choices=get_company_list(),
                    value="All",
                )
                db_search_btn = gr.Button("Search")

            product_table = gr.Dataframe(
                label="Products",
                value=get_product_counts(),
                interactive=True,
            )

            with gr.Row():
                count_table = gr.Dataframe(
                    label="Product Counts",
                    value=get_product_counts(),
                    interactive=True,
                )

            with gr.Row():
                export_btn = gr.Button("Export CSV")
                export_output = gr.File(label="Download CSV")
                reindex_btn = gr.Button("Re-index Vector Store")
                reindex_status = gr.Textbox(label="Re-index Status")

            with gr.Row():
                delete_id = gr.Textbox(
                    label="Delete Product (enter ID prefix)",
                    placeholder="e.g. 'a1b2c3d4'",
                    max_lines=1,
                )
                delete_btn = gr.Button("Delete", variant="stop")
                delete_status = gr.Textbox(label="Delete Status")

            db_search_btn.click(
                search_products,
                inputs=[db_search, db_company],
                outputs=[product_table],
            )
            export_btn.click(export_csv, outputs=[export_output])
            reindex_btn.click(reindex_vectors, outputs=[reindex_status])
            delete_btn.click(
                delete_product, inputs=[delete_id], outputs=[delete_status]
            )

        # ── Feedback Review Tab ───────────────────────────────────
        with gr.Tab("Feedback Review"):
            feedback_table = gr.Dataframe(
                label="Distributor Feedback",
                value=get_feedback(),
                interactive=True,
            )
            refresh_feedback_btn = gr.Button("Refresh")
            refresh_feedback_btn.click(
                get_feedback, outputs=[feedback_table]
            )

            gr.Markdown("### Manually Confirm an Equivalent")
            with gr.Row():
                confirm_comp_code = gr.Textbox(label="Competitor Model Code", max_lines=1)
                confirm_comp_company = gr.Textbox(label="Competitor Company", max_lines=1)
                confirm_my_code = gr.Textbox(label="Danfoss Model Code", max_lines=1)
                confirm_eq_btn = gr.Button("Confirm Equivalent", variant="primary")
                confirm_eq_status = gr.Textbox(label="Status")

            confirm_eq_btn.click(
                confirm_equivalent,
                inputs=[confirm_comp_code, confirm_my_code, confirm_comp_company],
                outputs=[confirm_eq_status],
            )

        # ── Settings Tab ──────────────────────────────────────────
        with gr.Tab("Settings"):
            gr.Markdown("### Brand Synonyms")
            gr.Markdown("Map alternative brand names to canonical names (e.g. 'Rexroth' -> 'Bosch Rexroth')")
            with gr.Row():
                syn_term = gr.Textbox(label="Term", placeholder="e.g. Rexroth", max_lines=1)
                syn_canonical = gr.Textbox(label="Canonical Name", placeholder="e.g. Bosch Rexroth", max_lines=1)
                syn_btn = gr.Button("Add Synonym")
                syn_status = gr.Textbox(label="Status")
            syn_btn.click(
                add_synonym,
                inputs=[syn_term, syn_canonical],
                outputs=[syn_status],
            )

            gr.Markdown("### Vector Store Status")
            vector_status = gr.Textbox(
                label="Collection Counts",
                value=get_vector_counts(),
            )
            refresh_vector_btn = gr.Button("Refresh")
            refresh_vector_btn.click(
                get_vector_counts, outputs=[vector_status]
            )

            gr.Markdown("### Configuration")
            gr.Markdown(
                f"- **Confidence Threshold:** {0.75:.0%} (change in `models.py` -> `CONFIDENCE_THRESHOLD`)\n"
                f"- **Sales Contact:** Edit `SALES_CONTACT` in `graph.py`\n"
                f"- **Model Code Patterns:** Automatically extracted from user guide uploads"
            )


if __name__ == "__main__":
    launch_kwargs = {
        "server_name": os.getenv("ADMIN_HOST", "127.0.0.1"),
        "server_port": int(os.getenv("ADMIN_PORT", "7861")),
        "share": False,
        "max_file_size": f"{MAX_UPLOAD_SIZE_MB}mb",
    }
    # Enable auth if credentials are set (recommended for any shared deployment)
    if ADMIN_USERNAME and ADMIN_PASSWORD:
        launch_kwargs["auth"] = (ADMIN_USERNAME, ADMIN_PASSWORD)
    admin_ui.launch(**launch_kwargs)
