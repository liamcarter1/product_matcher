"""
ProductMatchPro - Admin Console (Back-Office)
Used by {my_company} staff to upload PDFs, manage products, review feedback, and configure settings.
NOT visible to distributors.

Run: python admin_app.py
"""

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

# ── Initialise storage ────────────────────────────────────────────────

db = ProductDB()
vs = VectorStore()
pipeline = IngestionPipeline(db, vs)

# Temporary storage for extracted products awaiting confirmation
pending_extractions: list[ExtractedProduct] = []
pending_metadata: UploadMetadata | None = None


# ── Upload Tab ────────────────────────────────────────────────────────

def process_upload(file, company, doc_type, category):
    """Process an uploaded PDF and return extracted products for review."""
    global pending_extractions, pending_metadata

    if file is None:
        return "Please upload a PDF file.", None

    # Gradio 5 with type="filepath" passes a string path
    file_path = str(file)

    metadata = UploadMetadata(
        company=company.strip() if company else "Unknown",
        document_type=DocumentType(doc_type),
        category=category if category != "All / Auto-detect" else "",
        filename=Path(file_path).name,
    )

    try:
        extracted = pipeline.process_pdf(file_path, metadata)

        if not extracted:
            return "No products could be extracted from this PDF. Try a different document type setting.", None

        pending_extractions = extracted
        pending_metadata = metadata

        # Also index guide text
        chunk_count = pipeline.index_guide_text(file_path, metadata)

        # Build a review dataframe
        rows = []
        for ep in extracted:
            row = {
                "Model Code": ep.model_code,
                "Product Name": ep.product_name,
                "Category": ep.category,
                "Confidence": f"{ep.confidence:.0%}",
            }
            # Add key specs
            for key in ["max_pressure_bar", "max_flow_lpm", "coil_voltage",
                        "valve_size", "actuator_type", "port_size", "mounting"]:
                row[key] = ep.specs.get(key, "")
            rows.append(row)

        df = pd.DataFrame(rows)
        status = (
            f"Extracted {len(extracted)} products from {metadata.filename}. "
            f"Also indexed {chunk_count} text chunks from the document.\n\n"
            f"Review the products below and click 'Confirm & Index' to add them to the database."
        )
        return status, df

    except Exception as e:
        return f"Error processing PDF: {str(e)}", None


def confirm_extraction():
    """Confirm the extracted products and store them."""
    global pending_extractions, pending_metadata

    if not pending_extractions or not pending_metadata:
        return "No pending extractions to confirm. Upload a PDF first."

    count = pipeline.confirm_and_store(pending_extractions, pending_metadata)
    pending_extractions = []
    pending_metadata = None

    return f"Successfully indexed {count} products into the database and vector store."


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
            "Mounting": p.mounting or "",
        })

    return pd.DataFrame(rows) if rows else pd.DataFrame()


def export_csv():
    """Export all products as CSV."""
    import tempfile
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


def reindex_vectors():
    """Rebuild the vector store from the SQLite database."""
    products = db.get_all_products()
    vs.rebuild_from_products(products)
    return f"Re-indexed {len(products)} products in the vector store."


def delete_product(product_id):
    """Delete a product by ID prefix."""
    if not product_id:
        return "Enter a product ID to delete."
    products = db.get_all_products()
    for p in products:
        if p.id.startswith(product_id):
            db.delete_product(p.id)
            vs.delete_product(p.id, p.company)
            return f"Deleted product {p.model_code} ({p.id[:8]})"
    return f"No product found with ID starting with '{product_id}'"


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
        return "Both competitor and {my_company} model codes are required."
    db.insert_confirmed_equivalent(
        competitor_code=competitor_code.strip(),
        competitor_company=competitor_company.strip() if competitor_company else "Unknown",
        my_company_code=my_company_code.strip(),
        confirmed_by="admin",
    )
    return f"Confirmed: {competitor_code} → {my_company_code}"


# ── Settings Tab ──────────────────────────────────────────────────────

def add_synonym(term, canonical):
    """Add a brand synonym."""
    if not term or not canonical:
        return "Both term and canonical name are required."
    db.insert_synonym(term.strip(), canonical.strip())
    return f"Added synonym: '{term}' → '{canonical}'"


def get_vector_counts():
    """Get vector store collection counts."""
    counts = vs.get_collection_counts()
    return (
        f"My Company Products: {counts['my_company']}\n"
        f"Competitor Products: {counts['competitor']}\n"
        f"Guide Chunks: {counts['guides']}"
    )


# ── Build Gradio UI ──────────────────────────────────────────────────

def get_company_list():
    companies = db.get_companies()
    return ["All"] + companies


CATEGORIES = [
    "All / Auto-detect",
    "directional_valves", "pressure_valves", "flow_valves",
    "pumps", "motors", "cylinders",
    "filters", "accumulators", "hoses_fittings", "other",
]

with gr.Blocks(title="ProductMatchPro - Admin Console") as admin_ui:

    gr.Markdown("# ProductMatchPro - Admin Console")
    gr.Markdown("Manage product database, upload catalogues, and review distributor feedback.")

    with gr.Tabs():
        # ── Upload Tab ────────────────────────────────────────────
        with gr.Tab("Upload Documents"):
            with gr.Row():
                with gr.Column(scale=1):
                    upload_file = gr.File(
                        label="Upload PDF",
                        file_types=[".pdf"],
                        type="filepath",
                    )
                    upload_company = gr.Textbox(
                        label="Company Name",
                        placeholder="e.g. 'Eaton Vickers', 'Parker', or 'my_company'",
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
                        interactive=False,
                    )
                    confirm_btn = gr.Button("Confirm & Index", variant="secondary")
                    confirm_status = gr.Textbox(label="Confirmation Status")

            upload_btn.click(
                process_upload,
                inputs=[upload_file, upload_company, upload_doc_type, upload_category],
                outputs=[upload_status, review_table],
            )
            confirm_btn.click(
                confirm_extraction, inputs=[], outputs=[confirm_status]
            )

        # ── Product Database Tab ──────────────────────────────────
        with gr.Tab("Product Database"):
            with gr.Row():
                db_search = gr.Textbox(
                    label="Search",
                    placeholder="Search by model code or name...",
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
                interactive=False,
            )

            with gr.Row():
                count_table = gr.Dataframe(
                    label="Product Counts",
                    value=get_product_counts(),
                    interactive=False,
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
                interactive=False,
            )
            refresh_feedback_btn = gr.Button("Refresh")
            refresh_feedback_btn.click(
                get_feedback, outputs=[feedback_table]
            )

            gr.Markdown("### Manually Confirm an Equivalent")
            with gr.Row():
                confirm_comp_code = gr.Textbox(label="Competitor Model Code")
                confirm_comp_company = gr.Textbox(label="Competitor Company")
                confirm_my_code = gr.Textbox(label="{my_company} Model Code")
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
            gr.Markdown("Map alternative brand names to canonical names (e.g. 'Vickers' → 'Eaton')")
            with gr.Row():
                syn_term = gr.Textbox(label="Term", placeholder="e.g. Vickers")
                syn_canonical = gr.Textbox(label="Canonical Name", placeholder="e.g. Eaton")
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
                f"- **Confidence Threshold:** {0.75:.0%} (change in `models.py` → `CONFIDENCE_THRESHOLD`)\n"
                f"- **Sales Contact:** Edit `SALES_CONTACT` in `graph.py`\n"
                f"- **Model Code Patterns:** Automatically extracted from user guide uploads"
            )


if __name__ == "__main__":
    admin_ui.launch(
        server_name="0.0.0.0",
        server_port=7861,
        share=False,
    )
