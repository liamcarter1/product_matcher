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
            # Branch for cross-reference documents
            if doc_type == "cross_reference":
                cross_refs = pipeline.process_cross_reference_pdf(file_path, metadata)
                if cross_refs:
                    all_extracted.extend(cross_refs)  # dicts, not ExtractedProduct
                    file_summaries.append(
                        f"  {metadata.filename}: {len(cross_refs)} cross-reference mappings"
                    )
                else:
                    file_summaries.append(
                        f"  {metadata.filename}: no cross-references extracted"
                    )
                continue

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
        item_type = "cross-reference mappings" if doc_type == "cross_reference" else "products"
        summary = f"No {item_type} could be extracted from any of the uploaded files.\n\n"
        summary += "File results:\n" + "\n".join(file_summaries)
        summary += "\n\nTry a different document type setting."
        return summary, None, pending_state

    # ── Cross-reference branch: different pending state and review table ──
    if doc_type == "cross_reference":
        pending_state = {
            "cross_references": all_extracted,  # list of dicts
            "metadata": {
                "company": company_name,
                "document_type": doc_type,
                "category": "",
                "filename": f"{len(file_paths)} files",
            },
        }
        rows = []
        for ref in all_extracted:
            rows.append({
                "Danfoss Series": ref.get("my_company_series", ""),
                "Competitor Series": ref.get("competitor_series", ""),
                "Competitor": ref.get("competitor_company", ""),
                "Product Type": ref.get("product_type", ""),
                "Notes": ref.get("notes", ""),
            })
        df = pd.DataFrame(rows)
        status = (
            f"Extracted {len(all_extracted)} cross-reference mappings from {len(file_paths)} file(s).\n"
            f"Also indexed text chunks for LLM context.\n\n"
            f"File results:\n" + "\n".join(file_summaries) + "\n\n"
            f"Review the mappings below and click 'Confirm & Index' to store them."
        )
        return status, df, pending_state

    # ── Standard product extraction branch ──
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

    # First: discover ALL dynamic spec keys across all extracted products
    known_keys = {"max_pressure_bar", "max_flow_lpm", "coil_voltage",
                  "valve_size", "spool_type", "seal_material", "actuator_type",
                  "port_size", "mounting", "num_ports"}
    all_extra_keys = []
    seen_extra = set()
    for ep in all_extracted:
        for k in (ep.specs or {}):
            if not k.startswith("_") and k not in known_keys and k not in seen_extra:
                seen_extra.add(k)
                all_extra_keys.append(k)

    # Build a review dataframe with consistent columns
    rows = []
    for ep in all_extracted:
        row = {
            "Source": ep.source if hasattr(ep, "source") else "llm",
            "Model Code": ep.model_code,
            "Product Name": ep.product_name,
            "Category": ep.category,
            "Confidence": f"{ep.confidence:.0%}",
        }
        # Add known key specs
        for key in ["max_pressure_bar", "max_flow_lpm", "coil_voltage",
                    "valve_size", "spool_type", "seal_material", "actuator_type",
                    "port_size", "mounting", "num_ports"]:
            row[key] = ep.specs.get(key, "")
        # Add ALL discovered dynamic specs (consistent columns across all rows)
        for k in all_extra_keys:
            col_name = k.replace("_", " ").title()
            v = ep.specs.get(k, "")
            row[col_name] = str(v) if v is not None and v != "" else ""
        rows.append(row)

    df = pd.DataFrame(rows)

    # Check for gap-fill warnings from spool reference validation
    gapfill_warnings = getattr(pipeline, "_last_gapfill_warnings", [])
    gapfill_count = sum(1 for ep in all_extracted
                        if ep.specs.get("_spool_source") == "reference_gapfill")

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

    # Build dynamic column info for user feedback
    dynamic_col_info = ""
    if all_extra_keys:
        col_names = [k.replace("_", " ").title() for k in all_extra_keys]
        dynamic_col_info = (
            f"\nDynamic columns discovered ({len(all_extra_keys)}): "
            f"{', '.join(col_names)}\n"
        )
    else:
        dynamic_col_info = "\nNo dynamic columns found beyond standard fields.\n"

    gapfill_info = ""
    if gapfill_count:
        gapfill_info = (
            f"\nSpool gap-fill: {gapfill_count} products were added from the spool reference "
            f"database (spool types known from previous extractions but not found in this document).\n"
        )
        for w in gapfill_warnings:
            gapfill_info += f"  {w}\n"

    status = (
        f"Extracted {len(all_extracted)} products from {len(file_paths)} file(s). "
        f"Also indexed {all_chunk_counts} text chunks.\n"
        f"Sources: {', '.join(source_details)}.\n"
        f"{dynamic_col_info}"
        f"{gapfill_info}\n"
        f"File results:\n" + "\n".join(file_summaries) + "\n\n"
        f"Review the products below and click 'Confirm & Index' to add them to the database."
    )
    return status, df, pending_state


def confirm_extraction(pending_state):
    """Confirm the extracted products (or cross-references) and store them.
    Uses gr.State for session isolation (Fix #2).
    Wraps in try/except for robustness (Fix #3)."""

    # Cross-reference branch
    if pending_state and pending_state.get("cross_references"):
        try:
            cross_refs = pending_state["cross_references"]
            meta_dict = pending_state["metadata"]
            metadata = UploadMetadata(
                company=meta_dict["company"],
                document_type=DocumentType(meta_dict["document_type"]),
                category="",
                filename=meta_dict.get("filename", ""),
            )
            count = pipeline.confirm_and_store_cross_references(cross_refs, metadata)
            pending_state = {}
            return f"Successfully stored {count} cross-reference mappings.", pending_state
        except Exception as e:
            logger.error(f"Error storing cross-references: {e}", exc_info=True)
            return (
                f"Error storing cross-references: {_sanitize_error(e)}. "
                f"Your data is preserved — you can retry."
            ), pending_state

    # Standard product extraction branch
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

    # First pass: discover ALL dynamic column names across all products
    display_products = products[:100]
    all_extra_columns = []
    seen_extra = set()
    for p in display_products:
        if p.extra_specs:
            for k in p.extra_specs:
                if not k.startswith("_") and k not in seen_extra:
                    seen_extra.add(k)
                    all_extra_columns.append(k)

    # Second pass: build rows with consistent columns
    rows = []
    for p in display_products:
        row = {
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
        }
        # Add ALL discovered extra_specs columns (consistent across all rows)
        extra = p.extra_specs or {}
        for k in all_extra_columns:
            col_name = k.replace("_", " ").title()
            v = extra.get(k, "")
            row[col_name] = str(v) if v is not None and v != "" else ""
        rows.append(row)

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


def delete_all_products():
    """Delete ALL products from the database and vector store."""
    try:
        products = db.get_all_products()
        if not products:
            return "Database is already empty."
        count = len(products)
        for p in products:
            db.delete_product(p.id)
            try:
                vs.delete_product(p.id, p.company)
            except Exception:
                pass  # Vector cleanup is best-effort
        return f"Deleted all {count} products from the database."
    except Exception as e:
        logger.error(f"Delete all error: {e}", exc_info=True)
        return f"Error: {_sanitize_error(e)}"


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


# ── Spool Types Tab ──────────────────────────────────────────────────

def get_spool_type_table(manufacturer_filter="All"):
    """Load spool type references for display."""
    if manufacturer_filter and manufacturer_filter != "All":
        refs = db.get_spool_type_references(manufacturer=manufacturer_filter)
    else:
        refs = db.get_spool_type_references()
    if not refs:
        return pd.DataFrame(columns=[
            "ID", "Series", "Manufacturer", "Spool Code",
            "Description", "Center Condition", "Primary", "Source",
        ])
    rows = []
    for r in refs:
        rows.append({
            "ID": str(r.get("id", ""))[:8],
            "Series": r.get("series_prefix", ""),
            "Manufacturer": r.get("manufacturer", ""),
            "Spool Code": r.get("spool_code", ""),
            "Description": r.get("description", ""),
            "Center Condition": r.get("center_condition", ""),
            "Primary": "Yes" if r.get("is_primary") else "",
            "Source": r.get("source", ""),
        })
    return pd.DataFrame(rows)


def add_spool_type_ref(series_prefix, manufacturer, spool_code, description, center_condition, is_primary):
    """Manually add a spool type reference."""
    if not series_prefix or not manufacturer or not spool_code:
        return "Series prefix, manufacturer, and spool code are required.", get_spool_type_table()
    db.insert_spool_type_reference(
        series_prefix=series_prefix.strip()[:50],
        manufacturer=manufacturer.strip()[:MAX_COMPANY_NAME_LEN],
        spool_code=spool_code.strip()[:10],
        description=(description or "").strip()[:200],
        center_condition=(center_condition or "").strip()[:200],
        is_primary=(is_primary == "Yes"),
        source="manual",
    )
    label = f"Added spool type: {spool_code.strip()} for {manufacturer.strip()} {series_prefix.strip()}"
    if is_primary == "Yes":
        label += " [PRIMARY]"
    return label, get_spool_type_table()


def toggle_spool_primary(ref_id, set_primary):
    """Toggle the primary flag on a spool type reference."""
    if not ref_id:
        return "Reference ID is required.", get_spool_type_table()
    is_primary = set_primary == "Yes"
    db.update_spool_type_primary(ref_id.strip(), is_primary)
    status = "Set as primary" if is_primary else "Removed primary flag"
    return f"{status}: {ref_id.strip()}", get_spool_type_table()


def set_primary_spools_bulk(spool_codes_text, manufacturer_filter):
    """Bulk set primary flag from comma-separated spool codes."""
    if not spool_codes_text or not spool_codes_text.strip():
        return "Enter comma-separated spool codes.", get_spool_type_table(manufacturer_filter)

    codes = [c.strip() for c in spool_codes_text.split(",") if c.strip()]
    if not codes:
        return "No valid spool codes found.", get_spool_type_table(manufacturer_filter)

    manufacturer = manufacturer_filter if manufacturer_filter != "All" else None
    if not manufacturer:
        return "Select a specific manufacturer first (not 'All').", get_spool_type_table(manufacturer_filter)

    # Clear existing primary flags, then set the specified codes
    db.clear_all_spools_primary(manufacturer=manufacturer)
    db.set_all_spools_primary(manufacturer=manufacturer, spool_codes=codes)

    return (
        f"Set {len(codes)} primary spool types for {manufacturer}: {', '.join(codes)}",
        get_spool_type_table(manufacturer_filter),
    )


def delete_spool_type_ref(ref_id):
    """Delete a spool type reference by ID prefix."""
    if not ref_id:
        return "Reference ID is required.", get_spool_type_table()
    db.delete_spool_type_reference(ref_id.strip())
    return f"Deleted spool type reference: {ref_id.strip()}", get_spool_type_table()


def import_spools_from_database(manufacturer):
    """Scan existing products and auto-populate spool_type_reference from them."""
    if not manufacturer:
        return "Select a manufacturer.", get_spool_type_table()

    products = db.get_all_products(company=manufacturer)
    if not products:
        return f"No products found for {manufacturer}.", get_spool_type_table()

    series_spools: dict[str, dict[str, dict]] = {}
    for p in products:
        if not p.spool_type:
            continue
        # Infer series from model code patterns
        series = _infer_series_from_model(p.model_code)
        if not series:
            continue
        if series not in series_spools:
            series_spools[series] = {}
        extra = p.extra_specs or {}
        series_spools[series][p.spool_type] = {
            "center_condition": extra.get("center_condition", ""),
            "solenoid_a_function": extra.get("solenoid_a_energised", ""),
            "solenoid_b_function": extra.get("solenoid_b_energised", ""),
            "canonical_pattern": extra.get("canonical_spool_pattern", ""),
            "description": extra.get("spool_function_description", ""),
        }

    count = 0
    for series, spools in series_spools.items():
        for code, data in spools.items():
            db.insert_spool_type_reference(
                series_prefix=series,
                manufacturer=manufacturer,
                spool_code=code,
                source="auto_extracted",
                **data,
            )
            count += 1

    return (
        f"Imported {count} spool types across {len(series_spools)} series for {manufacturer}.",
        get_spool_type_table(manufacturer),
    )


def _infer_series_from_model(model_code: str) -> str:
    """Infer series prefix from a single model code.

    Heuristic: take everything before the spool type code typically changes.
    For hyphenated codes like DG4V-3-2A-M-..., take first two segments.
    For compact codes like D1VW004CNJW, take the alpha prefix.
    """
    if not model_code:
        return ""
    code = model_code.strip()

    # Hyphenated: DG4V-3-2A-M... -> DG4V-3
    parts = code.split("-")
    if len(parts) >= 3:
        return "-".join(parts[:2])

    # Compact: D1VW004CNJW -> D1VW (alpha prefix before digits begin)
    import re
    m = re.match(r'^([A-Za-z0-9]*?[A-Za-z])(?=\d{2,})', code)
    if m:
        return m.group(1)

    # Fallback: first 6 chars
    return code[:6]


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
                        choices=["catalogue", "user_guide", "datasheet", "cross_reference"],
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

            with gr.Row():
                delete_all_btn = gr.Button(
                    "Delete ALL Products",
                    variant="stop",
                )

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
            delete_all_btn.click(
                delete_all_products, outputs=[delete_status]
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

        # ── Spool Types Tab ──────────────────────────────────────────
        with gr.Tab("Spool Types"):
            gr.Markdown("### Known Spool Type Reference")
            gr.Markdown(
                "Manage known spool types per manufacturer/series. "
                "These are used during extraction to ensure all spool options are captured. "
                "Mark spool types as **Primary** to limit product generation to only "
                "key variants (main runners) instead of generating all combinations."
            )

            with gr.Row():
                spool_mfr_filter = gr.Dropdown(
                    label="Manufacturer Filter",
                    choices=["All"] + KNOWN_COMPANIES,
                    value="All",
                    allow_custom_value=True,
                )
                spool_refresh_btn = gr.Button("Refresh")

            spool_ref_table = gr.Dataframe(
                label="Known Spool Types",
                value=get_spool_type_table(),
                interactive=False,
            )
            spool_refresh_btn.click(
                get_spool_type_table,
                inputs=[spool_mfr_filter],
                outputs=[spool_ref_table],
            )

            gr.Markdown("### Add Spool Type")
            with gr.Row():
                spool_series = gr.Textbox(
                    label="Series Prefix", placeholder="e.g. DG4V-3", max_lines=1,
                )
                spool_mfr = gr.Dropdown(
                    label="Manufacturer", choices=KNOWN_COMPANIES,
                    allow_custom_value=True,
                )
                spool_code_input = gr.Textbox(
                    label="Spool Code", placeholder="e.g. 2A", max_lines=1,
                )
            with gr.Row():
                spool_desc = gr.Textbox(
                    label="Description", placeholder="e.g. Closed center, standard crossover",
                    max_lines=1,
                )
                spool_center = gr.Textbox(
                    label="Center Condition", placeholder="e.g. All ports blocked",
                    max_lines=1,
                )
                spool_is_primary = gr.Dropdown(
                    label="Primary?", choices=["No", "Yes"], value="No",
                )
            with gr.Row():
                spool_add_btn = gr.Button("Add Spool Type", variant="primary")
                spool_add_status = gr.Textbox(label="Status")

            spool_add_btn.click(
                add_spool_type_ref,
                inputs=[spool_series, spool_mfr, spool_code_input,
                        spool_desc, spool_center, spool_is_primary],
                outputs=[spool_add_status, spool_ref_table],
            )

            gr.Markdown("### Primary Spool Types (Main Runners)")
            gr.Markdown(
                "Set which spool types are **primary** to limit product generation. "
                "When primary spools are defined, only those variants are generated "
                "instead of all 30+ combinations. Enter codes comma-separated."
            )
            with gr.Row():
                spool_primary_codes = gr.Textbox(
                    label="Primary Spool Codes (comma-separated)",
                    placeholder="e.g. 2A, 2B, 2C, 6C, D, H, 0C, 33C",
                    max_lines=1,
                )
                spool_primary_btn = gr.Button("Set Primary Spools", variant="primary")
                spool_primary_status = gr.Textbox(label="Status")

            spool_primary_btn.click(
                set_primary_spools_bulk,
                inputs=[spool_primary_codes, spool_mfr_filter],
                outputs=[spool_primary_status, spool_ref_table],
            )

            gr.Markdown("### Toggle Individual Spool Primary")
            with gr.Row():
                spool_toggle_id = gr.Textbox(
                    label="Reference ID (prefix)", placeholder="e.g. a1b2c3",
                    max_lines=1,
                )
                spool_toggle_primary = gr.Dropdown(
                    label="Set Primary", choices=["Yes", "No"], value="Yes",
                )
                spool_toggle_btn = gr.Button("Toggle Primary")
                spool_toggle_status = gr.Textbox(label="Status")

            spool_toggle_btn.click(
                toggle_spool_primary,
                inputs=[spool_toggle_id, spool_toggle_primary],
                outputs=[spool_toggle_status, spool_ref_table],
            )

            gr.Markdown("### Import from Existing Products")
            gr.Markdown(
                "Scan products already in the database and auto-populate the spool "
                "reference table. Useful for bootstrapping from existing data."
            )
            with gr.Row():
                spool_import_mfr = gr.Dropdown(
                    label="Manufacturer", choices=KNOWN_COMPANIES,
                    allow_custom_value=True,
                )
                spool_import_btn = gr.Button("Import from Database")
                spool_import_status = gr.Textbox(label="Import Status")

            spool_import_btn.click(
                import_spools_from_database,
                inputs=[spool_import_mfr],
                outputs=[spool_import_status, spool_ref_table],
            )

            gr.Markdown("### Delete Spool Type")
            with gr.Row():
                spool_delete_id = gr.Textbox(
                    label="Reference ID (prefix)", placeholder="e.g. a1b2c3",
                    max_lines=1,
                )
                spool_delete_btn = gr.Button("Delete", variant="stop")
                spool_delete_status = gr.Textbox(label="Delete Status")

            spool_delete_btn.click(
                delete_spool_type_ref,
                inputs=[spool_delete_id],
                outputs=[spool_delete_status, spool_ref_table],
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
