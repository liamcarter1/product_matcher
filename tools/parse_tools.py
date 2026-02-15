"""
ProductMatchPro - PDF Parsing Tools
Extracts product data from hydraulic catalogues and user guides.
Uses pdfplumber for tables and pypdf + LLM for unstructured text.
"""

import re
import uuid
import itertools
import logging
from pathlib import Path
from typing import Optional
import pdfplumber
from openai import OpenAI
from dotenv import load_dotenv
import json

logger = logging.getLogger(__name__)

# PyMuPDF (fitz) for high-quality text extraction — optional fallback to pypdf
try:
    import fitz as _fitz  # PyMuPDF
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False
    from pypdf import PdfReader as _PdfReader

from models import (
    ExtractedProduct, HydraulicProduct, UploadMetadata,
    OrderingCodeSegment, OrderingCodeDefinition,
)

load_dotenv(override=True)

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client

# Common header patterns in hydraulic catalogues
HEADER_MAPPINGS = {
    # Model code variations
    "model": "model_code", "model no": "model_code", "model number": "model_code",
    "model code": "model_code", "part no": "model_code", "part number": "model_code",
    "ordering code": "model_code", "type": "model_code", "designation": "model_code",
    "order code": "model_code", "order number": "model_code", "item number": "model_code",
    # Pressure
    "pressure": "max_pressure_bar", "max pressure": "max_pressure_bar",
    "max. pressure": "max_pressure_bar", "rated pressure": "max_pressure_bar",
    "pressure (bar)": "max_pressure_bar", "pressure bar": "max_pressure_bar",
    "operating pressure": "max_pressure_bar", "max. operating pressure": "max_pressure_bar",
    "nominal pressure": "max_pressure_bar", "working pressure": "max_pressure_bar",
    # Flow
    "flow": "max_flow_lpm", "max flow": "max_flow_lpm", "max. flow": "max_flow_lpm",
    "rated flow": "max_flow_lpm", "flow (l/min)": "max_flow_lpm",
    "flow l/min": "max_flow_lpm", "flow lpm": "max_flow_lpm",
    "nominal flow": "max_flow_lpm", "flow rate": "max_flow_lpm",
    "qmax": "max_flow_lpm", "q max": "max_flow_lpm",
    # Port size
    "port": "port_size", "port size": "port_size", "connection": "port_size",
    "connection size": "port_size",
    # Num ports (separate from port_size)
    "ports": "num_ports", "no. of ports": "num_ports", "number of ports": "num_ports",
    # Num positions
    "positions": "num_positions", "no. of positions": "num_positions",
    "number of positions": "num_positions", "ways": "num_positions",
    # Voltage
    "voltage": "coil_voltage", "coil voltage": "coil_voltage",
    "coil": "coil_voltage", "supply voltage": "coil_voltage",
    "solenoid voltage": "coil_voltage", "operating voltage": "coil_voltage",
    "supply": "coil_voltage",
    # Coil type / connector
    "coil type": "coil_type", "solenoid type": "coil_type", "coil design": "coil_type",
    "connector": "coil_connector", "coil connector": "coil_connector",
    "electrical connection": "coil_connector", "plug type": "coil_connector",
    "connector type": "coil_connector",
    # Mounting
    "mounting": "mounting", "mounting type": "mounting",
    "mounting pattern": "mounting_pattern", "interface": "mounting_pattern",
    "interface standard": "mounting_pattern", "pattern": "mounting_pattern",
    "iso": "mounting_pattern", "nfpa": "mounting_pattern",
    # Valve size
    "size": "valve_size", "valve size": "valve_size", "cetop": "valve_size",
    "ng": "valve_size", "nominal size": "valve_size", "nom. size": "valve_size",
    # Weight
    "weight": "weight_kg", "weight (kg)": "weight_kg", "mass": "weight_kg",
    "mass (kg)": "weight_kg",
    # Product name
    "description": "product_name", "name": "product_name",
    "product": "product_name",
    # Actuation
    "actuation": "actuator_type", "operation": "actuator_type",
    "actuator": "actuator_type",
    # Seal
    "seal": "seal_material", "seal material": "seal_material",
    "seals": "seal_material",
    # Body
    "material": "body_material", "body material": "body_material",
    "body": "body_material",
    # Spool
    "spool": "spool_type", "spool type": "spool_type",
    "function": "spool_type", "valve function": "spool_type",
    # Port type / thread
    "thread": "port_type", "thread type": "port_type",
    "connection type": "port_type", "port type": "port_type",
    # Fluid
    "fluid": "fluid_type", "fluid type": "fluid_type",
    "hydraulic fluid": "fluid_type", "medium": "fluid_type",
    "operating fluid": "fluid_type",
    # Viscosity
    "viscosity": "viscosity_range_cst", "viscosity range": "viscosity_range_cst",
    "kinematic viscosity": "viscosity_range_cst",
    # Displacement (pumps/motors)
    "displacement": "displacement_cc", "cc/rev": "displacement_cc",
    "displacement cc": "displacement_cc", "geometric volume": "displacement_cc",
    # Speed (pumps/motors)
    "speed": "speed_rpm_max", "max speed": "speed_rpm_max",
    "speed (rpm)": "speed_rpm_max", "rated speed": "speed_rpm_max", "rpm": "speed_rpm_max",
    # Bore / Rod / Stroke (cylinders)
    "bore": "bore_diameter_mm", "bore diameter": "bore_diameter_mm",
    "rod": "rod_diameter_mm", "rod diameter": "rod_diameter_mm",
    "stroke": "stroke_mm", "stroke length": "stroke_mm",
    # Temperature
    "temperature": "operating_temp_max_c", "temp range": "operating_temp_max_c",
    "operating temperature": "operating_temp_max_c",
    "max temp": "operating_temp_max_c", "min temp": "operating_temp_min_c",
    "max. temperature": "operating_temp_max_c", "min. temperature": "operating_temp_min_c",
    # Subcategory
    "subcategory": "subcategory", "product type": "subcategory",
    "valve type": "subcategory",
}


def extract_tables_from_pdf(pdf_path: str) -> list[list[dict]]:
    """Extract tables from a PDF using pdfplumber.
    Returns list of tables, each table is a list of row dicts."""
    tables = []
    with pdfplumber.open(pdf_path) as pdf:
        for page_num, page in enumerate(pdf.pages):
            page_tables = page.extract_tables()
            if not page_tables:
                continue
            for table in page_tables:
                if not table or len(table) < 2:
                    continue
                # First row is likely headers
                raw_headers = [str(h).strip().lower() if h else "" for h in table[0]]
                # Map headers to our field names
                mapped_headers = []
                for h in raw_headers:
                    mapped = None
                    for pattern, field in HEADER_MAPPINGS.items():
                        if pattern in h.lower():
                            mapped = field
                            break
                    mapped_headers.append(mapped or h)

                # Build row dicts
                rows = []
                for row in table[1:]:
                    if not row or all(not cell for cell in row):
                        continue
                    row_dict = {"_page": page_num + 1}
                    for i, cell in enumerate(row):
                        if i < len(mapped_headers) and mapped_headers[i]:
                            row_dict[mapped_headers[i]] = str(cell).strip() if cell else ""
                    if row_dict.get("model_code"):
                        rows.append(row_dict)
                if rows:
                    tables.append(rows)
    return tables


def extract_text_from_pdf(pdf_path: str) -> list[dict]:
    """Extract text from each page of a PDF using the best available library.

    Uses PyMuPDF (fitz) when available — significantly better extraction of
    complex layouts, multi-column text, tables-as-text, and operating data.
    Falls back to pypdf if PyMuPDF is not installed.

    Also extracts text from pdfplumber as a supplementary source, since
    pdfplumber sometimes captures table-adjacent text that other parsers miss.

    Returns list of {page: int, text: str} for ALL pages with content.
    """
    pages = []

    # ── Primary extraction: PyMuPDF (fitz) or pypdf ─────────────────
    if HAS_PYMUPDF:
        try:
            doc = _fitz.open(pdf_path)
            for i, page in enumerate(doc):
                # "text" mode gives clean text; "blocks" preserves layout
                text = page.get_text("text")
                if text and text.strip():
                    pages.append({"page": i + 1, "text": text.strip()})
            doc.close()
            logger.info(f"PyMuPDF extracted text from {len(pages)}/{len(pages)} pages")
        except Exception as e:
            logger.warning(f"PyMuPDF extraction failed, falling back to pypdf: {e}")
            pages = _extract_text_pypdf(pdf_path)
    else:
        pages = _extract_text_pypdf(pdf_path)

    # ── Supplementary extraction: pdfplumber text ───────────────────
    # pdfplumber captures text differently — especially table-adjacent text.
    # Merge any pages where pdfplumber got more content than the primary extractor.
    try:
        plumber_pages = _extract_text_pdfplumber(pdf_path)
        existing_page_nums = {p["page"] for p in pages}
        existing_by_page = {p["page"]: p for p in pages}

        for pp in plumber_pages:
            page_num = pp["page"]
            if page_num not in existing_page_nums:
                # pdfplumber found a page the primary extractor missed entirely
                pages.append(pp)
                logger.info(f"pdfplumber rescued page {page_num} missed by primary extractor")
            else:
                # If pdfplumber got significantly more text, append it
                existing_text = existing_by_page[page_num]["text"]
                plumber_text = pp["text"]
                # Check if pdfplumber has unique content (at least 100 new chars)
                if len(plumber_text) > len(existing_text) + 100:
                    # Merge: keep primary text and append pdfplumber's extra content
                    merged = existing_text + "\n\n[Additional extracted content]\n" + plumber_text
                    existing_by_page[page_num]["text"] = merged

        # Re-sort by page number
        pages.sort(key=lambda p: p["page"])
    except Exception as e:
        logger.warning(f"Supplementary pdfplumber text extraction failed: {e}")

    logger.info(f"Total text extraction: {len(pages)} pages from {pdf_path}")
    return pages


def _extract_text_pypdf(pdf_path: str) -> list[dict]:
    """Fallback text extraction using pypdf."""
    from pypdf import PdfReader
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
        text = page.extract_text()
        if text and text.strip():
            pages.append({"page": i + 1, "text": text.strip()})
    return pages


def _extract_text_pdfplumber(pdf_path: str) -> list[dict]:
    """Extract raw text from each page using pdfplumber.
    Captures text that surrounds tables and structured elements."""
    pages = []
    with pdfplumber.open(pdf_path) as pdf:
        for i, page in enumerate(pdf.pages):
            text = page.extract_text()
            if text and text.strip():
                pages.append({"page": i + 1, "text": text.strip()})
    return pages


def table_rows_to_products(
    rows: list[dict], metadata: UploadMetadata
) -> list[ExtractedProduct]:
    """Convert table rows into ExtractedProduct objects."""
    products = []
    for row in rows:
        model_code = row.get("model_code", "").strip()
        if not model_code:
            continue

        specs = {}
        for key, value in row.items():
            if key.startswith("_") or key == "model_code":
                continue
            if value and value.strip():
                specs[key] = _parse_numeric_if_possible(value.strip())

        product = ExtractedProduct(
            model_code=model_code,
            product_name=specs.pop("product_name", ""),
            category=metadata.category or "",
            specs=specs,
            raw_text=json.dumps(row),
            page_number=row.get("_page"),
            confidence=0.8,
            source="table",
        )
        products.append(product)
    return products


def extract_products_with_llm(
    text: str, metadata: UploadMetadata
) -> list[ExtractedProduct]:
    """Use GPT-4o-mini to extract products from unstructured text.

    Processes text in batches to ensure ALL pages are covered — not just the first
    12,000 characters. Each batch is sent to the LLM independently, then results
    are merged and deduplicated.
    """
    if not text or not text.strip():
        return []

    # Split into batches of ~10,000 chars each (with overlap for context)
    BATCH_SIZE = 10000
    BATCH_OVERLAP = 500
    batches = []
    start = 0
    while start < len(text):
        end = start + BATCH_SIZE
        batches.append(text[start:end])
        start = end - BATCH_OVERLAP
        if start >= len(text):
            break

    logger.info(f"LLM product extraction: {len(batches)} batch(es) from {len(text)} chars")

    all_products = []
    for batch_idx, batch_text in enumerate(batches):
        batch_products = _extract_products_from_batch(batch_text, metadata, batch_idx + 1)
        all_products.extend(batch_products)

    # Deduplicate by model_code (keep the one with more specs)
    seen = {}
    for p in all_products:
        key = p.model_code.upper().strip()
        if key in seen:
            existing = seen[key]
            existing_count = sum(1 for v in existing.specs.values() if v is not None and v != "")
            new_count = sum(1 for v in p.specs.values() if v is not None and v != "")
            if new_count > existing_count:
                seen[key] = p
        else:
            seen[key] = p

    return list(seen.values())


def _extract_products_from_batch(
    text: str, metadata: UploadMetadata, batch_num: int = 1
) -> list[ExtractedProduct]:
    """Extract products from a single text batch using GPT-4o-mini."""
    prompt = f"""You are an expert at extracting hydraulic product specifications from technical documents.

Extract ALL products mentioned in the following text. For each product, extract:
- model_code: the full model/part number
- product_name: what the product is (e.g. "Directional Control Valve")
- category: one of: directional_valves, proportional_directional_valves, pressure_valves, flow_valves, pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
- subcategory: more specific type (e.g. "check_valve", "vane_pump", "servo_valve")
- max_pressure_bar: maximum operating pressure in bar (number only)
- max_flow_lpm: maximum flow rate in litres per minute (number only)
- valve_size: e.g. "CETOP 3", "CETOP 5", "NG6", "NG10"
- spool_type: CRITICAL FIELD. The spool/function designation code AND its center condition description.
  This defines which ports are connected in each valve position (especially the center/neutral position).
  Include both the manufacturer's code and the functional description. Examples:
  "2A (all ports open to tank in center)", "D (P blocked, A&B to T)", "E (P&T blocked, A&B open)",
  "H (all ports blocked)", "J (P to A&B, T blocked)", "33C (tandem center, P to T, A&B blocked)".
  Look for spool designation tables, schematic symbols showing flow paths, and center condition descriptions.
  For Danfoss/Vickers: codes like 0A, 2A, 6C, 23, 33C, 36, etc.
  For Bosch Rexroth: codes like D, E, EA, H, J, L, M, R, U, W, etc.
  For Parker: codes like 01, 02, 06, 11, 20, 30, etc.
  For MOOG: letter/number codes in the model string.
- num_positions: number of switching positions (e.g. 2, 3)
- num_ports: number of ports/ways (e.g. 2, 3, 4)
- actuator_type: solenoid, manual, pilot, proportional
- coil_voltage: KEEP the full string, e.g. "12VDC", "24VDC", "110VAC", "220VAC" — never a bare number
- coil_type: e.g. "wet_pin", "dry", "explosion_proof"
- coil_connector: e.g. "DIN 43650-A", "Deutsch", "M12"
- port_size: KEEP the full string, e.g. "G3/8", "SAE-10", "M22x1.5" — never strip the prefix
- port_type: BSP, SAE, metric, NPTF
- mounting: e.g. "subplate", "inline", "manifold"
- mounting_pattern: e.g. "ISO 4401-03", "NFPA D03"
- body_material: e.g. "cast_iron", "steel", "aluminium"
- seal_material: e.g. "NBR", "FKM", "Viton"
- fluid_type: e.g. "mineral_oil", "HFC", "HFD", "synthetic_ester"
- viscosity_range_cst: e.g. "10-400" or "15-100"
- operating_temp_min_c: minimum temperature in Celsius (number only)
- operating_temp_max_c: maximum temperature in Celsius (number only)
- weight_kg: weight in kg (number only)
- displacement_cc: displacement in cc/rev (number only, for pumps/motors)
- speed_rpm_max: maximum speed in RPM (number only, for pumps/motors)
- bore_diameter_mm: bore diameter in mm (number only, for cylinders)
- rod_diameter_mm: rod diameter in mm (number only, for cylinders)
- stroke_mm: stroke length in mm (number only, for cylinders)

IMPORTANT: coil_voltage must be a string like "24VDC" or "110VAC", NEVER a bare number like 24.
IMPORTANT: port_size must keep its prefix like "G3/8" or "SAE-10", NEVER strip to just "3/8".

Also extract ANY other specification fields found in the document as additional key-value pairs.
Use descriptive snake_case names for any field not in the list above (e.g. "response_time_ms",
"hysteresis_percent", "step_response_ms", "repeat_accuracy_percent", "design_number",
"flow_class", "interface_standard", "connector_type", "special_function").
Do NOT omit any data — capture every specification mentioned in the text.

Company: {metadata.company}
Document type: {metadata.document_type}

Return a JSON object with key "products" containing an array of objects. Only include fields where you found actual data. If no products found, return {{"products": []}}.

Text:
{text}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        items = data if isinstance(data, list) else data.get("products", [])

        products = []
        for item in items:
            model_code = item.pop("model_code", "")
            if not model_code:
                continue
            product_name = item.pop("product_name", "")
            category = item.pop("category", metadata.category or "")
            products.append(ExtractedProduct(
                model_code=model_code,
                product_name=product_name,
                category=category,
                specs=item,
                raw_text=text[:500],
                confidence=0.6,
                source="llm",
            ))
        return products
    except Exception as e:
        logger.error(f"LLM extraction error (batch {batch_num}): {e}")
        return []


def extract_model_code_patterns_with_llm(
    text: str, company: str
) -> list[dict]:
    """Use GPT-4o-mini to extract model code breakdown patterns from user guide pages.
    These are the 'How to Order' or 'Model Code Breakdown' sections."""

    prompt = f"""You are an expert at reading hydraulic product user guides and extracting model code breakdown information.

Many hydraulic manufacturers include a section showing what each part of a model code means. For example:
  4WE6 D6X / EG24N9K4
  Where: 4WE6 = Series (4-way 6mm), D = Spool type, 6X = Size, EG24 = 24VDC coil, N9K4 = Design/seals

Extract ALL model code segment definitions from the following text for company: {company}

SPOOL TYPE EXTRACTION IS CRITICAL:
The spool type segment defines the valve's flow path configuration (which ports connect in each switching position).
This is THE most important spec for cross-referencing between manufacturers. Look carefully for:
- Spool designation tables showing codes and their center condition (e.g., "D = P blocked, A&B to T")
- Schematic symbol descriptions near valve function diagrams
- "Spool type", "Function", "Center condition", "Neutral position" sections
Each spool code corresponds to a specific port connection pattern:
  - All ports blocked (closed center)
  - All ports open to tank (open center)
  - P blocked, A&B to T (float center)
  - P to A&B, T blocked
  - P to T, A&B blocked (tandem center)
  ...and many manufacturer-specific variations.
Always map spool/function segments to maps_to_field: "spool_type".
The decoded_value MUST describe the flow path function, e.g.:
"2A - All ports open to tank in center" or "D - P blocked, A&B connected to T"

For each segment definition, provide:
- series: the base series name (e.g. "4WE6", "D1VW", "DHI")
- segment_position: which segment this is (0=first after series, 1=second, etc.)
- segment_name: what this segment represents (use these exact names: valve_size, spool_type, actuator_type, coil_voltage, port_size, port_type, mounting, seal_material, design_series, option_code)
- code_value: the specific code (e.g. "H7", "3", "2A")
- decoded_value: what it means (e.g. "24VDC", "CETOP 3", "function_2A")
- maps_to_field: which product spec field this sets (e.g. "coil_voltage", "valve_size", "spool_type", "actuator_type", "mounting", "seal_material")

Return a JSON array. If no model code breakdowns found, return [].

Text:
{text[:8000]}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        patterns = data if isinstance(data, list) else data.get("patterns", [])
        return patterns
    except Exception as e:
        print(f"LLM pattern extraction error: {e}")
        return []


def _parse_numeric_if_possible(value: str):
    """Try to parse a string as a number, but preserve mixed alpha-numeric values.

    Keeps values like '24VDC', 'G3/8', 'FKM', 'ISO 4401-03' intact.
    Only parses pure numbers or numbers followed by known unit suffixes.
    """
    if not value:
        return value

    stripped = value.strip()

    # Direct integer/float parse (pure numbers like "315", "4.5", "-20")
    try:
        if "." in stripped:
            return float(stripped)
        return int(stripped)
    except ValueError:
        pass

    # Strip known unit suffixes and try again (e.g., "315 bar" -> 315)
    unit_pattern = r"^([\d.]+)\s*(bar|lpm|l/min|mm|kg|cc|rpm|mpa|psi|°c|°f|nm|kw|hp|ml|cm)$"
    match = re.match(unit_pattern, stripped, re.IGNORECASE)
    if match:
        num_str = match.group(1)
        try:
            return float(num_str) if "." in num_str else int(num_str)
        except ValueError:
            pass

    # Contains letters mixed with digits (e.g. "24VDC", "G3/8", "NBR") — keep as string
    return stripped


# ── Ordering Code Combinatorial Extraction ─────────────────────────────

# Valid HydraulicProduct field names for LLM mapping validation
_VALID_SPEC_FIELDS = {
    "max_pressure_bar", "max_flow_lpm", "valve_size", "spool_type",
    "num_positions", "num_ports", "actuator_type", "coil_voltage",
    "coil_type", "coil_connector", "port_size", "port_type",
    "mounting", "mounting_pattern", "body_material", "seal_material",
    "operating_temp_min_c", "operating_temp_max_c", "fluid_type",
    "viscosity_range_cst", "weight_kg", "displacement_cc",
    "speed_rpm_max", "bore_diameter_mm", "rod_diameter_mm", "stroke_mm",
    "subcategory",
}

MAX_COMBINATIONS = 500


def extract_ordering_code_with_llm(
    text: str, company: str, category: str = ""
) -> list[OrderingCodeDefinition]:
    """Use GPT-4o-mini to extract ordering code breakdown tables from datasheets.

    Ordering code tables (also called 'How to Order' tables) show how model codes
    are constructed from positional segments. Each segment has one or more options.
    Fixed segments have exactly one value; variable segments have multiple choices.

    Returns a list of OrderingCodeDefinition objects (one per table found).
    """

    prompt = f"""You are an expert at reading hydraulic product datasheets and extracting ordering code breakdown tables.

These tables show how a model code is constructed from positional segments. Each segment has a POSITION number.
Some segments are FIXED (only one value exists), others are VARIABLE (multiple options the customer chooses from).

Example: A Bosch Rexroth 4WRE ordering code "4WREE6...04-3XV/24A1" has:
- Position 01: "4" (fixed, 4 main ports → maps to num_ports=4)
- Position 02: "WRE" (fixed, series identifier)
- Position 06: flow rate (variable) — options: "04" (4 l/min), "08" (8 l/min), "16" (16 l/min), "32" (32 l/min)
- Position 09: seal material (variable) — options: "V" (FKM), "M" (NBR)

For the text below, extract ALL ordering code breakdown tables for company: {company}

For EACH ordering code table, return:
1. "series": the base series identifier (e.g. "4WRE", "D1VW")
2. "product_name": what the product is (e.g. "Proportional directional valve")
3. "category": one of: directional_valves, proportional_directional_valves, pressure_valves, flow_valves, pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
4. "code_template": a template showing how to assemble the model code, using {{01}}, {{02}}, etc. for each position.
   Include separators exactly as they appear (dashes "-", slashes "/", dots ".").
   Example: "{{01}}{{02}}{{03}}{{04}}{{05}}{{06}}{{07}}-{{08}}{{09}}/{{10}}{{11}}{{12}}"
5. "segments": array of segment definitions, each with:
   - "position": the position number (integer, starting from 1)
   - "segment_name": descriptive name for this segment (e.g. "num_ports", "series", "flow_rate", "seal_material")
   - "is_fixed": true if only one value, false if multiple options
   - "separator_before": character(s) immediately before this segment in the code ("", "-", "/", ".")
   - "options": array of value options, each with:
     - "code": the characters that appear in the model code (e.g. "04", "V", "WRE"). Use "" (empty string) for "no code" options.
     - "description": human-readable meaning (e.g. "4 l/min", "FKM seals")
     - "maps_to_field": which spec field this sets. Known fields: {', '.join(sorted(_VALID_SPEC_FIELDS))}. You may also use ANY descriptive snake_case field name for specs not in this list (e.g. "design_number", "flow_class", "interface_standard", "manual_override", "connection_type"). EVERY segment must map to a field — use the segment_name if no known field fits (e.g. "series_code" for a series identifier).
     - "maps_to_value": the value to store in that field (e.g. 4 for max_flow_lpm, "FKM" for seal_material, "24VDC" for coil_voltage). Use the appropriate type (number for numeric fields, string for text fields).
6. "shared_specs": object with any specifications that apply to ALL variants from this table (e.g. {{"max_pressure_bar": 315}}).
   Extract these from the document text around the ordering code table.

SPOOL TYPE / VALVE FUNCTION SEGMENT — CRITICAL:
The spool type (also called valve function, spool code, or center condition) is one of the most important
segments in a directional valve ordering code. It defines how ports connect in each valve position.
You MUST map this segment to maps_to_field: "spool_type".

Spool type codes vary by manufacturer:
- Danfoss/Vickers: letter codes like "D", "E", "H", "J", "K", "L", "M", "S", "T", "W", or compound
  codes like "2A", "2B", "6C", "7C", "8C", "33C", "34C", "60B". These appear in model codes such as
  D1VW001CNJW (where "C" is the spool type segment) or DG4V-3-2A-M-... (where "2A" is the spool).
- Bosch Rexroth: letter codes like "D", "E", "EA", "H", "J", "L", "R", "U", "W" in model codes
  such as 4WE6-D6X/... or 4WE10-E3X/... (the letter after the size number is the spool type).
- Parker: numeric codes like "01", "02", "06", "11", "12", "20", "30", "60", "70".
- MOOG: letter codes embedded in the model number.

The maps_to_value for spool type must be ONLY the code itself, e.g.: "D", "2A", "2C", "H", "01".
Do NOT include the functional description in maps_to_value for spool_type.

Instead, for each spool type option, ALSO create a SECOND entry in the same option with:
  maps_to_field: "spool_function_description" and maps_to_value: the functional description.
This is done by setting BOTH fields on the same option object — the system will store both.

Actually, since each option can only set one maps_to_field, instead set maps_to_field to "spool_type"
with maps_to_value as ONLY the code (e.g. "D", "2A"). The system's spool analysis will separately
determine center conditions and solenoid functions from the document symbols/tables.

Common spool function types to recognize in segment descriptions:
- All ports blocked (closed center)
- All ports open to tank (open center / tandem)
- P blocked, A&B to T (float center)
- P to A, B to T (motor/directional)
- P to B, A to T (reverse directional)
- P to A&B, T blocked (regenerative)
- A&B blocked, P to T (unloading)

SEGMENT NAMING RULES — CRITICAL:
- EVERY segment MUST have a meaningful "segment_name" in snake_case (e.g. "manual_override", "design_number", "connection_type")
- EVERY segment MUST have maps_to_field set to either a known spec field OR the segment_name itself
- Do NOT use "" for maps_to_field — every segment maps to something. If it's a series identifier, use maps_to_field: "series_code"
- Include ALL positions, both fixed and variable
- For "no code" options (where nothing appears in the model code), set "code" to "" (empty string)
- Skip positions marked as "free text" or "further details" — do not include them
- maps_to_value for coil_voltage must be a string like "24VDC", not a bare number
- maps_to_value for numeric fields (max_flow_lpm, max_pressure_bar, etc.) must be a number
- Spool type/valve function segments MUST use maps_to_field: "spool_type"

Return a JSON object: {{"ordering_codes": [...]}}
If no ordering code tables found, return {{"ordering_codes": []}}

Text:
{text[:12000]}"""

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        data = json.loads(content)
        raw_codes = data.get("ordering_codes", [])

        definitions = []
        for raw in raw_codes:
            try:
                segments = []
                for seg_data in raw.get("segments", []):
                    segments.append(OrderingCodeSegment(
                        position=seg_data.get("position", 0),
                        segment_name=seg_data.get("segment_name", ""),
                        is_fixed=seg_data.get("is_fixed", True),
                        separator_before=seg_data.get("separator_before", ""),
                        options=seg_data.get("options", []),
                    ))

                definition = OrderingCodeDefinition(
                    company=company,
                    series=raw.get("series", ""),
                    product_name=raw.get("product_name", ""),
                    category=raw.get("category", category),
                    code_template=raw.get("code_template", ""),
                    segments=segments,
                    shared_specs=raw.get("shared_specs", {}),
                )
                if definition.series and definition.segments:
                    definitions.append(definition)
            except Exception as e:
                print(f"Error parsing ordering code definition: {e}")
                continue

        return definitions
    except Exception as e:
        print(f"LLM ordering code extraction error: {e}")
        return []


def assemble_model_code(template: str, segment_values: dict) -> str:
    """Assemble a model code string from a code template and segment values.

    Template uses {01}, {02}, etc. for each position.
    segment_values maps position (int) to chosen code string.
    """
    result = template
    for pos, value in segment_values.items():
        placeholder = f"{{{pos:02d}}}"
        result = result.replace(placeholder, value if value else "")

    # Clean up artifacts from empty optional segments:
    # - Double separators (e.g., "--" from empty step function code)
    # - Trailing/leading separators
    result = re.sub(r"-{2,}", "-", result)
    result = re.sub(r"/{2,}", "/", result)
    result = result.strip("-/. ")

    return result


def generate_products_from_ordering_code(
    definition: OrderingCodeDefinition,
    metadata: UploadMetadata,
) -> list[ExtractedProduct]:
    """Generate all product combinations from an ordering code definition.

    For each combination of variable segment options, creates an ExtractedProduct
    with the assembled model code and populated spec fields.

    Caps output at MAX_COMBINATIONS to prevent combinatorial explosion.
    """
    if not definition.segments:
        return []

    # Separate fixed and variable segments
    fixed_segments = [s for s in definition.segments if s.is_fixed]
    variable_segments = [s for s in definition.segments if not s.is_fixed and len(s.options) > 0]

    # If no variable segments, generate a single product
    if not variable_segments:
        variable_segments_options = [[]]
    else:
        variable_segments_options = [seg.options for seg in variable_segments]

    # Calculate total combinations for logging
    total = 1
    for opts in variable_segments_options:
        total *= max(len(opts), 1)

    if total > MAX_COMBINATIONS:
        print(f"WARNING: Ordering code for series {definition.series} has {total} combinations, "
              f"capping at {MAX_COMBINATIONS}")

    products = []
    combos = itertools.product(*variable_segments_options) if variable_segments else [()]

    for combo in itertools.islice(combos, MAX_COMBINATIONS):
        segment_values = {}
        specs = dict(definition.shared_specs)

        # Fixed segments: use their single option
        for seg in fixed_segments:
            if seg.options:
                opt = seg.options[0]
                code = opt.get("code", "")
                segment_values[seg.position] = code
                field = opt.get("maps_to_field", "")
                value = opt.get("maps_to_value")
                if field and value is not None:
                    specs[field] = value
                # Always store segment by name so it becomes a visible column
                seg_name = seg.segment_name
                if seg_name and seg_name != field:
                    desc = opt.get("description", "")
                    specs[seg_name] = f"{code} - {desc}" if code and desc else (desc or code or "")

        # Variable segments: use the chosen option from this combination
        for seg, chosen_option in zip(variable_segments, combo):
            code = chosen_option.get("code", "")
            segment_values[seg.position] = code
            field = chosen_option.get("maps_to_field", "")
            value = chosen_option.get("maps_to_value")
            if field and value is not None:
                specs[field] = value
            # Always store segment by name so it becomes a visible column
            seg_name = seg.segment_name
            if seg_name and seg_name != field:
                desc = chosen_option.get("description", "")
                specs[seg_name] = f"{code} - {desc}" if code and desc else (desc or code or "")

        # Assemble the model code
        model_code = assemble_model_code(definition.code_template, segment_values)
        if not model_code:
            continue

        product = ExtractedProduct(
            model_code=model_code,
            product_name=definition.product_name,
            category=definition.category or metadata.category or "",
            specs=specs,
            raw_text=f"Generated from ordering code table: series {definition.series}",
            confidence=0.85,
            source="ordering_code",
        )
        products.append(product)

    return products


# ---------------------------------------------------------------------------
# Step 5: Deep spool function analysis
# ---------------------------------------------------------------------------

def compute_canonical_pattern(center_condition: str, sol_a_function: str, sol_b_function: str) -> str:
    """Normalise spool flow-path descriptions into a deterministic canonical string.

    This allows cross-manufacturer matching: Danfoss "2A" and Bosch "D" both
    produce "BLOCKED|PA-BT|PB-AT" because they share the same flow paths.

    Each position is normalised by:
      1. Uppercasing
      2. Extracting port connections (e.g. P→A, B→T becomes PA-BT)
      3. Sorting connections alphabetically within each position
      4. Joining positions with |
    """

    def _normalise_position(desc: str) -> str:
        if not desc:
            return "UNKNOWN"
        desc = desc.upper().strip()

        # Common center condition keywords — check more specific patterns FIRST
        # (float/tandem/open descriptions may contain "blocked" as a substring)
        float_keywords = ["FLOAT", "FLOATING"]
        tandem_keywords = ["TANDEM", "P-T CONNECTED", "PA-BT TANDEM"]
        open_keywords = ["ALL PORTS OPEN", "OPEN CENTER", "OPEN CENTRE", "ALL OPEN", "FREE FLOW"]
        blocked_keywords = ["ALL PORTS BLOCKED", "BLOCKED", "CLOSED", "ALL CLOSED"]

        for kw in float_keywords:
            if kw in desc:
                return "FLOAT"
        for kw in tandem_keywords:
            if kw in desc:
                return "TANDEM"
        for kw in open_keywords:
            if kw in desc:
                return "OPEN"
        for kw in blocked_keywords:
            if kw in desc:
                return "BLOCKED"

        # Extract port connections like P→A, B→T or P-A, B-T or PA, BT
        connections = re.findall(r'([PABTLR])\s*[→\->to\s]+\s*([PABTLR])', desc)
        if connections:
            parts = sorted([f"{a}{b}" for a, b in connections])
            return "-".join(parts)

        # Fallback: clean up and return as-is
        clean = re.sub(r'[^A-Z0-9\- ]', '', desc).strip()
        return clean if clean else "UNKNOWN"

    center = _normalise_position(center_condition)
    sol_a = _normalise_position(sol_a_function)
    sol_b = _normalise_position(sol_b_function)

    return f"{center}|{sol_a}|{sol_b}"


def analyze_spool_functions(text: str, company: str) -> list[dict]:
    """Perform a deep second-pass LLM analysis of spool/valve function symbols.

    Reads the full user guide text and extracts structured spool function data
    for every spool type mentioned.  Uses hydraulic engineering domain knowledge
    to understand centre conditions and solenoid functions.

    Returns a list of dicts, each with:
        spool_code, center_condition, solenoid_a_function, solenoid_b_function,
        description, canonical_pattern
    """
    load_dotenv()
    client = OpenAI()

    # Truncate to avoid token limits while keeping enough context
    max_chars = 80_000
    if len(text) > max_chars:
        text = text[:max_chars]

    prompt = f"""You are an expert hydraulic engineer analysing a product user guide from {company}.

Your task: identify EVERY spool type / valve function code mentioned in this document and
extract detailed functional information about each one.

For each spool type found, provide:
1. **spool_code**: the code designation (e.g. "2A", "2C", "4A", "D", "H", "K", "OC")
2. **center_condition**: what happens when the valve is in the center/neutral position
   (e.g. "All ports blocked", "All ports open", "P and T connected, A and B blocked",
   "Float - A, B, T connected, P blocked")
3. **solenoid_a_function**: flow path when solenoid A (or left solenoid) is energised
   (e.g. "P→A, B→T")
4. **solenoid_b_function**: flow path when solenoid B (or right solenoid) is energised
   (e.g. "P→B, A→T")
5. **description**: a brief human-readable summary of the spool function

Look carefully at:
- Ordering code breakdown tables (spool type segment)
- Hydraulic circuit symbol diagrams (described in text as flow paths)
- Centre condition tables
- Any tables showing port connections per spool position
- Valve function descriptions in the text

If the document shows a spool symbol diagram, interpret the flow arrows to determine
port connections in each position (left = solenoid A energised, center = neutral,
right = solenoid B energised).

For 2-position valves (no center condition), set center_condition to "N/A (2-position valve)"
and only fill solenoid_a_function.

Return valid JSON array. Example:
[
  {{
    "spool_code": "2A",
    "center_condition": "All ports blocked",
    "solenoid_a_function": "P→A, B→T",
    "solenoid_b_function": "P→B, A→T",
    "description": "Closed center, standard crossover"
  }},
  {{
    "spool_code": "2C",
    "center_condition": "All ports blocked (no crossover)",
    "solenoid_a_function": "P→A, B→T",
    "solenoid_b_function": "P→B, A→T",
    "description": "Closed center, no crossover in transition"
  }}
]

If NO spool types are found in the document, return an empty array: []

DOCUMENT TEXT:
{text}"""

    try:
        response = client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            response_format={"type": "json_object"},
        )

        raw = response.choices[0].message.content.strip()
        parsed = json.loads(raw)

        # Handle both {"spool_functions": [...]} and direct [...]
        if isinstance(parsed, dict):
            # Find the first list value
            for v in parsed.values():
                if isinstance(v, list):
                    parsed = v
                    break
            else:
                parsed = []

        if not isinstance(parsed, list):
            logger.warning("Spool analysis returned non-list: %s", type(parsed))
            return []

        # Enrich each result with canonical pattern
        results = []
        for item in parsed:
            if not isinstance(item, dict) or not item.get("spool_code"):
                continue

            item["canonical_pattern"] = compute_canonical_pattern(
                item.get("center_condition", ""),
                item.get("solenoid_a_function", ""),
                item.get("solenoid_b_function", ""),
            )
            results.append(item)

        logger.info("Spool function analysis found %d spool types for %s", len(results), company)
        return results

    except Exception as e:
        logger.error("Spool function analysis failed: %s", e)
        return []
