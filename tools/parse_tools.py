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

# Model used for structured extraction tasks (ordering codes, spool analysis, product extraction).
# GPT-4.1-mini has significantly better instruction-following than GPT-4o-mini,
# critical for correctly mapping ordering code segments to named fields.
EXTRACTION_MODEL = "gpt-4.1-mini"

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
- spool_type: The spool/valve function designation CODE ONLY — just the short alphanumeric code.
  Examples of CORRECT spool_type values: "D", "2A", "H", "01", "2C", "6C", "33C", "E", "J".
  Do NOT include descriptions — just the manufacturer's code letter/number.
  For Danfoss/Vickers: codes like 0A, 2A, 6C, 23, 33C, 36, etc.
  For Bosch Rexroth: codes like D, E, EA, H, J, L, M, R, U, W, etc.
  For Parker: codes like 01, 02, 06, 11, 20, 30, etc.
  For MOOG: letter/number codes in the model string.
- spool_function_description: the functional description of the spool type's center condition.
  Examples: "All ports blocked", "P to A, B to T", "Float - A&B to T, P blocked".
  This is SEPARATE from spool_type — do NOT combine them.
- manual_override: override option code and description if present.
  Examples: "Z - No overrides", "H - Water resistant", "Blank - No override".
  CRITICAL: Do NOT confuse manual override options with spool type codes.
  If a code means "No overrides" or "water resistant override", it is NOT a spool type.
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
            model=EXTRACTION_MODEL,
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
            model=EXTRACTION_MODEL,
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

# Section header patterns used for smart text selection
_ORDERING_CODE_HEADERS = re.compile(
    r'(?:ordering\s+code|how\s+to\s+order|model\s+(?:code|designation|number)\s+breakdown'
    r'|order\s+number\s+code|ordering\s+information|code\s+structure)',
    re.IGNORECASE,
)
_SPOOL_SECTION_HEADERS = re.compile(
    r'(?:spool\s+type|valve\s+function|center\s+condition|centre\s+condition'
    r'|spool\s+designation|spool\s+code|valve\s+spool)',
    re.IGNORECASE,
)


def _select_ordering_code_text(full_text: str, max_chars: int = 40000) -> str:
    """Select the most relevant text for ordering code extraction.

    Instead of blindly truncating at N characters, this scans for section headers
    like 'Ordering Code', 'Spool Type', 'Center Condition' and extracts those
    sections with generous surrounding context.  Overlapping ranges are merged.

    Falls back to first *max_chars* characters if nothing is found.
    """
    if len(full_text) <= max_chars:
        return full_text

    CONTEXT_BEFORE = 2000
    CONTEXT_AFTER = 4000  # spool tables often follow the heading

    ranges: list[tuple[int, int]] = []

    for pattern in (_ORDERING_CODE_HEADERS, _SPOOL_SECTION_HEADERS):
        for match in pattern.finditer(full_text):
            start = max(0, match.start() - CONTEXT_BEFORE)
            end = min(len(full_text), match.end() + CONTEXT_AFTER)
            ranges.append((start, end))

    if not ranges:
        return full_text[:max_chars]

    # Merge overlapping ranges
    ranges.sort()
    merged: list[tuple[int, int]] = [ranges[0]]
    for start, end in ranges[1:]:
        if start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))

    # Collect text from merged ranges, respecting max_chars
    parts: list[str] = []
    total = 0
    for start, end in merged:
        chunk = full_text[start:end]
        if total + len(chunk) > max_chars:
            remaining = max_chars - total
            if remaining > 500:
                parts.append(chunk[:remaining])
            break
        parts.append(chunk)
        total += len(chunk)

    return "\n\n...\n\n".join(parts)


def extract_ordering_code_with_llm(
    text: str, company: str, category: str = "",
    known_spool_codes: list[str] = None,
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

CRITICAL — BLANK/OPTIONAL POSITIONS:
Some panels in the model code breakdown diagram may appear BLANK or show no default code.
These blank panels are NOT to be skipped. They represent optional or variable positions.
You MUST:
1. Note the numbered position of the blank panel (e.g. position 7)
2. Find the CORRESPONDING numbered description in the description table/list below the diagram
   (these are usually numbered footnotes or a table with "Position 7: ..." or "7 = ...")
3. Extract ALL the option codes and descriptions listed for that position
4. Include the position as a variable segment with is_fixed=false and all its options
5. EVERY letter/number that can appear at ANY position in the model code MUST be captured

Do NOT collapse or skip ANY position. The resulting model codes must have ALL positions filled in.
If a position is truly optional (no code needed), include it with one option having code="" (empty string).

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

SPOOL OPTION COUNT — IMPORTANT:
For standard directional valve series (DG4V, 4WE, D1VW, D*FW, etc.), the spool type segment
typically has 10-30+ different options. If you find fewer than 5 spool type options for a
directional valve series, you are VERY LIKELY MISSING options. Look more carefully through the
entire document text for:
- Spool type / valve function tables listing all available codes
- Center condition tables with spool codes
- Ordering code description footnotes listing spool options
- Any table or list mapping letter codes to flow-path descriptions
The spool type segment MUST be is_fixed=false with ALL options listed.

SEGMENT NAMING RULES — CRITICAL:
- EVERY segment MUST have a meaningful "segment_name" in snake_case (e.g. "manual_override", "design_number", "connection_type")
- EVERY segment MUST have maps_to_field set to either a known spec field OR the segment_name itself
- Do NOT use "" for maps_to_field — every segment maps to something. If it's a series identifier, use maps_to_field: "series_code"
- Include ALL positions, both fixed and variable — NEVER skip a position number
- A blank panel in the diagram still has a position number — find its options in the description tables below
- For "no code" options (where nothing appears in the model code), set "code" to "" (empty string)
- Skip positions marked as "free text" or "further details" — do not include them
- maps_to_value for coil_voltage must be a string like "24VDC", not a bare number
- maps_to_value for numeric fields (max_flow_lpm, max_pressure_bar, etc.) must be a number
- Spool type/valve function segments MUST use maps_to_field: "spool_type"
- The total number of segments MUST match the total number of positions shown in the breakdown diagram

Return a JSON object: {{"ordering_codes": [...]}}
If no ordering code tables found, return {{"ordering_codes": []}}
"""

    # Inject known spool reference data when available
    if known_spool_codes:
        codes_str = ", ".join(f'"{c}"' for c in sorted(known_spool_codes))
        prompt += f"""
REFERENCE DATA — KNOWN SPOOL TYPES:
For {company} products in this series family, the following spool type codes are KNOWN to exist
from previous confirmed extractions: {codes_str}
You MUST ensure ALL of these appear as options in the spool type segment (with is_fixed=false).
If the document text does not explicitly mention some of these codes, still include them as
options with description "Known spool type (from reference data)".
Also include any ADDITIONAL spool types you find in the document that are NOT in this list.
"""

    # Use smart text selection to capture ordering code + spool sections
    selected_text = _select_ordering_code_text(text)
    prompt += f"\nText:\n{selected_text}"

    try:
        response = _get_client().chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        logger.info("Ordering code LLM response (first 2000 chars): %s", content[:2000])
        data = json.loads(content)
        raw_codes = data.get("ordering_codes", [])

        return _parse_ordering_code_response(raw_codes, company, category)
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
    primary_spool_codes: list[str] = None,
) -> list[ExtractedProduct]:
    """Generate all product combinations from an ordering code definition.

    For each combination of variable segment options, creates an ExtractedProduct
    with the assembled model code and populated spec fields.

    When primary_spool_codes is provided and non-empty, the spool type segment is
    filtered to only those codes before generating combinations. This dramatically
    reduces the combination count for valve series with many spool options.

    Caps output at MAX_COMBINATIONS to prevent combinatorial explosion.
    """
    if not definition.segments:
        return []

    # Separate fixed and variable segments
    fixed_segments = [s for s in definition.segments if s.is_fixed]
    variable_segments = [s for s in definition.segments if not s.is_fixed and len(s.options) > 0]

    # Filter spool segment to primary codes only (if specified)
    if primary_spool_codes:
        primary_upper = {c.upper() for c in primary_spool_codes}
        for seg in variable_segments:
            is_spool_seg = seg.segment_name == "spool_type"
            if not is_spool_seg:
                for opt in seg.options:
                    if opt.get("maps_to_field") == "spool_type":
                        is_spool_seg = True
                        break
            if is_spool_seg:
                original_options = list(seg.options)
                filtered = [
                    opt for opt in original_options
                    if opt.get("code", "").upper() in primary_upper
                ]
                if filtered:
                    logger.info(
                        "Primary spool filter: %d -> %d spool options for %s",
                        len(original_options), len(filtered), definition.series,
                    )
                    seg.options = filtered
                else:
                    logger.warning(
                        "Primary spool filter: no matching primary codes for %s, "
                        "keeping all %d options",
                        definition.series, len(original_options),
                    )
                break

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
    _logged_first = False

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
                # Also store segment by name for a visible column (human-readable)
                seg_name = seg.segment_name
                if seg_name and seg_name != field:
                    desc = opt.get("description", "")
                    readable = f"{code} - {desc}" if code and desc else (desc or code or "")
                    if readable:
                        specs[seg_name] = readable

        # Variable segments: use the chosen option from this combination
        for seg, chosen_option in zip(variable_segments, combo):
            code = chosen_option.get("code", "")
            segment_values[seg.position] = code
            field = chosen_option.get("maps_to_field", "")
            value = chosen_option.get("maps_to_value")
            if field and value is not None:
                specs[field] = value
            # Also store segment by name for a visible column (human-readable)
            seg_name = seg.segment_name
            if seg_name and seg_name != field:
                desc = chosen_option.get("description", "")
                readable = f"{code} - {desc}" if code and desc else (desc or code or "")
                if readable:
                    specs[seg_name] = readable

        # Assemble the model code
        model_code = assemble_model_code(definition.code_template, segment_values)
        if not model_code:
            continue

        # Log the first product's full spec keys for diagnostics
        if not _logged_first:
            logger.info("First generated product '%s' spec keys: %s",
                        model_code, sorted(specs.keys()))
            _logged_first = True

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
            model=EXTRACTION_MODEL,
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


# ---------------------------------------------------------------------------
# Step 6: Vision-based spool symbol extraction from PDF pages
# ---------------------------------------------------------------------------

SPOOL_VISION_MODEL = "gpt-4o"

_SPOOL_VISION_PROMPT = """You are an expert hydraulic engineer analysing a page from a product datasheet.

This page may contain HYDRAULIC SPOOL SYMBOL DIAGRAMS and/or a SPOOL OPTIONS TABLE.

## SPOOL SYMBOL DIAGRAMS
These are standardised schematic diagrams (ISO 1219) showing valve spool positions with
flow path arrows, blocked ports, and check valves.

Each spool symbol shows:
- A rectangle divided into sections (one section per valve switching position)
- Arrows showing flow paths between ports (P=pressure, T=tank, A/B=work ports)
- Blocked ports shown as T-symbols or dead-end lines
- A letter/number CODE labelling the symbol (e.g. "D", "E", "H", "2A", "01")

## SPOOL OPTIONS TABLE
Some pages contain a TABLE listing all available spool types with columns like:
- Spool code / type designation
- Center condition description
- Active (solenoid energised) function
- Crossover function during transition

If you see a spool options table, extract ALL rows from it.

## Your task:
Identify ALL spool symbols AND/OR table rows on this page. For each one extract:

1. "spool_code": the letter/number designation (e.g. "D", "2A", "H", "01", "EA")
2. "center_condition": what happens in the center/neutral position
   (e.g. "All ports blocked", "P and T connected, A and B blocked", "All ports open to tank")
3. "solenoid_a_function": flow path when left/solenoid-A is energised (e.g. "P→A, B→T")
4. "solenoid_b_function": flow path when right/solenoid-B is energised (e.g. "P→B, A→T")
5. "description": brief summary of the spool function
6. "symbol_description": describe the visual symbol pattern in detail (arrows, blocked ports,
   flow paths in each position from left to right). This text description of the VISUAL SYMBOL
   will be used to match equivalent spool functions across different manufacturers regardless
   of their letter codes. If from a table (no diagram), describe the functional behavior.

For 2-position valves, set center_condition to "N/A (2-position valve)" and only fill solenoid_a_function.

Return valid JSON: {"spool_symbols": [...]}
If NO spool symbols or spool tables are found on this page, return: {"spool_symbols": []}

IMPORTANT: Focus on the VISUAL flow path patterns shown by the arrows in the symbol diagrams.
Two manufacturers may use different letter codes for the exact same hydraulic function —
the symbol diagram is what matters for cross-referencing.
If both diagrams AND a table are present, extract from BOTH but deduplicate by spool_code."""


def extract_spool_symbols_from_pdf(pdf_path: str, company: str) -> list[dict]:
    """Extract spool symbols from PDF pages using GPT-4o vision.

    Renders each page as an image, sends to GPT-4o to identify hydraulic spool
    symbol diagrams, and returns structured spool data with canonical patterns.

    This catches spool information that text extraction misses because the symbols
    are graphical (ISO 1219 hydraulic schematic diagrams).

    Requires PyMuPDF (fitz) for page rendering.
    """
    if not HAS_PYMUPDF:
        logger.warning("PyMuPDF not available — skipping vision spool extraction")
        return []

    import base64

    try:
        doc = _fitz.open(pdf_path)
    except Exception as e:
        logger.error("Failed to open PDF for spool vision: %s", e)
        return []

    all_spools = []
    # Focus on pages likely to have spool symbols (typically later pages)
    # Process all pages but log which ones yield results
    total_pages = len(doc)
    logger.info("Spool vision extraction: scanning %d pages in %s", total_pages, pdf_path)

    for page_idx in range(total_pages):
        try:
            page = doc[page_idx]
            # Render page as PNG at 150 DPI (good balance of quality vs size)
            pix = page.get_pixmap(dpi=150)
            image_bytes = pix.tobytes("png")

            # Skip very small pages (probably blank or cover)
            if len(image_bytes) < 5000:
                continue

            image_b64 = base64.b64encode(image_bytes).decode("utf-8")

            response = _get_client().chat.completions.create(
                model=SPOOL_VISION_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": _SPOOL_VISION_PROMPT},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/png;base64,{image_b64}",
                                    "detail": "high",
                                },
                            },
                        ],
                    }
                ],
                response_format={"type": "json_object"},
                max_tokens=2000,
                temperature=0.1,
            )

            content = response.choices[0].message.content.strip()
            data = json.loads(content)
            symbols = data.get("spool_symbols", [])

            if symbols:
                logger.info("Page %d: found %d spool symbols via vision", page_idx + 1, len(symbols))
                for sym in symbols:
                    if not isinstance(sym, dict) or not sym.get("spool_code"):
                        continue
                    sym["source_page"] = page_idx + 1
                    sym["company"] = company
                    sym["extraction_method"] = "vision"
                    # Compute canonical pattern from the extracted flow paths
                    sym["canonical_pattern"] = compute_canonical_pattern(
                        sym.get("center_condition", ""),
                        sym.get("solenoid_a_function", ""),
                        sym.get("solenoid_b_function", ""),
                    )
                    all_spools.append(sym)

        except Exception as e:
            logger.warning("Spool vision extraction failed on page %d: %s", page_idx + 1, e)
            continue

    doc.close()

    # Deduplicate by spool_code (keep the one with most data)
    seen = {}
    for s in all_spools:
        code = s["spool_code"].strip().upper()
        if code in seen:
            existing = seen[code]
            # Keep the one with a longer symbol_description (more detail)
            if len(s.get("symbol_description", "")) > len(existing.get("symbol_description", "")):
                seen[code] = s
        else:
            seen[code] = s

    results = list(seen.values())
    logger.info("Spool vision extraction total: %d unique spool symbols for %s",
                len(results), company)
    return results


# ---------------------------------------------------------------------------
# Step 6b: Graphics-heavy PDF detection + Vision ordering code extraction
# ---------------------------------------------------------------------------


def _get_pdf_page_count(pdf_path: str) -> int:
    """Return the number of pages in a PDF."""
    if HAS_PYMUPDF:
        try:
            doc = _fitz.open(pdf_path)
            count = len(doc)
            doc.close()
            return count
        except Exception:
            pass
    # Fallback to pypdf
    try:
        from pypdf import PdfReader
        return len(PdfReader(pdf_path).pages)
    except Exception:
        return 0


def _is_graphics_heavy_pdf(pages: list[dict], total_page_count: int) -> bool:
    """Detect if a PDF is predominantly vector-graphic-based.

    Graphics-heavy PDFs (e.g. Danfoss datasheets) render ordering code
    diagrams, spool tables, and specs as vector graphics with very little
    extractable text.  This function checks the average text yield per page
    and classifies the PDF accordingly.

    Args:
        pages: Output from extract_text_from_pdf() — list of {page, text}.
        total_page_count: Total page count in the PDF (from _get_pdf_page_count).

    Returns:
        True if the PDF is likely graphics-heavy and needs vision extraction.
    """
    if total_page_count < 2:
        return False  # Very short PDFs are probably text-only or single-page

    if not pages:
        # No text at all — definitely graphics-heavy
        return total_page_count >= 2

    # Calculate average chars per page (across ALL pages, including blanks)
    total_chars = sum(len(p["text"]) for p in pages)
    avg_chars = total_chars / total_page_count

    # Count sparse pages (less than 300 chars of useful text — pdfplumber
    # supplementary extraction inflates counts with scattered labels)
    sparse_threshold = 300
    sparse_count = 0
    for page_num in range(1, total_page_count + 1):
        page_text = ""
        for p in pages:
            if p["page"] == page_num:
                page_text = p["text"]
                break
        if len(page_text) < sparse_threshold:
            sparse_count += 1

    sparse_ratio = sparse_count / total_page_count

    # Classify as graphics-heavy if EITHER condition is met:
    #   1. Very low average text (< 500 chars/page) AND most pages are sparse (> 40%)
    #   2. Almost all pages are sparse (> 70%) regardless of average
    is_heavy = (avg_chars < 500 and sparse_ratio > 0.4) or sparse_ratio > 0.7

    logger.info(
        "Graphics-heavy detection: avg_chars=%.0f, sparse_pages=%d/%d (%.0f%%), "
        "result=%s",
        avg_chars, sparse_count, total_page_count, sparse_ratio * 100,
        "GRAPHICS-HEAVY" if is_heavy else "text-ok",
    )

    return is_heavy


def _parse_ordering_code_response(
    raw_codes: list[dict], company: str, category: str,
) -> list[OrderingCodeDefinition]:
    """Parse raw ordering code JSON dicts into OrderingCodeDefinition objects.

    Shared parser used by both text-based (extract_ordering_code_with_llm)
    and vision-based (extract_ordering_code_from_images) extraction paths.
    """
    definitions = []
    for raw in raw_codes:
        try:
            segments = []
            for seg_data in raw.get("segments", []):
                seg_name = seg_data.get("segment_name", "")
                # Defensive fallback: auto-generate segment_name from description if missing
                if not seg_name:
                    desc = ""
                    opts = seg_data.get("options", [])
                    if opts:
                        desc = opts[0].get("description", "")
                    if not desc:
                        desc = f"position_{seg_data.get('position', 0)}"
                    # Convert description to snake_case
                    seg_name = re.sub(r'[^a-z0-9]+', '_', desc.lower()).strip('_')[:40]
                    logger.warning("Segment at position %d had no segment_name, "
                                   "auto-generated: '%s'", seg_data.get("position", 0), seg_name)
                    seg_data["segment_name"] = seg_name

                # Defensive fallback: ensure maps_to_field is set on all options
                for opt in seg_data.get("options", []):
                    if not opt.get("maps_to_field"):
                        opt["maps_to_field"] = seg_name
                        if opt.get("maps_to_value") is None:
                            opt["maps_to_value"] = opt.get("description", opt.get("code", ""))
                        logger.warning("Option code '%s' had no maps_to_field, "
                                       "set to segment_name: '%s'",
                                       opt.get("code", ""), seg_name)

                segments.append(OrderingCodeSegment(
                    position=seg_data.get("position", 0),
                    segment_name=seg_name,
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


# Vision prompt for ordering code diagram extraction
_ORDERING_CODE_VISION_PROMPT = """You are an expert hydraulic engineer reading a product datasheet.
These pages contain an ORDERING CODE BREAKDOWN DIAGRAM showing how
model codes are constructed from positional segments.

Read the diagram and ALL associated description tables. Extract the
COMPLETE ordering code structure:

1. The model code template using {01}, {02}, {03}, etc. as placeholders for each position.
   Include separators exactly as they appear (dashes "-", slashes "/", dots ".").
   Example: "{01}{02}{03}{04}{05}{06}{07}-{08}{09}/{10}{11}{12}"
   Another example: "{01}-{02}-{03}-{04}-{05}"
   Do NOT use underscores, asterisks, or blanks — ONLY use {01}, {02}, etc.
2. Each numbered position with ALL available option codes
3. Which positions are fixed vs. variable (customer selects)
4. Separators between positions (dashes, slashes, etc.) — put them literally in the template

SPOOL TYPE / VALVE FUNCTION SEGMENT — CRITICAL:
The spool type (also called valve function, spool code, or center condition) is one of the most important
segments in a directional valve ordering code. It defines how ports connect in each valve position.
You MUST map this segment to maps_to_field: "spool_type".

Spool type codes vary by manufacturer:
- Danfoss/Vickers: letter codes like "D", "E", "H", "J", "K", "L", "M", "S", "T", "W", or compound
  codes like "2A", "2B", "6C", "7C", "8C", "33C", "34C", "60B".
- Bosch Rexroth: letter codes like "D", "E", "EA", "H", "J", "L", "R", "U", "W".
- Parker: numeric codes like "01", "02", "06", "11", "12", "20", "30", "60", "70".
- MOOG: letter codes embedded in the model number.

The maps_to_value for spool type must be ONLY the code itself, e.g.: "D", "2A", "2C", "H", "01".
Do NOT include the functional description in maps_to_value for spool_type.

SPOOL OPTION COUNT — IMPORTANT:
For standard directional valve series (DG4V, 4WE, D1VW, D*FW, etc.), the spool type segment
typically has 10-30+ different options. If you find fewer than 5 spool type options for a
directional valve series, you are VERY LIKELY MISSING options. Look more carefully through the
entire document for spool type / valve function tables listing all available codes.

CRITICAL — BLANK/OPTIONAL POSITIONS:
Some panels in the model code breakdown diagram may appear BLANK or show no default code.
These blank panels are NOT to be skipped. They represent optional or variable positions.
You MUST:
1. Note the numbered position of the blank panel (e.g. position 7)
2. Find the CORRESPONDING numbered description in the description table/list nearby
3. Extract ALL the option codes and descriptions listed for that position
4. Include the position as a variable segment with is_fixed=false and all its options

SEGMENT NAMING RULES:
- EVERY segment MUST have a meaningful "segment_name" in snake_case
- EVERY segment MUST have maps_to_field set to a known spec field or the segment_name itself
- Known fields: actuator_type, body_material, bore_diameter_mm, coil_connector, coil_type, coil_voltage, displacement_cc, fluid_type, max_flow_lpm, max_pressure_bar, mounting, mounting_pattern, num_ports, num_positions, operating_temp_max_c, operating_temp_min_c, port_size, port_type, rod_diameter_mm, seal_material, speed_rpm_max, spool_type, stroke_mm, subcategory, valve_size, viscosity_range_cst, weight_kg
- You may also use ANY descriptive snake_case name for non-standard fields
- Include ALL positions, both fixed and variable — NEVER skip a position number
- For "no code" options (where nothing appears in the model code), set "code" to "" (empty string)
- The total number of segments MUST match the total number of positions shown in the breakdown diagram

CODE TEMPLATE FORMAT — CRITICAL:
The "code_template" field MUST use {01}, {02}, {03}, etc. as position placeholders.
Each placeholder number corresponds to the segment position number.
Example for "DG4V-3-2A-M-FW-B5-60": the template would be "{01}-{02}-{03}-{04}-{05}-{06}-{07}"

Return JSON: {"ordering_codes": [{"series": "...", "product_name": "...", "category": "...", "code_template": "template using {01},{02},etc.", "segments": [...], "shared_specs": {...}}]}
Each segment: {"position": N, "segment_name": "...", "is_fixed": bool, "separator_before": "", "options": [{"code": "...", "description": "...", "maps_to_field": "...", "maps_to_value": ...}]}

If no ordering code tables found, return {"ordering_codes": []}"""


def extract_ordering_code_from_images(
    pdf_path: str,
    company: str,
    category: str = "",
    known_spool_codes: list[str] = None,
    target_pages: list[int] = None,
    dpi: int = 200,
) -> list[OrderingCodeDefinition]:
    """Extract ordering code breakdown from PDF pages using GPT-4o vision.

    Renders target pages as PNG images and sends them in a single GPT-4o
    multi-image call for cross-page context. Designed for graphics-heavy
    PDFs where text extraction yields little usable content.

    Args:
        pdf_path: Path to the PDF file.
        company: Manufacturer name.
        category: Product category hint.
        known_spool_codes: Spool codes to inject as reference.
        target_pages: 0-indexed page numbers to render. Defaults to pages 1-5 (0-indexed).
        dpi: Render resolution. Higher = better OCR but larger tokens.

    Returns:
        List of OrderingCodeDefinition objects (same format as text-based extraction).
    """
    if not HAS_PYMUPDF:
        logger.warning("PyMuPDF not available — cannot do vision ordering code extraction")
        return []

    import base64

    try:
        doc = _fitz.open(pdf_path)
    except Exception as e:
        logger.error("Failed to open PDF for vision ordering code: %s", e)
        return []

    total_pages = len(doc)

    # Default: pages 1-10 (0-indexed) — ordering codes on early pages, spool tables on later pages
    if target_pages is None:
        target_pages = list(range(min(10, total_pages)))

    # Render pages to PNG
    images_b64 = []
    for page_idx in target_pages:
        if page_idx >= total_pages:
            continue
        try:
            page = doc[page_idx]
            pix = page.get_pixmap(dpi=dpi)
            image_bytes = pix.tobytes("png")
            if len(image_bytes) < 5000:
                continue  # Skip blank/tiny pages
            images_b64.append(base64.b64encode(image_bytes).decode("utf-8"))
            logger.info("Rendered page %d at %d DPI (%d bytes)", page_idx + 1, dpi, len(image_bytes))
        except Exception as e:
            logger.warning("Failed to render page %d: %s", page_idx + 1, e)

    doc.close()

    if not images_b64:
        logger.warning("No pages rendered for vision ordering code extraction")
        return []

    # Build the prompt with optional spool reference
    prompt = _ORDERING_CODE_VISION_PROMPT
    if known_spool_codes:
        codes_str = ", ".join(f'"{c}"' for c in sorted(known_spool_codes))
        prompt += f"""

REFERENCE DATA — KNOWN SPOOL TYPES:
For {company} products, these spool codes are KNOWN to exist: {codes_str}
Ensure ALL of these appear as spool segment options. Also include any ADDITIONAL
spool types visible in the diagram that are NOT in this list."""

    # Build multi-image message content
    content = [{"type": "text", "text": prompt}]
    for img_b64 in images_b64:
        content.append({
            "type": "image_url",
            "image_url": {
                "url": f"data:image/png;base64,{img_b64}",
                "detail": "high",
            },
        })

    try:
        response = _get_client().chat.completions.create(
            model="gpt-4o",
            messages=[{"role": "user", "content": content}],
            response_format={"type": "json_object"},
            max_tokens=4096,
            temperature=0.1,
        )

        raw_content = response.choices[0].message.content
        logger.info("Vision ordering code response (first 2000 chars): %s", raw_content[:2000])
        data = json.loads(raw_content)
        raw_codes = data.get("ordering_codes", [])

        definitions = _parse_ordering_code_response(raw_codes, company, category)
        logger.info("Vision ordering code extraction: %d definitions from %d pages",
                     len(definitions), len(images_b64))
        print(f"[DEBUG] Vision ordering code extraction: {len(definitions)} definitions "
              f"from {len(images_b64)} pages")

        return definitions

    except Exception as e:
        logger.error("Vision ordering code extraction error: %s", e)
        print(f"[DEBUG] Vision ordering code extraction error: {e}")
        return []


# ---------------------------------------------------------------------------
# Step 7: Cross-reference series extraction
# ---------------------------------------------------------------------------


def extract_cross_references_with_llm(
    text: str, source_company: str = "Danfoss"
) -> list[dict]:
    """Extract series-level cross-reference mappings from a cross-reference PDF.

    These documents map Danfoss/Vickers series prefixes to competitor equivalents.
    For example: Danfoss "KFDG4V-3" ↔ Parker "D1FW" (proportional directional valve).

    Returns a list of dicts with keys:
        my_company_series, competitor_series, competitor_company, product_type, notes
    """
    if not text or not text.strip():
        return []

    prompt = f"""You are an expert at reading hydraulic product cross-reference documents.

This document from {source_company} maps their product series to equivalent competitor series.
Each row typically shows a {source_company} model code prefix alongside the competitor's equivalent prefix.

Extract ALL series-level cross-reference mappings from the text below. For each mapping, provide:
- "my_company_series": the {source_company}/Vickers/Danfoss series prefix (e.g. "KFDG4V-3", "DG4V-3", "KDG4V-3")
- "competitor_series": the competitor's equivalent series prefix (e.g. "D1FW", "4WREE", "A10VSO")
- "competitor_company": the competitor manufacturer name (e.g. "Parker", "Bosch Rexroth", "Eaton")
- "product_type": what type of product this is (e.g. "Proportional Directional Valve", "Servo Valve", "Piston Pump")
- "notes": any additional notes about the equivalence (compatibility notes, differences, etc.)

IMPORTANT:
- Extract the SERIES PREFIX only, not full model codes with all options
- A single {source_company} series may map to multiple competitors — create one entry per competitor mapping
- Include ALL mappings found in the document, even if some are partial
- competitor_company should be the full official name (e.g. "Bosch Rexroth" not just "Rexroth")
- If the document uses Vickers model codes, still set my_company_series to the Vickers code (the system knows Vickers is now Danfoss)

Return a JSON object: {{"cross_references": [...]}}
If no cross-references found, return {{"cross_references": []}}

Text:
{text[:40000]}"""

    try:
        response = _get_client().chat.completions.create(
            model=EXTRACTION_MODEL,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
            temperature=0.1,
        )
        content = response.choices[0].message.content
        logger.info("Cross-reference LLM response (first 2000 chars): %s", content[:2000])
        data = json.loads(content)
        refs = data.get("cross_references", [])

        # Validate each entry has the required fields
        valid = []
        for ref in refs:
            if not isinstance(ref, dict):
                continue
            if ref.get("my_company_series") and ref.get("competitor_series") and ref.get("competitor_company"):
                valid.append(ref)
            else:
                logger.warning("Skipping incomplete cross-reference: %s", ref)

        logger.info("Extracted %d cross-reference mappings from %s document",
                     len(valid), source_company)
        return valid

    except Exception as e:
        logger.error("Cross-reference extraction error: %s", e)
        return []
