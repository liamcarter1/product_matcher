"""
ProductMatchPro - PDF Parsing Tools
Extracts product data from hydraulic catalogues and user guides.
Uses pdfplumber for tables and pypdf + LLM for unstructured text.
"""

import re
import uuid
from pathlib import Path
from typing import Optional
import pdfplumber
from pypdf import PdfReader
from openai import OpenAI
from dotenv import load_dotenv
import json

from models import ExtractedProduct, HydraulicProduct, UploadMetadata

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
    # Pressure
    "pressure": "max_pressure_bar", "max pressure": "max_pressure_bar",
    "max. pressure": "max_pressure_bar", "rated pressure": "max_pressure_bar",
    "pressure (bar)": "max_pressure_bar", "pressure bar": "max_pressure_bar",
    "operating pressure": "max_pressure_bar",
    # Flow
    "flow": "max_flow_lpm", "max flow": "max_flow_lpm", "max. flow": "max_flow_lpm",
    "rated flow": "max_flow_lpm", "flow (l/min)": "max_flow_lpm",
    "flow l/min": "max_flow_lpm", "flow lpm": "max_flow_lpm",
    "nominal flow": "max_flow_lpm",
    # Port size
    "port": "port_size", "port size": "port_size", "connection": "port_size",
    "connection size": "port_size", "ports": "port_size",
    # Voltage
    "voltage": "coil_voltage", "coil voltage": "coil_voltage",
    "coil": "coil_voltage", "supply voltage": "coil_voltage",
    "solenoid voltage": "coil_voltage",
    # Mounting
    "mounting": "mounting", "mounting type": "mounting",
    "mounting pattern": "mounting_pattern", "interface": "mounting_pattern",
    # Valve size
    "size": "valve_size", "valve size": "valve_size", "cetop": "valve_size",
    "ng": "valve_size",
    # Weight
    "weight": "weight_kg", "weight (kg)": "weight_kg", "mass": "weight_kg",
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
    """Extract text from each page of a PDF using pypdf.
    Returns list of {page: int, text: str}."""
    reader = PdfReader(pdf_path)
    pages = []
    for i, page in enumerate(reader.pages):
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
        )
        products.append(product)
    return products


def extract_products_with_llm(
    text: str, metadata: UploadMetadata
) -> list[ExtractedProduct]:
    """Use GPT-4o-mini to extract products from unstructured text."""
    prompt = f"""You are an expert at extracting hydraulic product specifications from technical documents.

Extract ALL products mentioned in the following text. For each product, extract:
- model_code: the full model/part number
- product_name: what the product is (e.g. "Directional Control Valve")
- category: one of: directional_valves, pressure_valves, flow_valves, pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
- max_pressure_bar: maximum operating pressure in bar (number only)
- max_flow_lpm: maximum flow rate in litres per minute (number only)
- valve_size: e.g. "CETOP 3", "CETOP 5", "NG6", "NG10"
- spool_type: spool/function designation
- actuator_type: solenoid, manual, pilot, proportional
- coil_voltage: e.g. "12VDC", "24VDC", "110VAC", "220VAC"
- port_size: e.g. "G3/8", "SAE-10"
- port_type: BSP, SAE, metric, NPTF
- mounting: e.g. "subplate", "inline", "manifold"
- mounting_pattern: e.g. "ISO 4401-03", "NFPA D03"
- body_material: e.g. "cast_iron", "steel", "aluminium"
- seal_material: e.g. "NBR", "FKM", "Viton"
- operating_temp_min_c: minimum temperature in Celsius
- operating_temp_max_c: maximum temperature in Celsius
- weight_kg: weight in kg

Company: {metadata.company}
Document type: {metadata.document_type}

Return a JSON array of objects. Only include fields where you found actual data. If no products found, return [].

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
            ))
        return products
    except Exception as e:
        print(f"LLM extraction error: {e}")
        return []


def extract_model_code_patterns_with_llm(
    text: str, company: str
) -> list[dict]:
    """Use GPT-4o-mini to extract model code breakdown patterns from user guide pages.
    These are the 'How to Order' or 'Model Code Breakdown' sections."""

    prompt = f"""You are an expert at reading hydraulic product user guides and extracting model code breakdown information.

Many hydraulic manufacturers include a section showing what each part of a model code means. For example:
  DG4V - 3 - 2A - M - U - H7 - 60
  Where: DG4V = Series, 3 = Valve size, 2A = Spool function, M = Solenoid actuation, H7 = 24VDC coil, 60 = Design

Extract ALL model code segment definitions from the following text for company: {company}

For each segment definition, provide:
- series: the base series name (e.g. "DG4V", "4WE6", "D1VW")
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
    """Try to parse a string as a number, return original if not possible."""
    cleaned = re.sub(r"[^\d.\-]", "", value)
    try:
        if "." in cleaned:
            return float(cleaned)
        return int(cleaned)
    except (ValueError, TypeError):
        return value
