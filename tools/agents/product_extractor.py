"""
ProductMatchPro - Product Extractor Agent
Extracts hydraulic product specifications from unstructured document text.
Uses TIER_MID (Sonnet / GPT-4.1-mini) for balanced quality and cost.
"""

import json
import logging

from models import ExtractedProduct, UploadMetadata
from tools.llm_client import call_llm_json, TIER_MID
from tools.agents.base import chunk_text

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at extracting hydraulic product specifications from technical documents.
Extract ALL products mentioned in the provided text. Return a JSON object: {"products": [...]}.
If no products found, return {"products": []}.
Only include fields where you found actual data."""

_USER_PROMPT_TEMPLATE = """Extract ALL products from the following text. For each product, extract:
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

Also extract ANY other specification fields found as additional key-value pairs.
Use descriptive snake_case names for any field not in the list above.
Do NOT omit any data — capture every specification mentioned in the text.

Company: {company}
Document type: {document_type}

Text:
{text}"""


# ── Public API ───────────────────────────────────────────────────────────

def extract_products(
    text: str, metadata: UploadMetadata
) -> list[ExtractedProduct]:
    """Extract products from unstructured text using LLM.

    Processes text in overlapping batches to cover the full document.
    Deduplicates results by model_code (keeps the one with more specs).
    """
    if not text or not text.strip():
        return []

    batches = chunk_text(text, chunk_size=10000, overlap=500)
    logger.info("Product extraction: %d batch(es) from %d chars", len(batches), len(text))

    all_products = []
    for batch_idx, batch_text in enumerate(batches):
        batch_products = _extract_batch(batch_text, metadata, batch_idx + 1)
        all_products.extend(batch_products)

    # Deduplicate by model_code (keep the one with more specs)
    seen: dict[str, ExtractedProduct] = {}
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


def _extract_batch(
    text: str, metadata: UploadMetadata, batch_num: int = 1
) -> list[ExtractedProduct]:
    """Extract products from a single text batch."""
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        company=metadata.company,
        document_type=metadata.document_type,
        text=text,
    )

    try:
        data = call_llm_json(
            TIER_MID,
            _SYSTEM_PROMPT,
            user_prompt,
        )
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
        logger.error("Product extraction error (batch %d): %s", batch_num, e)
        return []
