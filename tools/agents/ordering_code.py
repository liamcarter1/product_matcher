"""
ProductMatchPro - Ordering Code Extraction Agent
Extracts ordering code breakdown tables from datasheets (text and vision paths).
Uses TIER_HIGH (Opus / GPT-4.1) — the most critical extraction task.
"""

import logging

from models import OrderingCodeDefinition
from tools.llm_client import call_llm_json, TIER_HIGH
from tools.agents.base import render_pdf_pages, build_image_block, build_text_block
from tools.parse_tools import (
    _select_ordering_code_text,
    _parse_ordering_code_response,
    _VALID_SPEC_FIELDS,
)

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at reading hydraulic product datasheets and extracting ordering code breakdown tables.
Return a JSON object: {"ordering_codes": [...]}
If no ordering code tables found, return {"ordering_codes": []}"""

_USER_PROMPT_TEMPLATE = """You are an expert at reading hydraulic product datasheets and extracting ordering code breakdown tables.

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
     - "maps_to_field": which spec field this sets. Known fields: {valid_fields}. You may also use ANY descriptive snake_case field name for specs not in this list (e.g. "design_number", "flow_class", "interface_standard", "manual_override", "connection_type"). EVERY segment must map to a field — use the segment_name if no known field fits (e.g. "series_code" for a series identifier).
     - "maps_to_value": the value to store in that field (e.g. 4 for max_flow_lpm, "FKM" for seal_material, "24VDC" for coil_voltage). Use the appropriate type (number for numeric fields, string for text fields).
6. "shared_specs": object with any specifications that apply to ALL variants from this table (e.g. {{"max_pressure_bar": 315}}).
   Extract these from the document text around the ordering code table.

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

The system's spool analysis will separately determine center conditions and solenoid functions.

SPOOL OPTION COUNT — IMPORTANT:
For standard directional valve series (DG4V, 4WE, D1VW, D*FW, etc.), the spool type segment
typically has 10-30+ different options. If you find fewer than 5 spool type options for a
directional valve series, you are VERY LIKELY MISSING options. Look more carefully.

SEGMENT NAMING RULES — CRITICAL:
- EVERY segment MUST have a meaningful "segment_name" in snake_case
- EVERY segment MUST have maps_to_field set to either a known spec field OR the segment_name itself
- Do NOT use "" for maps_to_field — every segment maps to something
- Include ALL positions, both fixed and variable — NEVER skip a position number
- maps_to_value for coil_voltage must be a string like "24VDC", not a bare number
- maps_to_value for numeric fields (max_flow_lpm, max_pressure_bar, etc.) must be a number
- Spool type/valve function segments MUST use maps_to_field: "spool_type"
- The total number of segments MUST match the total number of positions shown in the breakdown diagram

{spool_reference_section}

Text:
{text}"""

_VISION_PROMPT_TEMPLATE = """You are an expert at reading hydraulic product datasheets and extracting ordering code breakdown tables from images.

The images show pages from a {company} datasheet. Look for "How to Order" or "Ordering Code" or "Model Code Breakdown" sections.

Extract ALL ordering code breakdown tables visible in the images.

For EACH ordering code table, return:
1. "series": the base series identifier
2. "product_name": what the product is
3. "category": one of: directional_valves, proportional_directional_valves, pressure_valves, flow_valves, pumps, motors, cylinders, filters, accumulators, hoses_fittings, other
4. "code_template": template using {{01}}, {{02}}, etc. with exact separators
5. "segments": array with position, segment_name, is_fixed, separator_before, options[]
6. "shared_specs": common specs for all variants

Each option needs: code, description, maps_to_field (from: {valid_fields} or custom snake_case), maps_to_value.

CRITICAL: Spool type segments map to maps_to_field: "spool_type". Include ALL spool options (typically 10-30+).
EVERY segment MUST have segment_name and maps_to_field. NEVER skip positions.

{spool_reference_section}

Return JSON: {{"ordering_codes": [...]}}"""


# ── Public API ───────────────────────────────────────────────────────────

def extract_ordering_code_text(
    text: str,
    company: str,
    category: str = "",
    known_spool_codes: list[str] | None = None,
) -> list[OrderingCodeDefinition]:
    """Extract ordering code breakdowns from document text.

    Uses TIER_HIGH for maximum accuracy on this critical task.
    """
    spool_ref = ""
    if known_spool_codes:
        codes_str = ", ".join(f'"{c}"' for c in sorted(known_spool_codes))
        spool_ref = (
            f"REFERENCE DATA — KNOWN SPOOL TYPES:\n"
            f"For {company} products in this series family, the following spool type codes are KNOWN "
            f"to exist from previous confirmed extractions: {codes_str}\n"
            f"You MUST ensure ALL of these appear as options in the spool type segment (with is_fixed=false).\n"
            f"Also include any ADDITIONAL spool types you find in the document that are NOT in this list.\n"
        )

    selected_text = _select_ordering_code_text(text)

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        company=company,
        valid_fields=", ".join(sorted(_VALID_SPEC_FIELDS)),
        spool_reference_section=spool_ref,
        text=selected_text,
    )

    try:
        data = call_llm_json(
            TIER_HIGH,
            _SYSTEM_PROMPT,
            user_prompt,
        )
        raw_codes = data.get("ordering_codes", [])
        logger.info("Ordering code extraction: %d raw tables found", len(raw_codes))
        return _parse_ordering_code_response(raw_codes, company, category)
    except Exception as e:
        logger.error("Ordering code text extraction error: %s", e)
        return []


def extract_ordering_code_vision(
    pdf_path: str,
    company: str,
    category: str = "",
    known_spool_codes: list[str] | None = None,
    target_pages: list[int] | None = None,
    dpi: int = 200,
) -> list[OrderingCodeDefinition]:
    """Extract ordering code breakdowns from PDF page images.

    Uses TIER_HIGH with vision for graphics-heavy PDFs.
    """
    if target_pages is None:
        target_pages = list(range(10))  # Default: first 10 pages

    rendered = render_pdf_pages(pdf_path, target_pages, dpi=dpi)
    if not rendered:
        logger.warning("No renderable pages for ordering code vision extraction")
        return []

    spool_ref = ""
    if known_spool_codes:
        codes_str = ", ".join(f'"{c}"' for c in sorted(known_spool_codes))
        spool_ref = (
            f"REFERENCE DATA — KNOWN SPOOL TYPES:\n"
            f"Known spool codes for {company}: {codes_str}\n"
            f"Ensure ALL appear in the spool type segment.\n"
        )

    prompt = _VISION_PROMPT_TEMPLATE.format(
        company=company,
        valid_fields=", ".join(sorted(_VALID_SPEC_FIELDS)),
        spool_reference_section=spool_ref,
    )

    # Build multi-image content
    content = [build_text_block(prompt)]
    for _, img_b64 in rendered:
        content.append(build_image_block(img_b64, "image/png"))

    try:
        data = call_llm_json(
            TIER_HIGH,
            _SYSTEM_PROMPT,
            content,
            vision=True,
            max_tokens=8192,
        )
        raw_codes = data.get("ordering_codes", [])
        logger.info("Ordering code vision extraction: %d raw tables found", len(raw_codes))
        return _parse_ordering_code_response(raw_codes, company, category)
    except Exception as e:
        logger.error("Ordering code vision extraction error: %s", e)
        return []
