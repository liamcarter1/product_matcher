"""
ProductMatchPro - Spec Extractor Agent
Extracts model code breakdown patterns ("How to Order" sections) from datasheets.
Uses TIER_MID (Sonnet / GPT-4.1-mini) for structured extraction.
"""

import logging

from tools.llm_client import call_llm_json, TIER_MID

logger = logging.getLogger(__name__)

# ── System prompt ────────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are an expert at reading hydraulic product user guides and extracting model code breakdown information.
Return a JSON object with key "patterns" containing an array of pattern objects.
If no model code breakdowns are found, return {"patterns": []}."""

_USER_PROMPT_TEMPLATE = """Many hydraulic manufacturers include a section showing what each part of a model code means. For example:
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

Text:
{text}"""


# ── Public API ───────────────────────────────────────────────────────────

def extract_model_code_patterns(
    text: str, company: str
) -> list[dict]:
    """Extract model code breakdown patterns from user guide / datasheet text.

    Returns a list of dicts with keys: series, segment_position, segment_name,
    code_value, decoded_value, maps_to_field.
    """
    user_prompt = _USER_PROMPT_TEMPLATE.format(
        company=company,
        text=text[:8000],
    )

    try:
        data = call_llm_json(
            TIER_MID,
            _SYSTEM_PROMPT,
            user_prompt,
        )
        patterns = data if isinstance(data, list) else data.get("patterns", [])
        return patterns
    except Exception as e:
        logger.error("Model code pattern extraction error: %s", e)
        return []
