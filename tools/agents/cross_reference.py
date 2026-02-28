"""
ProductMatchPro - Cross-Reference Extraction Agent
Extracts series-level cross-reference mappings from cross-reference PDFs.
Uses TIER_MID (Sonnet / GPT-4.1-mini) for structured table extraction.
"""

import logging

from tools.llm_client import call_llm_json, TIER_MID

logger = logging.getLogger(__name__)

_SYSTEM_PROMPT = """You are an expert at reading hydraulic product cross-reference documents.
Return a JSON object: {"cross_references": [...]}
If no cross-references found, return {"cross_references": []}"""

_USER_PROMPT_TEMPLATE = """This document from {source_company} maps their product series to equivalent competitor series.
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
- If the document uses Vickers model codes, still set my_company_series to the Vickers code

Text:
{text}"""


def extract_cross_references(
    text: str, source_company: str = "Danfoss"
) -> list[dict]:
    """Extract series-level cross-reference mappings from text.

    Returns a list of dicts with keys:
        my_company_series, competitor_series, competitor_company, product_type, notes
    """
    if not text or not text.strip():
        return []

    user_prompt = _USER_PROMPT_TEMPLATE.format(
        source_company=source_company,
        text=text[:40000],
    )

    try:
        data = call_llm_json(
            TIER_MID,
            _SYSTEM_PROMPT,
            user_prompt,
        )
        refs = data.get("cross_references", [])

        # Validate each entry
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
