"""
ProductMatchPro - Spool Analyzer Agent
Extracts spool type definitions from text and vision (ISO 1219 diagrams).
Text analysis uses TIER_MID (Sonnet), vision uses TIER_HIGH (Opus).
"""

import logging

from tools.llm_client import call_llm_json, TIER_MID, TIER_HIGH
from tools.agents.base import render_pdf_pages, build_image_block, build_text_block
from tools.parse_tools import compute_canonical_pattern, _deduplicate_spools

logger = logging.getLogger(__name__)


# ── Text analysis ────────────────────────────────────────────────────────

_TEXT_SYSTEM_PROMPT = """You are an expert hydraulic engineer analysing a product user guide.
Identify EVERY spool type / valve function code mentioned in the document.
Return a JSON array of spool objects. If no spool types found, return []."""

_TEXT_USER_PROMPT = """Your task: identify EVERY spool type / valve function code mentioned in this document from {company}.

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
  }}
]

If NO spool types are found in the document, return an empty array: []

DOCUMENT TEXT:
{text}"""


def analyze_spool_text(text: str, company: str) -> list[dict]:
    """Perform deep LLM analysis of spool/valve function symbols from text.

    Returns a list of dicts, each with:
        spool_code, center_condition, solenoid_a_function, solenoid_b_function,
        description, canonical_pattern
    """
    max_chars = 80_000
    if len(text) > max_chars:
        text = text[:max_chars]

    user_prompt = _TEXT_USER_PROMPT.format(company=company, text=text)

    try:
        data = call_llm_json(
            TIER_MID,
            _TEXT_SYSTEM_PROMPT,
            user_prompt,
        )

        # Handle both {"spool_functions": [...]} and direct [...]
        if isinstance(data, dict):
            for v in data.values():
                if isinstance(v, list):
                    data = v
                    break
            else:
                data = []

        if not isinstance(data, list):
            logger.warning("Spool analysis returned non-list: %s", type(data))
            return []

        # Enrich each result with canonical pattern
        results = []
        for item in data:
            if not isinstance(item, dict) or not item.get("spool_code"):
                continue
            item["canonical_pattern"] = compute_canonical_pattern(
                item.get("center_condition", ""),
                item.get("solenoid_a_function", ""),
                item.get("solenoid_b_function", ""),
            )
            results.append(item)

        logger.info("Spool text analysis found %d spool types for %s", len(results), company)
        return results

    except Exception as e:
        logger.error("Spool text analysis failed: %s", e)
        return []


# ── Vision analysis ──────────────────────────────────────────────────────

_VISION_SYSTEM_PROMPT = """You are an expert hydraulic engineer analysing pages from a product datasheet.
Extract ALL spool symbol diagrams and spool option table rows.
Return valid JSON: {"spool_symbols": [...]}
If NO spool symbols found, return: {"spool_symbols": []}"""

_VISION_PROMPT_V2 = """You are an expert hydraulic engineer analysing MULTIPLE PAGES from a product datasheet.

These pages may contain HYDRAULIC SPOOL SYMBOL DIAGRAMS and/or SPOOL OPTIONS TABLES.

## SPOOL SYMBOL DIAGRAMS (ISO 1219)
These are standardised schematic diagrams showing valve spool positions:
- A rectangle divided into 2-3 sections (one per switching position: left, center, right)
- Arrows showing flow paths between ports: P (pressure supply), T (tank return), A and B (work ports)
- Blocked ports shown as T-symbols or dead-end lines
- Each symbol has a letter/number CODE label (e.g. "D", "E", "H", "2A", "01", "6C")

IMPORTANT: The symbols can be SMALL. Look very carefully for tiny arrows and port labels.
The CENTER section of a 3-position symbol shows the NEUTRAL/CENTER CONDITION — this is
the most important part for cross-referencing across manufacturers.

## SPOOL OPTIONS TABLE
Some pages contain a TABLE listing all available spool types with columns like:
- Spool code / type designation
- Center condition description
- Active (solenoid energised) function
- Crossover function during transition

If you see a spool options table, extract ALL rows from it — do not skip any.

## EXPECTED COUNT
Directional valve series typically have **10-30+ spool type options**. If you find fewer
than 5 options across all pages shown, look more carefully at small diagrams and table rows.

{previously_found}
{known_spool_codes_section}
## Your task:
Identify ALL spool symbols AND/OR table rows across ALL pages shown. For each one extract:

1. "spool_code": the letter/number designation (e.g. "D", "2A", "H", "01", "EA", "6C", "0C")
2. "center_condition": what happens in the center/neutral position
3. "solenoid_a_function": flow path when left/solenoid-A is energised (e.g. "P→A, B→T")
4. "solenoid_b_function": flow path when right/solenoid-B is energised (e.g. "P→B, A→T")
5. "description": brief summary of the spool function
6. "symbol_description": describe the visual symbol pattern in detail

For 2-position valves, set center_condition to "N/A (2-position valve)" and only fill solenoid_a_function.

Return valid JSON: {{"spool_symbols": [...]}}
If NO spool symbols or spool tables are found on ANY page, return: {{"spool_symbols": []}}

IMPORTANT: Focus on the VISUAL flow path patterns. Two manufacturers may use different
letter codes for the exact same hydraulic function."""


def analyze_spool_vision(
    pdf_path: str,
    company: str,
    page_classifications: dict[int, str] | None = None,
    dpi: int = 250,
    max_tokens_per_batch: int = 4096,
    batch_size: int = 4,
    known_spool_codes: list[str] | None = None,
    retry_on_low_count: bool = True,
    min_expected_spools: int = 5,
) -> list[dict]:
    """Extract spool symbols from PDF using batched multi-page vision.

    Uses TIER_HIGH (Opus) for best vision accuracy on ISO 1219 diagrams.
    """
    # --- Page selection and ordering ----------------------------------------
    if page_classifications:
        spool_pages = sorted(i for i, c in page_classifications.items()
                             if c == "SPOOL_CONTENT")
        maybe_pages = sorted(i for i, c in page_classifications.items()
                             if c == "MAYBE_SPOOL")
        selected_pages = spool_pages + maybe_pages
    else:
        # Determine total pages
        try:
            import fitz
            doc = fitz.open(pdf_path)
            total = len(doc)
            doc.close()
            selected_pages = list(range(total))
        except Exception:
            selected_pages = list(range(20))  # Fallback guess

    if not selected_pages:
        selected_pages = list(range(20))

    logger.info(
        "Spool vision: %d pages selected at %d DPI, batch_size=%d",
        len(selected_pages), dpi, batch_size,
    )

    rendered = render_pdf_pages(pdf_path, selected_pages, dpi=dpi)
    if not rendered:
        logger.warning("No renderable pages found for spool extraction")
        return []

    # --- Batch pages and process with cumulative context --------------------
    all_spools: list[dict] = []
    previously_found: dict[str, dict] = {}
    total_batches = (len(rendered) + batch_size - 1) // batch_size

    for batch_idx in range(total_batches):
        batch_start = batch_idx * batch_size
        batch = rendered[batch_start:batch_start + batch_size]
        batch_page_indices = [p[0] for p in batch]

        # Build cumulative context
        if previously_found:
            prev_lines = []
            for code, data in sorted(previously_found.items()):
                desc = data.get("center_condition", "") or data.get("description", "")
                prev_lines.append(f"  - {code}: {desc}")
            prev_text = (
                "## PREVIOUSLY FOUND SPOOL TYPES\n"
                "Earlier pages already yielded these spool codes:\n"
                + "\n".join(prev_lines) + "\n"
                "Look for ADDITIONAL spool types not in this list.\n"
            )
        else:
            prev_text = ""

        # Build known codes section
        known_section = ""
        if known_spool_codes:
            known_section = (
                "## KNOWN SPOOL CODES FOR THIS MANUFACTURER\n"
                "Database contains: " + ", ".join(sorted(set(known_spool_codes))) + "\n"
                "Ensure you extract ALL codes on these pages.\n"
            )

        prompt = _VISION_PROMPT_V2.format(
            previously_found=prev_text,
            known_spool_codes_section=known_section,
        )

        # Build multi-image content
        content = [build_text_block(prompt)]
        for _, img_b64 in batch:
            content.append(build_image_block(img_b64, "image/png"))

        logger.info(
            "Spool vision batch %d/%d: pages %s, %d images",
            batch_idx + 1, total_batches, batch_page_indices, len(batch),
        )

        try:
            data = call_llm_json(
                TIER_HIGH,
                _VISION_SYSTEM_PROMPT,
                content,
                vision=True,
                max_tokens=max_tokens_per_batch,
            )
            symbols = data.get("spool_symbols", []) if isinstance(data, dict) else []

            batch_codes = []
            for sym in symbols:
                if not isinstance(sym, dict) or not sym.get("spool_code"):
                    continue
                sym["source_page"] = batch_page_indices[0] + 1
                sym["company"] = company
                sym["extraction_method"] = "vision_v2"
                sym["canonical_pattern"] = compute_canonical_pattern(
                    sym.get("center_condition", ""),
                    sym.get("solenoid_a_function", ""),
                    sym.get("solenoid_b_function", ""),
                )
                all_spools.append(sym)
                code_upper = sym["spool_code"].strip().upper()
                previously_found[code_upper] = sym
                batch_codes.append(sym["spool_code"])

            logger.info(
                "Spool vision batch %d: %d spools found: %s",
                batch_idx + 1, len(batch_codes), batch_codes,
            )

        except Exception as e:
            logger.warning(
                "Spool vision batch %d failed (pages %s): %s",
                batch_idx + 1, batch_page_indices, e,
            )

    # --- Deduplicate --------------------------------------------------------
    unique = _deduplicate_spools(all_spools)

    logger.info(
        "Spool vision total: %d unique spools for %s (from %d raw)",
        len(unique), company, len(all_spools),
    )

    # --- Retry with higher DPI if too few found ----------------------------
    if retry_on_low_count and len(unique) < min_expected_spools:
        retry_dpi = max(dpi + 50, 300)
        logger.warning(
            "Only %d spools found (expected >= %d). Retrying with DPI=%d, all pages",
            len(unique), min_expected_spools, retry_dpi,
        )
        retry_result = analyze_spool_vision(
            pdf_path, company,
            page_classifications=None,  # All pages
            dpi=retry_dpi,
            max_tokens_per_batch=max(max_tokens_per_batch + 2048, 6144),
            batch_size=batch_size,
            known_spool_codes=known_spool_codes,
            retry_on_low_count=False,  # No recursive retry
            min_expected_spools=min_expected_spools,
        )
        # Merge: keep best results
        combined = {s["spool_code"].strip().upper(): s for s in unique}
        for s in retry_result:
            code = s["spool_code"].strip().upper()
            if code not in combined:
                combined[code] = s
        unique = list(combined.values())

    return unique
