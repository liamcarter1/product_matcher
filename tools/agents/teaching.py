"""
ProductMatchPro - Teaching Mode Service
Manages extraction examples (few-shot) and builds prompt injection sections.
"""

import json
import logging
from typing import Optional

from tools.agents.base import load_teaching_image, build_image_block, build_text_block

logger = logging.getLogger(__name__)

# Page types that map to agent modules
PAGE_TYPES = [
    "ordering_code_table",
    "spool_diagram",
    "spool_table",
    "spec_table",
]


# ── Example selection ────────────────────────────────────────────────────

def get_relevant_examples(
    db,
    manufacturer: str,
    page_type: str,
    series_prefix: str = "",
    max_examples: int = 2,
) -> list[dict]:
    """Select the most relevant teaching examples for prompt injection.

    Priority:
    1. Same manufacturer + same series + same page_type  (best)
    2. Same manufacturer + same page_type                (good)
    Never injects cross-manufacturer examples.

    Returns at most max_examples dicts, each with:
        id, manufacturer, series_prefix, page_type,
        image_path, annotation, correct_output
    """
    if not manufacturer or not page_type:
        return []

    all_examples = db.get_extraction_examples(
        manufacturer=manufacturer,
        page_type=page_type,
        active_only=True,
    )

    if not all_examples:
        return []

    # Score each example by relevance
    scored = []
    for ex in all_examples:
        score = 2  # base: same manufacturer + same page_type
        if (
            series_prefix
            and ex.get("series_prefix", "").upper() == series_prefix.upper()
        ):
            score = 3  # same series — strongest match
        scored.append((score, ex))

    # Sort by relevance score desc, then times_used desc
    scored.sort(key=lambda t: (t[0], t[1].get("times_used", 0)), reverse=True)

    selected = [ex for _, ex in scored[:max_examples]]

    # Track usage
    for ex in selected:
        try:
            db.increment_example_usage(ex["id"])
        except Exception:
            pass

    if selected:
        logger.info(
            "Selected %d teaching example(s) for %s/%s (page_type=%s)",
            len(selected), manufacturer, series_prefix or "*", page_type,
        )

    return selected


# ── Text prompt injection ────────────────────────────────────────────────

def build_few_shot_section(
    examples: list[dict],
    page_type: str = "",
    max_output_chars: int = 3000,
) -> str:
    """Build a text few-shot section for insertion into agent prompts.

    Returns an empty string if no examples, so callers can just concatenate.
    """
    if not examples:
        return ""

    parts = [
        "\n## REFERENCE EXAMPLES FROM PREVIOUS CORRECT EXTRACTIONS\n",
        "The following example(s) show what a correct extraction looks like "
        "for this type of page. Use them as a guide for format, detail level, "
        "and what to look for.\n",
    ]

    for i, ex in enumerate(examples, 1):
        mfr = ex.get("manufacturer", "")
        series = ex.get("series_prefix", "")
        label = f"{mfr} - {series}" if series else mfr

        parts.append(f"### Example {i} ({label})")
        annotation = ex.get("annotation", "")
        if annotation:
            parts.append(f"**Admin note:** {annotation}")

        output = ex.get("correct_output", "{}")
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n... (truncated)"
        parts.append(f"Correct extraction output:\n```json\n{output}\n```\n")

    parts.append(
        "Use these examples as reference for the format, level of detail, "
        "and what to look for. Now extract from the NEW document below.\n"
        "---\n"
    )
    return "\n".join(parts)


# ── Vision prompt injection ──────────────────────────────────────────────

def build_few_shot_vision_content(
    examples: list[dict],
    page_type: str = "",
    max_image_examples: int = 1,
    max_output_chars: int = 3000,
) -> list[dict]:
    """Build vision content blocks with example images for prompt injection.

    Returns content blocks to prepend to the actual document images.
    Limited to max_image_examples images to stay within token budget
    (~3000-5000 tokens per 200 DPI page image).
    """
    if not examples:
        return []

    blocks: list[dict] = []
    blocks.append(build_text_block(
        "## REFERENCE EXAMPLES\n"
        "The following show CORRECT extractions from similar pages. "
        "Use them as reference for what to look for and how to "
        "structure your output.\n"
    ))

    images_included = 0
    for i, ex in enumerate(examples, 1):
        mfr = ex.get("manufacturer", "")
        series = ex.get("series_prefix", "")
        label = f"{mfr} - {series}" if series else mfr

        annotation = ex.get("annotation", "")
        output = ex.get("correct_output", "{}")
        if len(output) > max_output_chars:
            output = output[:max_output_chars] + "\n... (truncated)"

        header = f"### Reference Example {i} ({label})\n"
        if annotation:
            header += f"Admin note: {annotation}\n"
        header += f"Correct output for this page:\n```json\n{output}\n```"
        blocks.append(build_text_block(header))

        # Include the image only up to the limit
        if images_included < max_image_examples:
            image_path = ex.get("image_path", "")
            if image_path:
                image_b64 = load_teaching_image(image_path)
                if image_b64:
                    blocks.append(build_image_block(image_b64, "image/png"))
                    images_included += 1

    blocks.append(build_text_block(
        "\n---\nNow extract from the NEW pages below. "
        "Follow the same format and level of detail as the examples above.\n"
    ))
    return blocks
