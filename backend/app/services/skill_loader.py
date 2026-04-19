"""
Loads the hydraulics domain knowledge skill file once at import,
split into sections so callers can inject only what they need.
"""

import logging
from functools import lru_cache
from pathlib import Path
from typing import Dict

logger = logging.getLogger(__name__)

# Resolve path relative to project root (3 levels up from this file)
_SKILL_PATH = Path(__file__).resolve().parents[3] / "skills" / "hydraulics_engineer.md"


def _parse_sections(text: str) -> Dict[str, str]:
    """Split the skill file on '## N.' headers into named sections."""
    sections = {}
    current_key = None
    current_lines = []

    for line in text.splitlines():
        if line.startswith("## ") and line[3:4].isdigit():
            if current_key:
                sections[current_key] = "\n".join(current_lines).strip()
            # e.g. "## 2. Spool Type Identification" -> "spool_type"
            header = line.split(".", 1)[1].strip() if "." in line else line[3:].strip()
            current_key = header.lower().replace(" ", "_")
            current_lines = [line]
        elif current_key:
            current_lines.append(line)

    if current_key:
        sections[current_key] = "\n".join(current_lines).strip()

    return sections


@lru_cache(maxsize=1)
def _load() -> Dict[str, str]:
    """Load and cache the skill file sections. Returns empty dict on failure."""
    if not _SKILL_PATH.exists():
        logger.warning("Hydraulics skill file not found at %s", _SKILL_PATH)
        return {}
    try:
        text = _SKILL_PATH.read_text(encoding="utf-8")
        sections = _parse_sections(text)
        logger.info("Loaded hydraulics skill: %d sections, %d chars total",
                     len(sections), len(text))
        return sections
    except Exception as e:
        logger.error("Failed to load hydraulics skill: %s", e)
        return {}


def get_sections(*keys: str) -> str:
    """
    Return the concatenated text of requested sections.

    Args:
        *keys: Section key fragments to match (e.g. "spool", "unit", "failure").
               A key matches if it appears anywhere in the section name.

    Returns:
        Combined section text, or empty string if nothing matched.
    """
    sections = _load()
    if not sections:
        return ""

    matched = []
    for section_key, section_text in sections.items():
        for k in keys:
            if k in section_key:
                matched.append(section_text)
                break

    return "\n\n".join(matched)


def get_part_lookup_context() -> str:
    """Sections relevant to cross-referencing competitor parts to Danfoss."""
    return get_sections("spool", "unit", "failure")


def get_spec_query_context() -> str:
    """Sections relevant to specification queries."""
    return get_sections("spec", "unit")


def get_vision_context() -> str:
    """Minimal section for vision-based spool/nameplate extraction."""
    return get_sections("spool")
