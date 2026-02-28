"""
ProductMatchPro - Image Reader Agent
Reads product labels/nameplates from photos to extract model codes.
Uses TIER_MID (Sonnet / GPT-4o) for vision-based label reading.
"""

import logging
from pathlib import Path

from tools.llm_client import call_llm, TIER_MID
from tools.agents.base import encode_image_file, build_image_block, build_text_block

logger = logging.getLogger(__name__)

MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_SYSTEM_PROMPT = """You are an expert at reading hydraulic product labels and nameplates.
Examine the image and extract the model code, manufacturer, and any visible specifications."""

_VISION_PROMPT = """Examine this image of a hydraulic valve/pump/motor label or nameplate. Extract:
1. The model code / part number (the primary alphanumeric identifier)
2. The manufacturer / brand name if visible
3. Any visible specifications (voltage, pressure, flow rate, etc.)

Return ONLY the model code on the first line, followed by the manufacturer name on the second line (if visible).
If you can see additional specs, add them on subsequent lines in the format "spec: value".

If you cannot read a model code from the image, respond with: UNREADABLE

Examples of model codes: 4WE6D6X/EG24N9K4, D1VW020BN, DHI-0631/2-X 24DC, DSHG-04-3C4-T-A120-31"""


def extract_text_from_image(image_path: str) -> dict:
    """Extract model code text from a photo of a product label.

    Returns dict with keys: text, raw_response, error.
    """
    path = Path(image_path)

    if not path.exists():
        return {"text": "", "raw_response": "", "error": "Image file not found."}

    suffix = path.suffix.lower()
    if suffix not in ALLOWED_TYPES:
        return {"text": "", "raw_response": "",
                "error": f"Unsupported image type: {suffix}. Use JPG, PNG, or WebP."}

    size = path.stat().st_size
    if size > MAX_IMAGE_SIZE_BYTES:
        return {"text": "", "raw_response": "",
                "error": f"Image too large ({size / 1024 / 1024:.1f}MB). Maximum is 10MB."}
    if size == 0:
        return {"text": "", "raw_response": "", "error": "Image file is empty."}

    image_b64, mime_type = encode_image_file(image_path)

    content = [
        build_text_block(_VISION_PROMPT),
        build_image_block(image_b64, mime_type),
    ]

    try:
        raw_text = call_llm(
            TIER_MID,
            _SYSTEM_PROMPT,
            content,
            max_tokens=300,
            vision=True,
        ).strip()

        if "UNREADABLE" in raw_text.upper():
            return {
                "text": "",
                "raw_response": raw_text,
                "error": "Could not read a model code from the image. Please try a clearer photo.",
            }

        lines = [line.strip() for line in raw_text.split("\n") if line.strip()]
        model_code = lines[0] if lines else ""

        query_parts = [model_code]
        if len(lines) > 1 and ":" not in lines[1]:
            query_parts.append(lines[1])

        enriched_text = " ".join(query_parts)

        logger.info("Vision extraction: '%s' from image %s", enriched_text, path.name)
        return {
            "text": enriched_text,
            "raw_response": raw_text,
            "error": None,
        }

    except Exception as e:
        logger.error("Vision API error: %s", e, exc_info=True)
        return {
            "text": "",
            "raw_response": "",
            "error": "Failed to analyze the image. Please try again or type the model code manually.",
        }
