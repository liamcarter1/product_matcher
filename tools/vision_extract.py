"""
ProductMatchPro - Vision-based Model Code Extraction
Uses GPT-4o vision to extract competitor model codes from photos of product labels/nameplates.
"""

import base64
import logging
from pathlib import Path

from openai import OpenAI
from dotenv import load_dotenv

load_dotenv(override=True)

logger = logging.getLogger(__name__)

VISION_MODEL = "gpt-4o"
MAX_IMAGE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB

ALLOWED_TYPES = {
    ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg",
    ".png": "image/png",
    ".webp": "image/webp",
}

_client = None


def _get_client() -> OpenAI:
    global _client
    if _client is None:
        _client = OpenAI()
    return _client


VISION_PROMPT = """You are an expert at reading hydraulic product labels and nameplates.

Examine this image of a hydraulic valve/pump/motor label or nameplate. Extract:
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
        return {"text": "", "raw_response": "", "error": f"Unsupported image type: {suffix}. Use JPG, PNG, or WebP."}

    size = path.stat().st_size
    if size > MAX_IMAGE_SIZE_BYTES:
        return {"text": "", "raw_response": "", "error": f"Image too large ({size / 1024 / 1024:.1f}MB). Maximum is 10MB."}
    if size == 0:
        return {"text": "", "raw_response": "", "error": "Image file is empty."}

    with open(image_path, "rb") as f:
        image_data = base64.b64encode(f.read()).decode("utf-8")

    mime_type = ALLOWED_TYPES[suffix]

    try:
        response = _get_client().chat.completions.create(
            model=VISION_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": VISION_PROMPT},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:{mime_type};base64,{image_data}",
                                "detail": "high",
                            },
                        },
                    ],
                }
            ],
            max_tokens=300,
            temperature=0.1,
        )

        raw_text = response.choices[0].message.content.strip()

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
