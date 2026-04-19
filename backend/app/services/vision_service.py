"""
Vision service for extracting data from competitor nameplate images.
Uses GPT-4o vision to OCR and parse nameplate information.
"""

import base64
import json
import logging
from io import BytesIO
from typing import Optional

from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

from ..config import get_settings
from ..models.schemas import NameplateData
from .skill_loader import get_vision_context

logger = logging.getLogger(__name__)

_VISION_DOMAIN_CTX = get_vision_context()

NAMEPLATE_EXTRACTION_PROMPT = """You are an expert at reading industrial product nameplates and labels.
Analyze this image of a product nameplate and extract all visible information.

Return a JSON object with exactly these fields:
{
  "model_number": "the model or part number (null if not found)",
  "manufacturer": "the brand or manufacturer name (null if not found)",
  "specifications": {"key": "value"} dictionary of specs like voltage, current, power, dimensions, frequency, etc.,
  "raw_text": "all text visible on the nameplate, transcribed as-is"
}

Rules:
- Extract the model/part number as precisely as possible, preserving dashes and formatting
- Include ALL visible specifications as key-value pairs
- For raw_text, include everything you can read, line by line
- If a field is not visible or readable, use null for strings or {} for specifications
- Return ONLY valid JSON, no markdown formatting or explanation"""

# Augmented prompt used when the image is identified as a hydraulic valve or spool diagram
VALVE_EXTRACTION_PROMPT = NAMEPLATE_EXTRACTION_PROMPT + """

If this image shows a hydraulic directional valve nameplate, ordering code, or spool symbol diagram,
also include these fields in specifications:
- "spool_type": the manufacturer's spool code (e.g. "E", "2C")
- "spool_function": the hydraulic centre condition (e.g. "all_ports_blocked", "open_centre")
- "valve_size": the ISO size number if visible
- "voltage": the coil voltage if visible
- "voltage_type": "DC" or "AC"

""" + (_VISION_DOMAIN_CTX if _VISION_DOMAIN_CTX else "")


class VisionService:
    """Service for extracting nameplate data from images using GPT-4o vision."""

    def __init__(self):
        settings = get_settings()
        self.llm = ChatOpenAI(
            model=settings.openai_model,
            api_key=settings.openai_api_key,
            temperature=0.1,
            max_tokens=1000,
        )

    async def extract_nameplate_data(
        self, image_bytes: bytes, mime_type: str,
        is_valve_image: bool = False
    ) -> NameplateData:
        """
        Extract nameplate data from an image using GPT-4o vision.

        Args:
            image_bytes: Raw image bytes
            mime_type: MIME type of the image (e.g., image/jpeg)
            is_valve_image: If True, uses the valve-specific prompt with
                           spool/hydraulic domain knowledge (~400 extra tokens)

        Returns:
            NameplateData with extracted information and confidence score
        """
        prompt = VALVE_EXTRACTION_PROMPT if is_valve_image else NAMEPLATE_EXTRACTION_PROMPT
        image_b64 = base64.b64encode(image_bytes).decode("utf-8")
        data_url = f"data:{mime_type};base64,{image_b64}"

        message = HumanMessage(
            content=[
                {"type": "text", "text": prompt},
                {"type": "image_url", "image_url": {"url": data_url}},
            ]
        )

        try:
            response = await self.llm.ainvoke([message])
            parsed = self._parse_response(response.content)
            return parsed
        except Exception as e:
            logger.error(f"Vision extraction failed: {e}")
            raise

    def _parse_response(self, content: str) -> NameplateData:
        """Parse the LLM JSON response into NameplateData."""
        # Strip markdown code fences if present
        text = content.strip()
        if text.startswith("```"):
            lines = text.split("\n")
            # Remove first and last lines (``` markers)
            lines = [l for l in lines if not l.strip().startswith("```")]
            text = "\n".join(lines)

        data = json.loads(text)

        model_number = data.get("model_number")
        manufacturer = data.get("manufacturer")
        specifications = data.get("specifications") or {}
        raw_text = data.get("raw_text", "")

        # Calculate confidence based on what was found
        confidence = self._calculate_confidence(model_number, manufacturer, raw_text)

        return NameplateData(
            model_number=model_number,
            manufacturer=manufacturer,
            specifications=specifications,
            raw_text=raw_text,
            confidence=confidence,
        )

    def _calculate_confidence(
        self,
        model_number: Optional[str],
        manufacturer: Optional[str],
        raw_text: str,
    ) -> float:
        """Calculate confidence score based on extracted fields."""
        score = 0.0

        if raw_text and len(raw_text) > 5:
            score += 30.0

        if model_number:
            score += 40.0

        if manufacturer:
            score += 30.0

        return min(score, 100.0)
