"""
ProductMatchPro - Shared Agent Utilities
Image encoding, text chunking, and content block builders used across all agents.
"""

import base64
import logging
from pathlib import Path

from tools.llm_client import get_provider

logger = logging.getLogger(__name__)

# PyMuPDF for PDF page rendering (optional)
try:
    import fitz as _fitz
    HAS_PYMUPDF = True
except ImportError:
    HAS_PYMUPDF = False


# ── Text chunking ────────────────────────────────────────────────────────

def chunk_text(
    text: str,
    chunk_size: int = 10000,
    overlap: int = 500,
) -> list[str]:
    """Split text into overlapping chunks.

    Replaces langchain-text-splitters dependency with a simple implementation.
    """
    if not text:
        return []

    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end - overlap
        if start >= len(text):
            break
    return chunks


# ── Image encoding ───────────────────────────────────────────────────────

def encode_image_file(image_path: str) -> tuple[str, str]:
    """Read an image file and return (base64_data, mime_type)."""
    path = Path(image_path)
    suffix = path.suffix.lower()
    mime_map = {
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
        ".png": "image/png",
        ".webp": "image/webp",
        ".gif": "image/gif",
    }
    mime_type = mime_map.get(suffix, "image/png")

    with open(image_path, "rb") as f:
        image_b64 = base64.b64encode(f.read()).decode("utf-8")
    return image_b64, mime_type


def render_pdf_pages(
    pdf_path: str,
    page_indices: list[int],
    dpi: int = 200,
    min_bytes: int = 5000,
) -> list[tuple[int, str]]:
    """Render selected PDF pages to base64 PNG images.

    Args:
        pdf_path: Path to the PDF file
        page_indices: List of 0-based page indices to render
        dpi: Resolution for rendering
        min_bytes: Skip pages whose rendered image is smaller than this

    Returns:
        List of (page_idx, base64_png_string) tuples
    """
    if not HAS_PYMUPDF:
        logger.error("PyMuPDF (fitz) not available for PDF rendering")
        return []

    rendered: list[tuple[int, str]] = []
    doc = _fitz.open(pdf_path)
    try:
        for page_idx in page_indices:
            if page_idx < 0 or page_idx >= len(doc):
                continue
            try:
                page = doc[page_idx]
                pix = page.get_pixmap(dpi=dpi)
                image_bytes = pix.tobytes("png")
                if len(image_bytes) < min_bytes:
                    continue
                image_b64 = base64.b64encode(image_bytes).decode("utf-8")
                rendered.append((page_idx, image_b64))
            except Exception as e:
                logger.warning("Failed to render page %d at %d DPI: %s", page_idx, dpi, e)
    finally:
        doc.close()

    return rendered


# ── Content block builders (provider-agnostic) ───────────────────────────

def build_image_block(image_b64: str, mime_type: str = "image/png") -> dict:
    """Build a provider-appropriate image content block.

    For Anthropic: {"type": "image", "source": {"type": "base64", ...}}
    For OpenAI:    {"type": "image_url", "image_url": {"url": "data:...", "detail": "high"}}
    """
    if get_provider() == "anthropic":
        return {
            "type": "image",
            "source": {
                "type": "base64",
                "media_type": mime_type,
                "data": image_b64,
            },
        }
    else:
        return {
            "type": "image_url",
            "image_url": {
                "url": f"data:{mime_type};base64,{image_b64}",
                "detail": "high",
            },
        }


def build_text_block(text: str) -> dict:
    """Build a text content block (same for both providers)."""
    return {"type": "text", "text": text}


def build_vision_content(
    prompt_text: str,
    images_b64: list[str],
    mime_type: str = "image/png",
) -> list[dict]:
    """Build a multi-image vision content block list.

    Returns a list of content blocks: [text_block, image1, image2, ...]
    suitable for passing as user_content to call_llm / call_llm_json.
    """
    blocks = [build_text_block(prompt_text)]
    for img_b64 in images_b64:
        blocks.append(build_image_block(img_b64, mime_type))
    return blocks
