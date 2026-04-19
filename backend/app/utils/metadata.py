"""
Metadata extraction and normalization utilities.
"""

import re
from typing import Dict, Any, Optional


def normalize_part_number(part_number: str) -> str:
    """
    Normalize a part number for comparison and matching.

    Removes common separators (dashes, spaces, dots, slashes)
    and converts to uppercase.

    Args:
        part_number: Original part number string

    Returns:
        Normalized part number

    Examples:
        >>> normalize_part_number("ABC-123")
        'ABC123'
        >>> normalize_part_number("abc.123/x")
        'ABC123X'
    """
    if not part_number:
        return ""

    # Remove common separators and convert to uppercase
    normalized = re.sub(r'[-\s\.\/]', '', str(part_number).upper())
    return normalized


def extract_metadata(content: str) -> Dict[str, Any]:
    """
    Extract metadata from document content.

    Attempts to identify:
    - Part numbers
    - Voltages
    - Currents
    - Power ratings
    - Dimensions

    Args:
        content: Document text content

    Returns:
        Dictionary of extracted metadata
    """
    metadata = {}

    # Part number patterns
    part_patterns = [
        r'(?:Part\s*(?:No\.?|Number|#)?[:\s]*)?([A-Z]{2,5}[-\s]?\d{3,}[-\s]?\w*)',
        r'\b([A-Z]{1,3}\d{3,}[A-Z0-9]*)\b',
    ]

    for pattern in part_patterns:
        matches = re.findall(pattern, content, re.IGNORECASE)
        if matches:
            metadata['detected_part_numbers'] = list(set(matches[:5]))
            break

    # Voltage extraction
    voltage_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:V|VAC|VDC|Volts?)',
        content,
        re.IGNORECASE
    )
    if voltage_match:
        metadata['voltage'] = voltage_match.group(1) + 'V'

    # Current extraction
    current_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:A|Amps?|Amperes?)',
        content,
        re.IGNORECASE
    )
    if current_match:
        metadata['current'] = current_match.group(1) + 'A'

    # Power extraction
    power_match = re.search(
        r'(\d+(?:\.\d+)?)\s*(?:kW|W|Watts?|HP)',
        content,
        re.IGNORECASE
    )
    if power_match:
        metadata['power'] = power_match.group(0)

    # Dimension extraction (LxWxH format)
    dim_match = re.search(
        r'(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*[xX×]\s*(\d+(?:\.\d+)?)\s*(?:mm|cm|in)?',
        content
    )
    if dim_match:
        metadata['dimensions'] = f"{dim_match.group(1)}x{dim_match.group(2)}x{dim_match.group(3)}"

    # IP Rating extraction
    ip_match = re.search(r'IP\s*(\d{2})', content, re.IGNORECASE)
    if ip_match:
        metadata['ip_rating'] = f"IP{ip_match.group(1)}"

    return metadata


def format_specs_for_display(metadata: Dict[str, Any]) -> str:
    """
    Format metadata specifications for human-readable display.

    Args:
        metadata: Dictionary of metadata

    Returns:
        Formatted string of specifications
    """
    spec_keys = ['voltage', 'current', 'power', 'dimensions', 'ip_rating']
    specs = []

    for key in spec_keys:
        if key in metadata and metadata[key]:
            display_key = key.replace('_', ' ').title()
            specs.append(f"{display_key}: {metadata[key]}")

    return ', '.join(specs) if specs else "No specifications available"


def categorize_document(content: str, filename: str) -> str:
    """
    Categorize a document based on its content and filename.

    Args:
        content: Document text content
        filename: Original filename

    Returns:
        Category string
    """
    content_lower = content.lower()
    filename_lower = filename.lower()

    # Check for cross-reference indicators
    crossref_indicators = [
        'cross reference', 'crossref', 'equivalent', 'replaces',
        'substitute', 'compatible'
    ]
    if any(ind in content_lower or ind in filename_lower for ind in crossref_indicators):
        return 'parts_crossref'

    # Check for specification indicators
    spec_indicators = [
        'specification', 'datasheet', 'technical data', 'spec sheet'
    ]
    if any(ind in content_lower or ind in filename_lower for ind in spec_indicators):
        return 'specifications'

    # Check for manual/guide indicators
    guide_indicators = [
        'manual', 'guide', 'installation', 'user guide', 'instruction'
    ]
    if any(ind in content_lower or ind in filename_lower for ind in guide_indicators):
        return 'user_guide'

    # Default
    return 'general'
