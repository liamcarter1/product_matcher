"""
ProductMatchPro - Hydraulic Code Translator
Decodes competitor model codes into normalized specs, translates spool functions
using the canonical pattern cross-reference, and constructs the equivalent
Vickers by Danfoss model code.

Pipeline:  competitor_code
               -> decode()          (regex + DB patterns -> normalized spec dict)
               -> translate_spool() (canonical_pattern lookup -> Danfoss spool code)
               -> translate_voltage() (manufacturer voltage code -> Danfoss voltage code)
               -> construct()       (assemble DG4V-N-{spool}-M-{conn}-{volt}-{design})

This is deterministic and does not call an LLM. It is the primary matching path
when no confirmed_equivalent exists. The LLM (spec comparison + semantic search)
is the fallback when this path cannot fully resolve.
"""

import re
import json
import logging
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).resolve().parents[2] / "data"


def _load_json(filename: str) -> dict:
    path = _DATA_DIR / filename
    if not path.exists():
        logger.warning("Translation data file not found: %s", path)
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as e:
        logger.error("Failed to load %s: %s", filename, e)
        return {}


# Rexroth 4WE6/4WE10 decode regex
# Format: 4WE{size}{spool}{design}/{style}{voltage}{override?}{connector?}
# Example: 4WE6E62/EG24N9K4
#          4WE6G6X/EG24
#          4WE6E73 6X/EG24N9K4   (soft-shift spool)
_RE_REXROTH_4WE = re.compile(
    r"^4WE(\d+)"           # 1: valve_size (6 or 10)
    r"([A-Z][A-Z0-9]*?)"   # 2: spool_code (E, G, H, E73, G73, E1, E2, E3, ...)
    r"(\d+[X\d])"          # 3: design (6X, 62, 63, 64)
    r"/([EAB])"            # 4: style (E=central, A=sol-a side, B=sol-b side)
    r"([GW]\d+)"           # 5: voltage (G12, G24, G48, W110, W230)
    r"(N[9WHF])?"          # 6: override (optional: N9, NW, NH, NF)
    r"(K\d+)?$",           # 7: connector (optional: K4, K20)
    re.IGNORECASE,
)

# Rexroth 4WREE / 4WRE proportional
_RE_REXROTH_4WRE = re.compile(
    r"^4WRE(E?)(\d+)([A-Z])(\d+)-(.+)/(\d+)(.*)?$",
    re.IGNORECASE,
)

# Generic voltage normalisation patterns (bare numbers + common suffixes)
_RE_BARE_VOLTAGE = re.compile(
    r"^(\d+)\s*([VAC]{0,3}(?:DC)?)$",
    re.IGNORECASE,
)


class CodeTranslator:
    """Translates competitor hydraulic valve model codes to Danfoss equivalents."""

    def __init__(self, db=None):
        self.db = db  # ProductDB instance — optional; enables DB-backed decode
        self._voltage = _load_json("voltage_translations.json")
        self._connectors = _load_json("connector_translations.json")
        self._templates = _load_json("danfoss_code_templates.json")

    # ── Public API ──────────────────────────────────────────────────────

    def decode(self, model_code: str, manufacturer: str) -> dict:
        """Decode a competitor model code into a normalized spec dict.

        Returns a dict with keys such as:
            valve_size, spool_code, voltage_raw, voltage_normalized,
            style_raw, connector_raw, override_raw, design_raw, manufacturer
        Returns {} if the code cannot be decoded.
        """
        code = model_code.strip().upper().replace(" ", "")
        mfr_lower = manufacturer.lower()

        # 1. Try DB-backed decode (model_code_patterns table)
        if self.db:
            try:
                decoded = self.db.decode_model_code(model_code, manufacturer)
                if decoded:
                    return self._normalize_db_decoded(decoded, manufacturer)
            except Exception as e:
                logger.debug("DB decode failed for %s: %s", model_code, e)

        # 2. Manufacturer-specific regex decoders
        if "rexroth" in mfr_lower:
            result = self._decode_rexroth_4we(code)
            if result:
                return result

        # 3. Generic fallback — extract obvious fragments
        return self._generic_decode(code, manufacturer)

    def translate_spool(self, spool_code: str, manufacturer: str) -> Optional[str]:
        """Find the primary Danfoss spool code with the same canonical function.

        Returns the Danfoss spool code (e.g. '2A') or None if no match found.
        """
        if not spool_code or not self.db:
            return None

        # Get canonical pattern for competitor spool
        canonical = self._get_canonical(spool_code, manufacturer)
        if not canonical:
            logger.debug("No canonical pattern for %s spool '%s'", manufacturer, spool_code)
            return None

        # Find primary Danfoss spool with matching canonical pattern
        danfoss_refs = self.db.get_spool_type_references(manufacturer="Danfoss")
        # Primary match first
        for ref in danfoss_refs:
            if ref.get("canonical_pattern") == canonical and ref.get("is_primary"):
                logger.info(
                    "Spool match: %s '%s' -> Danfoss '%s' (canonical: %s)",
                    manufacturer, spool_code, ref["spool_code"], canonical,
                )
                return ref["spool_code"]
        # Any match (non-primary)
        for ref in danfoss_refs:
            if ref.get("canonical_pattern") == canonical:
                logger.info(
                    "Spool match (non-primary): %s '%s' -> Danfoss '%s'",
                    manufacturer, spool_code, ref["spool_code"],
                )
                return ref["spool_code"]

        logger.debug("No Danfoss spool found for canonical '%s'", canonical)
        return None

    def translate_voltage(self, raw_voltage: str, manufacturer: str) -> dict:
        """Translate a competitor voltage code to normalized form and Danfoss DG4V code.

        Returns dict with keys: normalized (e.g. '24VDC'), danfoss_dgv (e.g. 'H7'), volts (int).
        """
        if not raw_voltage:
            return {}

        # Check manufacturer-specific table first
        mfr_table = self._voltage.get("by_manufacturer", {})
        for mfr_key, codes in mfr_table.items():
            if mfr_key.lower() in manufacturer.lower() or manufacturer.lower() in mfr_key.lower():
                entry = codes.get(raw_voltage.upper())
                if entry:
                    normalized = entry["normalized"]
                    dgv_code = self._voltage.get("normalized_to_danfoss_dgv", {}).get(normalized)
                    return {"normalized": normalized, "danfoss_dgv": dgv_code,
                            "volts": entry.get("volts"), "dc": entry.get("dc", True)}

        # Try normalizing the raw code directly
        normalized = self._normalize_voltage_string(raw_voltage)
        if normalized:
            dgv_code = self._voltage.get("normalized_to_danfoss_dgv", {}).get(normalized)
            return {"normalized": normalized, "danfoss_dgv": dgv_code}

        return {"normalized": raw_voltage}

    def translate_connector(self, style_raw: str, connector_raw: str,
                            manufacturer: str) -> str:
        """Translate competitor style/connector codes to Danfoss DG4V connector code."""
        mfr_data = self._connectors.get("by_manufacturer", {})
        norm_to_danfoss = self._connectors.get("normalized_to_danfoss_dgv", {})

        for mfr_key, data in mfr_data.items():
            if mfr_key.lower() in manufacturer.lower() or manufacturer.lower() in mfr_key.lower():
                # Resolve style -> normalized
                normalized = None
                if style_raw:
                    style_entry = data.get("style", {}).get(style_raw.upper())
                    if style_entry:
                        normalized = style_entry.get("normalized")
                # Resolve connector override if bare leads
                if connector_raw:
                    conn_entry = data.get("connector", {}).get(connector_raw.upper())
                    if conn_entry and conn_entry.get("normalized") == "bare_leads":
                        normalized = "bare_leads"
                if normalized:
                    danfoss_code = norm_to_danfoss.get(normalized)
                    if danfoss_code:
                        return danfoss_code

        # Default: U (DIN plug, most common)
        return "U"

    def construct_danfoss_code(self, decoded: dict) -> Optional[str]:
        """Construct a Vickers by Danfoss DG4V model code from decoded specs.

        Returns the assembled code (e.g. 'DG4V-3-2A-M-U-H7-60') or None if
        critical fields are missing.
        """
        if not decoded:
            return None

        # Resolve Danfoss series from valve_size
        valve_size = decoded.get("valve_size", "")
        series_key = self._resolve_series(valve_size)
        template_data = self._templates.get("series", {}).get(series_key)
        if not template_data:
            logger.debug("No Danfoss template for valve_size '%s' (series '%s')", valve_size, series_key)
            return None

        segments = template_data.get("segments", {})

        # Resolve spool
        danfoss_spool = decoded.get("danfoss_spool")
        if not danfoss_spool:
            logger.debug("Cannot construct code: danfoss_spool not resolved")
            return None

        # Resolve voltage
        voltage_info = decoded.get("voltage_info", {})
        danfoss_voltage = voltage_info.get("danfoss_dgv")
        if not danfoss_voltage:
            logger.debug("Cannot construct code: voltage not resolved (raw: %s)",
                         decoded.get("voltage_raw"))
            return None

        # Resolve connector
        danfoss_connector = decoded.get("danfoss_connector") or template_data.get("default_connector", "U")
        design = template_data.get("default_design", "60")

        size_val = segments.get("size", {}).get("value", "3")
        code = f"DG4V-{size_val}-{danfoss_spool}-M-{danfoss_connector}-{danfoss_voltage}-{design}"
        logger.info("Constructed Danfoss code: %s (from decoded: %s)", code, decoded)
        return code

    def full_translate(self, competitor_code: str, manufacturer: str) -> dict:
        """Full pipeline: decode -> translate spool/voltage/connector -> construct.

        Returns dict with:
            decoded:          raw decoded spec dict
            danfoss_code:     constructed Danfoss code (or None)
            danfoss_spool:    translated Danfoss spool code
            voltage_info:     normalized voltage info
            explanation:      human-readable explanation of the translation steps
            confidence:       'high' | 'medium' | 'low' (how reliable the result is)
        """
        decoded = self.decode(competitor_code, manufacturer)
        if not decoded:
            return {
                "decoded": {},
                "danfoss_code": None,
                "confidence": "low",
                "explanation": f"Could not decode model code '{competitor_code}' for {manufacturer}.",
            }

        # Translate spool
        spool_raw = decoded.get("spool_code", "")
        danfoss_spool = self.translate_spool(spool_raw, manufacturer) if spool_raw else None
        decoded["danfoss_spool"] = danfoss_spool

        # Translate voltage
        voltage_raw = decoded.get("voltage_raw", "")
        voltage_info = self.translate_voltage(voltage_raw, manufacturer)
        decoded["voltage_info"] = voltage_info

        # Translate connector
        danfoss_connector = self.translate_connector(
            decoded.get("style_raw", ""),
            decoded.get("connector_raw", ""),
            manufacturer,
        )
        decoded["danfoss_connector"] = danfoss_connector

        # Construct code
        danfoss_code = self.construct_danfoss_code(decoded)

        # Assess confidence
        confidence = self._assess_confidence(decoded, danfoss_spool, voltage_info, danfoss_code)

        # Build explanation
        explanation = self._build_explanation(
            competitor_code, manufacturer, decoded, danfoss_spool,
            voltage_info, danfoss_connector, danfoss_code,
        )

        return {
            "decoded": decoded,
            "danfoss_code": danfoss_code,
            "danfoss_spool": danfoss_spool,
            "voltage_info": voltage_info,
            "danfoss_connector": danfoss_connector,
            "confidence": confidence,
            "explanation": explanation,
        }

    # ── Manufacturer-specific decoders ────────────────────────────────

    def _decode_rexroth_4we(self, code: str) -> dict:
        """Decode Bosch Rexroth 4WE6 / 4WE10 model code."""
        m = _RE_REXROTH_4WE.match(code)
        if not m:
            return {}
        size_num = m.group(1)          # "6" or "10"
        spool = m.group(2).upper()     # "E", "G", "E73", "G73", "E1"...
        design = m.group(3).upper()    # "6X", "62", "63", "64"
        style = m.group(4).upper()     # "E", "A", "B"
        voltage = m.group(5).upper()   # "G12", "G24", "G48", "W110", "W230"
        override = (m.group(6) or "").upper()   # "N9", "NW", "" etc.
        connector = (m.group(7) or "").upper()  # "K4", "K20", ""

        valve_size = f"NG{size_num}"

        return {
            "manufacturer": "Bosch Rexroth",
            "series": f"4WE{size_num}",
            "valve_size": valve_size,
            "spool_code": spool,
            "design_raw": design,
            "style_raw": style,
            "voltage_raw": voltage,
            "override_raw": override,
            "connector_raw": connector,
            "decode_method": "regex_rexroth_4we",
        }

    def _normalize_db_decoded(self, decoded: dict, manufacturer: str) -> dict:
        """Normalise a dict returned by db.decode_model_code()."""
        result = dict(decoded)
        result["manufacturer"] = manufacturer
        result["decode_method"] = "db_patterns"
        # Unify field names
        if "spool_type" in result and "spool_code" not in result:
            result["spool_code"] = result["spool_type"]
        if "coil_voltage" in result and "voltage_raw" not in result:
            result["voltage_raw"] = result["coil_voltage"]
        return result

    def _generic_decode(self, code: str, manufacturer: str) -> dict:
        """Extract obvious fragments from an unknown code format."""
        result = {"manufacturer": manufacturer, "decode_method": "generic"}

        # Voltage: look for bare voltage patterns (24VDC, 24V, G24, W230, etc.)
        voltage_match = re.search(r"(G\d+|W\d+|\d{2,3}V?(?:DC|AC)?)", code, re.IGNORECASE)
        if voltage_match:
            result["voltage_raw"] = voltage_match.group(1)

        # Valve size: look for NG or CETOP prefix
        size_match = re.search(r"(?:NG|CETOP)\s*(\d+)", code, re.IGNORECASE)
        if size_match:
            result["valve_size"] = f"NG{size_match.group(1)}"

        return result if len(result) > 2 else {}

    # ── Helpers ────────────────────────────────────────────────────────

    def _get_canonical(self, spool_code: str, manufacturer: str) -> Optional[str]:
        """Return canonical_pattern for a spool code from the DB reference table."""
        if not self.db:
            return None
        refs = self.db.get_spool_type_references(manufacturer=manufacturer)
        spool_upper = spool_code.strip().upper()
        for ref in refs:
            if ref.get("spool_code", "").upper() == spool_upper:
                return ref.get("canonical_pattern")
        return None

    def _resolve_series(self, valve_size: str) -> str:
        """Map NG6/NG10/CETOP3 etc. to Danfoss series key."""
        mapping = self._templates.get("size_mapping", {})
        normalized = valve_size.strip().upper().replace(" ", "")
        return mapping.get(normalized) or mapping.get(valve_size) or "DG4V-3"

    def _normalize_voltage_string(self, raw: str) -> Optional[str]:
        """Normalize a bare voltage string like '24VDC', '24', '110VAC'."""
        raw = raw.strip().upper()
        # Already normalized forms
        for normalized in self._voltage.get("normalized_to_danfoss_dgv", {}):
            if raw == normalized.upper():
                return normalized
        # Try numeric + suffix
        m = re.match(r"^(\d+)\s*([VACDC]*)$", raw)
        if m:
            volts = int(m.group(1))
            suffix = m.group(2)
            if "AC" in suffix:
                candidate = f"{volts}VAC"
            else:
                candidate = f"{volts}VDC"
            if candidate in self._voltage.get("normalized_to_danfoss_dgv", {}):
                return candidate
        return None

    def _assess_confidence(self, decoded: dict, danfoss_spool: Optional[str],
                           voltage_info: dict, danfoss_code: Optional[str]) -> str:
        if danfoss_code and danfoss_spool and voltage_info.get("danfoss_dgv"):
            return "high"
        if danfoss_code:
            return "medium"
        if danfoss_spool or voltage_info.get("normalized"):
            return "low"
        return "low"

    def _build_explanation(
        self, competitor_code: str, manufacturer: str, decoded: dict,
        danfoss_spool: Optional[str], voltage_info: dict,
        danfoss_connector: str, danfoss_code: Optional[str],
    ) -> str:
        lines = [f"**Code decode: {competitor_code} ({manufacturer})**"]
        valve_size = decoded.get("valve_size", "unknown")
        spool_raw = decoded.get("spool_code", "unknown")
        voltage_raw = decoded.get("voltage_raw", "unknown")

        lines.append(f"- Valve size: **{valve_size}**")
        lines.append(f"- Spool function (competitor code): **{spool_raw}**")

        if danfoss_spool:
            lines.append(f"- Spool function (Danfoss equivalent): **{danfoss_spool}** "
                         f"(matched via canonical hydraulic function)")
        else:
            lines.append(f"- Spool function: could not find Danfoss equivalent for '{spool_raw}'")

        if voltage_info.get("normalized"):
            lines.append(f"- Voltage: **{voltage_info['normalized']}** "
                         f"-> Danfoss code **{voltage_info.get('danfoss_dgv', '?')}**")
        else:
            lines.append(f"- Voltage: could not normalize '{voltage_raw}'")

        lines.append(f"- Connector: **{danfoss_connector}** (DIN plug standard)")

        if danfoss_code:
            lines.append(f"\n**Recommended Vickers by Danfoss code: `{danfoss_code}`**")
        else:
            lines.append("\nCould not construct a Danfoss code — see spec comparison results below.")

        return "\n".join(lines)
