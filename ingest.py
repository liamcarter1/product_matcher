"""
ProductMatchPro - PDF Ingestion Pipeline
Orchestrates: parse PDF → extract products → review → store in SQLite + ChromaDB.
"""

import logging
import re
import traceback
import uuid
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from models import (
    HydraulicProduct, ExtractedProduct, UploadMetadata,
    ModelCodePattern, DocumentType,
)
from tools.parse_tools import (
    extract_tables_from_pdf,
    extract_text_from_pdf,
    table_rows_to_products,
    extract_products_with_llm,
    extract_model_code_patterns_with_llm,
    extract_ordering_code_with_llm,
    extract_ordering_code_from_images,
    generate_products_from_ordering_code,
    analyze_spool_functions,
    extract_spool_symbols_from_pdf,
    extract_spool_symbols_from_pdf_v2,
    extract_cross_references_with_llm,
    _is_graphics_heavy_pdf,
    _get_pdf_page_count,
    _classify_spool_pages,
    _merge_spool_results,
    _merge_spool_results_v2,
)
from storage.product_db import ProductDB
from storage.vector_store import VectorStore

# Numeric fields that need float conversion in _apply_decoded_specs
_FLOAT_FIELDS = {
    "max_pressure_bar", "max_flow_lpm", "weight_kg",
    "operating_temp_min_c", "operating_temp_max_c",
    "displacement_cc", "speed_rpm_max",
    "bore_diameter_mm", "rod_diameter_mm", "stroke_mm",
}

# Integer fields that need int conversion in _apply_decoded_specs
_INT_FIELDS = {"num_positions", "num_ports"}

# All valid HydraulicProduct spec fields for _apply_decoded_specs
_ALL_SPEC_FIELDS = {
    "coil_voltage", "valve_size", "spool_type", "actuator_type",
    "port_size", "port_type", "mounting", "mounting_pattern",
    "seal_material", "body_material", "coil_type", "coil_connector",
    "subcategory", "fluid_type", "viscosity_range_cst",
} | _FLOAT_FIELDS | _INT_FIELDS

# Field aliases: map common LLM/table output names to canonical HydraulicProduct field names
_FIELD_ALIASES = {
    # Pressure variations
    "pressure": "max_pressure_bar", "pressure_bar": "max_pressure_bar",
    "max_pressure": "max_pressure_bar", "rated_pressure": "max_pressure_bar",
    "operating_pressure": "max_pressure_bar",
    # Flow variations
    "flow": "max_flow_lpm", "flow_rate": "max_flow_lpm",
    "max_flow": "max_flow_lpm", "rated_flow": "max_flow_lpm",
    "flow_lpm": "max_flow_lpm",
    # Voltage variations
    "voltage": "coil_voltage", "supply_voltage": "coil_voltage",
    "solenoid_voltage": "coil_voltage",
    # Size variations
    "size": "valve_size", "nominal_size": "valve_size",
    "cetop": "valve_size", "ng_size": "valve_size",
    # Port variations
    "port": "port_size", "connection": "port_size",
    "connection_size": "port_size", "thread": "port_type",
    # Mounting variations
    "mounting_type": "mounting", "interface": "mounting_pattern",
    # Material variations
    "seal": "seal_material", "seals": "seal_material",
    "material": "body_material",
    # Cylinder dimensions
    "bore": "bore_diameter_mm", "rod": "rod_diameter_mm",
    "stroke": "stroke_mm", "stroke_length": "stroke_mm",
    # Pump/motor specs
    "displacement": "displacement_cc", "speed": "speed_rpm_max",
    "max_speed": "speed_rpm_max", "rpm": "speed_rpm_max",
    # Positions/ports
    "positions": "num_positions", "ways": "num_positions",
    "ports": "num_ports", "number_of_ports": "num_ports",
    # Temperature
    "temp_min": "operating_temp_min_c", "temp_max": "operating_temp_max_c",
    "min_temp": "operating_temp_min_c", "max_temp": "operating_temp_max_c",
    # Actuation
    "actuation": "actuator_type", "operation": "actuator_type",
    # Other
    "weight": "weight_kg", "mass": "weight_kg",
    "fluid": "fluid_type", "medium": "fluid_type",
    "viscosity": "viscosity_range_cst",
    "spool": "spool_type", "function": "spool_type",
    "connector": "coil_connector",
    "product_type": "subcategory", "valve_type": "subcategory",
}

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1500,
    chunk_overlap=300,
    separators=["\n\n", "\n", ". ", " ", ""],
)


class IngestionPipeline:

    def __init__(self, db: ProductDB, vector_store: VectorStore):
        self.db = db
        self.vector_store = vector_store

    def process_pdf(
        self, pdf_path: str, metadata: UploadMetadata
    ) -> list[ExtractedProduct]:
        """Process a PDF and return extracted products for review.
        Does NOT store yet - admin must confirm first."""

        extracted_products = []

        # Step 1: Try table extraction (best for catalogues)
        tables = extract_tables_from_pdf(pdf_path)
        for table_rows in tables:
            products = table_rows_to_products(table_rows, metadata)
            extracted_products.extend(products)

        # Step 2: Extract full text for LLM processing
        pages = extract_text_from_pdf(pdf_path)
        full_text = "\n\n".join(p["text"] for p in pages)

        # Step 2.5: Detect graphics-heavy PDFs (ordering codes rendered as vector graphics)
        total_page_count = _get_pdf_page_count(pdf_path)
        graphics_heavy = _is_graphics_heavy_pdf(pages, total_page_count)
        self._last_extraction_method = "vision" if graphics_heavy else "text"
        if graphics_heavy:
            print(f"[DEBUG] PDF classified as graphics-heavy ({total_page_count} pages, "
                  f"avg {sum(len(p['text']) for p in pages) / max(total_page_count, 1):.0f} chars/page)")

        # Step 3: If tables didn't yield products, use LLM extraction
        if not extracted_products and full_text:
            llm_products = extract_products_with_llm(full_text, metadata)
            extracted_products.extend(llm_products)

        # Step 4: Extract model code patterns (from user guides/datasheets)
        if metadata.document_type in (DocumentType.USER_GUIDE, DocumentType.DATASHEET):
            patterns = extract_model_code_patterns_with_llm(full_text, metadata.company)
            for pattern_data in patterns:
                try:
                    pattern = ModelCodePattern(
                        company=metadata.company,
                        series=pattern_data.get("series", ""),
                        segment_position=pattern_data.get("segment_position", 0),
                        segment_name=pattern_data.get("segment_name", ""),
                        code_value=pattern_data.get("code_value", ""),
                        decoded_value=pattern_data.get("decoded_value", ""),
                        maps_to_field=pattern_data.get("maps_to_field", ""),
                    )
                    if pattern.series and pattern.code_value:
                        self.db.insert_model_code_pattern(pattern)
                except Exception as e:
                    print(f"Error storing pattern: {e}")

        # Step 5: Spool extraction — REFERENCE-FIRST pipeline
        # Priority: DB reference > text analysis > vision (last resort)
        #   5a: Infer series prefix from text + query spool_type_reference
        #   5b: Text-based spool analysis (secondary, discovers new codes)
        #   5c: Vision extraction ONLY if 5a+5b found ZERO spools
        #   5d: Three-way merge: reference > text > vision
        reference_spools: list[dict] = []
        text_spools: list[dict] = []
        vision_spools: list[dict] = []

        # Step 5a: Query spool_type_reference (zero API cost)
        inferred_series = self._infer_series_prefix_from_text(
            full_text, metadata.company,
        )
        if inferred_series:
            print(f"[DEBUG] Inferred series prefix: {inferred_series}")

        # Try series-specific lookup first, then manufacturer-wide
        if inferred_series:
            reference_spools = self.db.get_spool_type_references(
                inferred_series, metadata.company,
            )
        if not reference_spools:
            reference_spools = self.db.get_spool_type_references(
                metadata.category or "", metadata.company,
            )
        if not reference_spools:
            # Broadest: all refs for this manufacturer
            reference_spools = self.db.get_spool_type_references(
                manufacturer=metadata.company,
            )

        if reference_spools:
            print(f"Reference DB: {len(reference_spools)} spool types found "
                  f"for {metadata.company}")
            logger.info("Reference-first: %d spool refs from DB for %s",
                        len(reference_spools), metadata.company)

        # Step 5b: Text-based spool analysis (always runs — discovers new codes)
        if full_text:
            try:
                text_spools = analyze_spool_functions(full_text, metadata.company)
                if text_spools:
                    print(f"Spool text analysis: found {len(text_spools)} spool types")
            except Exception as e:
                print(f"Error in spool function analysis: {e}")

        # Step 5c: Vision extraction — ONLY if reference + text found ZERO spools
        if not reference_spools and not text_spools:
            print("[DEBUG] No reference or text spools — falling back to vision (last resort)")
            page_classifications = _classify_spool_pages(pdf_path, pages)
            try:
                known_spools_db = self.db.get_spool_codes_for_series(
                    metadata.category or "", metadata.company,
                )
                vision_spools = extract_spool_symbols_from_pdf_v2(
                    pdf_path, metadata.company,
                    page_classifications=page_classifications,
                    known_spool_codes=known_spools_db if known_spools_db else None,
                    retry_on_low_count=True,
                )
                if vision_spools:
                    # Flag all vision results as unconfirmed
                    for vs in vision_spools:
                        vs["source"] = "vision_unconfirmed"
                    print(f"Vision spool extraction (last resort): "
                          f"{len(vision_spools)} symbols found")
            except Exception as e:
                logger.warning("Vision spool extraction error (non-fatal): %s", e)
        else:
            print("[DEBUG] Skipping vision — reference/text data available")

        # Step 5d: Three-way merge (reference > text > vision)
        merged_spool_lookup = _merge_spool_results_v2(
            reference_spools, text_spools, vision_spools,
        )

        # Combine discovered spool codes for injection into ordering code prompt
        discovered_spool_codes = list(merged_spool_lookup.keys())

        print(f"[DEBUG] Merged spool lookup: {len(merged_spool_lookup)} entries, "
              f"codes: {sorted(set(s.get('spool_code', '').upper() for s in merged_spool_lookup.values() if s.get('spool_code')))}")

        # Step 6: Ordering code combinatorial generation
        # Extract ordering code breakdown tables and generate ALL product variants
        # Now receives spool codes discovered in Step 5 as additional reference data
        gapfill_warnings: list[str] = []
        try:
            # Look up known spool codes for this company to inject into the prompt
            print(f"[DEBUG] Starting ordering code extraction for {metadata.company}...")
            known_spools = self.db.get_spool_codes_for_series(
                metadata.category or "", metadata.company,
            )
            print(f"[DEBUG] Known spools (DB): {known_spools}")

            # Merge DB-known spools with freshly-discovered spools from Step 5
            all_known_spools = list(set(
                (known_spools or []) + discovered_spool_codes
            ))
            if discovered_spool_codes:
                print(f"[DEBUG] Injecting {len(discovered_spool_codes)} discovered spool codes "
                      f"into ordering code extraction")

            # Choose extraction method: vision-first for graphics-heavy, text for normal
            ordering_defs = []
            if graphics_heavy:
                print("[DEBUG] Using VISION extraction for ordering codes (graphics-heavy PDF)")
                ordering_defs = extract_ordering_code_from_images(
                    pdf_path, metadata.company, metadata.category or "",
                    known_spool_codes=all_known_spools if all_known_spools else None,
                )
                # If vision fails, fall back to text (may still yield partial results)
                if not ordering_defs and full_text:
                    print("[DEBUG] Vision extraction returned nothing, falling back to text extraction")
                    self._last_extraction_method = "text (vision fallback)"
                    ordering_defs = extract_ordering_code_with_llm(
                        full_text, metadata.company, metadata.category or "",
                        known_spool_codes=all_known_spools if all_known_spools else None,
                    )
            elif full_text:
                ordering_defs = extract_ordering_code_with_llm(
                    full_text, metadata.company, metadata.category or "",
                    known_spool_codes=all_known_spools if all_known_spools else None,
                )
                # Vision retry: if text extraction found NO ordering codes and
                # the PDF has enough pages, try vision as a fallback. This catches
                # PDFs that aren't flagged as graphics-heavy but still have their
                # ordering codes rendered as vector graphics.
                if not ordering_defs and total_page_count >= 3:
                    print("[DEBUG] Text ordering code extraction found nothing on "
                          f"{total_page_count}-page PDF, retrying with VISION")
                    ordering_defs = extract_ordering_code_from_images(
                        pdf_path, metadata.company, metadata.category or "",
                        known_spool_codes=all_known_spools if all_known_spools else None,
                    )
                    if ordering_defs:
                        self._last_extraction_method = "vision (text retry)"
                        print(f"[DEBUG] Vision retry found {len(ordering_defs)} definitions!")

            # Step 6a: Augment spool segments with discovered spool codes
            # GPT-4o often only finds 1 spool option from the ordering code diagram
            # (e.g. "0A" as the example). The spool extraction steps found the FULL
            # list of spool codes from symbol tables. Inject any missing codes into
            # the spool segment of each ordering code definition.
            if merged_spool_lookup:
                for definition in ordering_defs:
                    for seg in definition.segments:
                        is_spool = (seg.segment_name == "spool_type" or
                                    any(o.get("maps_to_field") == "spool_type"
                                        for o in seg.options))
                        if not is_spool:
                            continue
                        existing_codes = {
                            o.get("code", "").upper() for o in seg.options
                        }
                        added = 0
                        for code, spool_data in merged_spool_lookup.items():
                            # merged_spool_lookup has both "2A" and "2A" without
                            # leading zeros as keys — deduplicate
                            raw_code = spool_data.get("spool_code", code).strip()
                            if not raw_code or raw_code.upper() in existing_codes:
                                continue
                            existing_codes.add(raw_code.upper())
                            desc = spool_data.get("center_condition", "") or \
                                   spool_data.get("description", "")
                            seg.options.append({
                                "code": raw_code,
                                "description": desc or f"Spool type {raw_code}",
                                "maps_to_field": "spool_type",
                                "maps_to_value": raw_code,
                            })
                            added += 1
                        if added:
                            seg.is_fixed = False  # ensure variable
                            print(f"[DEBUG] Augmented spool segment for "
                                  f"{definition.series}: +{added} codes "
                                  f"(total: {len(seg.options)})")
                        break  # only one spool segment per definition

            for definition in ordering_defs:
                # Also look up spools by specific series (more precise)
                if definition.series and not known_spools:
                    series_spools = self.db.get_spool_codes_for_series(
                        definition.series, metadata.company,
                    )
                    if series_spools:
                        logger.info("Found %d known spool codes for series %s",
                                    len(series_spools), definition.series)

                # Look up primary (main runner) spool codes for this series
                print(f"[DEBUG] Looking up primary spools for series={definition.series}, company={metadata.company}")
                primary_spools = self.db.get_primary_spool_codes(
                    definition.series, metadata.company,
                )
                print(f"[DEBUG] Primary spools: {primary_spools}")
                if primary_spools:
                    logger.info(
                        "Primary spool filter active for %s: %d codes: %s",
                        definition.series, len(primary_spools), primary_spools,
                    )

                generated = generate_products_from_ordering_code(
                    definition, metadata,
                    primary_spool_codes=primary_spools if primary_spools else None,
                )
                if generated:
                    print(f"Generated {len(generated)} products from ordering code "
                          f"table for series: {definition.series} "
                          f"[{self._last_extraction_method}]")
                    # Diagnostic: log segment names and sample spec keys
                    seg_names = [s.segment_name for s in definition.segments]
                    logger.info("Ordering code segments: %s", seg_names)
                    if generated:
                        sample_keys = sorted(generated[0].specs.keys())
                        logger.info("Sample product spec keys: %s", sample_keys)
                    extracted_products.extend(generated)

                    # Also store as ModelCodePattern rows for future decode use
                    for seg in definition.segments:
                        for opt in seg.options:
                            try:
                                pattern = ModelCodePattern(
                                    company=metadata.company,
                                    series=definition.series,
                                    segment_position=seg.position,
                                    segment_name=seg.segment_name,
                                    code_value=opt.get("code", ""),
                                    decoded_value=opt.get("description", ""),
                                    maps_to_field=opt.get("maps_to_field", ""),
                                )
                                if pattern.series and pattern.code_value:
                                    self.db.insert_model_code_pattern(pattern)
                            except Exception as e:
                                print(f"Error storing ordering code pattern: {e}")

                # Step 6a: Validate spool options against reference & gap-fill
                warnings = self._validate_and_gapfill_spools(
                    definition, metadata, extracted_products,
                )
                gapfill_warnings.extend(warnings)

        except Exception as e:
            print(f"Error in ordering code extraction: {e}")
            traceback.print_exc()

        # Step 7: Merge spool function data into extracted products
        # merged_spool_lookup was built earlier (before ordering code extraction)
        if merged_spool_lookup:
            spool_merged_count = 0
            for product in extracted_products:
                spool = product.specs.get("spool_type", "")
                if not spool:
                    continue
                spool_upper = str(spool).strip().upper()
                matched = merged_spool_lookup.get(spool_upper)
                if not matched:
                    # Try partial match (e.g. product spool "2A" in "type 2A")
                    for code, data in merged_spool_lookup.items():
                        if code in spool_upper or spool_upper in code:
                            matched = data
                            break
                if matched:
                    if not product.specs.get("center_condition"):
                        product.specs["center_condition"] = matched.get("center_condition", "")
                    if not product.specs.get("solenoid_a_energised"):
                        product.specs["solenoid_a_energised"] = matched.get("solenoid_a_function", "")
                    if not product.specs.get("solenoid_b_energised"):
                        product.specs["solenoid_b_energised"] = matched.get("solenoid_b_function", "")
                    if not product.specs.get("spool_function_description"):
                        product.specs["spool_function_description"] = matched.get("description", "")
                    if not product.specs.get("canonical_spool_pattern"):
                        product.specs["canonical_spool_pattern"] = matched.get("canonical_pattern", "")
                    if matched.get("symbol_description"):
                        product.specs["spool_symbol_description"] = matched["symbol_description"]
                    spool_merged_count += 1

            logger.info("Spool merge: %d spool definitions applied to %d products",
                        len(merged_spool_lookup), spool_merged_count)
            print(f"Spool merge: {len(merged_spool_lookup)} spool definitions, "
                  f"applied to {spool_merged_count} products")

        # Diagnostic: log all unique spec keys across extracted products
        all_spec_keys = set()
        for p in extracted_products:
            all_spec_keys.update(p.specs.keys())
        dynamic_keys = all_spec_keys - {"max_pressure_bar", "max_flow_lpm", "coil_voltage",
                                         "valve_size", "spool_type", "seal_material",
                                         "actuator_type", "port_size", "mounting", "num_ports"}
        logger.info("Total extracted products: %d, all spec keys: %s", len(extracted_products), sorted(all_spec_keys))
        logger.info("Dynamic (non-standard) spec keys: %s", sorted(dynamic_keys))

        # Step 8: Post-process spool_type values (safety net for LLM non-compliance)
        self._clean_spool_types(extracted_products)

        # Step 9: Deduplicate by model_code (prefer ordering_code > table > llm)
        extracted_products = self._deduplicate_products(extracted_products)

        # Attach metadata to all products
        for product in extracted_products:
            if not product.category and metadata.category:
                product.category = metadata.category

        # Store gap-fill warnings for admin review UI
        self._last_gapfill_warnings = gapfill_warnings

        return extracted_products

    @staticmethod
    def _clean_spool_types(products: list[ExtractedProduct]) -> None:
        """Post-process spool_type to extract ONLY the code, move descriptions elsewhere.

        Safety net that catches dirty spool_type values regardless of which
        LLM extraction path produced them. Also detects override codes that
        were misplaced into spool_type.
        """
        _OVERRIDE_KEYWORDS = {"override", "no override", "water resistant", "emergency"}

        for p in products:
            spool = p.specs.get("spool_type", "")
            if not spool:
                continue

            # Detect override codes misplaced in spool_type
            spool_lower = str(spool).lower()
            if any(kw in spool_lower for kw in _OVERRIDE_KEYWORDS):
                p.specs.setdefault("manual_override", spool)
                p.specs["spool_type"] = ""
                logger.info("Moved override data from spool_type: '%s'", spool)
                continue

            # Pattern: "2A (description...)" or "D - description" or "H (all ports blocked)"
            match = re.match(
                r'^([A-Za-z0-9]{1,5})\s*[\(\-–—]\s*(.+?)\)?\s*$',
                str(spool),
            )
            if match:
                code = match.group(1).strip()
                description = match.group(2).strip()
                p.specs["spool_type"] = code
                if not p.specs.get("spool_function_description"):
                    p.specs["spool_function_description"] = description
                logger.info("Cleaned spool_type '%s' -> code='%s', desc='%s'",
                            spool, code, description)

    def _validate_and_gapfill_spools(
        self,
        definition,
        metadata: UploadMetadata,
        extracted_products: list[ExtractedProduct],
    ) -> list[str]:
        """Compare extracted spool options against the spool_type_reference table.

        If the reference has spool codes not found in the extraction:
        1. Add missing codes as synthetic options to the spool segment
        2. Generate the missing product combinations
        3. Mark gap-filled products with _spool_source = 'reference_gapfill'
        4. Return warning messages for admin review UI
        """
        warnings: list[str] = []

        # Find the spool segment
        spool_seg = None
        for seg in definition.segments:
            if seg.segment_name == "spool_type":
                spool_seg = seg
                break
            # Check if any option maps to spool_type
            for opt in seg.options:
                if opt.get("maps_to_field") == "spool_type":
                    spool_seg = seg
                    break
            if spool_seg:
                break

        if not spool_seg:
            return warnings

        extracted_codes = {opt.get("code", "").upper() for opt in spool_seg.options if opt.get("code")}
        known_codes = set(self.db.get_spool_codes_for_series(
            definition.series, metadata.company,
        ))

        if not known_codes:
            return warnings

        missing_codes = known_codes - extracted_codes

        # If primary filtering is active, only gap-fill primary spools
        primary_codes = set(self.db.get_primary_spool_codes(
            definition.series, metadata.company,
        ))
        if primary_codes:
            missing_codes = missing_codes & primary_codes

        if not missing_codes:
            logger.info("Spool validation: all %d known codes found in extraction for %s",
                        len(known_codes), definition.series)
            return warnings

        logger.info("Spool gap-fill: %d known codes missing from extraction for %s: %s",
                     len(missing_codes), definition.series, sorted(missing_codes))

        # Look up reference data to build synthetic options
        refs = self.db.get_spool_type_references(definition.series, metadata.company)
        ref_lookup = {r["spool_code"].upper(): r for r in refs}

        for code in sorted(missing_codes):
            ref = ref_lookup.get(code.upper(), {})
            synthetic_opt = {
                "code": code,
                "description": ref.get("description", "Known spool type (from reference data)"),
                "maps_to_field": "spool_type",
                "maps_to_value": code,
            }
            spool_seg.options.append(synthetic_opt)

        # Regenerate products for ONLY the missing spool codes
        # Temporarily narrow the spool segment to missing codes only
        original_options = spool_seg.options
        spool_seg.options = [opt for opt in original_options
                             if opt.get("code", "").upper() in missing_codes]

        new_products = generate_products_from_ordering_code(definition, metadata)
        for p in new_products:
            p.specs["_spool_source"] = "reference_gapfill"

        # Restore full options list
        spool_seg.options = original_options

        if new_products:
            extracted_products.extend(new_products)
            missing_str = ", ".join(sorted(missing_codes))
            msg = (f"Gap-filled {len(new_products)} products from spool reference data "
                   f"for series {definition.series}. Missing codes added: {missing_str}")
            warnings.append(msg)
            print(msg)

        return warnings

    def _auto_learn_spool_types(
        self,
        products: list[ExtractedProduct],
        metadata: UploadMetadata,
    ) -> int:
        """Extract unique spool types from confirmed products and store in
        spool_type_reference table for future extractions.

        Only learns from products NOT marked as gap-filled.
        """
        # Group by source — infer series from ordering_code source
        spool_data: dict[str, dict] = {}  # spool_code -> {center_condition, ...}

        for p in products:
            # Skip gap-filled products (don't learn from our own gap-fill)
            if p.specs.get("_spool_source") == "reference_gapfill":
                continue
            spool_code = p.specs.get("spool_type", "").strip()
            if not spool_code:
                continue
            if spool_code not in spool_data:
                spool_data[spool_code] = {
                    "center_condition": p.specs.get("center_condition", ""),
                    "solenoid_a_function": p.specs.get("solenoid_a_energised", ""),
                    "solenoid_b_function": p.specs.get("solenoid_b_energised", ""),
                    "canonical_pattern": p.specs.get("canonical_spool_pattern", ""),
                    "description": p.specs.get("spool_function_description", ""),
                }

        if not spool_data:
            return 0

        # Infer series from the ordering code source or model code patterns
        series_prefix = self._infer_series_from_products(products, metadata.company)

        refs = []
        for code, data in spool_data.items():
            refs.append({
                "series_prefix": series_prefix,
                "manufacturer": metadata.company,
                "spool_code": code,
                "description": data.get("description", ""),
                "center_condition": data.get("center_condition", ""),
                "solenoid_a_function": data.get("solenoid_a_function", ""),
                "solenoid_b_function": data.get("solenoid_b_function", ""),
                "canonical_pattern": data.get("canonical_pattern", ""),
                "source": "auto_confirmed",
                "source_document": metadata.filename,
            })

        count = self.db.bulk_insert_spool_type_references(refs)
        if count:
            logger.info("Auto-learned %d spool types for series %s from %s",
                        count, series_prefix, metadata.filename)
            print(f"Auto-learned {count} spool types for series {series_prefix}")
        return count

    def _infer_series_prefix_from_text(
        self, text: str, company: str,
    ) -> str | None:
        """Infer product series prefix from document text.

        Checks the first ~10 000 characters of the extracted text for known
        series prefixes stored in the ``spool_type_reference`` table.
        Returns the longest matching prefix, or ``None`` if no match.
        """
        if not text:
            return None

        # Get all known series prefixes for this manufacturer
        all_refs = self.db.get_spool_type_references(manufacturer=company)
        known_prefixes = sorted(
            {r["series_prefix"] for r in all_refs if r.get("series_prefix")},
            key=len, reverse=True,  # longest first for most specific match
        )

        text_upper = text[:10_000].upper()
        for prefix in known_prefixes:
            if prefix.upper() in text_upper:
                return prefix

        return None

    @staticmethod
    def _infer_series_from_products(
        products: list[ExtractedProduct], company: str,
    ) -> str:
        """Infer series prefix from product model codes.

        Looks at ordering_code-sourced products and finds the common prefix.
        """
        ordering_codes = [
            p.model_code for p in products
            if getattr(p, "source", "") == "ordering_code" and p.model_code
        ]
        if not ordering_codes:
            ordering_codes = [p.model_code for p in products if p.model_code]

        if not ordering_codes:
            return ""

        if len(ordering_codes) == 1:
            # Single code: take everything before the last hyphen-separated segment
            parts = ordering_codes[0].split("-")
            if len(parts) >= 2:
                return "-".join(parts[:2])
            return ordering_codes[0][:6]

        # Find longest common prefix across all model codes
        prefix = ordering_codes[0]
        for code in ordering_codes[1:]:
            while not code.startswith(prefix) and prefix:
                prefix = prefix[:-1]

        # Trim trailing separators
        prefix = prefix.rstrip("-/.")

        return prefix if len(prefix) >= 3 else ordering_codes[0][:6]

    @staticmethod
    def _deduplicate_products(products: list[ExtractedProduct]) -> list[ExtractedProduct]:
        """Deduplicate by model_code.

        Priority: ordering_code > table > llm.
        When a higher-priority product wins, it merges useful specs from the
        lower-priority one so no data is lost.
        """
        SOURCE_PRIORITY = {"ordering_code": 3, "table": 2, "llm": 1}
        seen: dict[str, ExtractedProduct] = {}

        for p in products:
            key = p.model_code.upper().strip()
            if key not in seen:
                seen[key] = p
                continue

            existing = seen[key]
            existing_pri = SOURCE_PRIORITY.get(getattr(existing, "source", "llm"), 1)
            new_pri = SOURCE_PRIORITY.get(getattr(p, "source", "llm"), 1)

            if new_pri > existing_pri:
                # New product has higher priority — keep it, merge specs from old
                for k, v in existing.specs.items():
                    if v is not None and v != "" and not p.specs.get(k):
                        p.specs[k] = v
                seen[key] = p
            elif new_pri < existing_pri:
                # Existing has higher priority — merge specs from new into existing
                for k, v in p.specs.items():
                    if v is not None and v != "" and not existing.specs.get(k):
                        existing.specs[k] = v
            else:
                # Same priority: keep the one with more non-empty spec fields
                existing_count = sum(1 for v in existing.specs.values()
                                     if v is not None and v != "")
                new_count = sum(1 for v in p.specs.values()
                                if v is not None and v != "")
                if new_count > existing_count:
                    seen[key] = p

        return list(seen.values())

    def confirm_and_store(
        self,
        extracted_products: list[ExtractedProduct],
        metadata: UploadMetadata,
        auto_learn_spools: bool = True,
    ) -> int:
        """Confirm extracted products and store them in SQLite + ChromaDB.
        Called after admin reviews and approves the extraction."""

        stored_count = 0
        for ep in extracted_products:
            product = self._extracted_to_hydraulic(ep, metadata)

            # Try to decode model code for additional specs
            decoded = self.db.decode_model_code(product.model_code, product.company)
            if decoded:
                product.model_code_decoded = decoded
                # Fill in specs from decoded model code
                self._apply_decoded_specs(product, decoded)

            # Store in SQLite
            self.db.insert_product(product)

            # Index in ChromaDB
            self.vector_store.index_product(product)

            stored_count += 1

        # Auto-learn spool types from confirmed products
        if auto_learn_spools and extracted_products:
            self._auto_learn_spool_types(extracted_products, metadata)

        return stored_count

    def index_guide_text(
        self, pdf_path: str, metadata: UploadMetadata
    ) -> int:
        """Index user guide text chunks for supplementary search.

        Uses a two-pass strategy for maximum retrieval coverage:
        1. Page-level chunking: Each page's text is chunked individually,
           preserving page numbers in metadata and prefixing each chunk
           with the page context. This ensures page-specific content
           (e.g. 'Operating data' on page 6) is findable.
        2. Full-document chunking: The entire document is also chunked
           as one continuous text to capture cross-page information.

        Larger chunks (1500 chars, 300 overlap) preserve more context
        for technical content like specifications and operating data.
        """
        pages = extract_text_from_pdf(pdf_path)

        if not pages:
            return 0

        chunk_count = 0

        # ── Pass 1: Page-level chunks (preserves page context) ──────
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]

            if not page_text or len(page_text.strip()) < 50:
                continue

            # Prefix each chunk with page info for better retrieval
            page_prefix = f"[Page {page_num} of {metadata.filename}]\n"

            page_chunks = text_splitter.split_text(page_text)
            for i, chunk in enumerate(page_chunks):
                chunk_id = f"{metadata.filename}_p{page_num}_chunk_{i}"
                self.vector_store.index_guide_chunk(
                    chunk_id=chunk_id,
                    text=page_prefix + chunk,
                    company=metadata.company,
                    category=metadata.category or "",
                    source_document=metadata.filename,
                )
                chunk_count += 1

        # ── Pass 2: Full-document chunks (captures cross-page info) ─
        full_text = "\n\n".join(p["text"] for p in pages)
        full_chunks = text_splitter.split_text(full_text)
        for i, chunk in enumerate(full_chunks):
            chunk_id = f"{metadata.filename}_full_chunk_{i}"
            self.vector_store.index_guide_chunk(
                chunk_id=chunk_id,
                text=chunk,
                company=metadata.company,
                category=metadata.category or "",
                source_document=metadata.filename,
            )
            chunk_count += 1

        return chunk_count

    def _extracted_to_hydraulic(
        self, ep: ExtractedProduct, metadata: UploadMetadata
    ) -> HydraulicProduct:
        """Convert an ExtractedProduct to a HydraulicProduct.

        First maps known field names directly, then uses _FIELD_ALIASES
        to rescue any remaining specs with non-canonical names.
        """
        specs = dict(ep.specs) if ep.specs else {}

        # Alias rescue: before extracting fields, remap any aliased keys
        aliased_specs = {}
        for key, value in list(specs.items()):
            canonical = _FIELD_ALIASES.get(key.lower().strip())
            if canonical and canonical not in specs:
                aliased_specs[canonical] = value
                del specs[key]
        specs.update(aliased_specs)

        product = HydraulicProduct(
            id=str(uuid.uuid4()),
            company=metadata.company,
            model_code=ep.model_code,
            product_name=str(ep.product_name) if ep.product_name else specs.pop("product_name", ""),
            category=ep.category or metadata.category or "",
            subcategory=self._safe_str(specs.pop("subcategory", None)),
            max_pressure_bar=self._safe_float(specs.pop("max_pressure_bar", None)),
            max_flow_lpm=self._safe_float(specs.pop("max_flow_lpm", None)),
            valve_size=self._safe_str(specs.pop("valve_size", None)),
            spool_type=self._safe_str(specs.pop("spool_type", None)),
            num_positions=self._safe_int(specs.pop("num_positions", None)),
            num_ports=self._safe_int(specs.pop("num_ports", None)),
            actuator_type=self._safe_str(specs.pop("actuator_type", None)),
            coil_voltage=self._safe_str(specs.pop("coil_voltage", None)),
            coil_type=self._safe_str(specs.pop("coil_type", None)),
            coil_connector=self._safe_str(specs.pop("coil_connector", None)),
            port_size=self._safe_str(specs.pop("port_size", None)),
            port_type=self._safe_str(specs.pop("port_type", None)),
            mounting=self._safe_str(specs.pop("mounting", None)),
            mounting_pattern=self._safe_str(specs.pop("mounting_pattern", None)),
            body_material=self._safe_str(specs.pop("body_material", None)),
            seal_material=self._safe_str(specs.pop("seal_material", None)),
            operating_temp_min_c=self._safe_float(specs.pop("operating_temp_min_c", None)),
            operating_temp_max_c=self._safe_float(specs.pop("operating_temp_max_c", None)),
            fluid_type=self._safe_str(specs.pop("fluid_type", None)),
            viscosity_range_cst=self._safe_str(specs.pop("viscosity_range_cst", None)),
            weight_kg=self._safe_float(specs.pop("weight_kg", None)),
            displacement_cc=self._safe_float(specs.pop("displacement_cc", None)),
            speed_rpm_max=self._safe_float(specs.pop("speed_rpm_max", None)),
            bore_diameter_mm=self._safe_float(specs.pop("bore_diameter_mm", None)),
            rod_diameter_mm=self._safe_float(specs.pop("rod_diameter_mm", None)),
            stroke_mm=self._safe_float(specs.pop("stroke_mm", None)),
            description=ep.product_name or "",
            source_document=metadata.filename,
            raw_text=ep.raw_text,
        )

        # Collect ALL leftover specs into extra_specs (these didn't match any known field)
        # Each becomes a visible column in the admin UI
        leftover = {
            k: v for k, v in specs.items()
            if v is not None and v != ""
        }
        if leftover:
            product.extra_specs = {**(product.extra_specs or {}), **leftover}
            logger.debug("Product %s extra_specs keys: %s", product.model_code, sorted(leftover.keys()))

        return product

    def _apply_decoded_specs(self, product: HydraulicProduct, decoded: dict):
        """Apply decoded model code specs to the product, filling in missing fields.

        Dynamically handles ALL spec fields, using appropriate type conversion.
        Uses maps_to_field when available (stored as _field_<segment_name> keys).
        Only fills fields that are currently None — never overwrites existing data.
        """
        for key, value in decoded.items():
            if key.startswith("_") or key == "series":
                continue

            # Check if there's an explicit field mapping from decode_model_code
            mapped_field = decoded.get(f"_field_{key}")
            target_field = mapped_field if mapped_field and mapped_field in _ALL_SPEC_FIELDS else key

            if target_field not in _ALL_SPEC_FIELDS:
                # Try alias lookup
                target_field = _FIELD_ALIASES.get(target_field.lower(), target_field)

            if target_field not in _ALL_SPEC_FIELDS:
                # Store in extra_specs instead of discarding
                if product.extra_specs is None:
                    product.extra_specs = {}
                if value is not None and value != "":
                    product.extra_specs[target_field] = str(value) if value else None
                continue

            # Only fill if currently None
            if getattr(product, target_field, None) is not None:
                continue

            # Apply with appropriate type conversion
            if target_field in _FLOAT_FIELDS:
                converted = self._safe_float(value)
                if converted is not None:
                    setattr(product, target_field, converted)
            elif target_field in _INT_FIELDS:
                converted = self._safe_int(value)
                if converted is not None:
                    setattr(product, target_field, converted)
            else:
                setattr(product, target_field, str(value) if value else None)

    @staticmethod
    def _safe_str(val) -> Optional[str]:
        if val is None:
            return None
        return str(val)

    @staticmethod
    def _safe_float(val) -> Optional[float]:
        if val is None:
            return None
        try:
            return float(val)
        except (ValueError, TypeError):
            return None

    @staticmethod
    def _safe_int(val) -> Optional[int]:
        if val is None:
            return None
        try:
            return int(float(val))
        except (ValueError, TypeError):
            return None

    # ── Cross-Reference PDF Processing ────────────────────────────────

    def process_cross_reference_pdf(
        self, pdf_path: str, metadata: UploadMetadata
    ) -> list[dict]:
        """Process a cross-reference PDF and return extracted mappings for review.

        Also indexes the document as guide text chunks for LLM context.
        Does NOT store cross-references yet — admin must confirm first.
        """
        # Extract text
        pages = extract_text_from_pdf(pdf_path)
        full_text = "\n\n".join(p["text"] for p in pages)

        if not full_text:
            return []

        # Index as guide chunks (for LLM context in KB Q&A and response generation)
        chunk_count = self.index_guide_text(pdf_path, metadata)
        logger.info("Cross-reference PDF indexed as %d guide chunks", chunk_count)

        # Extract structured cross-reference mappings
        cross_refs = extract_cross_references_with_llm(full_text, metadata.company)
        logger.info("Extracted %d cross-reference mappings from %s",
                     len(cross_refs), metadata.filename)

        return cross_refs

    def confirm_and_store_cross_references(
        self,
        cross_refs: list[dict],
        metadata: UploadMetadata,
    ) -> int:
        """Store confirmed cross-reference mappings in the database."""
        stored = 0
        for ref in cross_refs:
            try:
                self.db.insert_series_cross_reference(
                    my_company_series=ref.get("my_company_series", ""),
                    competitor_series=ref.get("competitor_series", ""),
                    competitor_company=ref.get("competitor_company", ""),
                    product_type=ref.get("product_type", ""),
                    notes=ref.get("notes", ""),
                    source_document=metadata.filename,
                    my_company_name=metadata.company,
                )
                stored += 1
            except Exception as e:
                logger.error("Error storing cross-reference: %s", e)
        logger.info("Stored %d cross-reference mappings from %s", stored, metadata.filename)
        return stored
