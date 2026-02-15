"""
ProductMatchPro - PDF Ingestion Pipeline
Orchestrates: parse PDF → extract products → review → store in SQLite + ChromaDB.
"""

import uuid
from pathlib import Path
from typing import Optional
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
    generate_products_from_ordering_code,
    analyze_spool_functions,
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

        # Step 5: Ordering code combinatorial generation
        # Extract ordering code breakdown tables and generate ALL product variants
        if full_text:
            try:
                ordering_defs = extract_ordering_code_with_llm(
                    full_text, metadata.company, metadata.category or ""
                )
                for definition in ordering_defs:
                    generated = generate_products_from_ordering_code(definition, metadata)
                    if generated:
                        print(f"Generated {len(generated)} products from ordering code "
                              f"table for series: {definition.series}")
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
            except Exception as e:
                print(f"Error in ordering code extraction: {e}")

        # Step 5b: Deep spool function analysis (second-pass LLM)
        # Analyse the full user guide text to understand spool symbols and functions
        if full_text and metadata.document_type in (DocumentType.USER_GUIDE, DocumentType.DATASHEET):
            try:
                spool_results = analyze_spool_functions(full_text, metadata.company)
                if spool_results:
                    # Build lookup: spool_code -> structured data
                    spool_lookup = {}
                    for sr in spool_results:
                        code = sr.get("spool_code", "").strip()
                        if code:
                            spool_lookup[code.upper()] = sr
                            # Also store without leading zeros etc.
                            spool_lookup[code.upper().lstrip("0")] = sr

                    # Merge spool function data into matching extracted products
                    for product in extracted_products:
                        spool = product.specs.get("spool_type", "")
                        if not spool:
                            continue
                        spool_upper = str(spool).strip().upper()
                        matched_spool = spool_lookup.get(spool_upper)
                        if not matched_spool:
                            # Try partial match (e.g. product spool "2A" in "type 2A")
                            for code, data in spool_lookup.items():
                                if code in spool_upper or spool_upper in code:
                                    matched_spool = data
                                    break
                        if matched_spool:
                            if not product.specs.get("_spool_function"):
                                product.specs["_spool_function"] = {
                                    "center_condition": matched_spool.get("center_condition", ""),
                                    "solenoid_a_function": matched_spool.get("solenoid_a_function", ""),
                                    "solenoid_b_function": matched_spool.get("solenoid_b_function", ""),
                                    "description": matched_spool.get("description", ""),
                                    "canonical_pattern": matched_spool.get("canonical_pattern", ""),
                                }
                    print(f"Spool analysis: found {len(spool_results)} spool types, "
                          f"merged into products")
            except Exception as e:
                print(f"Error in spool function analysis: {e}")

        # Step 6: Deduplicate by model_code (keep the one with more specs)
        extracted_products = self._deduplicate_products(extracted_products)

        # Attach metadata to all products
        for product in extracted_products:
            if not product.category and metadata.category:
                product.category = metadata.category

        return extracted_products

    @staticmethod
    def _deduplicate_products(products: list[ExtractedProduct]) -> list[ExtractedProduct]:
        """Deduplicate by model_code, keeping the product with the most populated specs."""
        seen = {}
        for p in products:
            key = p.model_code.upper().strip()
            if key in seen:
                existing = seen[key]
                # Keep the one with more non-empty spec fields
                existing_count = sum(1 for v in existing.specs.values()
                                     if v is not None and v != "")
                new_count = sum(1 for v in p.specs.values()
                                if v is not None and v != "")
                if new_count > existing_count:
                    seen[key] = p
            else:
                seen[key] = p
        return list(seen.values())

    def confirm_and_store(
        self,
        extracted_products: list[ExtractedProduct],
        metadata: UploadMetadata,
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

        # Collect leftover specs into extra_specs (these didn't match any known field)
        leftover = {
            k: v for k, v in specs.items()
            if v is not None and v != ""
            and not k.startswith("_")
        }
        # Also collect _ prefixed structured data (e.g. _spool_function)
        structured = {
            k: v for k, v in specs.items()
            if k.startswith("_") and v is not None and v != ""
        }
        if leftover or structured:
            product.extra_specs = {**(product.extra_specs or {}), **leftover, **structured}

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
