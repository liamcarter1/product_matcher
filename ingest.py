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
)
from storage.product_db import ProductDB
from storage.vector_store import VectorStore

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=1000,
    chunk_overlap=200,
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

        # Step 4: Extract model code patterns (from user guides)
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

        # Attach metadata to all products
        for product in extracted_products:
            if not product.category and metadata.category:
                product.category = metadata.category

        return extracted_products

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
        """Index user guide text chunks for supplementary search."""
        pages = extract_text_from_pdf(pdf_path)
        full_text = "\n\n".join(p["text"] for p in pages)

        if not full_text:
            return 0

        chunks = text_splitter.split_text(full_text)
        chunk_count = 0

        for i, chunk in enumerate(chunks):
            chunk_id = f"{metadata.filename}_chunk_{i}"
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
        """Convert an ExtractedProduct to a HydraulicProduct."""
        specs = ep.specs or {}

        product = HydraulicProduct(
            id=str(uuid.uuid4()),
            company=metadata.company,
            model_code=ep.model_code,
            product_name=ep.product_name or specs.pop("product_name", ""),
            category=ep.category or metadata.category or "",
            subcategory=specs.pop("subcategory", None),
            max_pressure_bar=self._safe_float(specs.pop("max_pressure_bar", None)),
            max_flow_lpm=self._safe_float(specs.pop("max_flow_lpm", None)),
            valve_size=specs.pop("valve_size", None),
            spool_type=specs.pop("spool_type", None),
            num_positions=self._safe_int(specs.pop("num_positions", None)),
            num_ports=self._safe_int(specs.pop("num_ports", None)),
            actuator_type=specs.pop("actuator_type", None),
            coil_voltage=specs.pop("coil_voltage", None),
            coil_type=specs.pop("coil_type", None),
            coil_connector=specs.pop("coil_connector", None),
            port_size=specs.pop("port_size", None),
            port_type=specs.pop("port_type", None),
            mounting=specs.pop("mounting", None),
            mounting_pattern=specs.pop("mounting_pattern", None),
            body_material=specs.pop("body_material", None),
            seal_material=specs.pop("seal_material", None),
            operating_temp_min_c=self._safe_float(specs.pop("operating_temp_min_c", None)),
            operating_temp_max_c=self._safe_float(specs.pop("operating_temp_max_c", None)),
            fluid_type=specs.pop("fluid_type", None),
            viscosity_range_cst=specs.pop("viscosity_range_cst", None),
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
        return product

    def _apply_decoded_specs(self, product: HydraulicProduct, decoded: dict):
        """Apply decoded model code specs to the product, filling in missing fields."""
        field_mapping = {
            "coil_voltage": "coil_voltage",
            "valve_size": "valve_size",
            "spool_type": "spool_type",
            "actuator_type": "actuator_type",
            "port_size": "port_size",
            "port_type": "port_type",
            "mounting": "mounting",
            "seal_material": "seal_material",
        }
        for decoded_key, product_field in field_mapping.items():
            if decoded_key in decoded and not getattr(product, product_field, None):
                setattr(product, product_field, decoded[decoded_key])

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
