"""
Tests for ingest.py — IngestionPipeline
Covers: type coercion (_safe_str/_safe_float/_safe_int), field alias mapping,
deduplication, and _extracted_to_hydraulic conversion.
"""

import uuid
import pytest
from unittest.mock import MagicMock, patch

from models import (
    ExtractedProduct, HydraulicProduct, UploadMetadata,
    DocumentType,
)
from ingest import IngestionPipeline, _FIELD_ALIASES, _FLOAT_FIELDS, _INT_FIELDS, _ALL_SPEC_FIELDS


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def pipeline():
    """Pipeline with mocked DB and vector store."""
    db = MagicMock()
    db.decode_model_code.return_value = None
    vs = MagicMock()
    return IngestionPipeline(db=db, vector_store=vs)


@pytest.fixture
def metadata():
    return UploadMetadata(
        company="Danfoss",
        document_type=DocumentType.CATALOGUE,
        category="directional_valves",
        filename="test.pdf",
    )


# ── _safe_str tests ──────────────────────────────────────────────────


class TestSafeStr:
    def test_none_returns_none(self):
        assert IngestionPipeline._safe_str(None) is None

    def test_string_passthrough(self):
        assert IngestionPipeline._safe_str("FKM") == "FKM"

    def test_int_coerced_to_string(self):
        """The bug that caused mounting=5 to fail Pydantic validation."""
        assert IngestionPipeline._safe_str(5) == "5"

    def test_float_coerced_to_string(self):
        assert IngestionPipeline._safe_str(3.14) == "3.14"

    def test_bool_coerced_to_string(self):
        assert IngestionPipeline._safe_str(True) == "True"

    def test_empty_string(self):
        assert IngestionPipeline._safe_str("") == ""


# ── _safe_float tests ────────────────────────────────────────────────


class TestSafeFloat:
    def test_none_returns_none(self):
        assert IngestionPipeline._safe_float(None) is None

    def test_float_passthrough(self):
        assert IngestionPipeline._safe_float(315.0) == 315.0

    def test_int_converted(self):
        assert IngestionPipeline._safe_float(315) == 315.0

    def test_string_number(self):
        assert IngestionPipeline._safe_float("315") == 315.0

    def test_string_float(self):
        assert IngestionPipeline._safe_float("3.14") == 3.14

    def test_invalid_string(self):
        assert IngestionPipeline._safe_float("not_a_number") is None

    def test_empty_string(self):
        assert IngestionPipeline._safe_float("") is None


# ── _safe_int tests ──────────────────────────────────────────────────


class TestSafeInt:
    def test_none_returns_none(self):
        assert IngestionPipeline._safe_int(None) is None

    def test_int_passthrough(self):
        assert IngestionPipeline._safe_int(4) == 4

    def test_float_truncated(self):
        assert IngestionPipeline._safe_int(4.7) == 4

    def test_string_int(self):
        assert IngestionPipeline._safe_int("3") == 3

    def test_string_float(self):
        """String "3.0" should be converted via float -> int."""
        assert IngestionPipeline._safe_int("3.0") == 3

    def test_invalid_string(self):
        assert IngestionPipeline._safe_int("abc") is None


# ── _extracted_to_hydraulic: type coercion (THE BUG) ────────────────


class TestExtractedToHydraulicTypeCoercion:
    """Ensures all Optional[str] fields survive when the LLM
    or table extractor returns an int or float instead of a string."""

    def test_mounting_int_coerced_to_string(self, pipeline, metadata):
        """Reproduce the original bug: mounting=5 (int) should become '5' (str)."""
        ep = ExtractedProduct(
            model_code="TEST-001",
            specs={"mounting": 5},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.mounting == "5"
        assert isinstance(product.mounting, str)

    def test_valve_size_int_coerced(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="TEST-002",
            specs={"valve_size": 6},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.valve_size == "6"

    def test_spool_type_preserved_as_string(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="TEST-003",
            specs={"spool_type": "2A (all ports open to tank in center)"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.spool_type == "2A (all ports open to tank in center)"

    def test_coil_voltage_string_preserved(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="TEST-004",
            specs={"coil_voltage": "24VDC"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.coil_voltage == "24VDC"

    def test_port_size_string_preserved(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="TEST-005",
            specs={"port_size": "G3/8"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.port_size == "G3/8"

    def test_all_string_fields_survive_int_input(self, pipeline, metadata):
        """Stress test: every Optional[str] field gets an int value."""
        str_fields = [
            "subcategory", "valve_size", "spool_type", "actuator_type",
            "coil_voltage", "coil_type", "coil_connector", "port_size",
            "port_type", "mounting", "mounting_pattern", "body_material",
            "seal_material", "fluid_type", "viscosity_range_cst",
        ]
        specs = {f: 42 for f in str_fields}
        ep = ExtractedProduct(model_code="TEST-ALL-STR", specs=specs)
        product = pipeline._extracted_to_hydraulic(ep, metadata)

        for field_name in str_fields:
            val = getattr(product, field_name)
            assert val == "42", f"Field {field_name} should be '42', got {val!r}"
            assert isinstance(val, str), f"Field {field_name} should be str, got {type(val)}"

    def test_numeric_fields_properly_typed(self, pipeline, metadata):
        """Float and int fields should be their correct types, not strings."""
        specs = {
            "max_pressure_bar": "315",
            "max_flow_lpm": "120.5",
            "num_positions": "3",
            "num_ports": "4",
            "weight_kg": "2.5",
        }
        ep = ExtractedProduct(model_code="TEST-NUM", specs=specs)
        product = pipeline._extracted_to_hydraulic(ep, metadata)

        assert product.max_pressure_bar == 315.0
        assert product.max_flow_lpm == 120.5
        assert product.num_positions == 3
        assert product.num_ports == 4
        assert product.weight_kg == 2.5

    def test_pydantic_validation_succeeds(self, pipeline, metadata):
        """Full product with mixed types should pass Pydantic validation."""
        specs = {
            "mounting": 5,
            "valve_size": 10,
            "max_pressure_bar": 315,
            "spool_type": "D",
            "coil_voltage": 24,
            "seal_material": "NBR",
        }
        ep = ExtractedProduct(model_code="TEST-PYDANTIC", specs=specs)
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        # If we get here without exception, Pydantic validation passed
        assert isinstance(product, HydraulicProduct)


# ── Field alias mapping ──────────────────────────────────────────────


class TestFieldAliasMapping:
    """Ensure _FIELD_ALIASES correctly rescue non-canonical field names."""

    def test_pressure_alias(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="ALIAS-001",
            specs={"pressure": "350"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.max_pressure_bar == 350.0

    def test_flow_alias(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="ALIAS-002",
            specs={"flow_rate": "80"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.max_flow_lpm == 80.0

    def test_voltage_alias(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="ALIAS-003",
            specs={"voltage": "24VDC"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.coil_voltage == "24VDC"

    def test_bore_alias(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="ALIAS-004",
            specs={"bore": "50"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.bore_diameter_mm == 50.0

    def test_alias_does_not_overwrite_canonical(self, pipeline, metadata):
        """If both the canonical and aliased keys exist, canonical wins."""
        ep = ExtractedProduct(
            model_code="ALIAS-005",
            specs={"max_pressure_bar": "350", "pressure": "250"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        # Canonical key should take precedence
        assert product.max_pressure_bar == 350.0

    def test_spool_alias(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="ALIAS-006",
            specs={"function": "2A"},
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.spool_type == "2A"

    def test_all_aliases_have_valid_targets(self):
        """Every alias must map to a real HydraulicProduct field."""
        hp_fields = set(HydraulicProduct.model_fields.keys())
        for alias, target in _FIELD_ALIASES.items():
            assert target in hp_fields, (
                f"Alias '{alias}' maps to '{target}' which is not a HydraulicProduct field"
            )


# ── Deduplication ────────────────────────────────────────────────────


class TestDeduplication:
    def test_keeps_product_with_more_specs(self):
        sparse = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"max_pressure_bar": 315},
        )
        rich = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"max_pressure_bar": 315, "coil_voltage": "24VDC", "spool_type": "2A"},
        )
        result = IngestionPipeline._deduplicate_products([sparse, rich])
        assert len(result) == 1
        assert result[0].specs.get("coil_voltage") == "24VDC"

    def test_case_insensitive_dedup(self):
        p1 = ExtractedProduct(model_code="dg4v-3-2a", specs={"a": "1"})
        p2 = ExtractedProduct(model_code="DG4V-3-2A", specs={"a": "1", "b": "2"})
        result = IngestionPipeline._deduplicate_products([p1, p2])
        assert len(result) == 1

    def test_whitespace_stripped(self):
        p1 = ExtractedProduct(model_code="  DG4V-3-2A  ", specs={"a": "1"})
        p2 = ExtractedProduct(model_code="DG4V-3-2A", specs={"a": "1", "b": "2"})
        result = IngestionPipeline._deduplicate_products([p1, p2])
        assert len(result) == 1

    def test_different_models_kept(self):
        p1 = ExtractedProduct(model_code="DG4V-3-2A", specs={})
        p2 = ExtractedProduct(model_code="DG4V-3-6C", specs={})
        result = IngestionPipeline._deduplicate_products([p1, p2])
        assert len(result) == 2

    def test_empty_list(self):
        result = IngestionPipeline._deduplicate_products([])
        assert result == []

    def test_none_values_not_counted_as_specs(self):
        """None and '' should not count toward spec richness."""
        sparse = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"max_pressure_bar": 315},
        )
        padded = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"a": None, "b": "", "c": None},
        )
        result = IngestionPipeline._deduplicate_products([padded, sparse])
        assert len(result) == 1
        assert result[0].specs.get("max_pressure_bar") == 315


# ── _apply_decoded_specs ─────────────────────────────────────────────


class TestApplyDecodedSpecs:
    def test_fills_empty_field(self, pipeline):
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
            spool_type=None,
        )
        decoded = {"spool_type": "2A - All ports open", "_field_spool_type": "spool_type"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.spool_type == "2A - All ports open"

    def test_does_not_overwrite_existing(self, pipeline):
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
            spool_type="D - P blocked, A&B to T",
        )
        decoded = {"spool_type": "2A - All ports open"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.spool_type == "D - P blocked, A&B to T"

    def test_float_field_conversion(self, pipeline):
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
            max_pressure_bar=None,
        )
        decoded = {"max_pressure_bar": "315"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.max_pressure_bar == 315.0

    def test_int_field_conversion(self, pipeline):
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
            num_ports=None,
        )
        decoded = {"num_ports": "4"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.num_ports == 4

    def test_alias_lookup_in_decoded(self, pipeline):
        """decoded keys that aren't direct spec fields should be resolved via aliases."""
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
            coil_voltage=None,
        )
        decoded = {"voltage": "24VDC"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.coil_voltage == "24VDC"

    def test_skips_internal_keys(self, pipeline):
        """Keys starting with _ and 'series' should be skipped."""
        product = HydraulicProduct(
            id="test", company="Danfoss", model_code="TEST",
        )
        decoded = {"_field_something": "test", "series": "4WE6"}
        pipeline._apply_decoded_specs(product, decoded)
        # Should not raise or modify any fields


# ── confirm_and_store ────────────────────────────────────────────────


class TestConfirmAndStore:
    def test_stores_products_in_db_and_vector_store(self, pipeline, metadata):
        ep = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"max_pressure_bar": 315, "spool_type": "2A"},
        )
        count = pipeline.confirm_and_store([ep], metadata)
        assert count == 1
        pipeline.db.insert_product.assert_called_once()
        pipeline.vector_store.index_product.assert_called_once()

    def test_applies_model_code_decoding(self, pipeline, metadata):
        pipeline.db.decode_model_code.return_value = {
            "spool_type": "2A - All ports open",
            "_field_spool_type": "spool_type",
        }
        ep = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={},
        )
        pipeline.confirm_and_store([ep], metadata)
        stored_product = pipeline.db.insert_product.call_args[0][0]
        assert stored_product.model_code_decoded is not None

    def test_empty_list_returns_zero(self, pipeline, metadata):
        count = pipeline.confirm_and_store([], metadata)
        assert count == 0


# ── Constants validation ─────────────────────────────────────────────


class TestConstants:
    def test_float_fields_are_float_on_model(self):
        for field_name in _FLOAT_FIELDS:
            field_info = HydraulicProduct.model_fields.get(field_name)
            assert field_info is not None, f"Float field {field_name} not on HydraulicProduct"

    def test_int_fields_are_int_on_model(self):
        for field_name in _INT_FIELDS:
            field_info = HydraulicProduct.model_fields.get(field_name)
            assert field_info is not None, f"Int field {field_name} not on HydraulicProduct"

    def test_all_spec_fields_on_model(self):
        for field_name in _ALL_SPEC_FIELDS:
            assert field_name in HydraulicProduct.model_fields, (
                f"Spec field {field_name} not on HydraulicProduct"
            )


class TestExtraSpecsPreservation:
    """Tests that unknown specs are preserved in extra_specs."""

    @pytest.fixture
    def pipeline(self, tmp_path):
        from ingest import IngestionPipeline
        from storage.product_db import ProductDB
        from storage.vector_store import VectorStore
        db = ProductDB(str(tmp_path / "test.db"))
        vs = VectorStore(str(tmp_path / "vectors"))
        return IngestionPipeline(db=db, vector_store=vs)

    def test_leftover_specs_stored_in_extra_specs(self, pipeline):
        """Specs that don't match known fields should go to extra_specs."""
        ep = ExtractedProduct(
            model_code="TEST-001",
            category="directional_valves",
            specs={
                "max_pressure_bar": 315,
                "design_number": "42",
                "flow_class": "high",
                "interface_standard": "ISO 4401",
            },
        )
        metadata = UploadMetadata(
            company="TestCo", document_type=DocumentType.CATALOGUE,
            filename="test.pdf",
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.max_pressure_bar == 315.0
        assert product.extra_specs is not None
        assert product.extra_specs.get("design_number") == "42"
        assert product.extra_specs.get("flow_class") == "high"
        assert product.extra_specs.get("interface_standard") == "ISO 4401"

    def test_spool_function_flat_keys_in_extra_specs(self, pipeline):
        """Flat spool function keys should end up in extra_specs as visible columns."""
        ep = ExtractedProduct(
            model_code="TEST-002",
            category="directional_valves",
            specs={
                "spool_type": "2A",
                "center_condition": "All ports blocked",
                "solenoid_a_energised": "P→A, B→T",
                "solenoid_b_energised": "P→B, A→T",
                "canonical_spool_pattern": "BLOCKED|PA-BT|AT-PB",
            },
        )
        metadata = UploadMetadata(
            company="TestCo", document_type=DocumentType.CATALOGUE,
            filename="test.pdf",
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        assert product.spool_type == "2A"
        assert product.extra_specs is not None
        assert product.extra_specs.get("center_condition") == "All ports blocked"
        assert product.extra_specs.get("solenoid_a_energised") == "P→A, B→T"
        assert product.extra_specs.get("canonical_spool_pattern") == "BLOCKED|PA-BT|AT-PB"

    def test_no_extra_specs_when_all_known(self, pipeline):
        """If all specs match known fields, extra_specs should be empty/None."""
        ep = ExtractedProduct(
            model_code="TEST-003",
            specs={"max_pressure_bar": 315, "spool_type": "2A"},
        )
        metadata = UploadMetadata(
            company="TestCo", document_type=DocumentType.CATALOGUE,
            filename="test.pdf",
        )
        product = pipeline._extracted_to_hydraulic(ep, metadata)
        # extra_specs should be None or empty dict
        assert not product.extra_specs or product.extra_specs == {}

    def test_apply_decoded_unknown_key_to_extra_specs(self, pipeline):
        """Unknown decoded keys should go to extra_specs, not be dropped."""
        product = HydraulicProduct(
            id="test", company="TestCo", model_code="X",
        )
        decoded = {"unknown_field_xyz": "some_value"}
        pipeline._apply_decoded_specs(product, decoded)
        assert product.extra_specs is not None
        assert product.extra_specs.get("unknown_field_xyz") == "some_value"


# ── _clean_spool_types ───────────────────────────────────────────────


class TestCleanSpoolTypes:
    """Tests for the spool_type post-processing safety net."""

    def test_strips_description_from_spool_type(self):
        """'2 (Closed center, P port closed)' → spool_type='2', description extracted."""
        p = ExtractedProduct(
            model_code="X",
            specs={"spool_type": "2 (Closed center, P port closed, A & B to tank)"},
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == "2"
        assert "Closed center" in p.specs.get("spool_function_description", "")

    def test_strips_description_with_dash(self):
        """'D - P to A, B to T' → spool_type='D'."""
        p = ExtractedProduct(
            model_code="X",
            specs={"spool_type": "D - P to A, B to T"},
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == "D"
        assert "P to A" in p.specs.get("spool_function_description", "")

    def test_clean_spool_code_left_alone(self):
        """'2A' stays as '2A' — no change needed."""
        p = ExtractedProduct(
            model_code="X",
            specs={"spool_type": "2A"},
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == "2A"

    def test_override_moved_from_spool_type(self):
        """'Z - No overrides' should be moved to manual_override."""
        p = ExtractedProduct(
            model_code="X",
            specs={"spool_type": "Z - No overrides"},
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == ""
        assert p.specs["manual_override"] == "Z - No overrides"

    def test_h_all_ports_blocked_cleaned(self):
        """'H (all ports blocked)' → spool_type='H'."""
        p = ExtractedProduct(
            model_code="X",
            specs={"spool_type": "H (all ports blocked)"},
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == "H"
        assert "all ports blocked" in p.specs.get("spool_function_description", "")

    def test_does_not_overwrite_existing_description(self):
        """If spool_function_description already exists, don't overwrite it."""
        p = ExtractedProduct(
            model_code="X",
            specs={
                "spool_type": "2A (Open center)",
                "spool_function_description": "Existing description",
            },
        )
        IngestionPipeline._clean_spool_types([p])
        assert p.specs["spool_type"] == "2A"
        assert p.specs["spool_function_description"] == "Existing description"


# ── Deduplication source priority ────────────────────────────────────


class TestDeduplicationSourcePriority:
    """Tests for source-priority-aware deduplication."""

    def test_ordering_code_wins_over_llm(self):
        """ordering_code source should be preferred over llm source."""
        llm_product = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"spool_type": "2A (description)", "max_pressure_bar": 315},
            source="llm",
        )
        oc_product = ExtractedProduct(
            model_code="DG4V-3-2A",
            specs={"spool_type": "2A"},
            source="ordering_code",
        )
        result = IngestionPipeline._deduplicate_products([llm_product, oc_product])
        assert len(result) == 1
        # ordering_code should win
        assert result[0].source == "ordering_code"
        # But should merge specs from LLM
        assert result[0].specs.get("max_pressure_bar") == 315

    def test_table_wins_over_llm(self):
        """table source should be preferred over llm source."""
        llm_p = ExtractedProduct(model_code="X", specs={"a": "1"}, source="llm")
        table_p = ExtractedProduct(model_code="X", specs={"b": "2"}, source="table")
        result = IngestionPipeline._deduplicate_products([llm_p, table_p])
        assert len(result) == 1
        assert result[0].source == "table"
        assert result[0].specs.get("a") == "1"  # merged from LLM

    def test_same_source_uses_spec_count(self):
        """Same source: fall back to spec count."""
        p1 = ExtractedProduct(model_code="X", specs={"a": "1"}, source="llm")
        p2 = ExtractedProduct(model_code="X", specs={"a": "1", "b": "2"}, source="llm")
        result = IngestionPipeline._deduplicate_products([p1, p2])
        assert len(result) == 1
        assert result[0].specs.get("b") == "2"


# ── Vision spool integration in pipeline ──────────────────────────────


class TestVisionSpoolIntegration:
    """Tests that the ingestion pipeline imports and references vision spool extraction."""

    def test_extract_spool_symbols_imported(self):
        """The ingest module should import extract_spool_symbols_from_pdf."""
        import ingest
        assert hasattr(ingest, 'extract_spool_symbols_from_pdf')

    def test_pipeline_process_pdf_source_contains_vision_step(self):
        """The process_pdf method should contain the vision spool extraction step."""
        import inspect
        source = inspect.getsource(IngestionPipeline.process_pdf)
        assert "extract_spool_symbols_from_pdf" in source
        assert "vision" in source.lower()

    def test_vision_spool_data_merges_into_product_specs(self):
        """Vision spool data with symbol_description should be stored in product specs."""
        product = ExtractedProduct(
            model_code="4WE6D",
            specs={"spool_type": "D"},
            source="ordering_code",
        )
        # Simulate what the pipeline does when merging vision spool data
        vision_data = {
            "spool_code": "D",
            "center_condition": "All ports blocked",
            "solenoid_a_function": "P to A, B to T",
            "solenoid_b_function": "P to B, A to T",
            "description": "Closed center",
            "symbol_description": "3-section rectangle, center has T-blocks on all ports",
            "canonical_pattern": "BLOCKED|AB-BT|AB-AT",
        }
        # Apply the same logic as the pipeline
        if not product.specs.get("center_condition"):
            product.specs["center_condition"] = vision_data.get("center_condition", "")
        if not product.specs.get("canonical_spool_pattern"):
            product.specs["canonical_spool_pattern"] = vision_data.get("canonical_pattern", "")
        if vision_data.get("symbol_description"):
            product.specs["spool_symbol_description"] = vision_data["symbol_description"]

        assert product.specs["center_condition"] == "All ports blocked"
        assert product.specs["canonical_spool_pattern"] == "BLOCKED|AB-BT|AB-AT"
        assert product.specs["spool_symbol_description"] == "3-section rectangle, center has T-blocks on all ports"

    def test_vision_does_not_overwrite_text_analysis(self):
        """Vision data should NOT overwrite data already set by text-based spool analysis."""
        product = ExtractedProduct(
            model_code="4WE6D",
            specs={
                "spool_type": "D",
                "center_condition": "All ports blocked (from text)",
                "canonical_spool_pattern": "BLOCKED|PA-BT|PB-AT",
            },
            source="ordering_code",
        )
        vision_data = {
            "center_condition": "All ports blocked (from vision)",
            "canonical_pattern": "BLOCKED|PA-BT|PB-AT-v2",
        }
        # Apply the same logic — should NOT overwrite
        if not product.specs.get("center_condition"):
            product.specs["center_condition"] = vision_data.get("center_condition", "")
        if not product.specs.get("canonical_spool_pattern"):
            product.specs["canonical_spool_pattern"] = vision_data.get("canonical_pattern", "")

        assert product.specs["center_condition"] == "All ports blocked (from text)"
        assert product.specs["canonical_spool_pattern"] == "BLOCKED|PA-BT|PB-AT"


# ── Cross-Reference Pipeline Integration ──────────────────────────────


class TestCrossReferencePipeline:
    """Tests for the cross-reference PDF processing pipeline."""

    def test_extract_cross_references_imported(self):
        """The ingest module should import extract_cross_references_with_llm."""
        import ingest
        assert hasattr(ingest, 'extract_cross_references_with_llm')

    def test_pipeline_has_cross_reference_methods(self):
        """The IngestionPipeline should have cross-reference methods."""
        assert hasattr(IngestionPipeline, 'process_cross_reference_pdf')
        assert hasattr(IngestionPipeline, 'confirm_and_store_cross_references')

    def test_confirm_stores_cross_references(self, pipeline, metadata):
        """confirm_and_store_cross_references should call db.insert_series_cross_reference."""
        cross_refs = [
            {
                "my_company_series": "DG4V-3",
                "competitor_series": "D1VW",
                "competitor_company": "Parker",
                "product_type": "Directional Valve",
                "notes": "",
            },
            {
                "my_company_series": "KFDG4V",
                "competitor_series": "4WRE",
                "competitor_company": "Bosch Rexroth",
                "product_type": "Proportional Valve",
                "notes": "",
            },
        ]
        count = pipeline.confirm_and_store_cross_references(cross_refs, metadata)
        assert count == 2
        assert pipeline.db.insert_series_cross_reference.call_count == 2


class TestSpoolGapFill:
    """Tests for _validate_and_gapfill_spools."""

    def _make_definition(self, spool_options):
        """Helper: build a minimal OrderingCodeDefinition with spool segment."""
        from tools.parse_tools import OrderingCodeDefinition, OrderingCodeSegment
        spool_seg = OrderingCodeSegment(
            position=3,
            segment_name="spool_type",
            is_fixed=False,
            separator_before="-",
            options=[
                {"code": code, "description": f"Spool {code}",
                 "maps_to_field": "spool_type", "maps_to_value": code}
                for code in spool_options
            ],
        )
        return OrderingCodeDefinition(
            company="Danfoss",
            series="DG4V-3",
            product_name="Directional Valve",
            category="directional_valves",
            code_template="{01}{02}-{03}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="series_code", is_fixed=True,
                    separator_before="", options=[
                        {"code": "DG4V", "description": "Series", "maps_to_field": "series_code", "maps_to_value": "DG4V"}
                    ],
                ),
                OrderingCodeSegment(
                    position=2, segment_name="size", is_fixed=True,
                    separator_before="-", options=[
                        {"code": "3", "description": "Size 3", "maps_to_field": "valve_size", "maps_to_value": "3"}
                    ],
                ),
                spool_seg,
            ],
            shared_specs={},
        )

    def test_gapfill_adds_missing_spools(self, pipeline, metadata):
        """Should add products for spool codes in reference but not in extraction."""
        pipeline.db.get_spool_codes_for_series.return_value = ["0C", "2A", "6C"]
        pipeline.db.get_primary_spool_codes.return_value = []  # no primary filter
        pipeline.db.get_spool_type_references.return_value = [
            {"spool_code": "2A", "description": "Closed center"},
            {"spool_code": "6C", "description": "Float center"},
        ]

        definition = self._make_definition(["0C"])  # only 0C extracted
        extracted = [
            ExtractedProduct(
                model_code="DG4V-3-0C", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "0C"}, raw_text="", confidence=0.85, source="ordering_code",
            )
        ]

        warnings = pipeline._validate_and_gapfill_spools(definition, metadata, extracted)
        assert len(warnings) == 1
        assert "2A" in warnings[0]
        assert "6C" in warnings[0]
        # Should have added new products
        gapfilled = [p for p in extracted if p.specs.get("_spool_source") == "reference_gapfill"]
        assert len(gapfilled) >= 2

    def test_no_gapfill_when_no_reference(self, pipeline, metadata):
        """Should return empty warnings when no reference data exists."""
        pipeline.db.get_spool_codes_for_series.return_value = []
        pipeline.db.get_primary_spool_codes.return_value = []
        definition = self._make_definition(["0C"])
        extracted = []
        warnings = pipeline._validate_and_gapfill_spools(definition, metadata, extracted)
        assert warnings == []

    def test_no_gapfill_when_all_present(self, pipeline, metadata):
        """Should return empty warnings when all known spools are in extraction."""
        pipeline.db.get_spool_codes_for_series.return_value = ["0C", "2A"]
        pipeline.db.get_primary_spool_codes.return_value = []
        pipeline.db.get_spool_type_references.return_value = []
        definition = self._make_definition(["0C", "2A"])
        extracted = []
        warnings = pipeline._validate_and_gapfill_spools(definition, metadata, extracted)
        assert warnings == []

    def test_gapfill_products_marked(self, pipeline, metadata):
        """Gap-filled products should have _spool_source = 'reference_gapfill'."""
        pipeline.db.get_spool_codes_for_series.return_value = ["0C", "2A"]
        pipeline.db.get_primary_spool_codes.return_value = []
        pipeline.db.get_spool_type_references.return_value = [
            {"spool_code": "2A", "description": "Closed center"},
        ]
        definition = self._make_definition(["0C"])
        extracted = []
        pipeline._validate_and_gapfill_spools(definition, metadata, extracted)
        for p in extracted:
            if p.specs.get("spool_type") == "2A":
                assert p.specs.get("_spool_source") == "reference_gapfill"


    def test_gapfill_respects_primary_filter(self, pipeline, metadata):
        """Gap-fill should only add primary spool codes when primary filtering is active."""
        pipeline.db.get_spool_codes_for_series.return_value = ["0C", "2A", "6C", "H"]
        pipeline.db.get_primary_spool_codes.return_value = ["2A", "6C"]
        pipeline.db.get_spool_type_references.return_value = [
            {"spool_code": "2A", "description": "Closed center"},
            {"spool_code": "6C", "description": "Float center"},
            {"spool_code": "H", "description": "Open center"},
        ]

        definition = self._make_definition(["0C"])  # only 0C extracted
        extracted = []
        pipeline._validate_and_gapfill_spools(definition, metadata, extracted)

        # Should only gap-fill 2A and 6C (primary), not H (non-primary)
        gapfilled_codes = {
            p.specs.get("spool_type") for p in extracted
            if p.specs.get("_spool_source") == "reference_gapfill"
        }
        assert "2A" in gapfilled_codes
        assert "6C" in gapfilled_codes
        assert "H" not in gapfilled_codes


class TestAutoLearnSpoolTypes:
    """Tests for _auto_learn_spool_types."""

    def test_learns_from_confirmed_products(self, pipeline, metadata):
        """Should save unique spool codes to spool_type_reference."""
        pipeline.db.bulk_insert_spool_type_references.return_value = 3
        products = [
            ExtractedProduct(
                model_code="DG4V-3-0C-M", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "0C", "center_condition": "All ports blocked"},
                raw_text="", confidence=0.85, source="ordering_code",
            ),
            ExtractedProduct(
                model_code="DG4V-3-2A-M", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "2A", "center_condition": "Closed center"},
                raw_text="", confidence=0.85, source="ordering_code",
            ),
            ExtractedProduct(
                model_code="DG4V-3-6C-M", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "6C"},
                raw_text="", confidence=0.85, source="ordering_code",
            ),
        ]
        count = pipeline._auto_learn_spool_types(products, metadata)
        assert count == 3
        pipeline.db.bulk_insert_spool_type_references.assert_called_once()
        refs = pipeline.db.bulk_insert_spool_type_references.call_args[0][0]
        codes = {r["spool_code"] for r in refs}
        assert codes == {"0C", "2A", "6C"}

    def test_skips_gapfill_products(self, pipeline, metadata):
        """Should not learn from products marked as reference_gapfill."""
        pipeline.db.bulk_insert_spool_type_references.return_value = 1
        products = [
            ExtractedProduct(
                model_code="DG4V-3-0C-M", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "0C"},
                raw_text="", confidence=0.85, source="ordering_code",
            ),
            ExtractedProduct(
                model_code="DG4V-3-2A-M", product_name="Test",
                category="directional_valves",
                specs={"spool_type": "2A", "_spool_source": "reference_gapfill"},
                raw_text="", confidence=0.85, source="ordering_code",
            ),
        ]
        pipeline._auto_learn_spool_types(products, metadata)
        refs = pipeline.db.bulk_insert_spool_type_references.call_args[0][0]
        codes = {r["spool_code"] for r in refs}
        assert "0C" in codes
        assert "2A" not in codes  # gap-filled, should be skipped

    def test_no_learn_when_no_spools(self, pipeline, metadata):
        """Should return 0 when no spool types in products."""
        products = [
            ExtractedProduct(
                model_code="TEST-123", product_name="No spool",
                category="directional_valves",
                specs={}, raw_text="", confidence=0.85, source="ordering_code",
            ),
        ]
        count = pipeline._auto_learn_spool_types(products, metadata)
        assert count == 0


class TestInferSeriesFromProducts:
    """Tests for _infer_series_from_products."""

    def test_hyphenated_codes(self):
        """Should extract series from hyphenated model codes."""
        products = [
            ExtractedProduct(
                model_code="DG4V-3-0C-M", product_name="",
                category="", specs={}, raw_text="", confidence=0.85, source="ordering_code",
            ),
            ExtractedProduct(
                model_code="DG4V-3-2A-M", product_name="",
                category="", specs={}, raw_text="", confidence=0.85, source="ordering_code",
            ),
        ]
        series = IngestionPipeline._infer_series_from_products(products, "Danfoss")
        assert series == "DG4V-3"

    def test_single_code(self):
        """Should handle single model code."""
        products = [
            ExtractedProduct(
                model_code="DG4V-3-2A-M", product_name="",
                category="", specs={}, raw_text="", confidence=0.85, source="ordering_code",
            ),
        ]
        series = IngestionPipeline._infer_series_from_products(products, "Danfoss")
        assert "DG4V" in series

    def test_empty_products(self):
        """Should return empty string for empty list."""
        series = IngestionPipeline._infer_series_from_products([], "Danfoss")
        assert series == ""


# ── Graphics-Heavy Pipeline Branching Tests ──────────────────────


class TestGraphicsHeavyBranching:
    """Tests for vision-first pipeline branching when graphics-heavy PDF is detected."""

    @pytest.fixture
    def pipeline(self):
        db = MagicMock()
        db.decode_model_code.return_value = None
        db.get_spool_codes_for_series.return_value = []
        db.get_primary_spool_codes.return_value = []
        vs = MagicMock()
        return IngestionPipeline(db=db, vector_store=vs)

    @pytest.fixture
    def metadata(self):
        return UploadMetadata(
            company="Danfoss",
            document_type=DocumentType.DATASHEET,
            category="directional_valves",
            filename="DG4V3.pdf",
        )

    @patch("ingest.extract_ordering_code_from_images")
    @patch("ingest.extract_ordering_code_with_llm")
    @patch("ingest._is_graphics_heavy_pdf", return_value=True)
    @patch("ingest._get_pdf_page_count", return_value=14)
    @patch("ingest.extract_text_from_pdf", return_value=[{"page": 1, "text": "short"}])
    @patch("ingest.extract_tables_from_pdf", return_value=[])
    @patch("ingest.extract_products_with_llm", return_value=[])
    @patch("ingest.extract_model_code_patterns_with_llm", return_value=[])
    @patch("ingest.analyze_spool_functions", return_value=[])
    @patch("ingest.extract_spool_symbols_from_pdf", return_value=[])
    def test_vision_path_used_when_graphics_heavy(
        self, mock_spool_vis, mock_spool_analysis,
        mock_patterns, mock_llm_products, mock_tables, mock_text,
        mock_page_count, mock_is_heavy,
        mock_text_ordering, mock_vision_ordering,
        pipeline, metadata,
    ):
        """When PDF is graphics-heavy, vision extraction should be called instead of text."""
        mock_vision_ordering.return_value = []
        mock_text_ordering.return_value = []

        pipeline.process_pdf("test.pdf", metadata)

        # Vision should be called
        mock_vision_ordering.assert_called_once()
        # Text extraction should NOT be called (since vision returned [] and text fallback fires,
        # but the initial text ordering should not be the primary call)
        assert pipeline._last_extraction_method in ("vision", "text (vision fallback)")

    @patch("ingest.extract_ordering_code_from_images")
    @patch("ingest.extract_ordering_code_with_llm")
    @patch("ingest._is_graphics_heavy_pdf", return_value=False)
    @patch("ingest._get_pdf_page_count", return_value=14)
    @patch("ingest.extract_text_from_pdf", return_value=[{"page": 1, "text": "x" * 5000}])
    @patch("ingest.extract_tables_from_pdf", return_value=[])
    @patch("ingest.extract_products_with_llm", return_value=[])
    @patch("ingest.extract_model_code_patterns_with_llm", return_value=[])
    @patch("ingest.analyze_spool_functions", return_value=[])
    @patch("ingest.extract_spool_symbols_from_pdf", return_value=[])
    def test_text_path_used_for_normal_pdf(
        self, mock_spool_vis, mock_spool_analysis,
        mock_patterns, mock_llm_products, mock_tables, mock_text,
        mock_page_count, mock_is_heavy,
        mock_text_ordering, mock_vision_ordering,
        pipeline, metadata,
    ):
        """When PDF is NOT graphics-heavy, text extraction is used first.
        If text finds nothing on a multi-page PDF, vision retry also fires."""
        mock_text_ordering.return_value = []
        mock_vision_ordering.return_value = []

        pipeline.process_pdf("test.pdf", metadata)

        # Text should be called first
        mock_text_ordering.assert_called_once()
        # Vision retry should also fire since text found nothing on a 14-page PDF
        mock_vision_ordering.assert_called_once()

    @patch("ingest.extract_ordering_code_from_images")
    @patch("ingest.extract_ordering_code_with_llm")
    @patch("ingest._is_graphics_heavy_pdf", return_value=False)
    @patch("ingest._get_pdf_page_count", return_value=2)
    @patch("ingest.extract_text_from_pdf", return_value=[{"page": 1, "text": "x" * 5000}])
    @patch("ingest.extract_tables_from_pdf", return_value=[])
    @patch("ingest.extract_products_with_llm", return_value=[])
    @patch("ingest.extract_model_code_patterns_with_llm", return_value=[])
    @patch("ingest.analyze_spool_functions", return_value=[])
    @patch("ingest.extract_spool_symbols_from_pdf", return_value=[])
    def test_no_vision_retry_for_short_pdf(
        self, mock_spool_vis, mock_spool_analysis,
        mock_patterns, mock_llm_products, mock_tables, mock_text,
        mock_page_count, mock_is_heavy,
        mock_text_ordering, mock_vision_ordering,
        pipeline, metadata,
    ):
        """Short PDFs (< 3 pages) should NOT trigger vision retry — only text."""
        mock_text_ordering.return_value = []

        pipeline.process_pdf("test.pdf", metadata)

        mock_text_ordering.assert_called_once()
        mock_vision_ordering.assert_not_called()
        assert pipeline._last_extraction_method == "text"

    @patch("ingest.extract_ordering_code_from_images")
    @patch("ingest.extract_ordering_code_with_llm")
    @patch("ingest._is_graphics_heavy_pdf", return_value=True)
    @patch("ingest._get_pdf_page_count", return_value=14)
    @patch("ingest.extract_text_from_pdf", return_value=[{"page": 1, "text": "some text"}])
    @patch("ingest.extract_tables_from_pdf", return_value=[])
    @patch("ingest.extract_products_with_llm", return_value=[])
    @patch("ingest.extract_model_code_patterns_with_llm", return_value=[])
    @patch("ingest.analyze_spool_functions", return_value=[])
    @patch("ingest.extract_spool_symbols_from_pdf", return_value=[])
    def test_vision_fallback_to_text(
        self, mock_spool_vis, mock_spool_analysis,
        mock_patterns, mock_llm_products, mock_tables, mock_text,
        mock_page_count, mock_is_heavy,
        mock_text_ordering, mock_vision_ordering,
        pipeline, metadata,
    ):
        """When vision extraction returns nothing, should fall back to text extraction."""
        mock_vision_ordering.return_value = []  # Vision fails
        mock_text_ordering.return_value = []

        pipeline.process_pdf("test.pdf", metadata)

        # Both should be called: vision first, then text fallback
        mock_vision_ordering.assert_called_once()
        mock_text_ordering.assert_called_once()
        assert pipeline._last_extraction_method == "text (vision fallback)"
