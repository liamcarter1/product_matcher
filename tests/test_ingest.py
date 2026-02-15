"""
Tests for ingest.py — IngestionPipeline
Covers: type coercion (_safe_str/_safe_float/_safe_int), field alias mapping,
deduplication, and _extracted_to_hydraulic conversion.
"""

import uuid
import pytest
from unittest.mock import MagicMock

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
