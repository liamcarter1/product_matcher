"""
Tests for tools/parse_tools.py — PDF parsing and extraction utilities.
Covers: _parse_numeric_if_possible, assemble_model_code, generate_products_from_ordering_code,
table_rows_to_products, HEADER_MAPPINGS, LLM prompt validation, and vision pipeline.
"""

import json
import pytest
from unittest.mock import patch, MagicMock

from models import (
    ExtractedProduct, UploadMetadata, DocumentType,
    OrderingCodeSegment, OrderingCodeDefinition,
)
from tools.parse_tools import (
    _parse_numeric_if_possible,
    assemble_model_code,
    generate_products_from_ordering_code,
    table_rows_to_products,
    compute_canonical_pattern,
    extract_spool_symbols_from_pdf,
    extract_spool_symbols_from_pdf_v2,
    extract_cross_references_with_llm,
    extract_ordering_code_from_images,
    _is_graphics_heavy_pdf,
    _get_pdf_page_count,
    _parse_ordering_code_response,
    _classify_spool_pages,
    _merge_spool_results,
    _deduplicate_spools,
    HEADER_MAPPINGS,
    _VALID_SPEC_FIELDS,
    MAX_COMBINATIONS,
)


# ── _parse_numeric_if_possible ───────────────────────────────────────


class TestParseNumericIfPossible:
    """This function is responsible for the mounting=5 bug when it converts
    pure numeric strings to int, which then gets passed to an Optional[str] field."""

    def test_pure_integer(self):
        assert _parse_numeric_if_possible("315") == 315
        assert isinstance(_parse_numeric_if_possible("315"), int)

    def test_pure_float(self):
        assert _parse_numeric_if_possible("3.14") == 3.14
        assert isinstance(_parse_numeric_if_possible("3.14"), float)

    def test_negative_number(self):
        result = _parse_numeric_if_possible("-20")
        assert result == -20

    def test_voltage_preserved_as_string(self):
        """'24VDC' has letters mixed with digits — must stay as string."""
        assert _parse_numeric_if_possible("24VDC") == "24VDC"

    def test_port_size_preserved(self):
        assert _parse_numeric_if_possible("G3/8") == "G3/8"

    def test_seal_material_preserved(self):
        assert _parse_numeric_if_possible("NBR") == "NBR"
        assert _parse_numeric_if_possible("FKM") == "FKM"

    def test_iso_standard_preserved(self):
        assert _parse_numeric_if_possible("ISO 4401-03") == "ISO 4401-03"

    def test_unit_suffix_bar(self):
        result = _parse_numeric_if_possible("315 bar")
        assert result == 315

    def test_unit_suffix_lpm(self):
        result = _parse_numeric_if_possible("120 lpm")
        assert result == 120

    def test_unit_suffix_mm(self):
        result = _parse_numeric_if_possible("50.5 mm")
        assert result == 50.5

    def test_unit_suffix_kg(self):
        result = _parse_numeric_if_possible("2.5 kg")
        assert result == 2.5

    def test_empty_string(self):
        assert _parse_numeric_if_possible("") == ""

    def test_none_passthrough(self):
        """None input should return None without error."""
        # The function checks `if not value` which is True for None
        assert _parse_numeric_if_possible(None) is None

    def test_whitespace_stripped(self):
        assert _parse_numeric_if_possible("  315  ") == 315

    def test_cetop_size_preserved(self):
        assert _parse_numeric_if_possible("CETOP 3") == "CETOP 3"

    def test_spool_code_preserved(self):
        """Spool codes like '2A', '33C' must NOT be parsed as numbers."""
        assert _parse_numeric_if_possible("2A") == "2A"
        assert _parse_numeric_if_possible("33C") == "33C"
        assert _parse_numeric_if_possible("6C") == "6C"


# ── assemble_model_code ──────────────────────────────────────────────


class TestAssembleModelCode:
    def test_basic_assembly(self):
        template = "{01}{02}{03}"
        values = {1: "4WE", 2: "6", 3: "D"}
        result = assemble_model_code(template, values)
        assert result == "4WE6D"

    def test_with_separators(self):
        template = "{01}{02}-{03}/{04}"
        values = {1: "4WE", 2: "6", 3: "EG24", 4: "K4"}
        result = assemble_model_code(template, values)
        assert result == "4WE6-EG24/K4"

    def test_empty_segment_cleaned(self):
        """Empty segments should not leave double separators."""
        template = "{01}-{02}-{03}"
        values = {1: "4WE", 2: "", 3: "K4"}
        result = assemble_model_code(template, values)
        assert result == "4WE-K4"  # Double dash cleaned

    def test_trailing_separator_stripped(self):
        template = "{01}-{02}/"
        values = {1: "4WE", 2: "6"}
        result = assemble_model_code(template, values)
        assert "4WE-6" in result
        assert not result.endswith("/")

    def test_all_empty_segments(self):
        template = "{01}{02}{03}"
        values = {1: "", 2: "", 3: ""}
        result = assemble_model_code(template, values)
        assert result == ""

    def test_danfoss_style_code(self):
        """Realistic Danfoss/Vickers model code assembly."""
        template = "{01}{02}{03}{04}{05}{06}{07}"
        values = {
            1: "D1VW",
            2: "001",
            3: "C",  # spool type
            4: "N",
            5: "J",
            6: "W",
            7: "",
        }
        result = assemble_model_code(template, values)
        assert result == "D1VW001CNJW"

    def test_bosch_rexroth_style_code(self):
        """Realistic Bosch Rexroth model code assembly."""
        template = "{01}{02}{03}{04}-{05}{06}/{07}{08}{09}"
        values = {
            1: "4",
            2: "WE",
            3: "6",
            4: "D",  # spool type
            5: "6",
            6: "X",
            7: "EG",
            8: "24",
            9: "N9K4",
        }
        result = assemble_model_code(template, values)
        assert result == "4WE6D-6X/EG24N9K4"


# ── generate_products_from_ordering_code ─────────────────────────────


class TestGenerateProductsFromOrderingCode:
    @pytest.fixture
    def metadata(self):
        return UploadMetadata(
            company="Bosch Rexroth",
            document_type=DocumentType.DATASHEET,
            category="directional_valves",
            filename="4WE6_datasheet.pdf",
        )

    def test_basic_generation(self, metadata):
        definition = OrderingCodeDefinition(
            company="Bosch Rexroth",
            series="4WE6",
            product_name="Directional Control Valve",
            category="directional_valves",
            code_template="{01}-{02}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="series", is_fixed=True,
                    options=[{"code": "4WE6", "maps_to_field": "", "maps_to_value": None}],
                ),
                OrderingCodeSegment(
                    position=2, segment_name="spool_type", is_fixed=False,
                    options=[
                        {"code": "D", "description": "P to A, B to T",
                         "maps_to_field": "spool_type", "maps_to_value": "D - P to A, B to T"},
                        {"code": "E", "description": "P&T blocked, A&B open",
                         "maps_to_field": "spool_type", "maps_to_value": "E - P&T blocked, A&B open"},
                    ],
                ),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) == 2
        codes = {p.model_code for p in products}
        assert "4WE6-D" in codes
        assert "4WE6-E" in codes

    def test_spool_type_in_specs(self, metadata):
        """Spool type should appear in the product specs with ONLY the code."""
        definition = OrderingCodeDefinition(
            company="Bosch Rexroth",
            series="4WE6",
            code_template="{01}-{02}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="series", is_fixed=True,
                    options=[{"code": "4WE6", "maps_to_field": "", "maps_to_value": None}],
                ),
                OrderingCodeSegment(
                    position=2, segment_name="spool_type", is_fixed=False,
                    options=[
                        {"code": "D", "maps_to_field": "spool_type",
                         "maps_to_value": "D", "description": "P to A, B to T"},
                    ],
                ),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) == 1
        assert products[0].specs.get("spool_type") == "D"

    def test_shared_specs_applied(self, metadata):
        definition = OrderingCodeDefinition(
            company="Bosch Rexroth",
            series="4WE6",
            code_template="{01}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="series", is_fixed=True,
                    options=[{"code": "4WE6", "maps_to_field": "", "maps_to_value": None}],
                ),
            ],
            shared_specs={"max_pressure_bar": 315, "num_ports": 4},
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) == 1
        assert products[0].specs["max_pressure_bar"] == 315
        assert products[0].specs["num_ports"] == 4

    def test_combinatorial_explosion_capped(self, metadata):
        """More than MAX_COMBINATIONS should be capped."""
        many_options = [{"code": str(i), "maps_to_field": "", "maps_to_value": None}
                        for i in range(100)]
        definition = OrderingCodeDefinition(
            company="Test",
            series="TEST",
            code_template="{01}-{02}-{03}",
            segments=[
                OrderingCodeSegment(position=1, segment_name="a", is_fixed=False, options=many_options),
                OrderingCodeSegment(position=2, segment_name="b", is_fixed=False, options=many_options),
                OrderingCodeSegment(position=3, segment_name="c", is_fixed=False, options=many_options),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) <= MAX_COMBINATIONS

    def test_empty_segments_returns_empty(self, metadata):
        definition = OrderingCodeDefinition(
            company="Test", series="TEST", segments=[],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert products == []

    def test_confidence_and_source(self, metadata):
        definition = OrderingCodeDefinition(
            company="Test", series="TEST",
            code_template="{01}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="s", is_fixed=True,
                    options=[{"code": "X", "maps_to_field": "", "maps_to_value": None}],
                ),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert products[0].confidence == 0.85
        assert products[0].source == "ordering_code"

    def test_invalid_maps_to_field_ignored(self, metadata):
        """Fields not in _VALID_SPEC_FIELDS should be silently ignored."""
        definition = OrderingCodeDefinition(
            company="Test", series="TEST",
            code_template="{01}",
            segments=[
                OrderingCodeSegment(
                    position=1, segment_name="bogus", is_fixed=True,
                    options=[{"code": "X", "maps_to_field": "nonexistent_field",
                              "maps_to_value": "ignored"}],
                ),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) == 1
        # After Step 4: unknown field names are now KEPT (stored in extra_specs later)
        assert "nonexistent_field" in products[0].specs
        assert products[0].specs["nonexistent_field"] == "ignored"

    def test_multi_variable_segments(self, metadata):
        """Two variable segments should produce cartesian product."""
        definition = OrderingCodeDefinition(
            company="Test", series="T",
            code_template="{01}-{02}-{03}",
            segments=[
                OrderingCodeSegment(position=1, segment_name="s", is_fixed=True,
                                    options=[{"code": "T", "maps_to_field": "", "maps_to_value": None}]),
                OrderingCodeSegment(position=2, segment_name="spool_type", is_fixed=False,
                                    options=[
                                        {"code": "D", "maps_to_field": "spool_type", "maps_to_value": "D"},
                                        {"code": "E", "maps_to_field": "spool_type", "maps_to_value": "E"},
                                    ]),
                OrderingCodeSegment(position=3, segment_name="seal", is_fixed=False,
                                    options=[
                                        {"code": "V", "maps_to_field": "seal_material", "maps_to_value": "FKM"},
                                        {"code": "M", "maps_to_field": "seal_material", "maps_to_value": "NBR"},
                                    ]),
            ],
        )
        products = generate_products_from_ordering_code(definition, metadata)
        assert len(products) == 4  # 2 spool * 2 seal
        codes = {p.model_code for p in products}
        assert "T-D-V" in codes
        assert "T-D-M" in codes
        assert "T-E-V" in codes
        assert "T-E-M" in codes


class TestPrimarySpoolFilter:
    """Tests for primary_spool_codes parameter in generate_products_from_ordering_code."""

    @pytest.fixture
    def metadata(self):
        return UploadMetadata(
            company="Test", document_type=DocumentType.DATASHEET,
            category="directional_valves", filename="test.pdf",
        )

    def _make_definition(self, spool_codes, extra_segments=None):
        segments = [
            OrderingCodeSegment(position=1, segment_name="series", is_fixed=True,
                                options=[{"code": "T", "maps_to_field": "", "maps_to_value": None}]),
            OrderingCodeSegment(position=2, segment_name="spool_type", is_fixed=False,
                                options=[
                                    {"code": c, "maps_to_field": "spool_type", "maps_to_value": c}
                                    for c in spool_codes
                                ]),
        ]
        if extra_segments:
            segments.extend(extra_segments)
        return OrderingCodeDefinition(
            company="Test", series="T", code_template="{01}-{02}",
            segments=segments,
        )

    def test_primary_filter_reduces_products(self, metadata):
        """When primary codes set, only those spools generate products."""
        defn = self._make_definition(["D", "E", "H", "J"])
        products = generate_products_from_ordering_code(
            defn, metadata, primary_spool_codes=["D", "H"],
        )
        assert len(products) == 2
        codes = {p.model_code for p in products}
        assert "T-D" in codes
        assert "T-H" in codes
        assert "T-E" not in codes

    def test_none_means_all(self, metadata):
        """primary_spool_codes=None should generate all products."""
        defn = self._make_definition(["D", "E"])
        products = generate_products_from_ordering_code(defn, metadata, primary_spool_codes=None)
        assert len(products) == 2

    def test_empty_list_means_all(self, metadata):
        """primary_spool_codes=[] should generate all products."""
        defn = self._make_definition(["D", "E"])
        products = generate_products_from_ordering_code(defn, metadata, primary_spool_codes=[])
        assert len(products) == 2

    def test_no_matching_codes_keeps_all(self, metadata):
        """If no primary codes match any option, keep all options."""
        defn = self._make_definition(["D", "E"])
        products = generate_products_from_ordering_code(
            defn, metadata, primary_spool_codes=["NONEXISTENT"],
        )
        assert len(products) == 2

    def test_only_affects_spool_segment(self, metadata):
        """Primary filter should not affect non-spool segments."""
        defn = self._make_definition(
            ["D", "E", "H"],
            extra_segments=[
                OrderingCodeSegment(position=3, segment_name="seal", is_fixed=False,
                                    options=[
                                        {"code": "V", "maps_to_field": "seal_material", "maps_to_value": "FKM"},
                                        {"code": "M", "maps_to_field": "seal_material", "maps_to_value": "NBR"},
                                    ]),
            ],
        )
        defn.code_template = "{01}-{02}-{03}"
        products = generate_products_from_ordering_code(
            defn, metadata, primary_spool_codes=["D"],
        )
        # 1 spool * 2 seal = 2 products
        assert len(products) == 2
        spool_types = {p.specs.get("spool_type") for p in products}
        assert spool_types == {"D"}
        seals = {p.specs.get("seal_material") for p in products}
        assert seals == {"FKM", "NBR"}

    def test_case_insensitive(self, metadata):
        """Primary spool filter should be case-insensitive."""
        defn = self._make_definition(["D", "E", "H"])
        products = generate_products_from_ordering_code(
            defn, metadata, primary_spool_codes=["d", "h"],
        )
        assert len(products) == 2
        codes = {p.model_code for p in products}
        assert "T-D" in codes
        assert "T-H" in codes


# ── table_rows_to_products ───────────────────────────────────────────


class TestTableRowsToProducts:
    @pytest.fixture
    def metadata(self):
        return UploadMetadata(
            company="Danfoss",
            document_type=DocumentType.CATALOGUE,
            filename="catalogue.pdf",
        )

    def test_basic_row(self, metadata):
        rows = [
            {"model_code": "DG4V-3-2A", "max_pressure_bar": "315", "max_flow_lpm": "120"},
        ]
        products = table_rows_to_products(rows, metadata)
        assert len(products) == 1
        assert products[0].model_code == "DG4V-3-2A"
        assert products[0].specs.get("max_pressure_bar") == 315

    def test_skips_row_without_model_code(self, metadata):
        rows = [
            {"max_pressure_bar": "315"},  # no model_code
            {"model_code": "DG4V", "max_pressure_bar": "315"},
        ]
        products = table_rows_to_products(rows, metadata)
        assert len(products) == 1

    def test_numeric_values_parsed(self, metadata):
        rows = [
            {"model_code": "TEST", "max_pressure_bar": "315 bar", "weight_kg": "2.5 kg"},
        ]
        products = table_rows_to_products(rows, metadata)
        specs = products[0].specs
        assert specs.get("max_pressure_bar") == 315
        assert specs.get("weight_kg") == 2.5

    def test_alphanumeric_values_preserved(self, metadata):
        rows = [
            {"model_code": "TEST", "coil_voltage": "24VDC", "port_size": "G3/8"},
        ]
        products = table_rows_to_products(rows, metadata)
        specs = products[0].specs
        assert specs.get("coil_voltage") == "24VDC"
        assert specs.get("port_size") == "G3/8"

    def test_source_is_table(self, metadata):
        rows = [{"model_code": "TEST"}]
        products = table_rows_to_products(rows, metadata)
        assert products[0].source == "table"
        assert products[0].confidence == 0.8

    def test_empty_rows(self, metadata):
        assert table_rows_to_products([], metadata) == []


# ── HEADER_MAPPINGS validation ───────────────────────────────────────


class TestHeaderMappings:
    def test_model_code_variants_mapped(self):
        """Common model code header variations should all map to 'model_code'."""
        for key in ["model", "model no", "part no", "part number",
                     "ordering code", "order code"]:
            assert HEADER_MAPPINGS.get(key) == "model_code", (
                f"Header '{key}' should map to model_code"
            )

    def test_pressure_variants_mapped(self):
        for key in ["pressure", "max pressure", "rated pressure", "pressure (bar)"]:
            assert HEADER_MAPPINGS.get(key) == "max_pressure_bar", (
                f"Header '{key}' should map to max_pressure_bar"
            )

    def test_spool_type_not_in_header_mappings(self):
        """Spool type typically comes from model code breakdown, not table headers.
        But if it does appear, verify the mapping exists or document the gap."""
        spool_keys = [k for k, v in HEADER_MAPPINGS.items() if v == "spool_type"]
        # This is informational — spool types usually come from LLM extraction
        # If no header maps to spool_type, that's expected but worth noting
        pass


# ── _VALID_SPEC_FIELDS validation ───────────────────────────────────


class TestValidSpecFields:
    def test_spool_type_included(self):
        assert "spool_type" in _VALID_SPEC_FIELDS

    def test_all_fields_exist_on_model(self):
        from models import HydraulicProduct
        for field in _VALID_SPEC_FIELDS:
            assert field in HydraulicProduct.model_fields, (
                f"_VALID_SPEC_FIELDS contains '{field}' which is not on HydraulicProduct"
            )

    def test_critical_fields_included(self):
        critical = {
            "max_pressure_bar", "max_flow_lpm", "valve_size",
            "spool_type", "coil_voltage", "actuator_type",
            "port_size", "mounting", "seal_material",
        }
        assert critical.issubset(_VALID_SPEC_FIELDS)


# ── LLM Prompt Content Validation ───────────────────────────────────


class TestLLMPromptContent:
    """Validate that LLM prompts contain the critical spool type guidance.
    These tests read the actual prompt strings from the functions to ensure
    the hydraulic engineering context is present."""

    def _get_prompt_from_function(self, func_name: str, **kwargs) -> str:
        """Extract the prompt string by mocking the OpenAI call."""
        captured_prompt = {}

        class MockResponse:
            class Choice:
                class Message:
                    content = '{"products": []}'
                message = Message()
            choices = [Choice()]

        def capture_call(**call_kwargs):
            msgs = call_kwargs.get("messages", [])
            if msgs:
                captured_prompt["text"] = msgs[0].get("content", "")
            return MockResponse()

        mock_client = MagicMock()
        mock_client.chat.completions.create.side_effect = capture_call

        with patch("tools.parse_tools._get_client", return_value=mock_client):
            import tools.parse_tools as pt
            func = getattr(pt, func_name)
            func(**kwargs)

        return captured_prompt.get("text", "")

    def test_extract_products_prompt_has_spool_guidance(self):
        prompt = self._get_prompt_from_function(
            "extract_products_with_llm",
            text="Sample hydraulic valve text for testing.",
            metadata=UploadMetadata(
                company="Danfoss", document_type=DocumentType.CATALOGUE, filename="test.pdf",
            ),
        )
        # The prompt should mention spool type extraction
        assert "spool_type" in prompt.lower() or "spool type" in prompt.lower()
        assert "center condition" in prompt.lower() or "center" in prompt.lower()

    def test_model_code_patterns_prompt_has_spool_guidance(self):
        prompt = self._get_prompt_from_function(
            "extract_model_code_patterns_with_llm",
            text="Sample model code breakdown text.",
            company="Danfoss",
        )
        assert "spool type" in prompt.lower()
        assert "maps_to_field" in prompt

    def test_ordering_code_prompt_has_spool_guidance(self):
        prompt = self._get_prompt_from_function(
            "extract_ordering_code_with_llm",
            text="Sample ordering code table text.",
            company="Bosch Rexroth",
            category="directional_valves",
        )
        assert "spool type" in prompt.lower() or "spool_type" in prompt.lower()
        assert "maps_to_field" in prompt

    def test_ordering_code_prompt_has_manufacturer_examples(self):
        prompt = self._get_prompt_from_function(
            "extract_ordering_code_with_llm",
            text="Sample text.",
            company="Danfoss",
            category="directional_valves",
        )
        # Should have manufacturer-specific spool code examples
        assert "danfoss" in prompt.lower() or "vickers" in prompt.lower()
        assert "bosch" in prompt.lower() or "rexroth" in prompt.lower()

    def test_ordering_code_prompt_has_center_conditions(self):
        prompt = self._get_prompt_from_function(
            "extract_ordering_code_with_llm",
            text="Sample text.",
            company="Danfoss",
            category="",
        )
        # Should describe common center conditions
        assert "blocked" in prompt.lower()
        assert "open" in prompt.lower()

    def test_ordering_code_prompt_references_valid_fields(self):
        """The prompt should reference the _VALID_SPEC_FIELDS set."""
        prompt = self._get_prompt_from_function(
            "extract_ordering_code_with_llm",
            text="Sample.",
            company="Test",
        )
        # Should list valid field names
        assert "spool_type" in prompt
        assert "coil_voltage" in prompt
        assert "seal_material" in prompt


class TestComputeCanonicalPattern:
    """Tests for the canonical spool pattern normalisation function."""

    def test_blocked_center_standard_crossover(self):
        """Danfoss 2A / Bosch D style: blocked center, standard crossover."""
        pattern = compute_canonical_pattern(
            "All ports blocked", "P→A, B→T", "P→B, A→T"
        )
        assert pattern.startswith("BLOCKED|")

    def test_open_center(self):
        pattern = compute_canonical_pattern(
            "All ports open", "P→A, B→T", "P→B, A→T"
        )
        assert pattern.startswith("OPEN|")

    def test_float_center(self):
        pattern = compute_canonical_pattern(
            "Float - A, B, T connected, P blocked", "P→A, B→T", "P→B, A→T"
        )
        assert pattern.startswith("FLOAT|")

    def test_tandem_center(self):
        pattern = compute_canonical_pattern(
            "P-T connected, A and B blocked", "P→A, B→T", "P→B, A→T"
        )
        assert pattern.startswith("TANDEM|")

    def test_empty_inputs(self):
        pattern = compute_canonical_pattern("", "", "")
        assert pattern == "UNKNOWN|UNKNOWN|UNKNOWN"

    def test_cross_manufacturer_equivalence(self):
        """Danfoss 2A and Bosch D should produce the same canonical pattern."""
        danfoss = compute_canonical_pattern(
            "All ports blocked", "P→A, B→T", "P→B, A→T"
        )
        bosch = compute_canonical_pattern(
            "All ports blocked", "P→A, B→T", "P→B, A→T"
        )
        assert danfoss == bosch

    def test_different_center_condition_differs(self):
        """Different center conditions should produce different patterns."""
        blocked = compute_canonical_pattern(
            "All ports blocked", "P→A, B→T", "P→B, A→T"
        )
        open_c = compute_canonical_pattern(
            "All ports open", "P→A, B→T", "P→B, A→T"
        )
        assert blocked != open_c

    def test_deterministic(self):
        """Same inputs should always produce the same output."""
        p1 = compute_canonical_pattern("Blocked", "P→A, B→T", "P→B, A→T")
        p2 = compute_canonical_pattern("Blocked", "P→A, B→T", "P→B, A→T")
        assert p1 == p2


# ── Vision-based spool symbol extraction ──────────────────────────────


class TestExtractSpoolSymbolsFromPdf:
    """Tests for extract_spool_symbols_from_pdf — GPT-4o vision spool extraction."""

    @patch("tools.parse_tools.HAS_PYMUPDF", False)
    def test_returns_empty_without_pymupdf(self):
        """Should gracefully return [] if PyMuPDF is not available."""
        result = extract_spool_symbols_from_pdf("fake.pdf", "Rexroth")
        assert result == []

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_extracts_spool_from_page(self, mock_client, mock_fitz):
        """Should extract spool data from a PDF page via GPT-4o vision."""
        # Mock PDF with one page
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000  # > 5000 bytes
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # Mock OpenAI response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": [{"spool_code": "D", "center_condition": "All ports blocked", "solenoid_a_function": "P to A, B to T", "solenoid_b_function": "P to B, A to T", "description": "Closed center", "symbol_description": "3-position, center blocked"}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_spool_symbols_from_pdf("test.pdf", "Rexroth")

        assert len(result) == 1
        assert result[0]["spool_code"] == "D"
        assert result[0]["company"] == "Rexroth"
        assert result[0]["extraction_method"] in ("vision", "vision_v2")
        assert "canonical_pattern" in result[0]
        assert result[0]["canonical_pattern"].startswith("BLOCKED|")

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_deduplicates_by_spool_code(self, mock_client, mock_fitz):
        """Should deduplicate spool symbols found on multiple pages."""
        # Mock PDF with 2 pages
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # Both pages return the same spool code "D"
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": [{"spool_code": "D", "center_condition": "All ports blocked", "solenoid_a_function": "P to A, B to T", "solenoid_b_function": "P to B, A to T", "description": "Closed center", "symbol_description": "basic"}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_spool_symbols_from_pdf("test.pdf", "Rexroth")
        assert len(result) == 1  # Deduplicated

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_skips_small_pages(self, mock_client, mock_fitz):
        """Should skip pages with very small rendered images (likely blank)."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 100  # < 5000 bytes
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        result = extract_spool_symbols_from_pdf("test.pdf", "Rexroth")
        assert result == []
        mock_client.return_value.chat.completions.create.assert_not_called()

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_no_spools_found(self, mock_client, mock_fitz):
        """Should return empty list when no spool symbols are found."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": []}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_spool_symbols_from_pdf("test.pdf", "Rexroth")
        assert result == []

    @patch("tools.parse_tools._fitz")
    def test_handles_pdf_open_failure(self, mock_fitz):
        """Should handle PDF open failure gracefully."""
        mock_fitz.open.side_effect = Exception("Cannot open PDF")
        result = extract_spool_symbols_from_pdf("bad.pdf", "Rexroth")
        assert result == []

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_multiple_spools_on_one_page(self, mock_client, mock_fitz):
        """Should extract multiple different spool symbols from one page."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": [{"spool_code": "D", "center_condition": "All ports blocked", "solenoid_a_function": "P to A, B to T", "solenoid_b_function": "P to B, A to T", "description": "Closed", "symbol_description": "blocked center"}, {"spool_code": "E", "center_condition": "All ports open", "solenoid_a_function": "P to A, B to T", "solenoid_b_function": "P to B, A to T", "description": "Open", "symbol_description": "open center"}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_spool_symbols_from_pdf("test.pdf", "Rexroth")
        assert len(result) == 2
        codes = {s["spool_code"] for s in result}
        assert codes == {"D", "E"}


# ── Ordering code prompt completeness ─────────────────────────────────


class TestOrderingCodePromptCompleteness:
    """Verify the ordering code LLM prompt includes instructions about blank panels."""

    def test_prompt_mentions_blank_panels(self):
        """The prompt should instruct LLM to handle blank/optional positions."""
        from tools.parse_tools import extract_ordering_code_with_llm
        import inspect
        source = inspect.getsource(extract_ordering_code_with_llm)
        assert "blank" in source.lower() or "BLANK" in source
        assert "numbered description" in source.lower() or "description table" in source.lower()

    def test_prompt_uses_smart_text_selection(self):
        """The prompt should use _select_ordering_code_text for smart text windowing."""
        from tools.parse_tools import extract_ordering_code_with_llm
        import inspect
        source = inspect.getsource(extract_ordering_code_with_llm)
        assert "_select_ordering_code_text" in source

    def test_prompt_requires_all_positions(self):
        """The prompt should explicitly require ALL positions to be present."""
        from tools.parse_tools import extract_ordering_code_with_llm
        import inspect
        source = inspect.getsource(extract_ordering_code_with_llm)
        assert "NEVER skip" in source or "Do NOT collapse" in source


# ── Cross-reference extraction ────────────────────────────────────────


class TestExtractCrossReferencesWithLLM:
    """Tests for extract_cross_references_with_llm — GPT-4.1-mini cross-reference extraction."""

    @patch("tools.parse_tools._get_client")
    def test_extracts_mappings(self, mock_client):
        """Should extract cross-reference mappings from text."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"cross_references": [{"my_company_series": "DG4V-3", "competitor_series": "D1VW", "competitor_company": "Parker", "product_type": "Directional Valve", "notes": ""}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_cross_references_with_llm("Some cross-reference text", "Danfoss")
        assert len(result) == 1
        assert result[0]["my_company_series"] == "DG4V-3"
        assert result[0]["competitor_series"] == "D1VW"
        assert result[0]["competitor_company"] == "Parker"

    @patch("tools.parse_tools._get_client")
    def test_skips_incomplete_entries(self, mock_client):
        """Should skip entries missing required fields."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"cross_references": [{"my_company_series": "DG4V-3", "competitor_series": "", "competitor_company": "Parker"}, {"my_company_series": "KFDG4V", "competitor_series": "4WRE", "competitor_company": "Bosch Rexroth"}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_cross_references_with_llm("Some text", "Danfoss")
        assert len(result) == 1  # Only the complete entry
        assert result[0]["my_company_series"] == "KFDG4V"

    def test_empty_text_returns_empty(self):
        """Should return empty list for empty input."""
        result = extract_cross_references_with_llm("", "Danfoss")
        assert result == []

    @patch("tools.parse_tools._get_client")
    def test_handles_api_failure(self, mock_client):
        """Should return empty list on API failure."""
        mock_client.return_value.chat.completions.create.side_effect = Exception("API error")
        result = extract_cross_references_with_llm("Some text", "Danfoss")
        assert result == []

    @patch("tools.parse_tools._get_client")
    def test_multiple_competitors(self, mock_client):
        """Should extract mappings for multiple competitors."""
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"cross_references": [{"my_company_series": "DG4V-3", "competitor_series": "D1VW", "competitor_company": "Parker", "product_type": "Directional Valve", "notes": ""}, {"my_company_series": "DG4V-3", "competitor_series": "4WE6", "competitor_company": "Bosch Rexroth", "product_type": "Directional Valve", "notes": ""}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_cross_references_with_llm("Cross-ref table", "Danfoss")
        assert len(result) == 2
        companies = {r["competitor_company"] for r in result}
        assert companies == {"Parker", "Bosch Rexroth"}


class TestSelectOrderingCodeText:
    """Tests for _select_ordering_code_text — smart text selection for ordering code extraction."""

    def test_short_text_returned_as_is(self):
        """Text shorter than max_chars should be returned unchanged."""
        from tools.parse_tools import _select_ordering_code_text
        text = "Short document text about ordering codes."
        result = _select_ordering_code_text(text, max_chars=50000)
        assert result == text

    def test_finds_ordering_code_section(self):
        """Should extract text around 'Ordering Code' headers."""
        from tools.parse_tools import _select_ordering_code_text
        # Create text with ordering code section in the middle
        padding = "x" * 20000
        text = padding + "\n\nOrdering Code Breakdown\nDG4V-3-xx-xx\n" + padding
        result = _select_ordering_code_text(text, max_chars=10000)
        assert "Ordering Code Breakdown" in result
        assert len(result) <= 10000

    def test_finds_spool_type_section(self):
        """Should extract text around 'spool type' / 'valve function' headers."""
        from tools.parse_tools import _select_ordering_code_text
        padding = "x" * 20000
        text = padding + "\n\nSpool Type Designation\n2A - All ports blocked\n" + padding
        result = _select_ordering_code_text(text, max_chars=10000)
        assert "Spool Type Designation" in result

    def test_fallback_to_first_n_chars(self):
        """Should fall back to first max_chars when no headers found."""
        from tools.parse_tools import _select_ordering_code_text
        text = "No relevant headers here. " * 5000
        result = _select_ordering_code_text(text, max_chars=10000)
        assert len(result) <= 10000
        assert result == text[:10000]

    def test_merges_overlapping_ranges(self):
        """Should merge overlapping text ranges from nearby sections."""
        from tools.parse_tools import _select_ordering_code_text
        padding = "x" * 500
        text = (
            padding
            + "\n\nOrdering Code\n"
            + padding
            + "\n\nSpool Type Table\n"
            + padding
            + "x" * 50000
        )
        result = _select_ordering_code_text(text, max_chars=40000)
        assert "Ordering Code" in result
        assert "Spool Type Table" in result


# ── _is_graphics_heavy_pdf ─────────────────────────────────────────


class TestIsGraphicsHeavyPdf:
    """Tests for graphics-heavy PDF detection."""

    def test_text_rich_pdf_returns_false(self):
        """A PDF with plenty of text per page should NOT be classified as graphics-heavy."""
        pages = [
            {"page": 1, "text": "x" * 500},
            {"page": 2, "text": "y" * 600},
            {"page": 3, "text": "z" * 700},
        ]
        assert _is_graphics_heavy_pdf(pages, 3) is False

    def test_sparse_pdf_returns_true(self):
        """A multi-page PDF with very little text should be classified as graphics-heavy."""
        pages = [
            {"page": 1, "text": "Title"},
            {"page": 2, "text": ""},
            {"page": 3, "text": "DG4V-3"},
        ]
        assert _is_graphics_heavy_pdf(pages, 10) is True

    def test_single_page_pdf_returns_false(self):
        """Very short PDFs (1 page) should not trigger graphics-heavy detection."""
        pages = [{"page": 1, "text": "Short"}]
        assert _is_graphics_heavy_pdf(pages, 1) is False

    def test_empty_pages_list_with_multi_page(self):
        """No text extracted at all from a multi-page PDF → graphics-heavy."""
        assert _is_graphics_heavy_pdf([], 5) is True

    def test_empty_pages_list_single_page(self):
        """No text from a single-page PDF → NOT graphics-heavy (just a cover)."""
        assert _is_graphics_heavy_pdf([], 1) is False

    def test_mixed_pages_below_threshold(self):
        """Some pages have text, but overall average is too low."""
        pages = [
            {"page": 1, "text": "x" * 300},  # decent text
            {"page": 2, "text": "y" * 50},    # sparse
            {"page": 3, "text": ""},           # empty
        ]
        # 3 pages from a 10-page PDF: avg = 350/10 = 35 chars/page
        assert _is_graphics_heavy_pdf(pages, 10) is True

    def test_borderline_stays_text(self):
        """Average at 500 chars/page with most pages having content should NOT be graphics-heavy."""
        pages = [
            {"page": 1, "text": "x" * 500},
            {"page": 2, "text": "y" * 500},
        ]
        assert _is_graphics_heavy_pdf(pages, 2) is False

    def test_danfoss_style_pdf_detected(self):
        """Typical Danfoss datasheet: 14 pages, ~100 chars extracted per page average,
        with pdfplumber sometimes inflating a few pages to 200-400 chars."""
        pages = [
            {"page": 1, "text": "x" * 400},   # cover page has some text
            {"page": 2, "text": "y" * 250},    # pdfplumber picked up scattered labels
            {"page": 3, "text": "DG4V-3"},     # ordering code page - almost no text
            {"page": 4, "text": "Spool"},       # spool page - almost no text
            {"page": 5, "text": "z" * 150},    # specs page - minimal text
        ]
        # 5 pages from a 14-page PDF. Pages 6-14 have no text at all.
        assert _is_graphics_heavy_pdf(pages, 14) is True


# ── _parse_ordering_code_response ──────────────────────────────────


class TestParseOrderingCodeResponse:
    """Tests for the shared ordering code JSON parser."""

    def test_parses_valid_response(self):
        """Should parse a well-formed ordering code response into definitions."""
        raw = [{
            "series": "DG4V-3",
            "product_name": "Directional Valve",
            "category": "directional_valves",
            "code_template": "{01}-{02}-{03}",
            "segments": [
                {
                    "position": 1, "segment_name": "series_code",
                    "is_fixed": True, "separator_before": "",
                    "options": [{"code": "DG4V", "description": "Series", "maps_to_field": "series_code", "maps_to_value": "DG4V"}],
                },
                {
                    "position": 2, "segment_name": "spool_type",
                    "is_fixed": False, "separator_before": "-",
                    "options": [
                        {"code": "2A", "description": "All ports blocked", "maps_to_field": "spool_type", "maps_to_value": "2A"},
                        {"code": "6C", "description": "P to T, A&B blocked", "maps_to_field": "spool_type", "maps_to_value": "6C"},
                    ],
                },
            ],
            "shared_specs": {"max_pressure_bar": 315},
        }]
        result = _parse_ordering_code_response(raw, "Danfoss", "directional_valves")
        assert len(result) == 1
        defn = result[0]
        assert defn.series == "DG4V-3"
        assert defn.company == "Danfoss"
        assert len(defn.segments) == 2
        assert defn.segments[1].segment_name == "spool_type"
        assert len(defn.segments[1].options) == 2

    def test_auto_generates_segment_name(self):
        """Should auto-generate segment_name when LLM omits it."""
        raw = [{
            "series": "4WE6",
            "product_name": "Valve",
            "category": "directional_valves",
            "code_template": "{01}",
            "segments": [{
                "position": 1, "segment_name": "",
                "is_fixed": True, "separator_before": "",
                "options": [{"code": "4WE6", "description": "4-way valve size 6"}],
            }],
        }]
        result = _parse_ordering_code_response(raw, "Rexroth", "")
        assert len(result) == 1
        seg = result[0].segments[0]
        assert seg.segment_name != ""
        # Should be auto-generated from description
        assert "4" in seg.segment_name or "way" in seg.segment_name

    def test_skips_entries_without_series(self):
        """Entries without a series should be skipped."""
        raw = [{"series": "", "product_name": "Unknown", "segments": []}]
        result = _parse_ordering_code_response(raw, "X", "")
        assert result == []

    def test_handles_empty_list(self):
        """Empty input should produce empty output."""
        assert _parse_ordering_code_response([], "X", "") == []


# ── extract_ordering_code_from_images ──────────────────────────────


class TestExtractOrderingCodeFromImages:
    """Tests for vision-based ordering code extraction."""

    @patch("tools.parse_tools.HAS_PYMUPDF", False)
    def test_returns_empty_without_pymupdf(self):
        """Should gracefully return [] if PyMuPDF is not available."""
        result = extract_ordering_code_from_images("fake.pdf", "Danfoss")
        assert result == []

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_extracts_ordering_code_via_vision(self, mock_client, mock_fitz):
        """Should extract ordering code definitions from rendered PDF pages."""
        # Mock PDF with 3 pages
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=3)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # Mock GPT-4o vision response with ordering code
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '''{
            "ordering_codes": [{
                "series": "DG4V-3",
                "product_name": "Directional Control Valve",
                "category": "directional_valves",
                "code_template": "{01}-{02}-{03}",
                "segments": [
                    {"position": 1, "segment_name": "series_code", "is_fixed": true,
                     "separator_before": "", "options": [{"code": "DG4V-3", "description": "Series",
                     "maps_to_field": "series_code", "maps_to_value": "DG4V-3"}]},
                    {"position": 2, "segment_name": "spool_type", "is_fixed": false,
                     "separator_before": "-", "options": [
                        {"code": "2A", "description": "All ports blocked",
                         "maps_to_field": "spool_type", "maps_to_value": "2A"},
                        {"code": "6C", "description": "P to T",
                         "maps_to_field": "spool_type", "maps_to_value": "6C"}
                    ]}
                ],
                "shared_specs": {"max_pressure_bar": 350}
            }]
        }'''
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_ordering_code_from_images("test.pdf", "Danfoss", "directional_valves")

        assert len(result) == 1
        assert result[0].series == "DG4V-3"
        assert result[0].company == "Danfoss"
        spool_seg = [s for s in result[0].segments if s.segment_name == "spool_type"]
        assert len(spool_seg) == 1
        assert len(spool_seg[0].options) == 2

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_handles_api_failure_gracefully(self, mock_client, mock_fitz):
        """Should return [] on API errors without crashing."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # Simulate API failure
        mock_client.return_value.chat.completions.create.side_effect = Exception("API error")

        result = extract_ordering_code_from_images("test.pdf", "Danfoss")
        assert result == []

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_skips_blank_pages(self, mock_client, mock_fitz):
        """Should skip pages that render to very small images (blank)."""
        # Page 0: blank (small image)
        mock_blank_page = MagicMock()
        mock_blank_pix = MagicMock()
        mock_blank_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 100  # < 5000
        mock_blank_page.get_pixmap.return_value = mock_blank_pix

        # Page 1: real content
        mock_real_page = MagicMock()
        mock_real_pix = MagicMock()
        mock_real_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_real_page.get_pixmap.return_value = mock_real_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(side_effect=[mock_blank_page, mock_real_page])
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"ordering_codes": []}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_ordering_code_from_images("test.pdf", "Danfoss")
        # Should have called the API only once (skipped the blank page)
        assert mock_client.return_value.chat.completions.create.call_count == 1


# ===========================================================================
# Tests for redesigned spool extraction pipeline (v2)
# ===========================================================================


class TestClassifySpoolPages:
    """Tests for _classify_spool_pages — page classification heuristics."""

    def test_spool_keywords_detected(self):
        """Pages with strong spool keywords → SPOOL_CONTENT."""
        pages = [
            {"page": 1, "text": "This page shows spool type options and center condition for each."},
        ]
        with patch("tools.parse_tools.HAS_PYMUPDF", True), \
             patch("tools.parse_tools._fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=1)
            mock_fitz.open.return_value = mock_doc
            result = _classify_spool_pages("test.pdf", pages)
        assert result[0] == "SPOOL_CONTENT"

    def test_maybe_spool_keywords(self):
        """Pages with weak keywords only → MAYBE_SPOOL."""
        pages = [
            {"page": 1, "text": "The solenoid is energised in this position for the left symbol side."},
        ]
        with patch("tools.parse_tools.HAS_PYMUPDF", True), \
             patch("tools.parse_tools._fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=1)
            mock_fitz.open.return_value = mock_doc
            result = _classify_spool_pages("test.pdf", pages)
        assert result[0] == "MAYBE_SPOOL"

    def test_non_spool_page(self):
        """Pages about dimensions, weight → NON_SPOOL."""
        pages = [
            {"page": 1, "text": "The overall dimensions are 150mm x 80mm x 60mm. Weight is 3.2kg. "
                                "Installation torque requirements must be followed. Maintenance "
                                "should be performed every 2000 hours of operation. Use proper tools "
                                "and follow all safety guidelines. The mounting bolt pattern is 4x M8 "
                                "on 100mm PCD. Ambient temperature range is -20C to +60C. Maximum "
                                "working pressure is 350 bar. Flow rate capacity is 100 LPM nominal."},
        ]
        with patch("tools.parse_tools.HAS_PYMUPDF", True), \
             patch("tools.parse_tools._fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=1)
            mock_fitz.open.return_value = mock_doc
            result = _classify_spool_pages("test.pdf", pages)
        assert result[0] == "NON_SPOOL"

    def test_sparse_text_classified_as_maybe(self):
        """Pages with very little text (<300 chars) → MAYBE_SPOOL (may be graphical)."""
        pages = [
            {"page": 1, "text": "DG4V-3"},  # Very sparse — likely a diagram page
        ]
        with patch("tools.parse_tools.HAS_PYMUPDF", True), \
             patch("tools.parse_tools._fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=1)
            mock_fitz.open.return_value = mock_doc
            result = _classify_spool_pages("test.pdf", pages)
        assert result[0] == "MAYBE_SPOOL"

    def test_returns_all_pages(self):
        """Should return a classification for every page in the PDF."""
        pages = [
            {"page": 1, "text": "Cover page"},
            {"page": 2, "text": "This shows spool type options with center condition blocked."},
            {"page": 3, "text": "Dimensions and weight specifications for mounting."},
        ]
        with patch("tools.parse_tools.HAS_PYMUPDF", True), \
             patch("tools.parse_tools._fitz") as mock_fitz:
            mock_doc = MagicMock()
            mock_doc.__len__ = MagicMock(return_value=3)
            mock_fitz.open.return_value = mock_doc
            result = _classify_spool_pages("test.pdf", pages)
        assert len(result) == 3
        assert 0 in result and 1 in result and 2 in result


class TestExtractSpoolSymbolsV2:
    """Tests for extract_spool_symbols_from_pdf_v2 — batched multi-page vision."""

    @patch("tools.parse_tools.HAS_PYMUPDF", False)
    def test_returns_empty_without_pymupdf(self):
        """Should gracefully return [] if PyMuPDF not available."""
        result = extract_spool_symbols_from_pdf_v2("fake.pdf", "Danfoss")
        assert result == []

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_uses_higher_dpi_than_v1(self, mock_client, mock_fitz):
        """Default DPI should be 250 (not 150)."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": [{"spool_code": "2A", "center_condition": "All ports blocked", "solenoid_a_function": "P to A", "solenoid_b_function": "P to B", "description": "test", "symbol_description": "test sym"}]}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss", retry_on_low_count=False,
        )
        # Check the DPI used for rendering
        mock_page.get_pixmap.assert_called_with(dpi=250)

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_batches_pages(self, mock_client, mock_fitz):
        """8 pages with batch_size=4 should create 2 API calls."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=8)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": []}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss", batch_size=4, retry_on_low_count=False,
        )
        # 8 pages / 4 batch = 2 API calls
        assert mock_client.return_value.chat.completions.create.call_count == 2

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_spool_content_pages_first(self, mock_client, mock_fitz):
        """SPOOL_CONTENT pages should be processed before MAYBE_SPOOL."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=4)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": []}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        # Page 3 is SPOOL_CONTENT, pages 0,1 are MAYBE, page 2 is NON_SPOOL
        classifications = {0: "MAYBE_SPOOL", 1: "MAYBE_SPOOL",
                           2: "NON_SPOOL", 3: "SPOOL_CONTENT"}

        extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss",
            page_classifications=classifications,
            batch_size=4, retry_on_low_count=False,
        )
        # Should process 3 pages (SPOOL_CONTENT + MAYBE_SPOOL, skip NON_SPOOL)
        assert mock_page.get_pixmap.call_count == 3

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_cumulative_context_injected(self, mock_client, mock_fitz):
        """Second batch should include spools found in first batch."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=2)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # First batch finds "2A", second batch should have it in context
        response_1 = MagicMock()
        response_1.choices = [MagicMock()]
        response_1.choices[0].message.content = '{"spool_symbols": [{"spool_code": "2A", "center_condition": "Blocked", "solenoid_a_function": "P→A", "solenoid_b_function": "P→B", "description": "Closed center", "symbol_description": "blocked"}]}'

        response_2 = MagicMock()
        response_2.choices = [MagicMock()]
        response_2.choices[0].message.content = '{"spool_symbols": [{"spool_code": "6C", "center_condition": "Open", "solenoid_a_function": "P→A", "solenoid_b_function": "P→B", "description": "Open center", "symbol_description": "open"}]}'

        mock_client.return_value.chat.completions.create.side_effect = [response_1, response_2]

        result = extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss", batch_size=1, retry_on_low_count=False,
        )

        # Both spools should be in result
        codes = {s["spool_code"] for s in result}
        assert "2A" in codes
        assert "6C" in codes

        # Check second call had "PREVIOUSLY FOUND" in the prompt
        second_call = mock_client.return_value.chat.completions.create.call_args_list[1]
        prompt_text = second_call[1]["messages"][0]["content"][0]["text"]
        assert "PREVIOUSLY FOUND" in prompt_text
        assert "2A" in prompt_text

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_known_spool_codes_injected(self, mock_client, mock_fitz):
        """Known spool codes from DB should appear in the prompt."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = '{"spool_symbols": []}'
        mock_client.return_value.chat.completions.create.return_value = mock_response

        extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss",
            known_spool_codes=["2A", "6C", "H"],
            retry_on_low_count=False,
        )

        call_args = mock_client.return_value.chat.completions.create.call_args
        prompt_text = call_args[1]["messages"][0]["content"][0]["text"]
        assert "KNOWN SPOOL CODES" in prompt_text
        assert "2A" in prompt_text
        assert "6C" in prompt_text

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_retry_on_low_count(self, mock_client, mock_fitz):
        """When <5 spools found with retry_on_low_count=True, should retry."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        # First pass: 1 spool. Retry: 2 more spools.
        response_1 = MagicMock()
        response_1.choices = [MagicMock()]
        response_1.choices[0].message.content = '{"spool_symbols": [{"spool_code": "2A", "center_condition": "Blocked", "solenoid_a_function": "P→A", "solenoid_b_function": "P→B", "description": "test", "symbol_description": "s"}]}'

        response_retry = MagicMock()
        response_retry.choices = [MagicMock()]
        response_retry.choices[0].message.content = '{"spool_symbols": [{"spool_code": "2A", "center_condition": "Blocked", "solenoid_a_function": "P→A", "solenoid_b_function": "P→B", "description": "test", "symbol_description": "s"}, {"spool_code": "6C", "center_condition": "Open", "solenoid_a_function": "P→A", "solenoid_b_function": "P→B", "description": "test", "symbol_description": "s"}]}'

        mock_client.return_value.chat.completions.create.side_effect = [response_1, response_retry]

        result = extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss", retry_on_low_count=True,
        )
        # Should have made 2 API calls (initial + retry)
        assert mock_client.return_value.chat.completions.create.call_count == 2
        codes = {s["spool_code"].upper() for s in result}
        assert "2A" in codes
        assert "6C" in codes

    @patch("tools.parse_tools._fitz")
    @patch("tools.parse_tools._get_client")
    def test_no_retry_when_enough_spools(self, mock_client, mock_fitz):
        """When >=5 spools found, should NOT retry."""
        mock_page = MagicMock()
        mock_pix = MagicMock()
        mock_pix.tobytes.return_value = b"\x89PNG" + b"\x00" * 10000
        mock_page.get_pixmap.return_value = mock_pix

        mock_doc = MagicMock()
        mock_doc.__len__ = MagicMock(return_value=1)
        mock_doc.__getitem__ = MagicMock(return_value=mock_page)
        mock_fitz.open.return_value = mock_doc

        spools_json = [
            {"spool_code": c, "center_condition": "test", "solenoid_a_function": "P→A",
             "solenoid_b_function": "P→B", "description": "test", "symbol_description": "s"}
            for c in ["2A", "2C", "6C", "0A", "H"]
        ]
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = json.dumps({"spool_symbols": spools_json})
        mock_client.return_value.chat.completions.create.return_value = mock_response

        result = extract_spool_symbols_from_pdf_v2(
            "test.pdf", "Danfoss", retry_on_low_count=True,
        )
        # Only 1 API call — no retry needed
        assert mock_client.return_value.chat.completions.create.call_count == 1
        assert len(result) == 5

    def test_legacy_wrapper_delegates(self):
        """extract_spool_symbols_from_pdf() should call v2."""
        with patch("tools.parse_tools.extract_spool_symbols_from_pdf_v2") as mock_v2:
            mock_v2.return_value = [{"spool_code": "X"}]
            result = extract_spool_symbols_from_pdf("test.pdf", "Danfoss")
            mock_v2.assert_called_once_with("test.pdf", "Danfoss")
            assert result == [{"spool_code": "X"}]


class TestDeduplicateSpools:
    """Tests for _deduplicate_spools helper."""

    def test_basic_dedup(self):
        """Same code twice — keep the one with more data."""
        spools = [
            {"spool_code": "2A", "symbol_description": "short",
             "center_condition": "Blocked", "solenoid_a_function": "",
             "solenoid_b_function": ""},
            {"spool_code": "2A", "symbol_description": "much longer description",
             "center_condition": "All ports blocked", "solenoid_a_function": "P→A",
             "solenoid_b_function": "P→B"},
        ]
        result = _deduplicate_spools(spools)
        assert len(result) == 1
        assert result[0]["solenoid_a_function"] == "P→A"

    def test_case_insensitive(self):
        """Spool codes should be deduped case-insensitively."""
        spools = [
            {"spool_code": "2a", "symbol_description": "x",
             "center_condition": "Blocked"},
            {"spool_code": "2A", "symbol_description": "xx",
             "center_condition": "Blocked"},
        ]
        result = _deduplicate_spools(spools)
        assert len(result) == 1

    def test_different_codes_preserved(self):
        """Different spool codes should all be kept."""
        spools = [
            {"spool_code": "2A", "symbol_description": "x"},
            {"spool_code": "6C", "symbol_description": "y"},
            {"spool_code": "H", "symbol_description": "z"},
        ]
        result = _deduplicate_spools(spools)
        assert len(result) == 3


class TestMergeSpoolResults:
    """Tests for _merge_spool_results — UNION merge of vision and text."""

    def test_vision_only(self):
        """Only vision spools → all preserved."""
        vision = [
            {"spool_code": "2A", "center_condition": "Blocked",
             "solenoid_a_function": "P→A", "solenoid_b_function": "P→B"},
            {"spool_code": "6C", "center_condition": "Open",
             "solenoid_a_function": "P→A", "solenoid_b_function": "P→B"},
        ]
        result = _merge_spool_results(vision, [])
        assert "2A" in result
        assert "6C" in result

    def test_text_only(self):
        """Only text spools → all preserved."""
        text = [
            {"spool_code": "2A", "center_condition": "Blocked",
             "solenoid_a_function": "P→A"},
        ]
        result = _merge_spool_results([], text)
        assert "2A" in result

    def test_union_no_overlap(self):
        """Different codes from each source → all present."""
        vision = [{"spool_code": "2A", "center_condition": "Blocked"}]
        text = [{"spool_code": "6C", "center_condition": "Open"}]
        result = _merge_spool_results(vision, text)
        assert "2A" in result
        assert "6C" in result

    def test_overlapping_codes_vision_wins(self):
        """Same code from both — vision data should be preserved."""
        vision = [{"spool_code": "2A", "center_condition": "All ports blocked",
                    "solenoid_a_function": "P→A, B→T"}]
        text = [{"spool_code": "2A", "center_condition": "Blocked",
                 "solenoid_a_function": "wrong data from text"}]
        result = _merge_spool_results(vision, text)
        assert result["2A"]["center_condition"] == "All ports blocked"
        assert result["2A"]["solenoid_a_function"] == "P→A, B→T"

    def test_text_fills_empty_vision_fields(self):
        """Vision has empty field, text has data → field should be filled."""
        vision = [{"spool_code": "2A", "center_condition": "Blocked",
                    "solenoid_a_function": "", "solenoid_b_function": ""}]
        text = [{"spool_code": "2A", "center_condition": "Blocked",
                 "solenoid_a_function": "P→A, B→T",
                 "solenoid_b_function": "P→B, A→T"}]
        result = _merge_spool_results(vision, text)
        assert result["2A"]["solenoid_a_function"] == "P→A, B→T"
        assert result["2A"]["solenoid_b_function"] == "P→B, A→T"

    def test_text_does_not_overwrite_vision_fields(self):
        """Vision has data, text has different data → vision should win."""
        vision = [{"spool_code": "2A", "center_condition": "All ports blocked",
                    "description": "Closed center standard"}]
        text = [{"spool_code": "2A", "center_condition": "Blocked center",
                 "description": "Different description"}]
        result = _merge_spool_results(vision, text)
        assert result["2A"]["center_condition"] == "All ports blocked"
        assert result["2A"]["description"] == "Closed center standard"

    def test_leading_zero_aliases(self):
        """Spool "0A" should also create alias "A"."""
        vision = [{"spool_code": "0A", "center_condition": "Blocked"}]
        result = _merge_spool_results(vision, [])
        assert "0A" in result
        assert "A" in result
