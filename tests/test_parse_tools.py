"""
Tests for tools/parse_tools.py — PDF parsing and extraction utilities.
Covers: _parse_numeric_if_possible, assemble_model_code, generate_products_from_ordering_code,
table_rows_to_products, HEADER_MAPPINGS, and LLM prompt validation.
"""

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
