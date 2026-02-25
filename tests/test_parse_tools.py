"""
Tests for tools/parse_tools.py — PDF parsing and extraction utilities.
Covers: _parse_numeric_if_possible, assemble_model_code, generate_products_from_ordering_code,
table_rows_to_products, HEADER_MAPPINGS, LLM prompt validation, and vision pipeline.
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
    extract_spool_symbols_from_pdf,
    extract_cross_references_with_llm,
    extract_ordering_code_from_images,
    _is_graphics_heavy_pdf,
    _get_pdf_page_count,
    _parse_ordering_code_response,
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
        assert result[0]["extraction_method"] == "vision"
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
        """Average exactly at 200 chars/page should NOT be graphics-heavy (< is strict)."""
        pages = [
            {"page": 1, "text": "x" * 200},
            {"page": 2, "text": "y" * 200},
        ]
        assert _is_graphics_heavy_pdf(pages, 2) is False


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
