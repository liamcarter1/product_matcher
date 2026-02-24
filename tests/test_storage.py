"""
Tests for storage/product_db.py — ProductDB
Covers: CRUD operations, spec_comparison scoring, fuzzy lookup,
model code decoding, confirmed equivalents, synonyms, and edge cases.
"""

import os
import uuid
import pytest
import tempfile

from models import (
    HydraulicProduct, ModelCodePattern, ScoreBreakdown,
    SCORE_WEIGHTS, CONFIDENCE_THRESHOLD,
)
from storage.product_db import ProductDB


# ── Fixtures ──────────────────────────────────────────────────────────


@pytest.fixture
def db():
    """Fresh in-memory-like DB for each test (temp file for SQLite)."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = ProductDB(db_path=path)
    yield database
    database.close()
    os.unlink(path)


def _make_product(**overrides) -> HydraulicProduct:
    defaults = dict(
        id=str(uuid.uuid4()),
        company="Danfoss",
        model_code="DG4V-3-2A-M-U-H7-60",
        product_name="Directional Control Valve",
        category="directional_valves",
        max_pressure_bar=315.0,
        max_flow_lpm=120.0,
        valve_size="CETOP 5",
        spool_type="2A - All ports open to tank in center",
        num_positions=3,
        num_ports=4,
        actuator_type="solenoid",
        coil_voltage="24VDC",
        port_size="G3/8",
        mounting="subplate",
        seal_material="NBR",
    )
    defaults.update(overrides)
    return HydraulicProduct(**defaults)


# ── CRUD Operations ──────────────────────────────────────────────────


class TestProductCRUD:
    def test_insert_and_retrieve(self, db):
        product = _make_product()
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved is not None
        assert retrieved.model_code == product.model_code
        assert retrieved.spool_type == product.spool_type
        assert retrieved.max_pressure_bar == 315.0

    def test_insert_product_with_string_mounting(self, db):
        """The original bug: mounting must be stored and retrieved as string."""
        product = _make_product(mounting="5")
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved.mounting == "5"
        assert isinstance(retrieved.mounting, str)

    def test_delete_product(self, db):
        product = _make_product()
        db.insert_product(product)
        db.delete_product(product.id)
        assert db.get_product(product.id) is None

    def test_get_all_products(self, db):
        p1 = _make_product(model_code="A")
        p2 = _make_product(id=str(uuid.uuid4()), model_code="B", company="Bosch Rexroth")
        db.insert_product(p1)
        db.insert_product(p2)
        all_products = db.get_all_products()
        assert len(all_products) == 2

    def test_get_all_products_filtered_by_company(self, db):
        p1 = _make_product(model_code="A")
        p2 = _make_product(id=str(uuid.uuid4()), model_code="B", company="Bosch Rexroth")
        db.insert_product(p1)
        db.insert_product(p2)
        danfoss = db.get_all_products(company="Danfoss")
        assert len(danfoss) == 1
        assert danfoss[0].company == "Danfoss"

    def test_get_products_by_category(self, db):
        p1 = _make_product(model_code="A", category="directional_valves")
        p2 = _make_product(id=str(uuid.uuid4()), model_code="B", category="pumps")
        db.insert_product(p1)
        db.insert_product(p2)
        valves = db.get_products_by_category("directional_valves")
        assert len(valves) == 1

    def test_get_companies(self, db):
        p1 = _make_product()
        p2 = _make_product(id=str(uuid.uuid4()), model_code="B", company="Parker")
        db.insert_product(p1)
        db.insert_product(p2)
        companies = db.get_companies()
        assert set(companies) == {"Danfoss", "Parker"}

    def test_get_all_model_codes(self, db):
        p1 = _make_product(model_code="CODE-A")
        p2 = _make_product(id=str(uuid.uuid4()), model_code="CODE-B")
        db.insert_product(p1)
        db.insert_product(p2)
        codes = db.get_all_model_codes()
        assert "CODE-A" in codes
        assert "CODE-B" in codes

    def test_model_code_decoded_roundtrip(self, db):
        """model_code_decoded dict should survive JSON serialisation."""
        decoded = {"series": "4WE6", "spool_type": "D - P to A, B to T"}
        product = _make_product(model_code_decoded=decoded)
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved.model_code_decoded == decoded

    def test_insert_or_replace_updates_existing(self, db):
        product = _make_product(spool_type="D")
        db.insert_product(product)
        product.spool_type = "2A"
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved.spool_type == "2A"

    def test_none_optional_fields(self, db):
        product = _make_product(
            spool_type=None, valve_size=None, coil_voltage=None,
            max_pressure_bar=None,
        )
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved.spool_type is None
        assert retrieved.valve_size is None
        assert retrieved.coil_voltage is None
        assert retrieved.max_pressure_bar is None


# ── Spec Comparison ──────────────────────────────────────────────────


class TestSpecComparison:
    def test_identical_products_high_score(self, db):
        p1 = _make_product()
        p2 = _make_product(id=str(uuid.uuid4()))
        score, breakdown = db.spec_comparison(p1, p2, semantic_score=1.0)
        assert score >= 0.9

    def test_different_category_caps_at_03(self, db):
        p1 = _make_product(category="directional_valves")
        p2 = _make_product(id=str(uuid.uuid4()), category="pumps")
        score, breakdown = db.spec_comparison(p1, p2, semantic_score=1.0)
        assert score <= 0.3
        assert breakdown.category_match == 0.0

    def test_unknown_category_scores_05(self, db):
        p1 = _make_product(category="")
        p2 = _make_product(id=str(uuid.uuid4()), category="directional_valves")
        _, breakdown = db.spec_comparison(p1, p2)
        assert breakdown.category_match == 0.5

    def test_spool_type_exact_match(self, db):
        p1 = _make_product(spool_type="2A")
        p2 = _make_product(id=str(uuid.uuid4()), spool_type="2A")
        _, breakdown = db.spec_comparison(p1, p2)
        assert breakdown.spool_function_match == 1.0

    def test_spool_type_mismatch(self, db):
        p1 = _make_product(spool_type="2A - All ports open")
        p2 = _make_product(id=str(uuid.uuid4()), spool_type="D - P blocked")
        _, breakdown = db.spec_comparison(p1, p2)
        assert breakdown.spool_function_match < 0.5

    def test_spool_type_unknown_scores_05(self, db):
        p1 = _make_product(spool_type=None)
        p2 = _make_product(id=str(uuid.uuid4()), spool_type="2A")
        _, breakdown = db.spec_comparison(p1, p2)
        assert breakdown.spool_function_match == 0.5

    def test_numerical_match_identical(self, db):
        assert ProductDB._numerical_match(315.0, 315.0) == 1.0

    def test_numerical_match_close(self, db):
        score = ProductDB._numerical_match(315.0, 310.0)
        assert score > 0.95  # 5/315 ≈ 1.6% diff

    def test_numerical_match_far(self, db):
        score = ProductDB._numerical_match(315.0, 100.0)
        assert score < 0.7

    def test_numerical_match_none(self, db):
        assert ProductDB._numerical_match(None, 315.0) == 0.5
        assert ProductDB._numerical_match(315.0, None) == 0.5

    def test_numerical_match_both_zero(self, db):
        assert ProductDB._numerical_match(0.0, 0.0) == 1.0

    def test_exact_match_case_insensitive(self, db):
        assert ProductDB._exact_match("24VDC", "24vdc") == 1.0

    def test_exact_match_normalised(self, db):
        """'24 VDC' vs '24VDC' should score high."""
        score = ProductDB._exact_match("24 VDC", "24VDC")
        assert score >= 0.9

    def test_exact_match_containment(self, db):
        score = ProductDB._exact_match("NBR", "NBR/FKM")
        assert score >= 0.7

    def test_exact_match_none(self, db):
        assert ProductDB._exact_match(None, "24VDC") == 0.5
        assert ProductDB._exact_match("24VDC", None) == 0.5

    def test_exact_match_completely_different(self, db):
        assert ProductDB._exact_match("solenoid", "manual") == 0.0

    def test_temp_range_covered(self, db):
        p1 = _make_product(operating_temp_min_c=-20.0, operating_temp_max_c=70.0)
        p2 = _make_product(
            id=str(uuid.uuid4()),
            operating_temp_min_c=-30.0, operating_temp_max_c=80.0,
        )
        score = ProductDB._temp_range_match(p1, p2)
        assert score == 1.0

    def test_temp_range_partial(self, db):
        p1 = _make_product(operating_temp_min_c=-20.0, operating_temp_max_c=70.0)
        p2 = _make_product(
            id=str(uuid.uuid4()),
            operating_temp_min_c=-10.0, operating_temp_max_c=80.0,
        )
        score = ProductDB._temp_range_match(p1, p2)
        assert score == 0.5  # max_ok but not min_ok

    def test_temp_range_unknown(self, db):
        p1 = _make_product(operating_temp_min_c=None, operating_temp_max_c=None)
        p2 = _make_product(id=str(uuid.uuid4()))
        score = ProductDB._temp_range_match(p1, p2)
        assert score == 0.5

    def test_spec_coverage_calculation(self, db):
        """Products with more populated specs should have higher coverage."""
        sparse = _make_product(
            id=str(uuid.uuid4()),
            max_pressure_bar=None, max_flow_lpm=None,
            valve_size=None, spool_type=None,
            coil_voltage=None, actuator_type=None,
            port_size=None, mounting=None, seal_material=None,
        )
        rich = _make_product(id=str(uuid.uuid4()))
        _, breakdown = db.spec_comparison(sparse, rich)
        # Coverage is based on both having a value
        assert breakdown.spec_coverage < 1.0

    def test_semantic_score_passthrough(self, db):
        p1 = _make_product()
        p2 = _make_product(id=str(uuid.uuid4()))
        _, breakdown = db.spec_comparison(p1, p2, semantic_score=0.85)
        assert breakdown.semantic_similarity == 0.85

    def test_score_clamped_to_01(self, db):
        p1 = _make_product()
        p2 = _make_product(id=str(uuid.uuid4()))
        score, _ = db.spec_comparison(p1, p2, semantic_score=1.0)
        assert 0.0 <= score <= 1.0

    def test_mounting_pattern_fallback(self, db):
        """When mounting doesn't match, mounting_pattern should be tried."""
        p1 = _make_product(mounting="type_A", mounting_pattern="ISO 4401-05")
        p2 = _make_product(
            id=str(uuid.uuid4()),
            mounting="type_B", mounting_pattern="ISO 4401-05",
        )
        _, breakdown = db.spec_comparison(p1, p2)
        assert breakdown.mounting_match >= 0.9  # Fallback to mounting_pattern


# ── Score Weights Validation ─────────────────────────────────────────


class TestScoreWeights:
    def test_weights_sum_close_to_one(self):
        """SCORE_WEIGHTS should sum to ~1.0 (minus spec_coverage which is not weighted)."""
        total = sum(SCORE_WEIGHTS.values())
        assert 0.99 <= total <= 1.01, f"SCORE_WEIGHTS sum to {total}, expected ~1.0"

    def test_spool_weight_nonzero(self):
        assert SCORE_WEIGHTS.get("spool_function_match", 0) > 0

    def test_all_weight_keys_are_breakdown_fields(self):
        for key in SCORE_WEIGHTS:
            assert key in ScoreBreakdown.model_fields, (
                f"Weight key '{key}' is not a ScoreBreakdown field"
            )


# ── Fuzzy Lookup ─────────────────────────────────────────────────────


class TestFuzzyLookup:
    def test_exact_match(self, db):
        product = _make_product(model_code="DG4V-3-2A-M-U-H7-60")
        db.insert_product(product)
        results = db.fuzzy_lookup_model("DG4V-3-2A-M-U-H7-60")
        assert len(results) >= 1
        assert results[0][0].model_code == product.model_code
        assert results[0][1] >= 0.9

    def test_partial_match(self, db):
        product = _make_product(model_code="DG4V-3-2A-M-U-H7-60")
        db.insert_product(product)
        results = db.fuzzy_lookup_model("DG4V-3-2A")
        assert len(results) >= 1

    def test_case_insensitive(self, db):
        product = _make_product(model_code="DG4V-3-2A")
        db.insert_product(product)
        results = db.fuzzy_lookup_model("dg4v-3-2a")
        assert len(results) >= 1

    def test_no_match_below_threshold(self, db):
        product = _make_product(model_code="DG4V-3-2A")
        db.insert_product(product)
        results = db.fuzzy_lookup_model("COMPLETELY-DIFFERENT-CODE", threshold=90)
        # Should be empty or very low score
        for _, score in results:
            assert score < 0.6

    def test_empty_database(self, db):
        results = db.fuzzy_lookup_model("DG4V-3-2A")
        assert results == []

    def test_company_filter(self, db):
        p1 = _make_product(model_code="DG4V-3-2A", company="Danfoss")
        p2 = _make_product(id=str(uuid.uuid4()), model_code="DG4V-3-2A", company="Parker")
        db.insert_product(p1)
        db.insert_product(p2)
        results = db.fuzzy_lookup_model("DG4V-3-2A", company="Danfoss")
        assert all(r[0].company == "Danfoss" for r in results)

    def test_limit_respected(self, db):
        for i in range(10):
            p = _make_product(id=str(uuid.uuid4()), model_code=f"VALVE-{i:03d}")
            db.insert_product(p)
        results = db.fuzzy_lookup_model("VALVE", limit=3)
        assert len(results) <= 3


# ── Model Code Decoding ─────────────────────────────────────────────


class TestModelCodeDecoding:
    def test_decode_with_matching_pattern(self, db):
        pattern = ModelCodePattern(
            company="Danfoss",
            series="DG4V",
            segment_position=0,
            segment_name="spool_type",
            code_value="2A",
            decoded_value="2A - All ports open to tank",
            maps_to_field="spool_type",
        )
        db.insert_model_code_pattern(pattern)
        decoded = db.decode_model_code("DG4V-2A-M", "Danfoss")
        assert decoded.get("series") == "DG4V"
        assert decoded.get("spool_type") == "2A - All ports open to tank"

    def test_decode_no_matching_series(self, db):
        decoded = db.decode_model_code("UNKNOWN-123", "Danfoss")
        assert decoded == {}

    def test_decode_stores_field_mapping(self, db):
        pattern = ModelCodePattern(
            company="Danfoss",
            series="DG4V",
            segment_position=0,
            segment_name="spool_type",
            code_value="D",
            decoded_value="D - P blocked, A&B to T",
            maps_to_field="spool_type",
        )
        db.insert_model_code_pattern(pattern)
        decoded = db.decode_model_code("DG4V-D-M", "Danfoss")
        assert decoded.get("_field_spool_type") == "spool_type"


# ── Confirmed Equivalents ───────────────────────────────────────────


class TestConfirmedEquivalents:
    def test_insert_and_retrieve(self, db):
        product = _make_product(model_code="DG4V-3-2A")
        db.insert_product(product)
        db.insert_confirmed_equivalent(
            competitor_code="4WE6-D6X",
            competitor_company="Bosch Rexroth",
            my_company_code="DG4V-3-2A",
        )
        result = db.get_confirmed_equivalent("4WE6-D6X")
        assert result is not None
        assert result.model_code == "DG4V-3-2A"

    def test_case_insensitive_lookup(self, db):
        product = _make_product(model_code="DG4V-3-2A")
        db.insert_product(product)
        db.insert_confirmed_equivalent(
            competitor_code="4WE6-D6X",
            competitor_company="Bosch Rexroth",
            my_company_code="DG4V-3-2A",
        )
        result = db.get_confirmed_equivalent("4we6-d6x")
        assert result is not None

    def test_no_match(self, db):
        result = db.get_confirmed_equivalent("NONEXISTENT")
        assert result is None


# ── Synonyms ─────────────────────────────────────────────────────────


class TestSynonyms:
    def test_insert_and_resolve(self, db):
        db.insert_synonym("Vickers", "Danfoss")
        assert db.resolve_synonym("Vickers") == "Danfoss"

    def test_case_insensitive(self, db):
        db.insert_synonym("vickers", "Danfoss")
        assert db.resolve_synonym("VICKERS") == "Danfoss"

    def test_unknown_returns_original(self, db):
        assert db.resolve_synonym("Unknown") == "Unknown"


# ── Feedback ─────────────────────────────────────────────────────────


class TestFeedback:
    def test_store_and_retrieve(self, db):
        db.store_feedback(
            query="Find Danfoss equiv for 4WE6",
            competitor_code="4WE6-D6X",
            my_company_code="DG4V-3-2A",
            confidence=0.85,
            thumbs_up=True,
        )
        feedback = db.get_feedback()
        assert len(feedback) == 1
        assert feedback[0]["thumbs_up"] == 1
        assert feedback[0]["confidence_score"] == 0.85

    def test_feedback_limit(self, db):
        for i in range(20):
            db.store_feedback(f"q{i}", f"c{i}", f"m{i}", 0.5, True)
        feedback = db.get_feedback(limit=5)
        assert len(feedback) == 5


class TestExtraSpecs:
    """Tests for extra_specs JSON column storage and retrieval."""

    def test_extra_specs_roundtrip(self, db):
        """Extra specs dict should survive insert → retrieve cycle."""
        product = HydraulicProduct(
            id=str(uuid.uuid4()), company="TestCo",
            model_code="ES-001",
            extra_specs={"design_number": "42", "flow_class": "high"},
        )
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved is not None
        assert retrieved.extra_specs == {"design_number": "42", "flow_class": "high"}

    def test_extra_specs_none_default(self, db):
        """Products without extra_specs should get empty dict."""
        product = HydraulicProduct(
            id=str(uuid.uuid4()), company="TestCo",
            model_code="ES-002",
        )
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved is not None
        assert retrieved.extra_specs == {} or retrieved.extra_specs is None

    def test_extra_specs_with_spool_function_flat(self, db):
        """Flat spool function keys should survive roundtrip."""
        product = HydraulicProduct(
            id=str(uuid.uuid4()), company="TestCo",
            model_code="ES-003",
            extra_specs={
                "center_condition": "All ports blocked",
                "solenoid_a_energised": "P→A, B→T",
                "solenoid_b_energised": "P→B, A→T",
                "canonical_spool_pattern": "BLOCKED|AB-PT|AT-PB",
                "design_number": "5",
            },
        )
        db.insert_product(product)
        retrieved = db.get_product(product.id)
        assert retrieved.extra_specs["canonical_spool_pattern"] == "BLOCKED|AB-PT|AT-PB"
        assert retrieved.extra_specs["center_condition"] == "All ports blocked"
        assert retrieved.extra_specs["design_number"] == "5"

    def test_schema_migration_idempotent(self, db):
        """Running migration twice should not error."""
        db._migrate_schema()
        db._migrate_schema()  # Should be safe to call again


class TestCanonicalSpoolMatching:
    """Tests for canonical spool pattern matching in spec_comparison."""

    def test_canonical_pattern_match_overrides_text(self, db):
        """Two products with different spool codes but same canonical pattern should score 1.0."""
        comp = HydraulicProduct(
            id="c1", company="Danfoss", model_code="C1",
            category="directional_valves", spool_type="2A",
            extra_specs={"canonical_spool_pattern": "BLOCKED|PA-BT|AT-PB"},
        )
        cand = HydraulicProduct(
            id="c2", company="Bosch", model_code="C2",
            category="directional_valves", spool_type="D",
            extra_specs={"canonical_spool_pattern": "BLOCKED|PA-BT|AT-PB"},
        )
        _, breakdown = db.spec_comparison(comp, cand)
        assert breakdown.spool_function_match == 1.0

    def test_canonical_pattern_mismatch(self, db):
        """Different canonical patterns should score 0.0."""
        comp = HydraulicProduct(
            id="c3", company="Danfoss", model_code="C3",
            category="directional_valves", spool_type="2A",
            extra_specs={"canonical_spool_pattern": "BLOCKED|PA-BT|AT-PB"},
        )
        cand = HydraulicProduct(
            id="c4", company="Bosch", model_code="C4",
            category="directional_valves", spool_type="H",
            extra_specs={"canonical_spool_pattern": "OPEN|PA-BT|AT-PB"},
        )
        _, breakdown = db.spec_comparison(comp, cand)
        assert breakdown.spool_function_match == 0.0

    def test_falls_back_to_text_when_no_canonical(self, db):
        """Without canonical patterns, falls back to text comparison."""
        comp = HydraulicProduct(
            id="c5", company="Danfoss", model_code="C5",
            category="directional_valves", spool_type="2A",
        )
        cand = HydraulicProduct(
            id="c6", company="Bosch", model_code="C6",
            category="directional_valves", spool_type="2A",
        )
        _, breakdown = db.spec_comparison(comp, cand)
        assert breakdown.spool_function_match == 1.0  # text match

    def test_extra_specs_coverage_bonus(self, db):
        """Common extra_specs keys that match should boost spec_coverage."""
        comp = HydraulicProduct(
            id="c7", company="A", model_code="C7",
            extra_specs={"design_number": "5", "flow_class": "high"},
        )
        cand = HydraulicProduct(
            id="c8", company="B", model_code="C8",
            extra_specs={"design_number": "5", "flow_class": "low"},
        )
        _, breakdown = db.spec_comparison(comp, cand)
        # design_number matches (5==5), flow_class doesn't — coverage should reflect this
        assert breakdown.spec_coverage > 0.0


# ── Series Cross-Reference ────────────────────────────────────────────


class TestSeriesCrossReference:
    """Tests for the series_cross_reference table and lookup methods."""

    def test_insert_and_lookup(self, db):
        """Should insert a cross-reference and find it by competitor prefix."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3",
            competitor_series="D1VW",
            competitor_company="Parker",
            product_type="Directional Valve",
        )
        results = db.lookup_series_by_competitor_prefix("D1VW004CNJW", "Parker")
        assert len(results) >= 1
        assert results[0]["my_company_series"] == "DG4V-3"

    def test_prefix_match_not_exact(self, db):
        """Lookup should match prefix, not require exact match."""
        db.insert_series_cross_reference(
            my_company_series="KFDG4V-3",
            competitor_series="4WRE",
            competitor_company="Bosch Rexroth",
        )
        results = db.lookup_series_by_competitor_prefix("4WREE6-04-3X", "Bosch Rexroth")
        assert len(results) >= 1
        assert results[0]["my_company_series"] == "KFDG4V-3"

    def test_no_match_returns_empty(self, db):
        """Should return empty list when no prefix matches."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3",
            competitor_series="D1VW",
            competitor_company="Parker",
        )
        results = db.lookup_series_by_competitor_prefix("XXXX-NOMATCH", "Parker")
        assert results == []

    def test_longest_prefix_first(self, db):
        """Should return longest (most specific) prefix match first."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3",
            competitor_series="D1V",
            competitor_company="Parker",
        )
        db.insert_series_cross_reference(
            my_company_series="DG4V-3S",
            competitor_series="D1VW",
            competitor_company="Parker",
        )
        results = db.lookup_series_by_competitor_prefix("D1VW004CNJW", "Parker")
        assert len(results) == 2
        # Most specific (D1VW) should come first
        assert results[0]["competitor_series"] == "D1VW"

    def test_company_filter(self, db):
        """Should only return matches for the specified competitor company."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3",
            competitor_series="D1VW",
            competitor_company="Parker",
        )
        db.insert_series_cross_reference(
            my_company_series="KFDG4V-3",
            competitor_series="D1VW",
            competitor_company="Bosch Rexroth",
        )
        results = db.lookup_series_by_competitor_prefix("D1VW004", "Parker")
        assert len(results) == 1
        assert results[0]["competitor_company"] == "Parker"

    def test_get_all_cross_references(self, db):
        """Should return all stored cross-references."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3", competitor_series="D1VW",
            competitor_company="Parker",
        )
        db.insert_series_cross_reference(
            my_company_series="KFDG4V", competitor_series="4WRE",
            competitor_company="Bosch Rexroth",
        )
        all_refs = db.get_all_cross_references()
        assert len(all_refs) == 2

    def test_delete_by_source(self, db):
        """Should delete cross-references from a specific source document."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3", competitor_series="D1VW",
            competitor_company="Parker", source_document="xref.pdf",
        )
        db.insert_series_cross_reference(
            my_company_series="KFDG4V", competitor_series="4WRE",
            competitor_company="Bosch Rexroth", source_document="other.pdf",
        )
        db.delete_cross_references_by_source("xref.pdf")
        remaining = db.get_all_cross_references()
        assert len(remaining) == 1
        assert remaining[0]["source_document"] == "other.pdf"

    def test_lookup_without_company_filter(self, db):
        """Should search all companies when competitor_company is None."""
        db.insert_series_cross_reference(
            my_company_series="DG4V-3", competitor_series="D1VW",
            competitor_company="Parker",
        )
        results = db.lookup_series_by_competitor_prefix("D1VW004CNJW")
        assert len(results) >= 1


class TestSpoolTypeReference:
    """Tests for the spool_type_reference table and CRUD methods."""

    def test_insert_and_retrieve(self, db):
        """Should insert a spool type reference and retrieve it."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3",
            manufacturer="Danfoss",
            spool_code="2A",
            description="Closed center, standard crossover",
            center_condition="All ports blocked",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert len(refs) >= 1
        assert refs[0]["spool_code"] == "2A"
        assert refs[0]["center_condition"] == "All ports blocked"

    def test_upsert_on_duplicate(self, db):
        """Should update (not duplicate) on same (series, manufacturer, code)."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C",
            description="Old description",
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C",
            description="New description",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        codes = [r["spool_code"] for r in refs if r["spool_code"] == "0C"]
        assert len(codes) == 1

    def test_fuzzy_series_match_stored_is_prefix(self, db):
        """Stored 'DG4V' should match query for 'DG4V-3'."""
        db.insert_spool_type_reference(
            series_prefix="DG4V", manufacturer="Danfoss", spool_code="2A",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert len(refs) >= 1
        assert refs[0]["spool_code"] == "2A"

    def test_fuzzy_series_match_query_is_prefix(self, db):
        """Query for 'DG4V' should match stored 'DG4V-3'."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="6C",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V", manufacturer="Danfoss")
        assert len(refs) >= 1
        assert refs[0]["spool_code"] == "6C"

    def test_get_spool_codes_for_series(self, db):
        """Should return sorted list of spool code strings."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C",
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="6C",
        )
        codes = db.get_spool_codes_for_series("DG4V-3", "Danfoss")
        assert codes == ["0C", "2A", "6C"]

    def test_bulk_insert(self, db):
        """Should bulk insert multiple references, skipping duplicates."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C",
        )
        refs = [
            {"series_prefix": "DG4V-3", "manufacturer": "Danfoss", "spool_code": "0C"},  # dupe
            {"series_prefix": "DG4V-3", "manufacturer": "Danfoss", "spool_code": "2A"},
            {"series_prefix": "DG4V-3", "manufacturer": "Danfoss", "spool_code": "6C"},
        ]
        count = db.bulk_insert_spool_type_references(refs)
        assert count == 2  # 0C already existed
        all_codes = db.get_spool_codes_for_series("DG4V-3", "Danfoss")
        assert "0C" in all_codes
        assert "2A" in all_codes
        assert "6C" in all_codes

    def test_delete_reference(self, db):
        """Should delete a spool type reference by ID."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        ref_id = refs[0]["id"]
        db.delete_spool_type_reference(ref_id)
        remaining = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert len(remaining) == 0

    def test_manufacturer_isolation(self, db):
        """Spool types for different manufacturers should not mix."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        db.insert_spool_type_reference(
            series_prefix="4WE6", manufacturer="Bosch Rexroth", spool_code="D",
        )
        danfoss = db.get_spool_codes_for_series("DG4V-3", "Danfoss")
        bosch = db.get_spool_codes_for_series("4WE6", "Bosch Rexroth")
        assert "2A" in danfoss
        assert "D" not in danfoss
        assert "D" in bosch
        assert "2A" not in bosch

    def test_is_primary_default_false(self, db):
        """New spool references should default to is_primary=0."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert refs[0].get("is_primary") == 0

    def test_insert_with_is_primary(self, db):
        """Should insert a spool type with is_primary=True."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
            is_primary=True,
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert refs[0].get("is_primary") == 1

    def test_update_spool_type_primary(self, db):
        """Should toggle is_primary on an existing reference."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        ref_id = refs[0]["id"]
        assert refs[0].get("is_primary") == 0

        db.update_spool_type_primary(ref_id, True)
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert refs[0].get("is_primary") == 1

        db.update_spool_type_primary(ref_id, False)
        refs = db.get_spool_type_references(series_prefix="DG4V-3", manufacturer="Danfoss")
        assert refs[0].get("is_primary") == 0

    def test_get_primary_spool_codes(self, db):
        """Should return only primary spool codes."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A", is_primary=True,
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="6C", is_primary=True,
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C", is_primary=False,
        )
        codes = db.get_primary_spool_codes("DG4V-3", "Danfoss")
        assert codes == ["2A", "6C"]
        assert "0C" not in codes

    def test_get_primary_spool_codes_empty_when_none_primary(self, db):
        """Should return empty list when no spools are marked primary."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        codes = db.get_primary_spool_codes("DG4V-3", "Danfoss")
        assert codes == []

    def test_set_all_spools_primary(self, db):
        """Bulk set primary for specific codes."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A",
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="6C",
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="0C",
        )
        db.set_all_spools_primary(manufacturer="Danfoss", spool_codes=["2A", "6C"])
        codes = db.get_primary_spool_codes("DG4V-3", "Danfoss")
        assert codes == ["2A", "6C"]

    def test_clear_all_spools_primary(self, db):
        """Bulk clear primary flags."""
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="2A", is_primary=True,
        )
        db.insert_spool_type_reference(
            series_prefix="DG4V-3", manufacturer="Danfoss", spool_code="6C", is_primary=True,
        )
        db.clear_all_spools_primary(manufacturer="Danfoss")
        codes = db.get_primary_spool_codes("DG4V-3", "Danfoss")
        assert codes == []
