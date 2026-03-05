"""
Tests for the Teaching Mode system — few-shot example injection for extraction.

Covers:
- DB CRUD for extraction_examples table
- Image save/load round-trip
- Example selection priority (series match > manufacturer match)
- Text prompt injection (build_few_shot_section)
- Vision prompt injection (build_few_shot_vision_content)
- Token budget / truncation limits
"""

import base64
import json
import os
import tempfile
import uuid

import pytest

from storage.product_db import ProductDB
from tools.agents.teaching import (
    PAGE_TYPES,
    get_relevant_examples,
    build_few_shot_section,
    build_few_shot_vision_content,
)


# ── Fixtures ────────────────────────────────────────────────────────────


@pytest.fixture
def db():
    """Fresh temp-file DB for each test."""
    fd, path = tempfile.mkstemp(suffix=".db")
    os.close(fd)
    database = ProductDB(db_path=path)
    yield database
    database.close()
    os.unlink(path)


@pytest.fixture
def tmp_images(tmp_path):
    """Provide a temporary teaching images directory."""
    return tmp_path / "teaching_images"


def _make_example(**overrides) -> dict:
    """Factory for teaching example dicts (as returned by DB)."""
    defaults = dict(
        id=str(uuid.uuid4()),
        manufacturer="Parker",
        series_prefix="D1VW",
        page_type="ordering_code_table",
        source_pdf="parker_d1vw.pdf",
        page_number=3,
        image_path="parker/page_3.png",
        annotation="Spool options are below the diagram",
        correct_output=json.dumps({"ordering_codes": [{"series": "D1VW"}]}),
        is_active=1,
        times_used=0,
    )
    defaults.update(overrides)
    return defaults


# ── DB CRUD Tests ───────────────────────────────────────────────────────


class TestExtractionExamplesDB:
    """Tests for extraction_examples table CRUD in ProductDB."""

    def test_insert_and_retrieve(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            series_prefix="D1VW",
            source_pdf="test.pdf",
            page_number=2,
            image_path="parker/page_2.png",
            annotation="Test annotation",
            correct_output='{"test": true}',
        )
        assert eid  # non-empty string

        examples = db.get_extraction_examples(
            manufacturer="Parker", page_type="ordering_code_table"
        )
        assert len(examples) == 1
        ex = examples[0]
        assert ex["id"] == eid
        assert ex["manufacturer"] == "Parker"
        assert ex["series_prefix"] == "D1VW"
        assert ex["page_type"] == "ordering_code_table"
        assert ex["annotation"] == "Test annotation"
        assert ex["correct_output"] == '{"test": true}'
        assert ex["is_active"] == 1
        assert ex["times_used"] == 0

    def test_insert_duplicate_replaces(self, db):
        eid1 = db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            source_pdf="test.pdf",
            page_number=2,
            annotation="original",
        )
        # Same manufacturer + source_pdf + page_number + page_type → replaced
        eid2 = db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            source_pdf="test.pdf",
            page_number=2,
            annotation="updated",
        )
        examples = db.get_extraction_examples(manufacturer="Parker")
        assert len(examples) == 1
        assert examples[0]["annotation"] == "updated"

    def test_filter_by_manufacturer(self, db):
        db.insert_extraction_example(manufacturer="Parker", page_type="spool_diagram", source_pdf="a.pdf")
        db.insert_extraction_example(manufacturer="Danfoss", page_type="spool_diagram", source_pdf="b.pdf")

        parker = db.get_extraction_examples(manufacturer="Parker")
        assert len(parker) == 1
        assert parker[0]["manufacturer"] == "Parker"

        danfoss = db.get_extraction_examples(manufacturer="Danfoss")
        assert len(danfoss) == 1

    def test_filter_active_only(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker", page_type="spool_table", source_pdf="a.pdf"
        )
        db.update_extraction_example(eid, is_active=0)

        active = db.get_extraction_examples(manufacturer="Parker", active_only=True)
        assert len(active) == 0

        all_examples = db.get_extraction_examples(manufacturer="Parker", active_only=False)
        assert len(all_examples) == 1

    def test_update_annotation(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker", page_type="ordering_code_table",
            source_pdf="a.pdf", annotation="old"
        )
        db.update_extraction_example(eid, annotation="new annotation")

        examples = db.get_extraction_examples(manufacturer="Parker")
        assert examples[0]["annotation"] == "new annotation"

    def test_update_correct_output(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker", page_type="ordering_code_table",
            source_pdf="a.pdf", correct_output="{}"
        )
        new_output = '{"ordering_codes": [{"series": "D1VW"}]}'
        db.update_extraction_example(eid, correct_output=new_output)

        examples = db.get_extraction_examples(manufacturer="Parker")
        assert examples[0]["correct_output"] == new_output

    def test_delete(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker", page_type="spool_diagram", source_pdf="a.pdf"
        )
        assert db.delete_extraction_example(eid) is True
        assert db.get_extraction_examples(manufacturer="Parker") == []

    def test_delete_nonexistent(self, db):
        assert db.delete_extraction_example("nonexistent-id") is False

    def test_increment_usage(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker", page_type="ordering_code_table", source_pdf="a.pdf"
        )
        db.increment_example_usage(eid)
        db.increment_example_usage(eid)

        examples = db.get_extraction_examples(manufacturer="Parker")
        assert examples[0]["times_used"] == 2


# ── Image Save/Load Tests ──────────────────────────────────────────────


class TestTeachingImages:
    """Tests for save_teaching_image / load_teaching_image round-trip."""

    def test_save_and_load_roundtrip(self, tmp_path, monkeypatch):
        from tools.agents import base as base_mod

        monkeypatch.setattr(base_mod, "_TEACHING_IMAGES_DIR", tmp_path)

        # Create a small valid PNG-like payload
        original_data = b"\x89PNG\r\n\x1a\n" + b"\x00" * 100
        original_b64 = base64.b64encode(original_data).decode("utf-8")

        rel_path = base_mod.save_teaching_image(
            original_b64, "Bosch Rexroth", "page_5.png"
        )

        assert rel_path == "bosch_rexroth/page_5.png"
        assert (tmp_path / "bosch_rexroth" / "page_5.png").exists()

        loaded_b64 = base_mod.load_teaching_image(rel_path)
        assert loaded_b64 == original_b64

    def test_load_missing_returns_none(self, tmp_path, monkeypatch):
        from tools.agents import base as base_mod

        monkeypatch.setattr(base_mod, "_TEACHING_IMAGES_DIR", tmp_path)

        result = base_mod.load_teaching_image("nonexistent/file.png")
        assert result is None

    def test_sanitize_slug(self):
        from tools.agents.base import _sanitize_slug

        assert _sanitize_slug("Bosch Rexroth") == "bosch_rexroth"
        assert _sanitize_slug("Parker-Hannifin") == "parker_hannifin"
        assert _sanitize_slug("MOOG") == "moog"
        assert _sanitize_slug("  Danfoss  ") == "danfoss"


# ── Example Selection Tests ────────────────────────────────────────────


class TestGetRelevantExamples:
    """Tests for get_relevant_examples() selection priority."""

    def test_empty_when_no_examples(self, db):
        result = get_relevant_examples(db, "Parker", "ordering_code_table")
        assert result == []

    def test_empty_when_no_manufacturer(self, db):
        result = get_relevant_examples(db, "", "ordering_code_table")
        assert result == []

    def test_empty_when_no_page_type(self, db):
        result = get_relevant_examples(db, "Parker", "")
        assert result == []

    def test_returns_matching_examples(self, db):
        db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            source_pdf="a.pdf",
            annotation="Test",
            correct_output='{"test": 1}',
        )
        result = get_relevant_examples(db, "Parker", "ordering_code_table")
        assert len(result) == 1
        assert result[0]["manufacturer"] == "Parker"

    def test_series_match_preferred(self, db):
        # Insert two examples: one with matching series, one without
        db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            series_prefix="D1VW",
            source_pdf="a.pdf",
            annotation="D1VW specific",
        )
        db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            series_prefix="D3W",
            source_pdf="b.pdf",
            annotation="D3W specific",
        )

        result = get_relevant_examples(
            db, "Parker", "ordering_code_table",
            series_prefix="D1VW", max_examples=1
        )
        assert len(result) == 1
        assert result[0]["series_prefix"] == "D1VW"

    def test_never_cross_manufacturer(self, db):
        db.insert_extraction_example(
            manufacturer="Danfoss",
            page_type="ordering_code_table",
            source_pdf="a.pdf",
        )
        result = get_relevant_examples(db, "Parker", "ordering_code_table")
        assert result == []

    def test_max_examples_respected(self, db):
        for i in range(5):
            db.insert_extraction_example(
                manufacturer="Parker",
                page_type="spool_diagram",
                source_pdf=f"file_{i}.pdf",
            )
        result = get_relevant_examples(
            db, "Parker", "spool_diagram", max_examples=2
        )
        assert len(result) == 2

    def test_increments_usage(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            source_pdf="a.pdf",
        )
        get_relevant_examples(db, "Parker", "ordering_code_table")

        examples = db.get_extraction_examples(manufacturer="Parker", active_only=False)
        assert examples[0]["times_used"] == 1

    def test_inactive_excluded(self, db):
        eid = db.insert_extraction_example(
            manufacturer="Parker",
            page_type="ordering_code_table",
            source_pdf="a.pdf",
        )
        db.update_extraction_example(eid, is_active=0)

        result = get_relevant_examples(db, "Parker", "ordering_code_table")
        assert result == []


# ── Text Prompt Injection Tests ────────────────────────────────────────


class TestBuildFewShotSection:
    """Tests for build_few_shot_section() text output."""

    def test_empty_when_no_examples(self):
        assert build_few_shot_section([]) == ""

    def test_contains_header(self):
        examples = [_make_example()]
        result = build_few_shot_section(examples)
        assert "REFERENCE EXAMPLES" in result

    def test_contains_manufacturer_label(self):
        examples = [_make_example(manufacturer="Parker", series_prefix="D1VW")]
        result = build_few_shot_section(examples)
        assert "Parker - D1VW" in result

    def test_contains_annotation(self):
        examples = [_make_example(annotation="Check the small table")]
        result = build_few_shot_section(examples)
        assert "Check the small table" in result

    def test_contains_correct_output(self):
        output = '{"ordering_codes": [{"series": "D1VW"}]}'
        examples = [_make_example(correct_output=output)]
        result = build_few_shot_section(examples)
        assert "D1VW" in result
        assert "```json" in result

    def test_output_truncated_when_too_long(self):
        long_output = "x" * 5000
        examples = [_make_example(correct_output=long_output)]
        result = build_few_shot_section(examples, max_output_chars=100)
        assert "(truncated)" in result

    def test_multiple_examples(self):
        examples = [
            _make_example(manufacturer="Parker", series_prefix="D1VW"),
            _make_example(manufacturer="Parker", series_prefix="D3W"),
        ]
        result = build_few_shot_section(examples)
        assert "Example 1" in result
        assert "Example 2" in result

    def test_ends_with_separator(self):
        examples = [_make_example()]
        result = build_few_shot_section(examples)
        assert "---" in result


# ── Vision Prompt Injection Tests ──────────────────────────────────────


class TestBuildFewShotVisionContent:
    """Tests for build_few_shot_vision_content() content blocks."""

    def test_empty_when_no_examples(self):
        assert build_few_shot_vision_content([]) == []

    def test_returns_content_blocks(self):
        examples = [_make_example(image_path="")]
        result = build_few_shot_vision_content(examples)
        assert len(result) >= 2  # header + example text + separator
        assert result[0]["type"] == "text"
        assert "REFERENCE EXAMPLES" in result[0]["text"]

    def test_includes_correct_output(self):
        examples = [_make_example(image_path="", correct_output='{"test": 1}')]
        result = build_few_shot_vision_content(examples)
        texts = [b["text"] for b in result if b.get("type") == "text"]
        combined = " ".join(texts)
        assert "test" in combined

    def test_max_image_examples_respected(self, tmp_path, monkeypatch):
        from tools.agents import base as base_mod

        monkeypatch.setattr(base_mod, "_TEACHING_IMAGES_DIR", tmp_path)

        # Create two image files
        for name in ["img1.png", "img2.png"]:
            img_dir = tmp_path / "parker"
            img_dir.mkdir(exist_ok=True)
            (img_dir / name).write_bytes(b"\x89PNG" + b"\x00" * 50)

        examples = [
            _make_example(image_path="parker/img1.png"),
            _make_example(image_path="parker/img2.png"),
        ]
        result = build_few_shot_vision_content(
            examples, max_image_examples=1
        )
        # Count image blocks
        image_blocks = [b for b in result if b.get("type") in ("image", "image_url")]
        assert len(image_blocks) <= 1

    def test_output_truncated_when_too_long(self):
        long_output = "y" * 5000
        examples = [_make_example(image_path="", correct_output=long_output)]
        result = build_few_shot_vision_content(
            examples, max_output_chars=100
        )
        texts = [b["text"] for b in result if b.get("type") == "text"]
        combined = " ".join(texts)
        assert "(truncated)" in combined

    def test_ends_with_separator(self):
        examples = [_make_example(image_path="")]
        result = build_few_shot_vision_content(examples)
        last_text = [b for b in result if b.get("type") == "text"][-1]
        assert "NEW pages below" in last_text["text"]


# ── PAGE_TYPES constant ────────────────────────────────────────────────


class TestPageTypes:
    """Verify the PAGE_TYPES constant is correct."""

    def test_contains_expected_types(self):
        assert "ordering_code_table" in PAGE_TYPES
        assert "spool_diagram" in PAGE_TYPES
        assert "spool_table" in PAGE_TYPES
        assert "spec_table" in PAGE_TYPES

    def test_exactly_four_types(self):
        assert len(PAGE_TYPES) == 4
