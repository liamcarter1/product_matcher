"""
Tests for DocumentLoader service.
Tests document loading, chunking, and column detection.
"""

import pytest
import pandas as pd
from pathlib import Path

from app.services.document_loader import DocumentLoader, ProcessedDocument


class TestDocumentLoaderInit:
    """Tests for DocumentLoader initialization."""

    def test_default_chunk_settings(self):
        """Test default chunk size and overlap."""
        loader = DocumentLoader()
        assert loader.chunk_size == 1000
        assert loader.chunk_overlap == 200

    def test_custom_chunk_settings(self):
        """Test custom chunk size and overlap."""
        loader = DocumentLoader(chunk_size=500, chunk_overlap=100)
        assert loader.chunk_size == 500
        assert loader.chunk_overlap == 100


class TestColumnDetection:
    """Tests for auto-detection of column types."""

    def test_detect_danfoss_part_column(self):
        """Test detection of Danfoss part number columns."""
        loader = DocumentLoader()

        test_cases = [
            ["danfoss_part", "other"],
            ["Danfoss Part Number", "description"],
            ["our_part", "competitor"],
            ["part_number", "brand"],
        ]

        for columns in test_cases:
            result = loader._detect_columns(columns)
            assert result["danfoss_part"] is not None, f"Failed for columns: {columns}"

    def test_detect_competitor_columns(self):
        """Test detection of competitor brand and part columns."""
        loader = DocumentLoader()

        columns = ["danfoss_part", "competitor_brand", "competitor_part", "description"]
        result = loader._detect_columns(columns)

        assert result["competitor_brand"] == "competitor_brand"
        assert result["competitor_part"] == "competitor_part"

    def test_detect_description_column(self):
        """Test detection of description columns."""
        loader = DocumentLoader()

        test_cases = [
            (["part", "description"], "description"),
            (["part", "product_name"], "product_name"),
            (["part", "item_name"], "item_name"),
        ]

        for columns, expected in test_cases:
            result = loader._detect_columns(columns)
            assert result["description"] == expected

    def test_detect_spec_columns(self):
        """Test detection of specification columns."""
        loader = DocumentLoader()

        columns = ["part", "voltage", "current", "temperature", "other"]
        result = loader._detect_columns(columns)

        assert "voltage" in result["specs"]
        assert "current" in result["specs"]
        assert "temperature" in result["specs"]
        assert "other" not in result["specs"]


class TestPartNumberNormalization:
    """Tests for part number normalization."""

    def test_normalize_removes_dashes(self):
        """Test that dashes are removed during normalization."""
        loader = DocumentLoader()
        assert loader._normalize_part_number("ABC-123") == "ABC123"

    def test_normalize_removes_spaces(self):
        """Test that spaces are removed during normalization."""
        loader = DocumentLoader()
        assert loader._normalize_part_number("ABC 123") == "ABC123"

    def test_normalize_uppercase(self):
        """Test that part numbers are uppercased."""
        loader = DocumentLoader()
        assert loader._normalize_part_number("abc-123") == "ABC123"

    def test_normalize_removes_dots_slashes(self):
        """Test that dots and slashes are removed."""
        loader = DocumentLoader()
        assert loader._normalize_part_number("ABC.123/X") == "ABC123X"


class TestCSVLoading:
    """Tests for CSV file loading."""

    def test_load_csv_parts_crossref(self, temp_csv_file):
        """Test loading a parts cross-reference CSV."""
        loader = DocumentLoader()
        documents = loader.load_csv(temp_csv_file)

        assert len(documents) == 3
        assert all(doc.metadata["document_type"] == "parts_crossref" for doc in documents)

    def test_csv_creates_semantic_content(self, temp_csv_file):
        """Test that CSV loading creates semantic document content."""
        loader = DocumentLoader()
        documents = loader.load_csv(temp_csv_file)

        # Check first document
        content = documents[0].page_content
        assert "Danfoss part ABC-123" in content
        assert "XYZ-111" in content

    def test_csv_includes_metadata(self, temp_csv_file):
        """Test that CSV documents include correct metadata."""
        loader = DocumentLoader()
        documents = loader.load_csv(temp_csv_file)

        doc = documents[0]
        assert "source_file" in doc.metadata
        assert doc.metadata["file_type"] == "csv"
        assert "danfoss_part" in doc.metadata
        assert "competitor_part" in doc.metadata

    def test_csv_normalized_part_numbers(self, temp_csv_file):
        """Test that CSV loading creates normalized part numbers."""
        loader = DocumentLoader()
        documents = loader.load_csv(temp_csv_file)

        doc = documents[0]
        assert "danfoss_part_normalized" in doc.metadata
        assert "competitor_part_normalized" in doc.metadata


class TestExcelLoading:
    """Tests for Excel file loading."""

    def test_load_excel_parts_crossref(self, temp_excel_file):
        """Test loading a parts cross-reference Excel file."""
        loader = DocumentLoader()
        documents = loader.load_excel(temp_excel_file)

        assert len(documents) == 3
        assert all(doc.metadata["document_type"] == "parts_crossref" for doc in documents)

    def test_excel_file_type_metadata(self, temp_excel_file):
        """Test that Excel documents have correct file_type."""
        loader = DocumentLoader()
        documents = loader.load_excel(temp_excel_file)

        assert all(doc.metadata["file_type"] == "excel" for doc in documents)


class TestGeneralDataLoading:
    """Tests for general (non-parts) data loading."""

    def test_general_data_creates_documents(self, sample_general_dataframe):
        """Test that general data creates documents."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_general_dataframe.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DocumentLoader()
            documents = loader.load_csv(temp_path)

            assert len(documents) == 2
            assert all(doc.metadata["document_type"] == "general_data" for doc in documents)
        finally:
            import os
            os.unlink(temp_path)

    def test_general_data_content_format(self, sample_general_dataframe):
        """Test that general data documents have correct content format."""
        import tempfile

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            sample_general_dataframe.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DocumentLoader()
            documents = loader.load_csv(temp_path)

            content = documents[0].page_content
            assert "Product Name: Widget A" in content
            assert "Category: Electronics" in content
        finally:
            import os
            os.unlink(temp_path)


class TestLoadFile:
    """Tests for the unified load_file method."""

    def test_load_file_csv(self, temp_csv_file):
        """Test load_file with CSV."""
        loader = DocumentLoader()
        documents = loader.load_file(temp_csv_file)
        assert len(documents) > 0

    def test_load_file_excel(self, temp_excel_file):
        """Test load_file with Excel."""
        loader = DocumentLoader()
        documents = loader.load_file(temp_excel_file)
        assert len(documents) > 0

    def test_load_file_unsupported_type(self):
        """Test load_file with unsupported file type."""
        loader = DocumentLoader()
        with pytest.raises(ValueError, match="Unsupported file type"):
            loader.load_file("test.txt")


class TestGetFileInfo:
    """Tests for get_file_info method."""

    def test_get_csv_info(self, temp_csv_file):
        """Test getting info for a CSV file."""
        loader = DocumentLoader()
        info = loader.get_file_info(temp_csv_file)

        assert info["file_type"] == "csv"
        assert info["row_count"] == 3
        assert info["column_count"] == 5
        assert "columns" in info

    def test_get_excel_info(self, temp_excel_file):
        """Test getting info for an Excel file."""
        loader = DocumentLoader()
        info = loader.get_file_info(temp_excel_file)

        assert info["file_type"] == "xlsx"
        assert info["row_count"] == 3
        assert "file_size_bytes" in info


class TestEdgeCases:
    """Tests for edge cases and error handling."""

    def test_empty_dataframe(self):
        """Test handling of empty DataFrame."""
        import tempfile

        df = pd.DataFrame(columns=["danfoss_part", "competitor_part"])

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DocumentLoader()
            documents = loader.load_csv(temp_path)
            assert len(documents) == 0
        finally:
            import os
            os.unlink(temp_path)

    def test_missing_required_columns(self):
        """Test handling when key columns have missing values."""
        import tempfile

        df = pd.DataFrame({
            "danfoss_part": ["ABC-123", None, "GHI-789"],
            "competitor_part": [None, "XYZ-222", "XYZ-333"],
        })

        with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
            df.to_csv(f, index=False)
            temp_path = f.name

        try:
            loader = DocumentLoader()
            documents = loader.load_csv(temp_path)
            # Only the third row has both values
            assert len(documents) == 1
        finally:
            import os
            os.unlink(temp_path)

    def test_get_value_handles_none_column(self):
        """Test _get_value handles None column gracefully."""
        loader = DocumentLoader()
        row = pd.Series({"a": 1, "b": 2})
        assert loader._get_value(row, None) is None

    def test_get_value_handles_missing_column(self):
        """Test _get_value handles missing column gracefully."""
        loader = DocumentLoader()
        row = pd.Series({"a": 1, "b": 2})
        assert loader._get_value(row, "c") is None

    def test_get_value_handles_nan(self):
        """Test _get_value handles NaN values."""
        loader = DocumentLoader()
        row = pd.Series({"a": float("nan"), "b": 2})
        assert loader._get_value(row, "a") is None
