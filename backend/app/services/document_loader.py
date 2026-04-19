"""
Document loading and processing for PDF, Excel, and CSV files.
Handles part cross-reference detection and semantic document creation.
"""

import os
import re
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from pypdf import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


@dataclass
class ProcessedDocument:
    """Represents a processed document ready for vector storage."""
    content: str
    metadata: Dict[str, Any]


class DocumentLoader:
    """
    Handles loading and processing of various document types.
    Supports PDF, Excel (.xlsx, .xls), and CSV files.
    """

    # Column name patterns for auto-detection
    DANFOSS_PART_PATTERNS = [
        r'danfoss.*part', r'our.*part', r'part.*number', r'danfoss.*pn',
        r'replacement.*part', r'danfoss.*item', r'danfoss_part', r'part_number'
    ]
    COMPETITOR_BRAND_PATTERNS = [
        r'competitor.*brand', r'brand', r'manufacturer', r'oem',
        r'competitor.*name', r'original.*brand', r'competitor_brand'
    ]
    COMPETITOR_PART_PATTERNS = [
        r'competitor.*part', r'oem.*part', r'original.*part',
        r'cross.*ref', r'replaces', r'competitor_part', r'oem_part'
    ]
    DESCRIPTION_PATTERNS = [
        r'description', r'desc', r'product.*name', r'item.*name', r'name'
    ]
    SPEC_PATTERNS = [
        r'voltage', r'current', r'power', r'dimension', r'size',
        r'weight', r'temp', r'rating', r'capacity', r'frequency'
    ]

    def __init__(self, chunk_size: int = 1000, chunk_overlap: int = 200):
        """
        Initialize the document loader.

        Args:
            chunk_size: Size of text chunks for PDFs
            chunk_overlap: Overlap between chunks
        """
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", " ", ""]
        )

    def load_file(self, file_path: str) -> List[Document]:
        """
        Load a file and return processed documents.

        Args:
            file_path: Path to the file

        Returns:
            List of Document objects ready for embedding
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        if extension == '.pdf':
            return self.load_pdf(file_path)
        elif extension in ['.xlsx', '.xls']:
            return self.load_excel(file_path)
        elif extension == '.csv':
            return self.load_csv(file_path)
        else:
            raise ValueError(f"Unsupported file type: {extension}")

    def load_pdf(self, file_path: str) -> List[Document]:
        """
        Load and process a PDF file.

        Args:
            file_path: Path to the PDF file

        Returns:
            List of Document objects with chunks
        """
        path = Path(file_path)
        reader = PdfReader(file_path)

        # Extract text from all pages
        full_text = ""
        for page_num, page in enumerate(reader.pages):
            page_text = page.extract_text() or ""
            full_text += f"\n--- Page {page_num + 1} ---\n{page_text}"

        # Split into chunks
        chunks = self.text_splitter.split_text(full_text)

        documents = []
        for i, chunk in enumerate(chunks):
            doc = Document(
                page_content=chunk,
                metadata={
                    "source_file": path.name,
                    "file_path": str(path),
                    "file_type": "pdf",
                    "chunk_index": i,
                    "total_chunks": len(chunks),
                    "document_type": "guide"  # Default for PDFs
                }
            )
            documents.append(doc)

        return documents

    def load_excel(self, file_path: str) -> List[Document]:
        """
        Load and process an Excel file.
        Auto-detects if it's a parts cross-reference or general data.

        Args:
            file_path: Path to the Excel file

        Returns:
            List of Document objects
        """
        path = Path(file_path)
        df = pd.read_excel(file_path, engine='openpyxl')
        return self._process_dataframe(df, path)

    def load_csv(self, file_path: str) -> List[Document]:
        """
        Load and process a CSV file.

        Args:
            file_path: Path to the CSV file

        Returns:
            List of Document objects
        """
        path = Path(file_path)
        df = pd.read_csv(file_path)
        return self._process_dataframe(df, path)

    def _process_dataframe(self, df: pd.DataFrame, path: Path) -> List[Document]:
        """
        Process a DataFrame into documents.
        Detects if it's a parts cross-reference and creates semantic documents.

        Args:
            df: Pandas DataFrame
            path: Path object for the source file

        Returns:
            List of Document objects
        """
        # Clean column names
        df.columns = [str(col).strip().lower() for col in df.columns]

        # Detect column mappings
        column_map = self._detect_columns(df.columns.tolist())

        # Check if this is a parts cross-reference file
        is_parts_crossref = (
            column_map.get('danfoss_part') is not None and
            column_map.get('competitor_part') is not None
        )

        documents = []

        if is_parts_crossref:
            documents = self._create_parts_documents(df, column_map, path)
        else:
            documents = self._create_general_documents(df, path)

        return documents

    def _detect_columns(self, columns: List[str]) -> Dict[str, Optional[str]]:
        """
        Auto-detect column types based on naming patterns.

        Args:
            columns: List of column names

        Returns:
            Dictionary mapping column types to actual column names
        """
        column_map = {
            'danfoss_part': None,
            'competitor_brand': None,
            'competitor_part': None,
            'description': None,
            'specs': []
        }

        for col in columns:
            col_lower = col.lower()

            # Check Danfoss part patterns
            if column_map['danfoss_part'] is None:
                for pattern in self.DANFOSS_PART_PATTERNS:
                    if re.search(pattern, col_lower):
                        column_map['danfoss_part'] = col
                        break

            # Check competitor brand patterns
            if column_map['competitor_brand'] is None:
                for pattern in self.COMPETITOR_BRAND_PATTERNS:
                    if re.search(pattern, col_lower):
                        column_map['competitor_brand'] = col
                        break

            # Check competitor part patterns
            if column_map['competitor_part'] is None:
                for pattern in self.COMPETITOR_PART_PATTERNS:
                    if re.search(pattern, col_lower):
                        column_map['competitor_part'] = col
                        break

            # Check description patterns
            if column_map['description'] is None:
                for pattern in self.DESCRIPTION_PATTERNS:
                    if re.search(pattern, col_lower):
                        column_map['description'] = col
                        break

            # Check spec patterns
            for pattern in self.SPEC_PATTERNS:
                if re.search(pattern, col_lower):
                    column_map['specs'].append(col)
                    break

        return column_map

    def _create_parts_documents(
        self,
        df: pd.DataFrame,
        column_map: Dict[str, Any],
        path: Path
    ) -> List[Document]:
        """
        Create semantic documents from parts cross-reference data.

        Args:
            df: DataFrame with parts data
            column_map: Detected column mappings
            path: Source file path

        Returns:
            List of Document objects
        """
        documents = []

        for idx, row in df.iterrows():
            # Skip rows with missing key data
            danfoss_part = self._get_value(row, column_map['danfoss_part'])
            competitor_part = self._get_value(row, column_map['competitor_part'])

            if not danfoss_part or not competitor_part:
                continue

            # Build semantic content
            competitor_brand = self._get_value(row, column_map['competitor_brand'])
            description = self._get_value(row, column_map['description'])

            # Build the semantic document text
            content_parts = []

            if competitor_brand:
                content_parts.append(
                    f"Danfoss part {danfoss_part} is equivalent to {competitor_brand} "
                    f"part {competitor_part}."
                )
            else:
                content_parts.append(
                    f"Danfoss part {danfoss_part} replaces competitor part {competitor_part}."
                )

            if description:
                content_parts.append(f"Product: {description}.")

            # Add specifications
            specs = []
            for spec_col in column_map.get('specs', []):
                spec_value = self._get_value(row, spec_col)
                if spec_value:
                    spec_name = spec_col.replace('_', ' ').title()
                    specs.append(f"{spec_name}: {spec_value}")

            if specs:
                content_parts.append(f"Specifications: {', '.join(specs)}.")

            content = " ".join(content_parts)

            # Build metadata
            metadata = {
                "source_file": path.name,
                "file_path": str(path),
                "file_type": "excel" if path.suffix in ['.xlsx', '.xls'] else "csv",
                "document_type": "parts_crossref",
                "danfoss_part": str(danfoss_part).upper(),
                "competitor_part": str(competitor_part).upper(),
                "row_index": idx
            }

            if competitor_brand:
                metadata["competitor_brand"] = str(competitor_brand)
            if description:
                metadata["description"] = str(description)

            # Add normalized part numbers for fuzzy matching
            metadata["danfoss_part_normalized"] = self._normalize_part_number(
                str(danfoss_part)
            )
            metadata["competitor_part_normalized"] = self._normalize_part_number(
                str(competitor_part)
            )

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def _create_general_documents(
        self,
        df: pd.DataFrame,
        path: Path
    ) -> List[Document]:
        """
        Create documents from general tabular data.

        Args:
            df: DataFrame with data
            path: Source file path

        Returns:
            List of Document objects
        """
        documents = []

        for idx, row in df.iterrows():
            # Create a text representation of the row
            content_parts = []
            for col in df.columns:
                value = self._get_value(row, col)
                if value:
                    col_name = col.replace('_', ' ').title()
                    content_parts.append(f"{col_name}: {value}")

            if not content_parts:
                continue

            content = ". ".join(content_parts) + "."

            metadata = {
                "source_file": path.name,
                "file_path": str(path),
                "file_type": "excel" if path.suffix in ['.xlsx', '.xls'] else "csv",
                "document_type": "general_data",
                "row_index": idx
            }

            doc = Document(page_content=content, metadata=metadata)
            documents.append(doc)

        return documents

    def _get_value(self, row: pd.Series, column: Optional[str]) -> Optional[str]:
        """
        Safely get a value from a DataFrame row.

        Args:
            row: DataFrame row
            column: Column name

        Returns:
            String value or None
        """
        if column is None or column not in row.index:
            return None

        value = row[column]
        if pd.isna(value):
            return None

        return str(value).strip()

    def _normalize_part_number(self, part_number: str) -> str:
        """
        Normalize a part number for fuzzy matching.
        Removes dashes, spaces, and converts to uppercase.

        Args:
            part_number: Original part number

        Returns:
            Normalized part number
        """
        # Remove common separators and convert to uppercase
        normalized = re.sub(r'[-\s\.\/]', '', part_number.upper())
        return normalized

    def get_file_info(self, file_path: str) -> Dict[str, Any]:
        """
        Get information about a file without fully processing it.

        Args:
            file_path: Path to the file

        Returns:
            Dictionary with file information
        """
        path = Path(file_path)
        extension = path.suffix.lower()

        info = {
            "filename": path.name,
            "file_type": extension[1:],  # Remove the dot
            "file_size_bytes": path.stat().st_size,
        }

        if extension == '.pdf':
            reader = PdfReader(file_path)
            info["page_count"] = len(reader.pages)
        elif extension in ['.xlsx', '.xls']:
            df = pd.read_excel(file_path, engine='openpyxl')
            info["row_count"] = len(df)
            info["column_count"] = len(df.columns)
            info["columns"] = df.columns.tolist()
        elif extension == '.csv':
            df = pd.read_csv(file_path)
            info["row_count"] = len(df)
            info["column_count"] = len(df.columns)
            info["columns"] = df.columns.tolist()

        return info
