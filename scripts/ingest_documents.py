#!/usr/bin/env python3
"""
CLI tool to bulk ingest documents into the Danfoss RAG system.

Usage:
    python ingest_documents.py --dir ./data
    python ingest_documents.py --file sample_parts.xlsx
    python ingest_documents.py --dir ./data --clear-existing
"""

import argparse
import os
import sys
import time
from pathlib import Path
from typing import List

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv

# Load environment variables
load_dotenv(Path(__file__).parent.parent / ".env")


def get_services():
    """Lazy import and initialize services."""
    from backend.app.services.document_loader import DocumentLoader
    from backend.app.services.pinecone_service import PineconeService

    return DocumentLoader(), PineconeService()


def find_documents(directory: str) -> List[Path]:
    """Find all supported documents in a directory."""
    supported_extensions = {".pdf", ".xlsx", ".xls", ".csv"}
    documents = []

    dir_path = Path(directory)
    if not dir_path.exists():
        print(f"Error: Directory '{directory}' does not exist")
        sys.exit(1)

    for ext in supported_extensions:
        documents.extend(dir_path.glob(f"**/*{ext}"))

    return sorted(documents)


def format_size(size_bytes: int) -> str:
    """Format file size for display."""
    for unit in ["B", "KB", "MB", "GB"]:
        if size_bytes < 1024:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024
    return f"{size_bytes:.1f} TB"


def print_progress_bar(current: int, total: int, prefix: str = "", width: int = 40):
    """Print a progress bar."""
    percent = current / total if total > 0 else 0
    filled = int(width * percent)
    bar = "=" * filled + "-" * (width - filled)
    print(f"\r{prefix} [{bar}] {current}/{total} ({percent*100:.1f}%)", end="", flush=True)


def ingest_file(
    file_path: Path,
    document_loader,
    pinecone_service,
    verbose: bool = False
) -> dict:
    """Ingest a single file and return results."""
    result = {
        "file": file_path.name,
        "status": "success",
        "documents": 0,
        "time": 0.0,
        "error": None
    }

    try:
        start_time = time.time()

        # Load and process the document
        documents = document_loader.load_file(str(file_path))

        if not documents:
            result["status"] = "empty"
            result["error"] = "No content extracted"
            return result

        # Upsert to Pinecone
        docs_added = pinecone_service.upsert_documents(documents)

        result["documents"] = docs_added
        result["time"] = time.time() - start_time

        if verbose:
            print(f"\n  Processed {file_path.name}: {docs_added} documents")

    except Exception as e:
        result["status"] = "error"
        result["error"] = str(e)
        if verbose:
            print(f"\n  Error processing {file_path.name}: {e}")

    return result


def main():
    parser = argparse.ArgumentParser(
        description="Bulk ingest documents into the Danfoss RAG system"
    )

    parser.add_argument(
        "--dir", "-d",
        type=str,
        help="Directory containing documents to ingest"
    )

    parser.add_argument(
        "--file", "-f",
        type=str,
        help="Single file to ingest"
    )

    parser.add_argument(
        "--clear-existing",
        action="store_true",
        help="Clear existing documents before ingesting"
    )

    parser.add_argument(
        "--verbose", "-v",
        action="store_true",
        help="Show detailed progress"
    )

    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="List files without processing"
    )

    args = parser.parse_args()

    if not args.dir and not args.file:
        parser.print_help()
        print("\nError: Either --dir or --file must be specified")
        sys.exit(1)

    # Find documents to process
    if args.file:
        file_path = Path(args.file)
        if not file_path.exists():
            print(f"Error: File '{args.file}' does not exist")
            sys.exit(1)
        documents = [file_path]
    else:
        documents = find_documents(args.dir)

    if not documents:
        print("No supported documents found")
        sys.exit(0)

    # Display summary
    print("\n" + "=" * 60)
    print("  DANFOSS RAG - Document Ingestion Tool")
    print("=" * 60)
    print(f"\nFound {len(documents)} document(s) to process:\n")

    total_size = 0
    for doc in documents:
        size = doc.stat().st_size
        total_size += size
        print(f"  - {doc.name} ({format_size(size)})")

    print(f"\nTotal size: {format_size(total_size)}")

    if args.dry_run:
        print("\n[DRY RUN] No documents were processed")
        sys.exit(0)

    # Confirm before proceeding
    if not args.verbose:
        response = input("\nProceed with ingestion? [y/N]: ")
        if response.lower() != "y":
            print("Aborted")
            sys.exit(0)

    # Initialize services
    print("\nInitializing services...")
    try:
        document_loader, pinecone_service = get_services()
    except Exception as e:
        print(f"Error initializing services: {e}")
        print("Make sure environment variables are set correctly")
        sys.exit(1)

    # Clear existing if requested
    if args.clear_existing:
        print("\nClearing existing documents...")
        for doc in documents:
            try:
                pinecone_service.delete_by_source(doc.name)
            except Exception:
                pass

    # Process documents
    print("\nProcessing documents...")
    results = []
    start_time = time.time()

    for i, doc in enumerate(documents):
        if not args.verbose:
            print_progress_bar(i + 1, len(documents), "Progress:")

        result = ingest_file(doc, document_loader, pinecone_service, args.verbose)
        results.append(result)

    total_time = time.time() - start_time

    # Print summary
    print("\n\n" + "=" * 60)
    print("  INGESTION COMPLETE")
    print("=" * 60)

    successful = [r for r in results if r["status"] == "success"]
    failed = [r for r in results if r["status"] == "error"]
    empty = [r for r in results if r["status"] == "empty"]

    total_docs = sum(r["documents"] for r in successful)

    print(f"""
Summary:
  - Files processed: {len(results)}
  - Successful: {len(successful)}
  - Failed: {len(failed)}
  - Empty: {len(empty)}
  - Total documents created: {total_docs}
  - Total time: {total_time:.1f}s
""")

    if failed:
        print("Failed files:")
        for r in failed:
            print(f"  - {r['file']}: {r['error']}")

    if empty:
        print("\nEmpty files (no content extracted):")
        for r in empty:
            print(f"  - {r['file']}")

    # Get final stats
    try:
        stats = pinecone_service.get_stats()
        print(f"\nVector Database Stats:")
        print(f"  Total vectors: {stats.get('total_vectors', 'N/A')}")
        if "namespaces" in stats:
            for ns, count in stats["namespaces"].items():
                print(f"  - {ns}: {count} vectors")
    except Exception:
        pass

    print("\nDone!")


if __name__ == "__main__":
    main()
