"""
Pinecone vector database service for document storage and retrieval.
"""

import hashlib
from typing import List, Dict, Any, Optional
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pinecone import Pinecone, ServerlessSpec
from rapidfuzz import fuzz
import re

from ..config import get_settings


class PineconeService:
    """
    Handles all Pinecone vector database operations.
    Supports document upsert, query, and deletion.
    """

    NAMESPACE_PRODUCTS = "products"
    NAMESPACE_GUIDES = "guides"

    def __init__(self):
        """Initialize Pinecone client and embeddings."""
        self.settings = get_settings()

        # Initialize Pinecone
        self.pc = Pinecone(api_key=self.settings.pinecone_api_key)

        # Initialize embeddings
        self.embeddings = OpenAIEmbeddings(
            model=self.settings.openai_embedding_model,
            openai_api_key=self.settings.openai_api_key
        )

        # Get or create index
        self._ensure_index()
        self.index = self.pc.Index(self.settings.pinecone_index)

    def _ensure_index(self):
        """Create the Pinecone index if it doesn't exist."""
        existing_indexes = [idx.name for idx in self.pc.list_indexes()]

        if self.settings.pinecone_index not in existing_indexes:
            self.pc.create_index(
                name=self.settings.pinecone_index,
                dimension=self.settings.embedding_dimension,
                metric="cosine",
                spec=ServerlessSpec(
                    cloud="aws",
                    region=self.settings.pinecone_environment
                )
            )

    def _generate_doc_id(self, content: str, metadata: Dict[str, Any]) -> str:
        """
        Generate a unique ID for a document.

        Args:
            content: Document content
            metadata: Document metadata

        Returns:
            Unique document ID
        """
        # Create a hash from content and source
        source = metadata.get("source_file", "")
        row_idx = metadata.get("row_index", "")
        chunk_idx = metadata.get("chunk_index", "")

        hash_input = f"{source}:{row_idx}:{chunk_idx}:{content[:100]}"
        return hashlib.md5(hash_input.encode()).hexdigest()

    def upsert_documents(
        self,
        documents: List[Document],
        namespace: Optional[str] = None
    ) -> int:
        """
        Upsert documents into Pinecone.

        Args:
            documents: List of Document objects
            namespace: Pinecone namespace (defaults based on document type)

        Returns:
            Number of documents upserted
        """
        if not documents:
            return 0

        # Determine namespace if not provided
        if namespace is None:
            doc_type = documents[0].metadata.get("document_type", "")
            if doc_type == "parts_crossref":
                namespace = self.NAMESPACE_PRODUCTS
            else:
                namespace = self.NAMESPACE_GUIDES

        # Prepare vectors for upsert
        vectors = []
        batch_size = 100

        for doc in documents:
            # Generate embedding
            embedding = self.embeddings.embed_query(doc.page_content)

            # Generate unique ID
            doc_id = self._generate_doc_id(doc.page_content, doc.metadata)

            # Prepare metadata (Pinecone has limits on metadata size)
            metadata = {
                k: str(v)[:500] if isinstance(v, str) else v
                for k, v in doc.metadata.items()
            }
            metadata["text"] = doc.page_content[:1000]  # Store text for retrieval

            vectors.append({
                "id": doc_id,
                "values": embedding,
                "metadata": metadata
            })

            # Batch upsert
            if len(vectors) >= batch_size:
                self.index.upsert(vectors=vectors, namespace=namespace)
                vectors = []

        # Upsert remaining vectors
        if vectors:
            self.index.upsert(vectors=vectors, namespace=namespace)

        return len(documents)

    def query(
        self,
        query_text: str,
        namespace: Optional[str] = None,
        filter_dict: Optional[Dict[str, Any]] = None,
        top_k: int = None
    ) -> List[Dict[str, Any]]:
        """
        Query the vector database.

        Args:
            query_text: Search query
            namespace: Namespace to search (None searches all)
            filter_dict: Metadata filters
            top_k: Number of results to return

        Returns:
            List of matching documents with scores
        """
        if top_k is None:
            top_k = self.settings.default_top_k

        # Generate query embedding
        query_embedding = self.embeddings.embed_query(query_text)

        # Search in specified namespace or default
        if namespace is None:
            # Search both namespaces and combine results
            results_products = self._query_namespace(
                query_embedding, self.NAMESPACE_PRODUCTS, filter_dict, top_k
            )
            results_guides = self._query_namespace(
                query_embedding, self.NAMESPACE_GUIDES, filter_dict, top_k
            )

            # Combine and sort by score
            all_results = results_products + results_guides
            all_results.sort(key=lambda x: x["score"], reverse=True)
            return all_results[:top_k]
        else:
            return self._query_namespace(
                query_embedding, namespace, filter_dict, top_k
            )

    def _query_namespace(
        self,
        query_embedding: List[float],
        namespace: str,
        filter_dict: Optional[Dict[str, Any]],
        top_k: int
    ) -> List[Dict[str, Any]]:
        """Query a specific namespace."""
        try:
            response = self.index.query(
                vector=query_embedding,
                namespace=namespace,
                top_k=top_k,
                include_metadata=True,
                filter=filter_dict
            )

            results = []
            for match in response.matches:
                results.append({
                    "id": match.id,
                    "score": match.score,
                    "content": match.metadata.get("text", ""),
                    "metadata": {
                        k: v for k, v in match.metadata.items()
                        if k != "text"
                    },
                    "namespace": namespace
                })

            return results
        except Exception:
            # Namespace might not exist yet
            return []

    def query_by_part_number(
        self,
        part_number: str,
        is_competitor_part: bool = True,
        top_k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Query specifically for part number matches with fuzzy matching.

        Args:
            part_number: Part number to search for
            is_competitor_part: If True, search competitor_part field
            top_k: Number of results

        Returns:
            List of matching documents with scores
        """
        # Normalize the input part number
        normalized_query = self._normalize_part_number(part_number)

        # Create a semantic query
        if is_competitor_part:
            query_text = f"Danfoss equivalent replacement for part {part_number}"
        else:
            query_text = f"Danfoss part {part_number} specifications and details"

        # Get semantic search results
        results = self.query(
            query_text,
            namespace=self.NAMESPACE_PRODUCTS,
            top_k=top_k * 2  # Get more results for fuzzy filtering
        )

        # Apply fuzzy matching boost
        enhanced_results = []
        for result in results:
            metadata = result.get("metadata", {})

            # Check for exact or fuzzy part number match
            fuzzy_boost = 0.0

            if is_competitor_part:
                stored_normalized = metadata.get(
                    "competitor_part_normalized", ""
                )
                if stored_normalized:
                    similarity = fuzz.ratio(normalized_query, stored_normalized)
                    if similarity >= self.settings.fuzzy_match_threshold:
                        fuzzy_boost = (similarity / 100) * 0.3
            else:
                stored_normalized = metadata.get(
                    "danfoss_part_normalized", ""
                )
                if stored_normalized:
                    similarity = fuzz.ratio(normalized_query, stored_normalized)
                    if similarity >= self.settings.fuzzy_match_threshold:
                        fuzzy_boost = (similarity / 100) * 0.3

            result["score"] = min(1.0, result["score"] + fuzzy_boost)
            result["fuzzy_boost"] = fuzzy_boost
            enhanced_results.append(result)

        # Sort by enhanced score
        enhanced_results.sort(key=lambda x: x["score"], reverse=True)
        return enhanced_results[:top_k]

    def _normalize_part_number(self, part_number: str) -> str:
        """Normalize a part number for matching."""
        return re.sub(r'[-\s\.\/]', '', part_number.upper())

    def delete_by_source(self, source_file: str, namespace: Optional[str] = None) -> int:
        """
        Delete all documents from a specific source file.

        Args:
            source_file: Source filename to delete
            namespace: Namespace to delete from (None = all)

        Returns:
            Approximate number of documents deleted
        """
        namespaces = [namespace] if namespace else [
            self.NAMESPACE_PRODUCTS,
            self.NAMESPACE_GUIDES
        ]

        total_deleted = 0

        for ns in namespaces:
            try:
                # Query for documents with this source
                # Note: We need to iterate and delete by ID
                results = self.index.query(
                    vector=[0.0] * self.settings.embedding_dimension,
                    namespace=ns,
                    top_k=10000,
                    include_metadata=True,
                    filter={"source_file": {"$eq": source_file}}
                )

                if results.matches:
                    ids_to_delete = [match.id for match in results.matches]
                    self.index.delete(ids=ids_to_delete, namespace=ns)
                    total_deleted += len(ids_to_delete)

            except Exception:
                continue

        return total_deleted

    def get_stats(self) -> Dict[str, Any]:
        """Get index statistics."""
        try:
            stats = self.index.describe_index_stats()
            return {
                "total_vectors": stats.total_vector_count,
                "namespaces": {
                    ns: data.vector_count
                    for ns, data in stats.namespaces.items()
                }
            }
        except Exception as e:
            return {"error": str(e)}
