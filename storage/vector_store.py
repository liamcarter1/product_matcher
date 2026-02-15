"""
ProductMatchPro - Vector Store (numpy-based)
Stores embedded product descriptions for semantic search with reranking.
Uses numpy + sentence-transformers instead of ChromaDB for Python 3.14 compatibility.
Persists embeddings to disk as .npz files alongside a JSON index.

Thread-safety: All collection mutations are protected by a threading.Lock.
Persistence: Writes go to a temp file first, then atomic rename, to prevent corruption.
"""

import json
import tempfile
import threading
import logging
import numpy as np
from pathlib import Path
from typing import Optional
from sentence_transformers import SentenceTransformer, CrossEncoder

from models import HydraulicProduct

logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
RERANKER_MODEL = "cross-encoder/ms-marco-MiniLM-L-6-v2"


class VectorCollection:
    """A single collection of embedded documents with numpy-based cosine search.
    Thread-safe: all mutations are serialised through self._lock.
    Persistence: writes use atomic temp-file-then-rename pattern."""

    def __init__(self, name: str, data_dir: Path):
        self.name = name
        self.data_dir = data_dir
        self._lock = threading.Lock()
        self.ids: list[str] = []
        self.documents: list[str] = []
        self.metadatas: list[dict] = []
        self.vectors: Optional[np.ndarray] = None
        self._load()

    def _index_path(self) -> Path:
        return self.data_dir / f"{self.name}_index.json"

    def _vectors_path(self) -> Path:
        return self.data_dir / f"{self.name}_vectors.npz"

    def _load(self):
        idx_path = self._index_path()
        vec_path = self._vectors_path()
        if idx_path.exists() and vec_path.exists():
            try:
                with open(idx_path, "r") as f:
                    data = json.load(f)
                self.ids = data.get("ids", [])
                self.documents = data.get("documents", [])
                self.metadatas = data.get("metadatas", [])
                loaded = np.load(vec_path)
                self.vectors = loaded["vectors"] if "vectors" in loaded else None
            except Exception as e:
                logger.error(f"Error loading vector collection '{self.name}': {e}")
                # Start fresh if files are corrupt
                self.ids, self.documents, self.metadatas, self.vectors = [], [], [], None

    def _save(self):
        """Write index and vectors to disk atomically (temp file + rename)."""
        self.data_dir.mkdir(parents=True, exist_ok=True)
        # Write JSON index atomically
        idx_path = self._index_path()
        try:
            with tempfile.NamedTemporaryFile(
                mode="w", dir=self.data_dir, suffix=".tmp", delete=False
            ) as tmp:
                json.dump({
                    "ids": self.ids,
                    "documents": self.documents,
                    "metadatas": self.metadatas,
                }, tmp)
                tmp_idx = Path(tmp.name)
            tmp_idx.replace(idx_path)
        except Exception as e:
            logger.error(f"Error saving index for '{self.name}': {e}")
            # Clean up temp file if rename failed
            try:
                tmp_idx.unlink(missing_ok=True)
            except Exception:
                pass
            raise

        # Write vectors atomically
        if self.vectors is not None:
            vec_path = self._vectors_path()
            try:
                with tempfile.NamedTemporaryFile(
                    dir=self.data_dir, suffix=".tmp", delete=False
                ) as tmp:
                    np.savez_compressed(tmp, vectors=self.vectors)
                    tmp_vec = Path(tmp.name)
                tmp_vec.replace(vec_path)
            except Exception as e:
                logger.error(f"Error saving vectors for '{self.name}': {e}")
                try:
                    tmp_vec.unlink(missing_ok=True)
                except Exception:
                    pass
                raise

    def upsert(self, doc_id: str, document: str, embedding: np.ndarray, metadata: dict):
        with self._lock:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.documents[idx] = document
                self.metadatas[idx] = metadata
                if self.vectors is not None:
                    self.vectors[idx] = embedding
            else:
                self.ids.append(doc_id)
                self.documents.append(document)
                self.metadatas.append(metadata)
                emb = embedding.reshape(1, -1)
                if self.vectors is None:
                    self.vectors = emb
                else:
                    self.vectors = np.vstack([self.vectors, emb])
            self._save()

    def upsert_batch(self, items: list[tuple[str, str, np.ndarray, dict]]):
        """Batch upsert multiple items with a single save at the end."""
        with self._lock:
            for doc_id, document, embedding, metadata in items:
                if doc_id in self.ids:
                    idx = self.ids.index(doc_id)
                    self.documents[idx] = document
                    self.metadatas[idx] = metadata
                    if self.vectors is not None:
                        self.vectors[idx] = embedding
                else:
                    self.ids.append(doc_id)
                    self.documents.append(document)
                    self.metadatas.append(metadata)
                    emb = embedding.reshape(1, -1)
                    if self.vectors is None:
                        self.vectors = emb
                    else:
                        self.vectors = np.vstack([self.vectors, emb])
            self._save()

    def delete(self, doc_id: str):
        with self._lock:
            if doc_id in self.ids:
                idx = self.ids.index(doc_id)
                self.ids.pop(idx)
                self.documents.pop(idx)
                self.metadatas.pop(idx)
                if self.vectors is not None and len(self.ids) > 0:
                    self.vectors = np.delete(self.vectors, idx, axis=0)
                else:
                    self.vectors = None
                self._save()

    def search(
        self, query_embedding: np.ndarray, n_results: int = 10,
        where: Optional[dict] = None,
    ) -> list[tuple[str, str, dict, float]]:
        """Search by cosine similarity. Returns (id, document, metadata, score)."""
        if self.vectors is None or len(self.ids) == 0:
            return []

        # Normalize query
        q = query_embedding.astype(np.float32)
        q_norm = np.linalg.norm(q)
        if q_norm > 0:
            q = q / q_norm

        # Normalize stored vectors
        norms = np.linalg.norm(self.vectors, axis=1, keepdims=True) + 1e-10
        normed = self.vectors / norms

        # Cosine similarity
        scores = normed @ q

        # Apply metadata filter
        valid_mask = np.ones(len(self.ids), dtype=bool)
        if where:
            for key, value in where.items():
                for i, meta in enumerate(self.metadatas):
                    if meta.get(key) != value:
                        valid_mask[i] = False
            scores = scores * valid_mask

        # Get top results
        n_valid = int(valid_mask.sum())
        k = min(n_results, n_valid)
        if k == 0:
            return []

        # argpartition kth must be < array length
        if k >= len(scores):
            top_indices = np.argsort(-scores)[:k]
        else:
            top_indices = np.argpartition(-scores, k)[:k]
            top_indices = top_indices[np.argsort(-scores[top_indices])]

        results = []
        for idx in top_indices:
            if scores[idx] > 0:
                results.append((
                    self.ids[idx],
                    self.documents[idx],
                    self.metadatas[idx],
                    float(scores[idx]),
                ))

        return results

    def count(self) -> int:
        return len(self.ids)

    def clear(self):
        with self._lock:
            self.ids = []
            self.documents = []
            self.metadatas = []
            self.vectors = None
            self._index_path().unlink(missing_ok=True)
            self._vectors_path().unlink(missing_ok=True)


class VectorStore:
    """Manages multiple vector collections for product search."""

    def __init__(self, data_dir: str = str(DATA_DIR)):
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(parents=True, exist_ok=True)
        self.embedder = SentenceTransformer(EMBEDDING_MODEL)
        self.reranker = CrossEncoder(RERANKER_MODEL)

        self.my_company_col = VectorCollection("danfoss_products", self.data_dir)
        self.competitor_col = VectorCollection("competitor_products", self.data_dir)
        self.guides_col = VectorCollection("product_guides", self.data_dir)

    def index_product(self, product: HydraulicProduct):
        """Index a product's description into the appropriate collection."""
        text = self._build_indexable_text(product)
        embedding = self.embedder.encode(text)

        metadata = {
            "company": product.company,
            "model_code": product.model_code,
            "category": product.category or "",
            "subcategory": product.subcategory or "",
            "source_document": product.source_document or "",
        }
        if product.coil_voltage:
            metadata["coil_voltage"] = product.coil_voltage
        if product.valve_size:
            metadata["valve_size"] = product.valve_size
        if product.actuator_type:
            metadata["actuator_type"] = product.actuator_type
        if product.mounting:
            metadata["mounting"] = product.mounting

        collection = self._get_collection(product.company)
        collection.upsert(product.id, text, embedding, metadata)

    def index_guide_chunk(
        self, chunk_id: str, text: str, company: str,
        model_code: str = "", category: str = "", source_document: str = "",
    ):
        """Index a user guide text chunk."""
        embedding = self.embedder.encode(text)
        metadata = {
            "company": company,
            "model_code": model_code,
            "category": category,
            "source_document": source_document,
        }
        self.guides_col.upsert(chunk_id, text, embedding, metadata)

    def search_my_company(
        self, query: str, category: Optional[str] = None,
        n_results: int = 20, rerank_top_k: int = 10,
    ) -> list[tuple[str, float]]:
        """Search Danfoss products with reranking. Returns (product_id, score)."""
        return self._search_with_rerank(
            self.my_company_col, query, category, n_results, rerank_top_k
        )

    def search_competitor(
        self, query: str, company: Optional[str] = None,
        category: Optional[str] = None, n_results: int = 10,
    ) -> list[tuple[str, float]]:
        """Search competitor products. Returns (product_id, score)."""
        where = {}
        if company:
            where["company"] = company
        if category:
            where["category"] = category

        query_embedding = self.embedder.encode(query)
        results = self.competitor_col.search(
            query_embedding, n_results, where=where if where else None
        )
        return [(doc_id, score) for doc_id, _, _, score in results]

    def search_guides(
        self, query: str, company: Optional[str] = None,
        model_code: Optional[str] = None, n_results: int = 10,
    ) -> list[tuple[str, str, float]]:
        """Search user guide chunks. Returns (chunk_id, text, score)."""
        where = {}
        if company:
            where["company"] = company
        if model_code:
            where["model_code"] = model_code

        query_embedding = self.embedder.encode(query)
        results = self.guides_col.search(
            query_embedding, n_results, where=where if where else None
        )
        return [(doc_id, text, score) for doc_id, text, _, score in results]

    def search_guides_with_metadata(
        self, query: str, company: Optional[str] = None,
        model_code: Optional[str] = None, n_results: int = 15,
    ) -> list[tuple[str, str, dict, float]]:
        """Search guide chunks with metadata. Returns (chunk_id, text, metadata, score)."""
        where = {}
        if company:
            where["company"] = company
        if model_code:
            where["model_code"] = model_code

        query_embedding = self.embedder.encode(query)
        return self.guides_col.search(
            query_embedding, n_results, where=where if where else None
        )

    def delete_product(self, product_id: str, company: str):
        collection = self._get_collection(company)
        collection.delete(product_id)

    def rebuild_from_products(self, products: list[HydraulicProduct]):
        """Rebuild vector store from scratch."""
        self.my_company_col.clear()
        self.competitor_col.clear()
        for product in products:
            self.index_product(product)

    def get_collection_counts(self) -> dict:
        return {
            "my_company": self.my_company_col.count(),
            "competitor": self.competitor_col.count(),
            "guides": self.guides_col.count(),
        }

    # ── Internal ──────────────────────────────────────────────────────

    def _search_with_rerank(
        self, collection: VectorCollection, query: str,
        category: Optional[str], n_results: int, rerank_top_k: int,
    ) -> list[tuple[str, float]]:
        """Search + cross-encoder rerank."""
        where = {"category": category} if category else None
        query_embedding = self.embedder.encode(query)
        results = collection.search(query_embedding, n_results, where=where)

        if not results:
            return []

        # Rerank with cross-encoder
        pairs = [[query, doc_text] for _, doc_text, _, _ in results]
        rerank_scores = self.reranker.predict(pairs)

        scored = list(zip(results, rerank_scores))
        scored.sort(key=lambda x: x[1], reverse=True)

        # Normalize scores to 0-1 range and CLAMP to valid bounds
        if len(scored) > 1:
            max_s = max(s for _, s in scored)
            min_s = min(s for _, s in scored)
            rng = max_s - min_s if max_s != min_s else 1.0
        else:
            min_s, rng = 0.0, 1.0

        output = []
        for (doc_id, _, _, _), raw_score in scored[:rerank_top_k]:
            normalized = (raw_score - min_s) / rng
            # Clamp to [0.0, 1.0] — cross-encoder scores can go negative
            normalized = max(0.0, min(1.0, float(normalized)))
            output.append((doc_id, normalized))

        return output

    def _get_collection(self, company: str) -> VectorCollection:
        if company.lower() in ("danfoss",):
            return self.my_company_col
        return self.competitor_col

    @staticmethod
    def _build_indexable_text(product: HydraulicProduct) -> str:
        """Build a rich text representation for embedding."""
        parts = [
            f"Model: {product.model_code}",
            f"Name: {product.product_name}" if product.product_name else "",
            f"Category: {product.category}" if product.category else "",
        ]

        spec_parts = []
        if product.max_pressure_bar:
            spec_parts.append(f"Max pressure: {product.max_pressure_bar} bar")
        if product.max_flow_lpm:
            spec_parts.append(f"Max flow: {product.max_flow_lpm} lpm")
        if product.valve_size:
            spec_parts.append(f"Valve size: {product.valve_size}")
        if product.spool_type:
            spec_parts.append(f"Spool type: {product.spool_type}")
        if product.actuator_type:
            spec_parts.append(f"Actuation: {product.actuator_type}")
        if product.coil_voltage:
            spec_parts.append(f"Coil voltage: {product.coil_voltage}")
        if product.port_size:
            spec_parts.append(f"Port size: {product.port_size}")
        if product.mounting:
            spec_parts.append(f"Mounting: {product.mounting}")
        if product.mounting_pattern:
            spec_parts.append(f"Mounting pattern: {product.mounting_pattern}")
        if product.seal_material:
            spec_parts.append(f"Seal material: {product.seal_material}")
        if product.body_material:
            spec_parts.append(f"Body material: {product.body_material}")
        if product.displacement_cc:
            spec_parts.append(f"Displacement: {product.displacement_cc} cc/rev")
        if product.bore_diameter_mm:
            spec_parts.append(f"Bore: {product.bore_diameter_mm} mm")

        if spec_parts:
            parts.append("Specs: " + ", ".join(spec_parts))
        if product.description:
            parts.append(product.description)

        # Include extra_specs in indexable text
        if product.extra_specs:
            extra_parts = []
            for k, v in product.extra_specs.items():
                if k == "_spool_function" and isinstance(v, dict):
                    # Include spool function description for semantic search
                    sf_desc = v.get("description", "")
                    sf_center = v.get("center_condition", "")
                    if sf_desc:
                        extra_parts.append(f"Spool function: {sf_desc}")
                    if sf_center:
                        extra_parts.append(f"Center condition: {sf_center}")
                elif not k.startswith("_"):
                    extra_parts.append(f"{k.replace('_', ' ')}: {v}")
            if extra_parts:
                parts.append("Additional: " + ", ".join(extra_parts))

        return "\n".join(p for p in parts if p)
