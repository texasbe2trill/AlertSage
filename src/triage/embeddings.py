"""
Semantic embeddings for incident similarity and search.

Uses sentence-transformers for fast, high-quality semantic search across incidents.
Enables features like:
- Finding similar past incidents
- Detecting duplicate reports
- Semantic clustering
- Smart search beyond keyword matching
"""

from __future__ import annotations

import os
from typing import TYPE_CHECKING, List, Optional, Tuple
import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer
else:
    try:
        from sentence_transformers import SentenceTransformer  # type: ignore
    except ImportError:
        SentenceTransformer = None  # type: ignore

# Default model: all-MiniLM-L6-v2 (384 dims, 90MB, fast)
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
EMBEDDING_MODEL = os.getenv("TRIAGE_EMBEDDING_MODEL", DEFAULT_EMBEDDING_MODEL)


class IncidentEmbeddings:
    """Semantic embeddings for cybersecurity incidents.

    Features:
    - Fast encoding of incident descriptions
    - Cosine similarity search
    - Duplicate detection
    - Semantic clustering

    Example:
        >>> embedder = IncidentEmbeddings()
        >>> embed = embedder.encode("Phishing email detected")
        >>> similar = embedder.find_similar(embed, corpus_embeddings, top_k=5)
    """

    def __init__(self, model_name: Optional[str] = None):
        """Initialize embedding model.

        Args:
            model_name: HuggingFace model name (default: all-MiniLM-L6-v2)
        """
        if SentenceTransformer is None:
            raise RuntimeError(
                "sentence-transformers not installed. "
                "Install with: pip install sentence-transformers"
            )

        self.model_name = model_name or EMBEDDING_MODEL
        self._model: Optional[SentenceTransformer] = None

    @property
    def model(self) -> "SentenceTransformer":
        """Lazy-load the embedding model."""
        if self._model is None:
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def encode(
        self,
        texts: str | List[str],
        normalize: bool = True,
        batch_size: int = 32,
    ) -> np.ndarray:
        """Encode text(s) into semantic embeddings.

        Args:
            texts: Single text or list of texts
            normalize: L2 normalize for cosine similarity (recommended)
            batch_size: Batch size for encoding

        Returns:
            Numpy array of shape (n_texts, embedding_dim)
        """
        if isinstance(texts, str):
            texts = [texts]

        embeddings = self.model.encode(
            texts,
            batch_size=batch_size,
            normalize_embeddings=normalize,
            show_progress_bar=len(texts) > 100,
        )

        return embeddings

    def similarity(
        self,
        embed1: np.ndarray,
        embed2: np.ndarray,
    ) -> float:
        """Compute cosine similarity between two embeddings.

        Args:
            embed1: First embedding vector (1D or 2D with shape (1, dim))
            embed2: Second embedding vector (1D or 2D with shape (1, dim))

        Returns:
            Similarity score in [0, 1] (1 = identical)
        """
        # Ensure both embeddings are 1D
        if embed1.ndim == 2:
            embed1 = embed1.flatten()
        if embed2.ndim == 2:
            embed2 = embed2.flatten()

        return float(np.dot(embed1, embed2))

    def find_similar(
        self,
        query_embedding: np.ndarray,
        corpus_embeddings: np.ndarray,
        top_k: int = 5,
    ) -> List[Tuple[int, float]]:
        """Find most similar incidents in corpus.

        Args:
            query_embedding: Query incident embedding (1D array or 2D with shape (1, dim))
            corpus_embeddings: All incident embeddings (2D array)
            top_k: Number of results to return

        Returns:
            List of (index, similarity_score) tuples, sorted by similarity
        """
        # Ensure query_embedding is 1D
        if query_embedding.ndim == 2:
            query_embedding = query_embedding.flatten()

        # Compute all similarities at once (fast matrix multiplication)
        similarities = np.dot(corpus_embeddings, query_embedding)

        # Get top-k indices
        top_indices = np.argsort(similarities)[::-1][:top_k]

        # Return (index, score) pairs
        return [(int(idx), float(similarities[idx])) for idx in top_indices]

    def is_duplicate(
        self,
        embed1: np.ndarray,
        embed2: np.ndarray,
        threshold: float = 0.90,
    ) -> bool:
        """Check if two incidents are likely duplicates.

        Args:
            embed1: First incident embedding
            embed2: Second incident embedding
            threshold: Similarity threshold (default: 0.90 = 90% similar)

        Returns:
            True if similarity exceeds threshold
        """
        sim = self.similarity(embed1, embed2)
        return sim >= threshold

    def cluster_incidents(
        self,
        embeddings: np.ndarray,
        n_clusters: int = 10,
    ) -> np.ndarray:
        """Cluster incidents by semantic similarity.

        Args:
            embeddings: Incident embeddings (2D array)
            n_clusters: Number of clusters

        Returns:
            Cluster labels array (length = n_incidents)
        """
        from sklearn.cluster import KMeans

        kmeans = KMeans(n_clusters=n_clusters, random_state=42)
        labels = kmeans.fit_predict(embeddings)

        return labels


# Singleton for easy access
_embedder: Optional[IncidentEmbeddings] = None


def get_embedder() -> IncidentEmbeddings:
    """Get or create global embeddings instance."""
    global _embedder
    if _embedder is None:
        _embedder = IncidentEmbeddings()
    return _embedder


__all__ = ["IncidentEmbeddings", "get_embedder"]
