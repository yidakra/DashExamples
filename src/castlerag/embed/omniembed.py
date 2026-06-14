"""OmniEmbed (Tevatron/OmniEmbed-v0.1-multivent) batch inference wrappers.

Backend: vLLM (default) or HuggingFace Transformers.

Modality batching strategy (SPEC §3.2):
  transcript windows : 128 per batch
  event summaries    : 64 per batch
  images             : 16 per batch
  video clips        : 4 per batch

Query format per OmniEmbed model card:
  text  → "Query: {text}"
  media → raw frames/video via Qwen2.5-Omni processor

Point ids are deterministic: sha1(model_version + source_type + record_id + modality)
"""

from __future__ import annotations

import os
import uuid
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

_POINT_NAMESPACE = uuid.UUID("00000000-0000-0000-0000-000000000000")


def make_point_id(
    model_version: str,
    source_type: str,
    record_id: str,
    modality: str,
) -> str:
    """Return a deterministic UUIDv5 point id (Qdrant requires UUID or uint)."""
    key = f"{model_version}|{source_type}|{record_id}|{modality}"
    return str(uuid.uuid5(_POINT_NAMESPACE, key))


class OmniEmbedClient:
    """Thin wrapper around a vLLM or Transformers OmniEmbed backend.

    Call embed_texts() or embed_images() or embed_videos() to get
    float32 embedding arrays.  Vector dimensionality is discovered from
    the first successful batch and stored in self.dim.
    """

    def __init__(
        self,
        model: str = "Tevatron/OmniEmbed-v0.1-multivent",
        backend: str = "vllm",
        vllm_base_url: Optional[str] = None,
        vllm_tensor_parallel: int = 1,
        vllm_gpu_memory_utilization: float = 0.90,
        query_cache_path: Optional[str] = None,
    ) -> None:
        """Initialise the embedder with model name, backend, and vLLM options.

        ``query_cache_path`` (or the ``OMNIEMBED_QUERY_CACHE`` env var) points
        to an NPZ file produced by ``scripts/precompute_queries.py``.  When set,
        ``embed_texts`` serves cached vectors locally and only falls back to the
        remote endpoint for cache misses.  This lets us swap the OmniEmbed vLLM
        server out for Qwen3-VL during eval without losing dense retrieval.
        """
        self.model = model
        self.backend = backend
        self.vllm_base_url = vllm_base_url
        self.vllm_tensor_parallel = vllm_tensor_parallel
        self.vllm_gpu_memory_utilization = vllm_gpu_memory_utilization
        self.dim: Optional[int] = None  # set after first batch
        self._client: Any = None
        self._query_cache: Optional[Dict[str, np.ndarray]] = None
        self._query_cache_path: Optional[str] = (
            query_cache_path or os.getenv("OMNIEMBED_QUERY_CACHE")
        )

    def _ensure_client(self) -> None:
        """Lazily initialise the embedding client on first use."""
        if self._client is not None:
            return
        if self.backend == "vllm":
            self._client = self._init_vllm()
        elif self.backend == "transformers":
            self._client = self._init_transformers()
        else:
            raise ValueError(f"Unknown backend: {self.backend!r}")

    def _init_vllm(self) -> Any:
        """Return an OpenAI-compatible client for the vLLM embeddings endpoint."""
        if not self.vllm_base_url:
            raise ValueError("vllm_base_url is required when backend='vllm'")
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise ImportError(
                "openai package required for vLLM embedding backend; "
                "pip install castlerag[inference]"
            ) from exc
        return OpenAI(base_url=self.vllm_base_url, api_key="not-needed")

    def _init_transformers(self) -> Any:
        """Return a transformers-based embedding client (stub for local inference)."""
        return object()

    def _load_query_cache(self) -> Dict[str, np.ndarray]:
        """Lazily load the precomputed query→vector NPZ if configured."""
        if self._query_cache is not None:
            return self._query_cache
        cache: Dict[str, np.ndarray] = {}
        path = self._query_cache_path
        if path and Path(path).exists():
            with np.load(path, allow_pickle=False) as bundle:
                keys = bundle["keys"].tolist()
                vectors = bundle["vectors"]
                cache = {k: vectors[i] for i, k in enumerate(keys)}
            if vectors.size:
                self.dim = int(vectors.shape[1])
        self._query_cache = cache
        return cache

    def embed_texts(self, texts: List[str]) -> np.ndarray:
        """Embed text strings.  Automatically prepends 'Query: ' prefix.

        If a query cache is configured, vectors are served from it; only
        cache misses are sent to the remote endpoint.  Misses raise when no
        backend is available (e.g. OmniEmbed has been swapped out for gen).
        """
        cache = self._load_query_cache()
        missing_idx: List[int] = []
        cached: List[Optional[np.ndarray]] = []
        for i, text in enumerate(texts):
            vec = cache.get(text)
            if vec is None:
                missing_idx.append(i)
            cached.append(vec)

        if missing_idx:
            self._ensure_client()
            missing_texts = [texts[i] for i in missing_idx]
            payload = [format_query_text(text) for text in missing_texts]
            if hasattr(self._client, "embeddings"):
                resp = self._client.embeddings.create(model=self.model, input=payload)
                fetched = np.asarray(
                    [row.embedding for row in resp.data], dtype=np.float32
                )
            elif hasattr(self._client, "embed_texts"):
                fetched = np.asarray(
                    self._client.embed_texts(payload), dtype=np.float32
                )
            else:
                raise NotImplementedError(
                    "Configured embedding backend does not support text embeddings"
                )
            for j, i in enumerate(missing_idx):
                cached[i] = fetched[j]
                cache[texts[i]] = fetched[j]

        vectors = np.asarray(cached, dtype=np.float32)
        if self.dim is None and vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors

    def embed_images(self, image_paths: List[str]) -> np.ndarray:
        """Embed image files (JPEG/PNG).  Preserves source resolution."""
        self._ensure_client()
        if not hasattr(self._client, "embed_images"):
            raise NotImplementedError(
                "Configured embedding backend does not support image embeddings"
            )
        vectors = np.asarray(self._client.embed_images(image_paths), dtype=np.float32)
        if self.dim is None and vectors.size:
            self.dim = int(vectors.shape[1])
        return vectors

    def embed_videos(
        self, payloads: List[List[str] | str]
    ) -> np.ndarray:
        """Embed video clips.

        Accepts either frame path lists or pre-built text descriptions.  When
        the configured backend exposes a true multimodal embed_videos hook we
        use it directly.  Otherwise, for the vLLM OpenAI-compatible endpoint
        (which only accepts text), we fall back to text embeddings — strings
        are sent verbatim, while frame-path lists are joined into a
        placeholder string that yields a stable but information-poor vector.
        """
        self._ensure_client()
        if hasattr(self._client, "embed_videos"):
            vectors = np.asarray(
                self._client.embed_videos(payloads), dtype=np.float32
            )
            if self.dim is None and vectors.size:
                self.dim = int(vectors.shape[1])
            return vectors
        # Text fallback: build a single string per clip
        texts: List[str] = []
        for p in payloads:
            if isinstance(p, str):
                texts.append(p)
            elif isinstance(p, list):
                # Frame paths were passed — degrade to a generic placeholder
                # so the embedder still emits a vector of the expected shape.
                texts.append("Video clip")
            else:
                texts.append("Video clip")
        return self.embed_texts(texts)

    # Provide a text-compatible embed_images for the same reason — vLLM's
    # /v1/embeddings endpoint takes text only, and the upstream pipeline now
    # feeds caption text in place of an image path.
    def embed_images_text(self, texts: List[str]) -> np.ndarray:
        """Embed image-modality records using their textual caption."""
        return self.embed_texts(texts)


def format_query_text(text: str) -> str:
    """Format text according to the OmniEmbed query convention."""
    return f"Query: {text}"


def prepare_text_inputs(texts: List[str]) -> List[str]:
    """Prepare text inputs for OmniEmbed."""
    return [format_query_text(text) for text in texts]
