"""Sentence embeddings used by text-aware graph edit distance."""

from __future__ import annotations

from typing import Dict, Optional, Sequence

import torch
import torch.nn.functional as F
from tqdm import tqdm


DEFAULT_TEXT_SIMILARITY_MODEL = "sentence-transformers/all-MiniLM-L6-v2"


class TextSimilarityEncoder:
    """Encode compressed CoT-node text with normalized sentence embeddings.

    The implementation intentionally uses the project's existing Transformers
    dependency rather than requiring the sentence-transformers package. The
    pooling operation is attention-mask-aware mean pooling followed by L2
    normalization, matching the standard usage of the default model.
    """

    def __init__(
        self,
        model_name: str = DEFAULT_TEXT_SIMILARITY_MODEL,
        *,
        revision: Optional[str] = None,
        device: str = "auto",
        batch_size: int = 64,
        max_length: int = 256,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be positive")
        if max_length <= 0:
            raise ValueError("max_length must be positive")

        # Keep this import lazy: reading an existing graph cache should not
        # initialize Transformers or require loading the embedding model.
        from transformers import AutoModel, AutoTokenizer

        self.model_name = model_name
        self.revision = revision
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = self._resolve_device(device)

        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
        )
        self.model = AutoModel.from_pretrained(
            model_name,
            revision=revision,
        )
        self.model.eval()
        self.model.to(self.device)

        self.embedding_dimension = int(self.model.config.hidden_size)
        self.resolved_revision = getattr(self.model.config, "_commit_hash", None)

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        if device == "auto":
            return torch.device("cuda" if torch.cuda.is_available() else "cpu")
        return torch.device(device)

    @property
    def cache_metadata(self) -> Dict[str, object]:
        """Configuration that determines the cached embedding values."""
        return {
            "model_name": self.model_name,
            "requested_revision": self.revision,
            "resolved_revision": self.resolved_revision,
            "pooling": "attention_mask_mean",
            "l2_normalized": True,
            "cosine_range": "clip_0_1",
            "max_length": self.max_length,
            "embedding_dimension": self.embedding_dimension,
        }

    def encode(self, texts: Sequence[str]) -> torch.Tensor:
        """Return an ``[N, D]`` CPU float32 tensor of normalized embeddings."""
        count = len(texts)
        encoded = torch.empty(
            (count, self.embedding_dimension),
            dtype=torch.float32,
            device="cpu",
        )
        if count == 0:
            return encoded

        starts = range(0, count, self.batch_size)
        for start in tqdm(starts, desc="Encoding compressed DAG nodes"):
            batch_texts = list(texts[start : start + self.batch_size])
            inputs = self.tokenizer(
                batch_texts,
                padding=True,
                truncation=True,
                max_length=self.max_length,
                return_tensors="pt",
            )
            inputs = {name: value.to(self.device) for name, value in inputs.items()}

            with torch.inference_mode():
                hidden = self.model(**inputs).last_hidden_state
                mask = inputs["attention_mask"].unsqueeze(-1).to(hidden.dtype)
                pooled = (hidden * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1)
                pooled = F.normalize(pooled.float(), p=2, dim=1)

            end = start + len(batch_texts)
            encoded[start:end] = pooled.cpu()

        return encoded
