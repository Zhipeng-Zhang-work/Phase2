# SPDX-License-Identifier: Apache-2.0
"""Persistent per-document KV cache for standalone TurboRAG evaluation."""

# Standard
from pathlib import Path
from typing import Any
import hashlib
import json

# Third Party
from safetensors.torch import load_file, save_file
import torch


class DocumentCacheStore:
    """Store one complete layered KV cache per document."""

    def __init__(self, root: str) -> None:
        """Create a cache rooted at `root`."""
        self.root = Path(root)
        self.root.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def key(token_ids: list[int]) -> str:
        """Return a stable SHA-256 key for token IDs."""
        payload = json.dumps(token_ids, separators=(",", ":")).encode()
        return hashlib.sha256(payload).hexdigest()

    def contains(self, key: str) -> bool:
        """Return whether a complete cache entry exists."""
        return self._tensor_path(key).exists() and self._metadata_path(key).exists()

    def save(
        self,
        key: str,
        token_ids: list[int],
        past_key_values: Any,
    ) -> None:
        """Atomically save one document's KV tensors."""
        legacy = (
            past_key_values.to_legacy_cache()
            if hasattr(past_key_values, "to_legacy_cache")
            else past_key_values
        )
        tensors: dict[str, torch.Tensor] = {}
        for layer_idx, (key_states, value_states) in enumerate(legacy):
            tensors[f"layer.{layer_idx}.key"] = key_states.detach().cpu().contiguous()
            tensors[f"layer.{layer_idx}.value"] = (
                value_states.detach().cpu().contiguous()
            )
        tensor_path = self._tensor_path(key)
        temporary_path = tensor_path.with_suffix(".safetensors.tmp")
        save_file(tensors, str(temporary_path))
        temporary_path.replace(tensor_path)
        self._metadata_path(key).write_text(
            json.dumps(
                {
                    "token_ids": token_ids,
                    "num_layers": len(legacy),
                    "sequence_length": len(token_ids),
                }
            ),
            encoding="utf-8",
        )

    def load(
        self,
        key: str,
    ) -> tuple[list[int], list[tuple[torch.Tensor, torch.Tensor]]]:
        """Load token IDs and CPU KV tensors."""
        metadata = json.loads(self._metadata_path(key).read_text(encoding="utf-8"))
        tensors = load_file(str(self._tensor_path(key)), device="cpu")
        layers = [
            (
                tensors[f"layer.{layer_idx}.key"],
                tensors[f"layer.{layer_idx}.value"],
            )
            for layer_idx in range(metadata["num_layers"])
        ]
        return list(metadata["token_ids"]), layers

    def _tensor_path(self, key: str) -> Path:
        return self.root / f"{key}.safetensors"

    def _metadata_path(self, key: str) -> Path:
        return self.root / f"{key}.json"
