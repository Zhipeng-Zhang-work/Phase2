# SPDX-License-Identifier: Apache-2.0
"""Standalone Transformers engine for TurboRAG-reordered."""

# Standard
from dataclasses import dataclass
from typing import Any
import time

# Third Party
from transformers import AutoModelForCausalLM, AutoTokenizer, DynamicCache
import torch

# Local
from .cache_store import DocumentCacheStore


@dataclass(frozen=True)
class GenerationResult:
    """Generated text and online latency measurements."""

    text: str
    ttft: float
    tpot: float
    total_time: float
    output_tokens: int


class TurboRAGEngine:
    """Precompute, reorder, concatenate, and consume document KV caches."""

    def __init__(
        self,
        model_path: str,
        cache_dir: str,
        device: str = "cuda",
    ) -> None:
        """Load a TurboRAG-trained model and tokenizer."""
        self.device = torch.device(device)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path,
            dtype=torch.bfloat16,
            attn_implementation="eager",
            low_cpu_mem_usage=False,
        )
        self.model.to(self.device)
        self.model.eval()
        if getattr(self.model.config, "model_type", None) != "qwen2":
            raise ValueError("the standalone TurboRAG engine currently supports Qwen2")
        if getattr(self.model.config, "rope_scaling", None):
            raise ValueError("scaled RoPE is not supported by cache relocation")
        self.store = DocumentCacheStore(cache_dir)
        self.document_start = self._required_token("<|doc_start|>")
        self.document_end = self._required_token("<|doc_end|>")
        self.rope_theta = float(getattr(self.model.config, "rope_theta", 10000.0))

    def document_token_ids(self, text: str) -> list[int]:
        """Tokenize and wrap a document exactly as TurboRAG training does."""
        content = self.tokenizer(
            text,
            add_special_tokens=False,
        ).input_ids
        return [self.document_start, *content, self.document_end]

    @torch.inference_mode()
    def precompute_document(self, text: str) -> tuple[str, bool]:
        """Compute and persist one independently encoded document."""
        token_ids = self.document_token_ids(text)
        key = self.store.key(token_ids)
        if self.store.contains(key):
            return key, False
        cache = DynamicCache(config=self.model.config)
        output = self._forward(token_ids, cache)
        self.store.save(key, token_ids, output.past_key_values)
        return key, True

    @torch.inference_mode()
    def generate(
        self,
        system_prompt: str,
        documents: list[str],
        query_prompt: str,
        max_new_tokens: int,
    ) -> GenerationResult:
        """Generate using concatenated TurboRAG document caches."""
        started = time.perf_counter()
        if max_new_tokens <= 0:
            raise ValueError("max_new_tokens must be positive")
        cache = DynamicCache(config=self.model.config)
        system_ids = self.tokenizer(
            system_prompt,
            add_special_tokens=True,
        ).input_ids
        self._forward(system_ids, cache)

        for document in documents:
            token_ids = self.document_token_ids(document)
            key = self.store.key(token_ids)
            cached_ids, layers = self.store.load(key)
            if cached_ids != token_ids:
                raise RuntimeError(f"cache token mismatch for {key}")
            self._append_document(cache, layers, cache.get_seq_length())

        query_ids = self.tokenizer(
            query_prompt,
            add_special_tokens=False,
        ).input_ids
        output = self._forward(query_ids, cache)
        first_token = int(output.logits[:, -1].argmax(dim=-1).item())
        self._synchronize()
        first_token_time = time.perf_counter()
        generated = [first_token]
        next_token = first_token
        while len(generated) < max_new_tokens:
            if next_token == self.tokenizer.eos_token_id:
                break
            output = self._forward([next_token], cache)
            next_token = int(output.logits[:, -1].argmax(dim=-1).item())
            generated.append(next_token)
        self._synchronize()
        finished = time.perf_counter()
        decode_tokens = max(0, len(generated) - 1)
        return GenerationResult(
            text=self.tokenizer.decode(generated, skip_special_tokens=True),
            ttft=first_token_time - started,
            tpot=(
                (finished - first_token_time) / decode_tokens if decode_tokens else 0.0
            ),
            total_time=finished - started,
            output_tokens=len(generated),
        )

    def _required_token(self, token: str) -> int:
        token_id = self.tokenizer.convert_tokens_to_ids(token)
        if token_id == self.tokenizer.unk_token_id:
            raise ValueError(
                f"{token} is absent from the checkpoint tokenizer; "
                "use the TurboRAG-trained checkpoint"
            )
        return int(token_id)

    def _synchronize(self) -> None:
        if self.device.type == "cuda":
            torch.cuda.synchronize(self.device)

    def _forward(
        self,
        token_ids: list[int],
        cache: DynamicCache,
    ) -> Any:
        start = cache.get_seq_length()
        input_ids = torch.tensor(
            [token_ids],
            dtype=torch.long,
            device=self.device,
        )
        positions = torch.arange(
            start,
            start + len(token_ids),
            dtype=torch.long,
            device=self.device,
        )
        attention_mask = torch.ones(
            (1, start + len(token_ids)),
            dtype=torch.long,
            device=self.device,
        )
        return self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=positions.unsqueeze(0),
            cache_position=positions,
            past_key_values=cache,
            use_cache=True,
            return_dict=True,
        )

    def _append_document(
        self,
        cache: DynamicCache,
        layers: list[tuple[torch.Tensor, torch.Tensor]],
        new_start: int,
    ) -> None:
        for layer_idx, (key_states, value_states) in enumerate(layers):
            length = key_states.shape[-2]
            old_positions = torch.arange(
                length,
                dtype=torch.float32,
                device=self.device,
            )
            new_positions = old_positions + new_start
            key_states = key_states.to(self.device, non_blocking=True)
            key_states = rerotate_key(
                key_states,
                old_positions,
                new_positions,
                self.rope_theta,
            )
            cache.update(
                key_states,
                value_states.to(self.device, non_blocking=True),
                layer_idx,
            )


def rerotate_key(
    key_states: torch.Tensor,
    old_positions: torch.Tensor,
    new_positions: torch.Tensor,
    rope_theta: float,
) -> torch.Tensor:
    """Move post-RoPE Qwen keys from local to global positions.

    Args:
        key_states: Key tensor shaped `[batch, heads, sequence, head_dim]`.
        old_positions: Original absolute positions for the sequence dimension.
        new_positions: Desired absolute positions for the sequence dimension.
        rope_theta: Qwen RoPE frequency base.

    Returns:
        Repositioned key tensor with the input shape and dtype.
    """
    head_dim = key_states.shape[-1]
    frequencies = 1.0 / (
        rope_theta
        ** (
            torch.arange(
                0,
                head_dim,
                2,
                dtype=torch.float32,
                device=key_states.device,
            )
            / head_dim
        )
    )
    delta = new_positions - old_positions
    angles = torch.outer(delta, frequencies)
    cos = torch.cat((angles.cos(), angles.cos()), dim=-1)[None, None, :, :]
    sin = torch.cat((angles.sin(), angles.sin()), dim=-1)[None, None, :, :]
    first, second = key_states.float().chunk(2, dim=-1)
    rotated_half = torch.cat((-second, first), dim=-1)
    return (key_states.float() * cos + rotated_half * sin).to(key_states.dtype)
