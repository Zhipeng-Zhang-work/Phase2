# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Optional, Union
import math

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.v1.compute.attention.metadata import LMCAttnMetadata
from lmcache.v1.compute.blend.metadata import LMCBlendCommonMetadata, LMCBlendMetadata
from lmcache.v1.compute.models.utils import infer_model_from_vllm
from lmcache.v1.config import LMCacheEngineConfig

logger = init_logger(__name__)


@dataclass(frozen=True)
class BlendPromptSpans:
    """Semantic token positions in one complete CacheBlend prompt."""

    system_indices: torch.Tensor
    document_indices: torch.Tensor
    separator_indices: torch.Tensor
    instruction_indices: torch.Tensor
    question_indices: torch.Tensor
    answer_prefix_indices: torch.Tensor

    @classmethod
    def from_dict(
        cls,
        spans: dict,
        prompt_length: int,
        device: torch.device,
    ) -> "BlendPromptSpans":
        """Validate and convert serialized prompt spans to device tensors.

        Args:
            spans: Mapping produced by the RAG prompt builder.
            prompt_length: Number of tokens in the complete prompt.
            device: Device on which selection is performed.

        Returns:
            Validated semantic prompt spans.

        Raises:
            ValueError: If a required region is missing, overlaps another
                region, or does not cover the complete prompt.
        """
        required = (
            "system_indices",
            "document_indices",
            "separator_indices",
            "instruction_indices",
            "question_indices",
            "answer_prefix_indices",
        )
        missing = [name for name in required if name not in spans]
        if missing:
            raise ValueError(f"Missing CacheBlend prompt spans: {missing}")

        tensors = {
            name: torch.tensor(spans[name], dtype=torch.int64, device=device)
            for name in required
        }
        owner = torch.full(
            (prompt_length,),
            -1,
            dtype=torch.int8,
            device=device,
        )
        for region_id, name in enumerate(required):
            indices = tensors[name]
            if indices.numel() == 0:
                continue
            if torch.any(indices < 0) or torch.any(indices >= prompt_length):
                raise ValueError(f"Prompt span {name} contains invalid positions")
            if torch.unique(indices).numel() != indices.numel():
                raise ValueError(f"Prompt span {name} contains duplicate positions")
            if torch.any(owner[indices] != -1):
                raise ValueError(f"Prompt span {name} overlaps another region")
            owner[indices] = region_id

        if torch.any(owner == -1):
            missing_positions = torch.where(owner == -1)[0][:16].tolist()
            raise ValueError(
                "Semantic prompt spans do not cover the complete prompt; "
                f"first missing positions: {missing_positions}"
            )
        if tensors["document_indices"].numel() == 0:
            raise ValueError("Query-aware CacheBlend requires document tokens")
        if tensors["question_indices"].numel() == 0:
            raise ValueError("Query-aware CacheBlend requires question tokens")
        return cls(**tensors)


def _repeat_kv_heads(keys: torch.Tensor, groups: int) -> torch.Tensor:
    """Repeat KV heads to align with query heads under GQA."""
    return keys.repeat_interleave(groups, dim=0)


def score_documents_by_observer_attention(
    queries: torch.Tensor,
    keys: torch.Tensor,
    observer_indices: torch.Tensor,
    document_indices: torch.Tensor,
    positions: torch.Tensor,
    num_heads: int,
    num_kv_heads: int,
    head_dim: int,
) -> torch.Tensor:
    """Score documents using observer attention under the original softmax.

    Softmax is evaluated over every causally visible key before document
    columns are selected. The result contains one scalar per document token.

    Args:
        queries: RoPE-applied query projections for the complete prompt.
        keys: RoPE-applied key projections for the complete prompt.
        observer_indices: Global positions of instruction/question observers.
        document_indices: Global positions of reusable document candidates.
        positions: Original positions corresponding to query/key rows.
        num_heads: Number of query heads.
        num_kv_heads: Number of KV heads.
        head_dim: Attention head dimension.

    Returns:
        Mean full-softmax attention score for each document token.

    Raises:
        ValueError: If the tensors do not describe a full prompt or GQA heads
            are incompatible.
    """
    sequence_length = keys.shape[0]
    if queries.shape[0] != sequence_length or positions.numel() != sequence_length:
        raise ValueError("Observer scoring requires the complete prompt Q/K tensors")
    if num_heads % num_kv_heads != 0:
        raise ValueError("Query heads must be divisible by KV heads for GQA")

    query_heads = queries.view(sequence_length, num_heads, head_dim).transpose(0, 1)
    key_heads = keys.view(sequence_length, num_kv_heads, head_dim).transpose(0, 1)
    repeated_keys = _repeat_kv_heads(key_heads, num_heads // num_kv_heads)

    observer_queries = query_heads[:, observer_indices, :]
    logits = torch.matmul(
        observer_queries.to(torch.float32),
        repeated_keys.transpose(1, 2).to(torch.float32),
    ) / math.sqrt(head_dim)

    observer_positions = positions[observer_indices]
    causal_mask = positions.view(1, 1, -1) > observer_positions.view(1, -1, 1)
    logits = logits.masked_fill(causal_mask, float("-inf"))

    full_attention = torch.softmax(logits, dim=-1)
    document_attention = full_attention[..., document_indices]
    document_scores = document_attention.mean(dim=1).mean(dim=0)
    if document_scores.numel() != document_indices.numel():
        raise AssertionError("Document score count does not match candidates")
    return document_scores


def select_document_repair_and_active_positions(
    document_scores: torch.Tensor,
    document_indices: torch.Tensor,
    mandatory_indices: torch.Tensor,
    recompute_ratio: float,
    prompt_length: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Select the global document repair set and budget-free active set.

    Args:
        document_scores: One importance value per reused document token.
        document_indices: Global positions of reused document tokens.
        mandatory_indices: Global positions that must remain online-active.
        recompute_ratio: Fraction of reused document tokens to repair.
        prompt_length: Number of tokens in the complete prompt.

    Returns:
        Sorted selected-document positions and their union with mandatory
        online positions.

    Raises:
        ValueError: If candidates are empty or the ratio is invalid.
    """
    if document_indices.numel() == 0:
        raise ValueError("Cannot select repair tokens from an empty document set")
    if document_scores.numel() != document_indices.numel():
        raise ValueError("Document scores and positions must have equal length")
    if recompute_ratio < 0.0 or recompute_ratio > 1.0:
        raise ValueError("Recompute ratio must be between zero and one")

    repair_budget = max(1, int(document_indices.numel() * recompute_ratio))
    repair_budget = min(repair_budget, document_indices.numel())
    selected_local = torch.topk(document_scores, k=repair_budget).indices
    selected_documents = document_indices[selected_local]
    selected_documents, _ = torch.sort(selected_documents)
    active_positions = torch.unique(
        torch.cat((selected_documents, mandatory_indices)),
        sorted=True,
    )

    document_membership = torch.zeros(
        prompt_length,
        dtype=torch.bool,
        device=document_indices.device,
    )
    document_membership[document_indices] = True
    if not torch.all(document_membership[selected_documents]):
        raise AssertionError("Repair set contains a non-document token")
    if selected_documents.numel() != repair_budget:
        raise AssertionError("Repair set does not match the document budget")
    if not torch.all(torch.isin(mandatory_indices, active_positions)):
        raise AssertionError("Mandatory online tokens left the active set")
    return selected_documents, active_positions


class QueryAwareDocumentBlender:
    """Shared implementation for query-aware document-only KV repair."""

    include_instruction_observers = False
    requires_semantic_prompt_spans = True
    variant_name = "query-aware-document-repair"

    def __init__(
        self,
        cache_engine,
        gpu_connector,
        vllm_model,
        config: LMCacheEngineConfig,
    ) -> None:
        self.cache_engine = cache_engine
        self.gpu_connector = gpu_connector

        enable_sparse = False
        if config.extra_config is not None:
            enable_sparse = config.extra_config.get("enable_sparse", False)
        self.layerwise_model = infer_model_from_vllm(
            vllm_model,
            self,
            enable_sparse,
        )
        self.num_layers = len(vllm_model.model.layers)

        check_layers = config.blend_check_layers or []
        if len(check_layers) != 1:
            raise ValueError(
                "Query-aware CacheBlend requires exactly one checking layer"
            )
        if config.blend_pyramid_enabled:
            raise ValueError("Query-aware CacheBlend does not support pyramid mode")
        ratios = config.blend_recompute_ratios
        if ratios is None or len(ratios) != 1:
            raise ValueError(
                "Query-aware CacheBlend requires exactly one recompute ratio"
            )
        self.common_metadata = LMCBlendCommonMetadata(
            check_layers=check_layers,
            recomp_ratios=ratios,
            thresholds=config.blend_thresholds,
        )
        self.metadata = LMCBlendMetadata(
            imp_indices=None,
            attn_mask=None,
            positions=None,
        )
        self.prompt_spans: Optional[BlendPromptSpans] = None
        self.cached_token_count: Optional[int] = None
        self.full_prompt_length: Optional[int] = None

    def _observer_indices(self) -> torch.Tensor:
        """Return the exact observer positions for this variant."""
        assert self.prompt_spans is not None
        if self.include_instruction_observers:
            return torch.cat(
                (
                    self.prompt_spans.instruction_indices,
                    self.prompt_spans.question_indices,
                )
            )
        return self.prompt_spans.question_indices

    def _candidate_and_mandatory_indices(
        self,
        device: torch.device,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return reused-document candidates and budget-free active positions."""
        assert self.prompt_spans is not None
        assert self.cached_token_count is not None
        assert self.full_prompt_length is not None

        documents = self.prompt_spans.document_indices
        reused_documents = documents[documents < self.cached_token_count]
        if reused_documents.numel() == 0:
            raise ValueError("No reused document tokens are available for repair")

        reusable = torch.zeros(
            self.full_prompt_length,
            dtype=torch.bool,
            device=device,
        )
        cached_system = self.prompt_spans.system_indices[
            self.prompt_spans.system_indices < self.cached_token_count
        ]
        reusable[cached_system] = True
        reusable[reused_documents] = True
        mandatory = torch.where(~reusable)[0]
        return reused_documents, mandatory

    @staticmethod
    def _copy_active_rows_to_cache(
        cache: torch.Tensor,
        fresh: torch.Tensor,
        active_positions: torch.Tensor,
    ) -> None:
        """Write fresh active rows that belong to the cacheable prefix."""
        cached_rows = active_positions < cache.shape[0]
        if torch.any(cached_rows):
            cache[active_positions[cached_rows]] = fresh[cached_rows]

    @staticmethod
    def _assemble_full_keys(
        cache: torch.Tensor,
        fresh: torch.Tensor,
        active_positions: torch.Tensor,
        prompt_length: int,
    ) -> torch.Tensor:
        """Combine cached prefix rows with fresh mandatory online rows."""
        if cache.shape[0] > prompt_length:
            raise ValueError("Retrieved cache is longer than the complete prompt")
        assembled = torch.empty(
            (prompt_length, cache.shape[1]),
            dtype=cache.dtype,
            device=cache.device,
        )
        assembled[: cache.shape[0]].copy_(cache)
        online_rows = active_positions >= cache.shape[0]
        online_positions = active_positions[online_rows]
        expected_online = torch.arange(
            cache.shape[0],
            prompt_length,
            dtype=torch.int64,
            device=cache.device,
        )
        if not torch.equal(online_positions, expected_online):
            raise ValueError("Mandatory active set does not cover the online suffix")
        assembled[online_positions] = fresh[online_rows]
        return assembled

    def process_qkv(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        residual: torch.Tensor,
        layer_id: int,
        attn_output: Optional[torch.Tensor],
        attn_metadata: LMCAttnMetadata,
    ):
        """Apply query-aware selection while preserving cached KV writeback."""
        old_k, old_v = self.gpu_connector.get_kv(layer_id)
        if attn_output is None:
            attn_output = torch.empty(q.shape, dtype=q.dtype, device=q.device)

        if self.metadata.positions is None:
            self.metadata.positions = torch.arange(
                q.shape[0],
                device=q.device,
                dtype=torch.int64,
            )
        active_positions = self.metadata.positions
        assert active_positions is not None
        assert self.full_prompt_length is not None
        assert self.cached_token_count is not None
        if old_k.shape[0] != self.cached_token_count:
            raise ValueError(
                "Retrieved layerwise cache length does not match the reusable "
                f"prefix: {old_k.shape[0]} != {self.cached_token_count}"
            )

        layer = self.layerwise_model.vllm_model.model.layers[layer_id]
        q, k = layer.self_attn.rotary_emb(active_positions, q, k)

        check_layer = self.common_metadata.check_layers[0]
        if layer_id < check_layer:
            self._copy_active_rows_to_cache(old_k, k, active_positions)
            self._copy_active_rows_to_cache(old_v, v, active_positions)
            return q, k, v, residual, attn_output, attn_metadata

        if layer_id == check_layer:
            expected_positions = torch.arange(
                self.full_prompt_length,
                dtype=torch.int64,
                device=q.device,
            )
            if not torch.equal(active_positions, expected_positions):
                raise ValueError(
                    "The checking layer requires complete prompt Q/K positions"
                )
            document_indices, mandatory_indices = self._candidate_and_mandatory_indices(
                q.device
            )
            observer_indices = self._observer_indices()
            num_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_heads
            num_kv_heads = self.layerwise_model.vllm_attn_layers[layer_id].num_kv_heads
            head_dim = self.layerwise_model.vllm_attn_layers[layer_id].head_size
            scores = score_documents_by_observer_attention(
                q,
                k,
                observer_indices,
                document_indices,
                active_positions,
                num_heads,
                num_kv_heads,
                head_dim,
            )

            assert self.common_metadata.recomp_ratios is not None
            ratio = self.common_metadata.recomp_ratios[0]
            selected_documents, active_positions = (
                select_document_repair_and_active_positions(
                    scores,
                    document_indices,
                    mandatory_indices,
                    ratio,
                    self.full_prompt_length,
                )
            )
            repair_budget = selected_documents.numel()

            local_active = torch.searchsorted(
                self.metadata.positions,
                active_positions,
            )
            active_k = k[local_active]
            active_v = v[local_active]
            self._copy_active_rows_to_cache(
                old_k,
                active_k,
                active_positions,
            )
            self._copy_active_rows_to_cache(
                old_v,
                active_v,
                active_positions,
            )
            full_k = self._assemble_full_keys(
                old_k,
                active_k,
                active_positions,
                self.full_prompt_length,
            )
            full_v = self._assemble_full_keys(
                old_v,
                active_v,
                active_positions,
                self.full_prompt_length,
            )

            q = q[local_active]
            residual = residual[local_active]
            self.metadata.imp_indices = selected_documents
            self.metadata.positions = active_positions
            attn_output = attn_output[: active_positions.numel()]
            attn_metadata.update_from_top_indices(active_positions)
            logger.info(
                "%s selected %d/%d reused document tokens; active=%d",
                self.variant_name,
                repair_budget,
                document_indices.numel(),
                active_positions.numel(),
            )
            return q, full_k, full_v, residual, attn_output, attn_metadata

        self._copy_active_rows_to_cache(old_k, k, active_positions)
        self._copy_active_rows_to_cache(old_v, v, active_positions)
        full_k = self._assemble_full_keys(
            old_k,
            k,
            active_positions,
            self.full_prompt_length,
        )
        full_v = self._assemble_full_keys(
            old_v,
            v,
            active_positions,
            self.full_prompt_length,
        )
        return q, full_k, full_v, residual, attn_output, attn_metadata

    def blend_layer(
        self,
        tokens: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ):
        """Run full-prompt auxiliary compute and cached-prefix retrieval."""
        retrieval_tokens = kwargs.pop("retrieval_tokens")
        spans = kwargs.pop("blend_prompt_spans")
        cached_token_count = int(kwargs.pop("cached_token_count"))
        if cached_token_count != len(retrieval_tokens):
            raise ValueError("Cached token count and retrieval prefix disagree")

        self.full_prompt_length = len(tokens)
        self.cached_token_count = cached_token_count
        self.prompt_spans = BlendPromptSpans.from_dict(
            spans,
            self.full_prompt_length,
            tokens.device,
        )

        layerwise_model_executor = self.layerwise_model.compute_layer(tokens)
        layerwise_retriever = self.cache_engine.retrieve_layer(
            retrieval_tokens,
            mask,
            **kwargs,
        )
        next(layerwise_retriever)
        yield
        for _ in range(self.num_layers):
            next(layerwise_retriever)
            next(layerwise_model_executor)
            yield
        next(layerwise_retriever)
        self.metadata.clean()
        self.prompt_spans = None
        self.cached_token_count = None
        self.full_prompt_length = None
        yield

    def blend(
        self,
        tokens: Union[torch.Tensor, list[int]],
        mask: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> None:
        """Perform query-aware blending for one complete prompt."""
        if isinstance(tokens, list):
            tokens = torch.tensor(tokens).cuda()
        retrieval_tokens = kwargs.get("retrieval_tokens")
        if isinstance(retrieval_tokens, list):
            kwargs["retrieval_tokens"] = torch.tensor(retrieval_tokens).cuda()

        layerwise_blender = self.blend_layer(tokens, mask, **kwargs)
        for _ in range(self.num_layers + 2):
            next(layerwise_blender)
