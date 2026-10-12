# SPDX-License-Identifier: Apache-2.0
# Standard
from typing import TYPE_CHECKING

# Third Party
from vllm.attention import Attention
from vllm.v1.attention.backends.flash_attn import FlashAttentionImpl
from vllm.vllm_flash_attn import flash_attn_varlen_func, get_scheduler_metadata
import torch

# First Party
from lmcache.v1.compute.attention.abstract import AttentionInterface
from lmcache.v1.compute.attention.metadata import LMCFlashAttnMetadata

if TYPE_CHECKING:
    # First Party
    from lmcache.v1.compute.attention.metadata import LMCAttnMetadata


class LMCFlashAttnBackend(AttentionInterface):
    """
    FlashAttention backend for LMCache.
    This backend uses the FlashAttention implementation
    for efficient attention computation.
    """

    def __init__(
        self,
        vllm_attn: Attention,
    ):
        self.vllm_attn = vllm_attn
        self.vllm_attn_impl: FlashAttentionImpl = vllm_attn.impl

        # TODO(Jiayi): remove this hardcode
        self.aot_schedule = False

        idx = torch.cuda.current_device()
        self.device = torch.device(f"cuda:{idx}")

    def forward_contiguous(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: "LMCAttnMetadata",
        **kwargs,
    ) -> torch.Tensor:
        assert isinstance(attn_metadata, LMCFlashAttnMetadata)

        if self._should_use_xformers_selected_path(attn_metadata, key):
            return self._forward_selected_xformers(
                query,
                key,
                value,
                output,
                attn_metadata,
            )

        cu_seqlens_q = attn_metadata.query_start_loc
        seqused_k = attn_metadata.seq_lens
        cu_seqlens_k = attn_metadata.cu_seqlens_k
        max_seqlen_q = attn_metadata.max_query_len
        max_seqlen_k = attn_metadata.max_seq_len

        descale_shape = (cu_seqlens_q.shape[0] - 1, key.shape[1])

        # TODO(Jiayi): Figure out how to use aot_schedule.
        scheduler_metadata = self._schedule(
            batch_size=1,  # NOTE(Jiayi): Assuming batch size is 1,
            # since we are processing request by request.
            cu_query_lens=cu_seqlens_q,
            max_query_len=max_seqlen_q,
            seqlens=seqused_k,
            max_seq_len=max_seqlen_k,
            causal=True,  # Assuming causal attention
        )

        flash_attn_varlen_func(
            q=query,  # contiguous
            k=key,  # contiguous
            v=value,  # contiguous
            out=output,
            cu_seqlens_q=cu_seqlens_q,
            max_seqlen_q=max_seqlen_q,
            cu_seqlens_k=cu_seqlens_k,
            # seqused_k=seqused_k,
            max_seqlen_k=max_seqlen_k,
            softmax_scale=self.vllm_attn_impl.scale,
            causal=True,
            alibi_slopes=self.vllm_attn_impl.alibi_slopes,
            window_size=self.vllm_attn_impl.sliding_window,
            block_table=None,
            softcap=self.vllm_attn_impl.logits_soft_cap,
            scheduler_metadata=scheduler_metadata,
            fa_version=self.vllm_attn_impl.vllm_flash_attn_version,
            q_descale=self.vllm_attn._q_scale.expand(descale_shape),
            k_descale=self.vllm_attn._k_scale.expand(descale_shape),
            v_descale=self.vllm_attn._v_scale.expand(descale_shape),
        )

        return output

    def _forward_selected_xformers(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        output: torch.Tensor,
        attn_metadata: LMCFlashAttnMetadata,
    ) -> torch.Tensor:
        # Third Party
        import xformers.ops as xops

        self._validate_xformers_selected_path()

        selected_positions = attn_metadata.selected_query_positions
        assert selected_positions is not None

        seq_len = key.shape[0]
        key_positions = torch.arange(seq_len, device=query.device, dtype=torch.int64)

        # Preserve original token positions after q is compacted. Each selected
        # query row can only attend to keys up to its original position.
        mask = key_positions.unsqueeze(0) > selected_positions.unsqueeze(1)
        padded_seq_len = ((seq_len + 7) // 8) * 8
        padded_attn_bias = torch.zeros(
            (1, 1, query.shape[0], padded_seq_len),
            dtype=query.dtype,
            device=query.device,
        )
        padded_attn_bias[:, :, :, seq_len:] = float("-inf")
        padded_attn_bias[:, :, :, :seq_len].masked_fill_(
            mask.view(1, 1, query.shape[0], seq_len),
            float("-inf"),
        )
        attn_bias = padded_attn_bias[:, :, :, :seq_len]

        if query.shape[1] == key.shape[1]:
            head_attn_bias = attn_bias.expand(
                -1,
                query.shape[1],
                -1,
                -1,
            )
            out = xops.memory_efficient_attention(
                query.contiguous().unsqueeze(0),
                key.contiguous().unsqueeze(0),
                value.contiguous().unsqueeze(0),
                attn_bias=head_attn_bias,
                p=0.0,
                scale=self.vllm_attn_impl.scale,
            )
            output.copy_(out.squeeze(0))
            return output

        if query.shape[1] % key.shape[1] != 0:
            raise ValueError(
                "The number of query heads must be divisible by the number "
                "of KV heads for xFormers selected-token recompute."
            )

        group_size = query.shape[1] // key.shape[1]
        grouped_attn_bias = attn_bias.unsqueeze(2).expand(
            -1,
            key.shape[1],
            group_size,
            -1,
            -1,
        )

        query_5d = query.contiguous().view(
            1,
            query.shape[0],
            key.shape[1],
            group_size,
            query.shape[2],
        )
        key_5d = key.contiguous().view(
            1,
            key.shape[0],
            key.shape[1],
            1,
            key.shape[2],
        )
        value_5d = value.contiguous().view(
            1,
            value.shape[0],
            value.shape[1],
            1,
            value.shape[2],
        )

        # xFormers supports GQA/MQA with 5D [B, M, G, H, D] tensors.  Expand
        # KV heads logically across each query-head group so all KV groups are
        # handled by one kernel call instead of one call per KV head.
        out = xops.memory_efficient_attention(
            query_5d,
            key_5d.expand(-1, -1, -1, group_size, -1),
            value_5d.expand(-1, -1, -1, group_size, -1),
            attn_bias=grouped_attn_bias,
            p=0.0,
            scale=self.vllm_attn_impl.scale,
        )
        output.copy_(out.reshape_as(output))

        return output

    def _should_use_xformers_selected_path(
        self,
        attn_metadata: LMCFlashAttnMetadata,
        key: torch.Tensor,
    ) -> bool:
        # Use the same xFormers path for every recompute selection, including
        # ratio=1.0, so TTFT comparisons do not mix attention backends.
        return attn_metadata.selected_query_positions is not None

    def _validate_xformers_selected_path(self) -> None:
        if self.vllm_attn_impl.alibi_slopes is not None:
            raise NotImplementedError(
                "xFormers selected-token recompute does not support ALiBi yet."
            )

        sliding_window = self.vllm_attn_impl.sliding_window
        if sliding_window not in (None, -1, (-1, -1)):
            raise NotImplementedError(
                "xFormers selected-token recompute does not support sliding "
                "window attention yet."
            )

        logits_soft_cap = self.vllm_attn_impl.logits_soft_cap
        if logits_soft_cap not in (None, 0.0):
            raise NotImplementedError(
                "xFormers selected-token recompute does not support logits "
                "soft cap yet."
            )

    def _schedule(
        self, batch_size, cu_query_lens, max_query_len, seqlens, max_seq_len, causal
    ):
        if self.aot_schedule:
            return get_scheduler_metadata(
                batch_size=batch_size,
                max_seqlen_q=max_query_len,
                max_seqlen_k=max_seq_len,
                cache_seqlens=seqlens,
                num_heads_q=self.vllm_attn_impl.num_heads_q,
                num_heads_kv=self.vllm_attn_impl.num_heads_kv,
                headdim=self.vllm_attn_impl.headdim,
                page_size=self.vllm_attn_impl.block_size,
                cu_seqlens_q=cu_query_lens,
                causal=causal,
                window_size=self.vllm_attn_impl.aot_sliding_window,
            )
        return None

    def init_attn_metadata(
        self,
        input_ids: torch.tensor,
        **kwargs,
    ) -> LMCFlashAttnMetadata:
        seq_len = input_ids.shape[0]
        device = input_ids.device
        return LMCFlashAttnMetadata(
            query_start_loc=torch.tensor(
                [0, seq_len], dtype=torch.int32, device=device
            ),
            seq_lens=torch.tensor([seq_len], device=device),
            cu_seqlens_k=torch.tensor([0, seq_len], dtype=torch.int32, device=device),
            max_query_len=seq_len,
            max_seq_len=seq_len,
        )
