# SPDX-License-Identifier: Apache-2.0
# Local
from .flash_attn import LMCFlashAttnBackend


def _get_flash_infer_sparse_backend(vllm_attn):
    try:
        from .flash_infer_sparse import LMCFlashInferSparseBackend
    except Exception as exc:  # noqa: BLE001
        raise ImportError(
            "FlashInfer sparse backend is unavailable. Install a compatible "
            "flashinfer or disable enable_sparse."
        ) from exc
    return LMCFlashInferSparseBackend(vllm_attn)


def infer_attn_backend_from_vllm(vllm_attn, enable_sparse=False):
    attn_name = type(vllm_attn.impl).__name__
    if attn_name == "FlashInferImpl" and enable_sparse:
        return _get_flash_infer_sparse_backend(vllm_attn)
    elif attn_name == "FlashAttentionImpl" and not enable_sparse:
        return LMCFlashAttnBackend(vllm_attn)
    else:
        raise ValueError(f"Attention backend {attn_name} is not supported in LMCache.")
