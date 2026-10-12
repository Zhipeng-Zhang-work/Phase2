# SPDX-License-Identifier: Apache-2.0
# Standard
import os

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.token_database import ChunkedTokenDatabase, SegmentTokenDatabase

# Local
from .utils import dumb_metadata, dumb_metadata_with_model_name, generate_tokens


def hf_credentials_available() -> bool:
    token_env = os.getenv("HF_TOKEN")
    hf_home = os.getenv("HF_HOME")
    default_token_file = os.path.expanduser("~/.cache/huggingface/token")
    token_file = os.path.join(hf_home, "token") if hf_home else ""
    return bool(
        token_env or os.path.exists(default_token_file) or os.path.exists(token_file)
    )


def test_segment_token_database_preserves_separator_without_bos(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    class Qwen3StyleTokenizer:
        """Tokenizer whose default encoding does not prepend a BOS token."""

        def __init__(self) -> None:
            self.encode_calls: list[bool] = []

        def encode(
            self,
            text: str,
            add_special_tokens: bool = True,
        ) -> list[int]:
            assert text == " # # "
            self.encode_calls.append(add_special_tokens)
            return [671, 671, 220]

    tokenizer = Qwen3StyleTokenizer()

    def fake_from_pretrained(model_name: str) -> Qwen3StyleTokenizer:
        assert model_name == "Qwen/Qwen3-32B"
        return tokenizer

    monkeypatch.setattr(
        "lmcache.v1.token_database.AutoTokenizer.from_pretrained",
        fake_from_pretrained,
    )

    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("Qwen/Qwen3-32B")
    db = SegmentTokenDatabase(cfg, metadata)

    assert tokenizer.encode_calls == [False]
    assert db.sep_tokens.tolist() == [671, 671, 220]

    segments = [
        torch.tensor([101, 102]),
        torch.tensor([201, 202, 203]),
        torch.tensor([301]),
    ]
    combined_tokens = torch.cat(
        [segments[0], db.sep_tokens, segments[1], db.sep_tokens, segments[2]]
    )

    combined_results = list(db.process_tokens(tokens=combined_tokens, make_key=False))
    independent_hashes = [
        list(db.process_tokens(tokens=segment, make_key=False))[0][2]
        for segment in segments
    ]

    assert [(start, end) for start, end, _ in combined_results] == [
        (0, 2),
        (5, 8),
        (11, 12),
    ]
    assert [chunk_hash for _, _, chunk_hash in combined_results] == independent_hashes

    precompute_results = list(
        db.process_tokens(
            tokens=combined_tokens,
            request_configs={
                "lmcache.store_full_prompt_on_last_prefill": True,
            },
        )
    )
    online_results = list(db.process_tokens(tokens=combined_tokens))

    assert [key for _, _, key in precompute_results] == [
        key for _, _, key in online_results
    ]


@pytest.mark.parametrize("chunk_length", [16, 64, 256])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
def test_chunked_token_database(chunk_length, save_unfull_chunk):
    cfg = LMCacheEngineConfig.from_legacy(
        chunk_size=chunk_length, backend="cpu", save_unfull_chunk=save_unfull_chunk
    )
    metadata = dumb_metadata()

    test_length = 2500
    tokens = generate_tokens(test_length, "cpu")
    mask = torch.full([test_length], True, dtype=torch.bool, device="cpu")

    num_falses = [i * chunk_length for i in range(0, test_length // chunk_length)]

    db = ChunkedTokenDatabase(cfg, metadata)

    # Process without mask
    original_results = list(db.process_tokens(tokens=tokens))
    end = (
        test_length if save_unfull_chunk else (test_length - test_length % chunk_length)
    )
    for i in range(0, end, chunk_length):
        st, ed, key = original_results[i // chunk_length]
        assert st == i
        if save_unfull_chunk:
            assert ed == min(i + chunk_length, test_length)
        else:
            assert ed == i + chunk_length

    for i in range(0, test_length // chunk_length):
        mask[: num_falses[i]] = False
        new_results = list(db.process_tokens(tokens=tokens, mask=mask))
        assert len(new_results) == len(original_results) - i

        for j in range(len(new_results)):
            st, ed, key = new_results[j]
            assert st == original_results[j + i][0]
            assert ed == original_results[j + i][1]


@pytest.mark.parametrize("prefix_length", [0, 16, 64, 256])
@pytest.mark.parametrize("chunk_lengths", [[256, 512, 256], [1024, 512, 256]])
@pytest.mark.skipif(
    not hf_credentials_available(), reason="No Hugging Face credentials found"
)
def test_segment_token_database(prefix_length, chunk_lengths):
    cfg = LMCacheEngineConfig.from_legacy(blend_special_str=" # # ")
    metadata = dumb_metadata_with_model_name("facebook/opt-125m")

    db = SegmentTokenDatabase(cfg, metadata)
    sep_tokens = db.sep_tokens

    sys_length = 25
    query_length = 50
    sys_tokens = generate_tokens(sys_length, "cpu", fixed=True)
    query_tokens = generate_tokens(query_length, "cpu", fixed=True)

    token_chunks = []
    starts = [0]
    ends = [sys_length]
    sys_tuple = tuple(sys_tokens.cpu().tolist())
    sys_hash = hash((None, sys_tuple, None))
    hashes = [sys_hash]
    start = sys_length + len(sep_tokens)
    for idx, chunk_length in enumerate(chunk_lengths):
        token_chunk = generate_tokens(chunk_length, "cpu", fixed=True)

        token_tuple = tuple(token_chunk.cpu().tolist())
        token_hash = hash((None, token_tuple, None))
        hashes.append(token_hash)

        token_chunk = torch.cat([sep_tokens, token_chunk])
        token_chunks.append(token_chunk)
        starts.append(start)
        ends.append(start + chunk_length)
        start += chunk_length + len(sep_tokens)

    query_tuple = tuple(query_tokens.cpu().tolist())
    query_hash = hash((None, query_tuple, None))
    hashes.append(query_hash)
    starts.append(start)
    ends.append(start + query_length)

    tokens = torch.cat([sys_tokens, *token_chunks, sep_tokens, query_tokens])
    total_length = len(tokens)
    mask = torch.full([total_length], True, dtype=torch.bool, device="cpu")
    mask[:prefix_length] = False

    chunk_lists = [sys_tokens, *token_chunks, sep_tokens, query_tokens]
    skip_chunk_num = 0
    cum_length = 0
    for chunk in chunk_lists:
        if prefix_length > cum_length:
            skip_chunk_num += 1
        cum_length += len(chunk)

    starts = starts[skip_chunk_num:]
    ends = ends[skip_chunk_num:]
    hashes = hashes[skip_chunk_num:]

    original_results = list(db.process_tokens(tokens=tokens, mask=mask))
    for i in range(len(original_results)):
        st, ed, key = original_results[i]
        assert st == starts[i]
        assert ed == ends[i]
        assert key.chunk_hash == hashes[i]
        # print(st, starts[i])
        # print(ed, ends[i])
