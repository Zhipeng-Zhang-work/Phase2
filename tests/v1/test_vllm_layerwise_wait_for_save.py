# SPDX-License-Identifier: Apache-2.0

# Standard
from types import SimpleNamespace
from unittest.mock import Mock

# Third Party
import pytest
import torch

pytest.importorskip("vllm")

# First Party
from lmcache.integration.vllm.vllm_v1_adapter import (
    LMCacheConnectorMetadata,
    LMCacheConnectorV1Impl,
    LoadSpec,
    ReqMeta,
    RequestTracker,
    SaveSpec,
)


class _FakeParent:
    def __init__(self, metadata):
        self._connector_metadata = metadata

    def _get_connector_metadata(self):
        return self._connector_metadata


class _FakeEngine:
    def __init__(self):
        self.unpinned: list[str] = []
        self.store_steps: dict[str, int] = {}
        self.store_calls: list[str] = []

    def lookup_unpin(self, req_id: str) -> None:
        self.unpinned.append(req_id)

    def store_layer(self, token_ids, **kwargs):
        req_id = kwargs["req_id"]
        self.store_calls.append(req_id)
        self.store_steps.setdefault(req_id, 0)

        def _storer():
            while True:
                self.store_steps[req_id] += 1
                yield None

        return _storer()


class _FakeManager:
    def __init__(self, engine: _FakeEngine):
        self.lmcache_engine = engine


def _make_req(req_id: str, can_save: bool = True):
    return SimpleNamespace(
        req_id=req_id,
        token_ids=[1, 2, 3, 4],
        slot_mapping=torch.arange(4, dtype=torch.long),
        save_spec=SaveSpec(skip_leading_tokens=0, can_save=can_save),
    )


def _make_connector(requests):
    metadata = LMCacheConnectorMetadata(requests=requests)
    engine = _FakeEngine()
    connector = LMCacheConnectorV1Impl.__new__(LMCacheConnectorV1Impl)
    connector._parent = _FakeParent(metadata)
    connector._manager = _FakeManager(engine)
    connector.kv_role = "kv_producer"
    connector.use_layerwise = True
    connector.device = "cpu"
    connector._lmcache_chunk_size = 8
    connector.kv_caches = {"layer0": torch.zeros(1)}
    connector._layerwise_save_storers = {}
    return connector, metadata, engine


def test_layerwise_storer_is_request_scoped_across_interleaved_finalize() -> None:
    connector, metadata, engine = _make_connector(
        [_make_req("req-1"), _make_req("req-2")]
    )

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1", "req-2"]
    assert engine.store_steps["req-1"] == 1
    assert engine.store_steps["req-2"] == 1

    metadata.requests = [_make_req("req-1")]
    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert engine.store_steps["req-2"] == 1
    assert engine.unpinned == ["req-1"]
    assert set(connector._layerwise_save_storers.keys()) == {"req-2"}

    metadata.requests = [_make_req("req-2")]
    connector.wait_for_save()
    assert engine.store_steps["req-2"] == 2
    assert engine.unpinned == ["req-1", "req-2"]
    assert connector._layerwise_save_storers == {}


def test_wait_for_save_repeated_call_does_not_readvance_finalized_storer() -> None:
    connector, metadata, engine = _make_connector([_make_req("req-1")])
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_steps["req-1"] == 1

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2


def test_multi_layer_save_and_finalize() -> None:
    connector, _, engine = _make_connector([_make_req("req-1"), _make_req("req-2")])
    num_layers = 4

    for _ in range(num_layers):
        connector.save_kv_layer("layer_x", torch.zeros(1), None)

    assert engine.store_steps["req-1"] == num_layers
    assert engine.store_steps["req-2"] == num_layers

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == num_layers + 1
    assert engine.store_steps["req-2"] == num_layers + 1
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_skips_requests_that_cannot_save() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])
    connector.kv_role = "kv_both"
    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == []
    assert connector._layerwise_save_storers == {}


def test_layerwise_save_kv_producer_ignores_can_save_flag() -> None:
    connector, _, engine = _make_connector([_make_req("req-1", can_save=False)])

    connector.save_kv_layer("layer0", torch.zeros(1), None)
    assert engine.store_calls == ["req-1"]
    assert engine.store_steps["req-1"] == 1
    assert set(connector._layerwise_save_storers.keys()) == {"req-1"}

    connector.wait_for_save()
    assert engine.store_steps["req-1"] == 2
    assert connector._layerwise_save_storers == {}


def test_full_prompt_precompute_defers_store_until_last_prefill() -> None:
    prompt_len = 9532
    prefill_chunk_len = 2048
    block_size = 16
    tracker = RequestTracker(
        req_id="precompute-request",
        prompt_len=prompt_len,
        token_ids=list(range(prefill_chunk_len)),
        allocated_block_ids=list(range(prefill_chunk_len // block_size)),
        request_configs={
            "lmcache.store_full_prompt_on_last_prefill": True,
        },
    )

    intermediate_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )

    assert intermediate_meta is None
    assert tracker.num_saved_tokens == 0

    next_token_id = prefill_chunk_len
    next_block_id = prefill_chunk_len // block_size
    for _ in range(3):
        new_token_ids = list(range(next_token_id, next_token_id + prefill_chunk_len))
        new_block_ids = list(
            range(next_block_id, next_block_id + prefill_chunk_len // block_size)
        )
        tracker.update(new_token_ids, new_block_ids)
        intermediate_meta = ReqMeta.from_request_tracker(
            tracker,
            block_size=block_size,
            lmcache_chunk_size=256,
            discard_partial_chunks=True,
        )

        assert intermediate_meta is None
        assert tracker.num_saved_tokens == 0
        next_token_id += prefill_chunk_len
        next_block_id += prefill_chunk_len // block_size

    final_chunk_len = prompt_len - next_token_id
    tracker.update(
        list(range(next_token_id, prompt_len)),
        list(
            range(
                next_block_id,
                next_block_id + (final_chunk_len + block_size - 1) // block_size,
            )
        ),
    )
    final_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )

    assert final_meta is not None
    assert final_meta.is_last_prefill
    assert final_meta.save_spec is not None
    assert final_meta.save_spec.can_save
    assert final_meta.save_spec.skip_leading_tokens == 0
    assert len(final_meta.token_ids) == prompt_len

    tracker.update([prompt_len], [])
    decode_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )

    assert tracker.is_decode_phase
    assert decode_meta is None
    assert tracker.num_saved_tokens == prompt_len


def test_regular_chunked_prefill_still_saves_intermediate_chunks() -> None:
    prefill_len = 2048
    block_size = 16
    tracker = RequestTracker(
        req_id="online-request",
        prompt_len=9532,
        token_ids=list(range(prefill_len)),
        allocated_block_ids=list(range(prefill_len // block_size)),
    )

    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )

    assert meta is not None
    assert not meta.is_last_prefill
    assert meta.save_spec is not None
    assert meta.save_spec.can_save
    assert meta.save_spec.skip_leading_tokens == 0
    assert len(meta.token_ids) == prefill_len
    assert tracker.num_saved_tokens == prefill_len


def test_one_token_final_prefill_is_not_treated_as_decode() -> None:
    prompt_len = 2049
    prefill_chunk_len = 2048
    block_size = 16
    tracker = RequestTracker(
        req_id="one-token-final-prefill",
        prompt_len=prompt_len,
        token_ids=list(range(prefill_chunk_len)),
        allocated_block_ids=list(range(prefill_chunk_len // block_size)),
        request_configs={
            "lmcache.store_full_prompt_on_last_prefill": True,
        },
    )

    intermediate_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )
    assert intermediate_meta is None

    tracker.update([prefill_chunk_len], [prefill_chunk_len // block_size])
    final_meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=block_size,
        lmcache_chunk_size=256,
        discard_partial_chunks=True,
    )

    assert not tracker.is_decode_phase
    assert final_meta is not None
    assert final_meta.is_last_prefill
    assert final_meta.save_spec is not None
    assert final_meta.save_spec.can_save
    assert final_meta.save_spec.skip_leading_tokens == 0
    assert len(final_meta.token_ids) == prompt_len
    assert not final_meta.is_decode_phase


def test_preempted_decode_metadata_keeps_prompt_spans_separate() -> None:
    tracker = RequestTracker(
        req_id="preempted-decode",
        prompt_len=3,
        token_ids=[1, 2, 3],
        allocated_block_ids=[0],
        blend_prompt_spans={"question_indices": [2]},
    )
    tracker.update(
        [4],
        [0],
        preempted=True,
        lmcache_cached_tokens=3,
        all_token_ids=[1, 2, 3, 4],
    )

    meta = ReqMeta.from_request_tracker(
        tracker,
        block_size=8,
        lmcache_chunk_size=1,
        load_spec=LoadSpec(0, 3, True),
    )

    assert meta is not None
    assert meta.is_decode_phase
    assert meta.token_ids == [1, 2, 3, 4]
    assert meta.blend_prompt_spans == {"question_indices": [2]}


def test_decode_recovery_retrieves_kv_without_prompt_blending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector, _, engine = _make_connector(
        [
            ReqMeta(
                req_id="preempted-decode",
                token_ids=[1, 2, 3, 4],
                slot_mapping=torch.arange(4, dtype=torch.long),
                is_decode_phase=True,
                load_spec=LoadSpec(0, 3, True),
                blend_prompt_spans={"question_indices": [2]},
            )
        ]
    )
    connector.enable_blending = True
    connector.blender = SimpleNamespace(
        requires_semantic_prompt_spans=True,
        blend=Mock(side_effect=AssertionError("decode must not blend")),
    )
    connector._stats_monitor = SimpleNamespace(
        update_interval_vllm_hit_tokens=Mock(),
        update_interval_prompt_tokens=Mock(),
    )
    engine.retrieve_layer = Mock(return_value=iter((None, None)))
    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_v1_adapter._append_mask_debug_record",
        lambda record: None,
    )

    connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    engine.retrieve_layer.assert_called_once()
    assert engine.retrieve_layer.call_args.args[0] == [1, 2, 3]
    connector.blender.blend.assert_not_called()


def test_prompt_prefill_still_uses_semantic_blending(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    connector, _, engine = _make_connector(
        [
            ReqMeta(
                req_id="prompt-prefill",
                token_ids=[1, 2, 3],
                slot_mapping=torch.arange(3, dtype=torch.long),
                is_decode_phase=False,
                load_spec=LoadSpec(0, 2, True),
                blend_prompt_spans={"question_indices": [2]},
            )
        ]
    )
    connector.enable_blending = True
    connector.blender = SimpleNamespace(
        requires_semantic_prompt_spans=True,
        blend=Mock(),
    )
    connector._stats_monitor = SimpleNamespace(
        update_interval_vllm_hit_tokens=Mock(),
        update_interval_prompt_tokens=Mock(),
    )
    engine.retrieve_layer = Mock(side_effect=AssertionError("prefill must blend"))
    monkeypatch.setattr(
        "lmcache.integration.vllm.vllm_v1_adapter._append_mask_debug_record",
        lambda record: None,
    )

    connector.start_load_kv(SimpleNamespace(attn_metadata=object()))

    connector.blender.blend.assert_called_once()
    engine.retrieve_layer.assert_not_called()
