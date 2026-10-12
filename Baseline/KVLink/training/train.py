# SPDX-License-Identifier: Apache-2.0
"""Distributed full-parameter fine-tuning for the KVLink baseline."""

# Standard
from functools import partial
from pathlib import Path
import argparse
import math
import os

# Third Party
from torch.distributed.fsdp import FullStateDictConfig
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
from torch.distributed.fsdp import MixedPrecision, StateDictType
from torch.distributed.fsdp.wrap import transformer_auto_wrap_policy
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen2.modeling_qwen2 import Qwen2DecoderLayer
import torch
import torch.distributed as dist

# Local
from .dataset import KVLinkCollator, KVLinkDataset
from .masking import register_kvlink_tokens


def parse_args() -> argparse.Namespace:
    """Parse KVLink training arguments."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--train-data", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-documents", type=int, default=40)
    parser.add_argument("--link-tokens-per-document", type=int, default=5)
    parser.add_argument("--micro-batch-size", type=int, default=1)
    parser.add_argument("--global-batch-size", type=int, default=64)
    parser.add_argument("--steps", type=int, default=6000)
    parser.add_argument("--warmup-steps", type=int, default=600)
    parser.add_argument("--learning-rate", type=float, default=5e-6)
    parser.add_argument("--weight-decay", type=float, default=0.1)
    parser.add_argument("--max-grad-norm", type=float, default=1.0)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--log-every", type=int, default=10)
    return parser.parse_args()


def main() -> None:
    """Run FSDP fine-tuning and export a Hugging Face checkpoint."""
    args = parse_args()
    dist.init_process_group("nccl")
    rank = dist.get_rank()
    world_size = dist.get_world_size()
    local_rank = int(os.environ["LOCAL_RANK"])
    torch.cuda.set_device(local_rank)
    torch.manual_seed(args.seed + rank)

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokens = register_kvlink_tokens(
        tokenizer,
        args.max_documents,
        args.link_tokens_per_document,
    )
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        dtype=torch.bfloat16,
        attn_implementation="eager",
        low_cpu_mem_usage=True,
    )
    model.resize_token_embeddings(len(tokenizer))
    model.config.use_cache = False
    model.gradient_checkpointing_enable()

    auto_wrap_policy = partial(
        transformer_auto_wrap_policy,
        transformer_layer_cls={Qwen2DecoderLayer},
    )
    mixed_precision = MixedPrecision(
        param_dtype=torch.bfloat16,
        reduce_dtype=torch.float32,
        buffer_dtype=torch.bfloat16,
    )
    model = FSDP(
        model,
        auto_wrap_policy=auto_wrap_policy,
        mixed_precision=mixed_precision,
        device_id=local_rank,
        use_orig_params=True,
    )
    dataset = KVLinkDataset(
        args.train_data,
        tokenizer,
        tokens,
        args.max_length,
        args.max_documents,
    )
    if not dataset:
        raise ValueError("training dataset has no valid examples")
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        seed=args.seed,
    )
    loader = DataLoader(
        dataset,
        batch_size=args.micro_batch_size,
        sampler=sampler,
        collate_fn=KVLinkCollator(
            tokenizer.pad_token_id,
            torch.bfloat16,
        ),
        num_workers=2,
        pin_memory=True,
    )
    accumulation_steps = args.global_batch_size // (args.micro_batch_size * world_size)
    if accumulation_steps <= 0 or (
        args.global_batch_size % (args.micro_batch_size * world_size) != 0
    ):
        raise ValueError(
            "global batch size must be divisible by micro batch size * world size"
        )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay,
    )
    data_iterator = iter(loader)
    optimizer.zero_grad(set_to_none=True)
    for step in range(args.steps):
        learning_rate = _learning_rate(
            step,
            args.steps,
            args.warmup_steps,
            args.learning_rate,
        )
        for group in optimizer.param_groups:
            group["lr"] = learning_rate

        accumulated_loss = torch.zeros((), device=local_rank)
        for micro_step in range(accumulation_steps):
            try:
                batch = next(data_iterator)
            except StopIteration:
                sampler.set_epoch(step)
                data_iterator = iter(loader)
                batch = next(data_iterator)
            batch = {
                key: value.to(local_rank, non_blocking=True)
                for key, value in batch.items()
            }
            should_sync = micro_step == accumulation_steps - 1
            context = model.no_sync() if not should_sync else torch.enable_grad()
            with context:
                output = model(**batch)
                loss = output.loss / accumulation_steps
                loss.backward()
                accumulated_loss += loss.detach()

        model.clip_grad_norm_(args.max_grad_norm)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        dist.all_reduce(accumulated_loss)
        if rank == 0 and (step + 1) % args.log_every == 0:
            print(
                f"step={step + 1} "
                f"loss={accumulated_loss.item() / world_size:.6f} "
                f"lr={learning_rate:.3e}",
                flush=True,
            )

    _save_model(model, tokenizer, args.output_dir, rank)
    dist.destroy_process_group()


def _learning_rate(
    step: int,
    total_steps: int,
    warmup_steps: int,
    peak: float,
) -> float:
    if step < warmup_steps:
        return peak * float(step + 1) / max(1, warmup_steps)
    progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
    return peak * 0.5 * (1.0 + math.cos(math.pi * progress))


def _save_model(
    model: FSDP,
    tokenizer: object,
    output_dir: str,
    rank: int,
) -> None:
    output_path = Path(output_dir)
    state_config = FullStateDictConfig(
        offload_to_cpu=True,
        rank0_only=True,
    )
    with FSDP.state_dict_type(
        model,
        StateDictType.FULL_STATE_DICT,
        state_config,
    ):
        state_dict = model.state_dict()
    if rank == 0:
        output_path.mkdir(parents=True, exist_ok=True)
        model.module.save_pretrained(
            output_path,
            state_dict=state_dict,
            safe_serialization=True,
        )
        tokenizer.save_pretrained(output_path)
    dist.barrier()


if __name__ == "__main__":
    main()
