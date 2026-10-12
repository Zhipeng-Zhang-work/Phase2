# SPDX-License-Identifier: Apache-2.0
# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Tuple
import argparse
import os

# Third Party
from openai import OpenAI
from transformers import AutoConfig, AutoTokenizer
import yaml
from utils import (
    PromptBuildMethodType,
    build_fewshot_prompt,
    build_qa_prompt,
    get_non_thinking_chat_template_parts,
    load_dataset,
    system_prompt_set,
    tokenize_segment_for_cache,
)


def _load_blend_add_special_in_precomp() -> bool:
    config_file = os.getenv("LMCACHE_CONFIG_FILE")
    if not config_file:
        return False
    try:
        with open(config_file) as f:
            config = yaml.safe_load(f) or {}
    except OSError:
        return False
    return bool(config.get("blend_add_special_in_precomp", False))


class KVPreCompute(ABC):
    def __init__(self):
        self._blend_add_special_in_precomp = _load_blend_add_special_in_precomp()

    @abstractmethod
    def precompute_kv(self, text_chunk, force_special_tokens: bool = False):
        pass


class OnlineKVPreCompute(KVPreCompute):
    def __init__(self, openai_api_key, openai_api_base, tokenizer=None):
        self.client = OpenAI(
            api_key=openai_api_key,
            base_url=openai_api_base,
        )
        self.model = self.client.models.list().data[0].id
        super().__init__()
        # NOTE: Make sure to configure this tokenizer exactly the same with the one
        # in openai api server.
        self._tokenizer = tokenizer

    def _gen_inputs_for_precompute(
        self,
        text_chunk: str,
        force_special_tokens: bool,
    ):
        add_special_tokens = self._blend_add_special_in_precomp or force_special_tokens
        return tokenize_segment_for_cache(
            self._tokenizer,
            text_chunk,
            add_special_tokens=add_special_tokens,
        )

    def precompute_kv(self, text_chunk: str, force_special_tokens: bool = False):
        inputs_precomp = self._gen_inputs_for_precompute(
            text_chunk,
            force_special_tokens,
        )
        self.precompute_kv_by_ids(inputs_precomp)

    def precompute_kv_by_ids(self, input_ids: list[int]) -> None:
        """Precompute and persist the complete prompt KV cache.

        Args:
            input_ids: Token IDs for one independently reusable cache segment.

        Returns:
            None.

        Raises:
            Exception: Propagates request errors reported by the OpenAI client.

        Notes:
            The request marker makes LMCache defer segment storage until vLLM
            finishes the final chunked-prefill iteration. This prevents a long
            segment from being persisted under a partial-prefix cache key.
        """
        self.client.completions.create(
            prompt=input_ids,
            model=self.model,
            max_tokens=1,
            extra_body={
                "kv_transfer_params": {
                    "lmcache.store_full_prompt_on_last_prefill": True,
                }
            },
        )


@dataclass
class PrecomputeConfig:
    # Model name.
    model: str
    # Tokenizer name.
    tokenizer: str
    # Model config path.
    model_config: str
    # Dataset.
    dataset: str
    # Start index.
    start_idx: int
    # End index.
    end_idx: int
    # KV storage size.
    kv_storage_size: int
    # KV chunk size.
    kv_chunk_size: int
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # API key
    api_key: str
    # Base url
    base_url: str
    # KV cache precision.
    kv_precision: int
    # Max model length for precompute. If <= 0, no limit.
    max_model_len: int


class KVSizeCalculator:
    def __init__(
        self,
        num_key_value_heads: int,
        head_dim: int,
        num_layers: int,
        precision: int,
    ):
        self.ratio = num_key_value_heads * head_dim * num_layers * precision * 2

    def get_kv_size(self, token_cnt: int) -> int:
        return token_cnt * self.ratio


def precompute_all_kv(config: PrecomputeConfig) -> Tuple[int, int, str]:
    tokenizer = AutoTokenizer.from_pretrained(config.tokenizer)
    model_config = AutoConfig.from_pretrained(config.model_config)
    num_kv_heads = getattr(
        model_config, "num_key_value_heads", model_config.num_attention_heads
    )
    head_dim = getattr(model_config, "head_dim", None)
    if head_dim is None:
        # Fallback for configs like Qwen2: head_dim is not explicit.
        head_dim = model_config.hidden_size // model_config.num_attention_heads
    kv_size_calculator = KVSizeCalculator(
        num_kv_heads,
        head_dim,
        model_config.num_hidden_layers,
        config.kv_precision,
    )
    eval_dataset = load_dataset(config.dataset)
    start_idx = config.start_idx
    end_idx = config.end_idx
    if end_idx >= 0:
        assert end_idx <= len(eval_dataset), (
            f"end_index {end_idx} > length of dataset {len(eval_dataset)}"
        )
    assert start_idx >= 0, f"start_idx {start_idx} < 0"
    assert start_idx < len(eval_dataset), (
        f"start_idx {start_idx} >= length of dataset {len(eval_dataset)}"
    )
    precompute_kv = OnlineKVPreCompute(config.api_key, config.base_url, tokenizer)
    with_bos = precompute_kv._blend_add_special_in_precomp
    current_size_taken = 0
    size_upper_bound = config.kv_storage_size
    assert size_upper_bound > 0, f"size_upper_bound {size_upper_bound} <= 0"
    current_idx = start_idx
    round_up_token_cnt = config.kv_chunk_size
    assert round_up_token_cnt >= 1
    system_prompt = system_prompt_set[config.prompt_build_method]
    chat_prefix, _ = get_non_thinking_chat_template_parts(tokenizer)
    system_prompt_ids = tokenize_segment_for_cache(
        tokenizer,
        chat_prefix + system_prompt,
        add_special_tokens=False,
    )
    system_prompt_size = kv_size_calculator.get_kv_size(len(system_prompt_ids))
    if current_size_taken + system_prompt_size <= size_upper_bound:
        # Reuse the exact IDs used at the beginning of the online RAG prompt.
        precompute_kv.precompute_kv_by_ids(system_prompt_ids)
        current_size_taken += system_prompt_size
        print(
            "Precompute system prompt "
            f"token_cnt={len(system_prompt_ids)} size={system_prompt_size} bytes",
            flush=True,
        )
    while True:
        if end_idx >= 0:
            if current_idx >= end_idx:
                break
        else:
            if current_size_taken >= size_upper_bound or current_idx >= len(
                eval_dataset
            ):
                break
        example = eval_dataset[current_idx]
        this_case_size = 0
        doc_prompts = None
        if config.prompt_build_method in {
            PromptBuildMethodType.QA,
            PromptBuildMethodType.DUREADER,
            PromptBuildMethodType.QA_ZH,
        }:
            doc_prompts, _ = build_qa_prompt(example, "")
        elif config.prompt_build_method == PromptBuildMethodType.FEW_SHOT:
            doc_prompts, _ = build_fewshot_prompt(example)
        assert doc_prompts is not None
        token_cnt = 0
        processed_prompts = []
        skip_example = False
        for doc_prompt in doc_prompts:
            assert len(doc_prompt) > 0
            input_comps = tokenize_segment_for_cache(
                tokenizer,
                doc_prompt,
                add_special_tokens=with_bos,
            )
            assert len(input_comps) > 0
            if config.max_model_len > 0 and len(input_comps) > config.max_model_len:
                print(
                    "Skip example due to prompt length "
                    f"{len(input_comps)} > max_model_len {config.max_model_len}"
                )
                skip_example = True
                break
            temp_cnt = len(input_comps)
            # Add doc token count before round up.
            temp_cnt = (
                (temp_cnt + round_up_token_cnt - 1) // round_up_token_cnt
            ) * round_up_token_cnt
            token_cnt += temp_cnt
            processed_prompts.append(doc_prompt)
        if skip_example:
            current_idx += 1
            continue
        assert token_cnt > 0, f"token_cnt {token_cnt} <= 0"
        this_case_size = kv_size_calculator.get_kv_size(token_cnt)
        print(
            f"Precompute idx={current_idx} token_cnt={token_cnt} "
            f"size={this_case_size} bytes",
            flush=True,
        )
        if current_size_taken + this_case_size > size_upper_bound:
            break
        for prompt in processed_prompts:
            precompute_kv.precompute_kv(prompt)
        current_idx += 1
        current_size_taken += this_case_size

    return start_idx, current_idx, precompute_kv.model


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG precompute configurations.")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name")
    parser.add_argument(
        "--model-config", type=str, default="", help="Model config path"
    )
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
    )
    parser.add_argument(
        "--kv-storage-size", type=str, default="", help="KV storage size"
    )
    parser.add_argument(
        "--kv-chunk-size", type=int, default=256, help="KV storage chunk size"
    )
    parser.add_argument(
        "--kv-precision-bit",
        type=int,
        default=16,
        help="KV cache precision bit",
    )
    parser.add_argument(
        "--max-model-len",
        type=int,
        default=-1,
        help="Max model length used for precompute. If <=0, no limit.",
    )
    parser.add_argument(
        "--base-url",
        type=str,
        required=True,
        help="Base URL of the serving engine endpoint",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default="EMPTY",
        help="API key of the serving engine endpoint",
    )
    args = parser.parse_args()
    return args


def parse_size(size: str) -> int:
    if len(size) == 0:
        return -1
    else:
        size = size.upper()
        if size.endswith("KB"):
            return int(size[:-2]) * 1024
        elif size.endswith("MB"):
            return int(size[:-2]) * 1024 * 1024
        elif size.endswith("GB"):
            return int(size[:-2]) * 1024 * 1024 * 1024
        elif size.endswith("TB"):
            return int(size[:-2]) * 1024 * 1024 * 1024 * 1024
        elif size.endswith("B"):
            return int(size[:-1])
        else:
            raise ValueError(f"Invalid size unit {size}")


def parse_prompt_build_method(
    prompt_build_method: str,
) -> PromptBuildMethodType:
    prompt_build_method = prompt_build_method.upper()
    if prompt_build_method == "QA":
        return PromptBuildMethodType.QA
    elif prompt_build_method == "DUREADER":
        return PromptBuildMethodType.DUREADER
    elif prompt_build_method == "QA_ZH":
        return PromptBuildMethodType.QA_ZH
    elif prompt_build_method == "FEW_SHOT":
        return PromptBuildMethodType.FEW_SHOT
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")


def run_precompute(args):
    kv_storage_size = parse_size(args.kv_storage_size)
    kv_chunk_size = args.kv_chunk_size
    prompt_build_method = parse_prompt_build_method(args.prompt_build_method)
    kv_precision_bit = args.kv_precision_bit
    assert kv_precision_bit % 8 == 0, (
        f"kv_precision_bit {kv_precision_bit} is not a multiple of 8"
    )
    kv_precision = kv_precision_bit // 8
    config = PrecomputeConfig(
        model=args.model,
        tokenizer=args.tokenizer,
        model_config=args.model_config,
        dataset=args.dataset,
        start_idx=args.start_index,
        end_idx=args.end_index,
        kv_storage_size=kv_storage_size,
        kv_chunk_size=kv_chunk_size,
        prompt_build_method=prompt_build_method,
        api_key=args.api_key,
        base_url=args.base_url,
        kv_precision=kv_precision,
        max_model_len=args.max_model_len,
    )
    start_idx, end_idx, model_name = precompute_all_kv(config)
    return start_idx, end_idx, model_name


def main():
    args = parse_arguments()
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model
    if len(args.model_config) == 0:
        args.model_config = args.model
    start_idx, end_idx, model_name = run_precompute(args)
    print(f"Precompute from {start_idx} to {end_idx} for model {model_name}")


if __name__ == "__main__":
    main()
