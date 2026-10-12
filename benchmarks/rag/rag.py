# SPDX-License-Identifier: Apache-2.0
# Standard
from dataclasses import dataclass
from typing import Any, Optional
import argparse
import asyncio
import logging
import random
import time

# Third Party
from transformers import AutoTokenizer
from utils import (
    AsyncLoopWrapper,
    PromptBuildMethodType,
    build_rag_prompt_token_ids_with_spans,
    compute_f1,
    compute_rl,
    get_non_thinking_chat_template_parts,
    init_logger,
    load_dataset,
    process_model_output,
    query_prompt_set,
    system_prompt_set,
    tokenize_segment_for_cache,
)
import openai
import pandas as pd
import requests

logger = init_logger(__name__, logging.INFO)


@dataclass
class WorkloadConfig:
    # Overall QPS
    qps: float
    # Model name
    model: str
    # Tokenizer name
    tokenizer: str
    # Dataset.
    dataset: str
    # Start index of the workload
    start_index: int
    # End index of the workload
    end_index: int
    # Random shuffle.
    shuffle: bool
    # System prompt.
    system_prompt: str
    # Separator.
    separator: str
    # Query prompt.
    query_prompt: str
    # Prompt build method.
    prompt_build_method: PromptBuildMethodType
    # Max tokens for each generation.
    max_tokens: int
    # Per-request timeout in seconds.
    request_timeout: float


@dataclass
class Response:
    request_id: int
    body: str
    ttft: float
    generation_time: float
    prompt_tokens: int
    generation_tokens: int
    launch_time: float
    finish_time: float


def parse_arguments():
    parser = argparse.ArgumentParser(description="Parse RAG benchmark configurations.")
    parser.add_argument("--qps", type=float, required=True, help="Overall QPS")
    parser.add_argument("--model", type=str, required=True, help="Model name")
    parser.add_argument("--tokenizer", type=str, default="", help="Tokenizer name")
    parser.add_argument("--dataset", type=str, required=True, help="The dataset path")
    parser.add_argument(
        "--start-index", type=int, default=0, help="Start index of the workload"
    )
    parser.add_argument(
        "--end-index", type=int, default=-1, help="End index of the workload"
    )
    parser.add_argument("--shuffle", action="store_true", help="Random shuffle")
    parser.add_argument("--system-prompt", type=str, default="", help="System prompt")
    parser.add_argument("--separator", type=str, default="", help="Separator")
    parser.add_argument("--query-prompt", type=str, default="", help="Query prompt")
    parser.add_argument(
        "--prompt-build-method",
        type=str,
        required=True,
        help="Prompt build method",
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
    parser.add_argument(
        "--output",
        type=str,
        default="summary.csv",
        help="The output file name (ended with csv or txt) for the summary csv and txt",
    )
    parser.add_argument(
        "--debug-output",
        type=str,
        default="",
        help="Optional per-request debug output CSV path",
    )
    parser.add_argument(
        "--warmup", action="store_true", help="Whether to enable warmup"
    )
    parser.add_argument(
        "--time",
        type=int,
        default=None,
        help="The total running time in seconds",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Whether to enable verbose logging",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=32,
        help="Max tokens for each generation",
    )
    parser.add_argument(
        "--request-timeout",
        type=float,
        default=1800.0,
        help="Per-request timeout in seconds",
    )
    parser.add_argument(
        "--step-interval", type=float, default=0.02, help="Step interval"
    )
    args = parser.parse_args()
    return args


class RequestExecutor:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        prompt_build_method: PromptBuildMethodType,
        model: str,
        timeout_seconds: float = 1800.0,
    ):
        self.client = openai.AsyncOpenAI(
            api_key=api_key, base_url=base_url, timeout=timeout_seconds
        )
        self.model = model
        self.loop = AsyncLoopWrapper.GetOrStartLoop()
        self.prompt_build_method = prompt_build_method

    def _extract_chunk_text(self, choice: Any) -> Optional[str]:
        if hasattr(choice, "delta"):
            return getattr(choice.delta, "content", None)
        return getattr(choice, "text", None)

    async def _async_launch_request(
        self,
        request_id: int,
        prompt: str | list[int],
        max_tokens: int,
        blend_prompt_spans: Optional[dict[str, list[int]]] = None,
    ) -> Response:
        start_time = time.time()
        first_token_time = None
        words = ""
        tokens_out = 0
        tokens_prefill = 0
        request_extras: dict[str, Any] = {}
        if blend_prompt_spans is not None:
            request_extras["extra_body"] = {
                "kv_transfer_params": {
                    "lmcache.blend_prompt_spans": blend_prompt_spans,
                }
            }
        response = await self.client.completions.create(
            prompt=prompt,
            model=self.model,
            temperature=0,
            stream=True,
            max_tokens=max_tokens,
            stream_options={"include_usage": True},
            **request_extras,
        )
        async for tok in response:
            if tok.usage is not None:
                if tok.usage.completion_tokens is not None:
                    tokens_out = tok.usage.completion_tokens
                if tok.usage.prompt_tokens is not None:
                    tokens_prefill = tok.usage.prompt_tokens
            if not tok.choices:
                continue
            chunk_message = self._extract_chunk_text(tok.choices[0])
            if chunk_message:
                if first_token_time is None:
                    first_token_time = time.time()
                words += chunk_message
        finish_time = time.time()
        if first_token_time is None:
            first_token_time = finish_time
        if tokens_out <= 0:
            tokens_out = 1
        return Response(
            request_id=request_id,
            body=words,
            ttft=first_token_time - start_time,
            generation_time=finish_time - first_token_time,
            prompt_tokens=tokens_prefill,
            generation_tokens=tokens_out,
            launch_time=start_time,
            finish_time=finish_time,
        )

    def launch_request(
        self,
        request_id: int,
        prompt,
        max_tokens,
        finish_callback,
        ignore_errors: bool = False,
        blend_prompt_spans: Optional[dict[str, list[int]]] = None,
    ):
        """
        finish_callback: Callable[[Response], None]
        """

        def real_callback(fut):
            try:
                result = fut.result()
            except Exception as exc:  # noqa: BLE001
                if ignore_errors:
                    logger.warning("Warmup request failed: %s", exc)
                    return
                logger.error(
                    "Request failed: %s (%s)",
                    repr(exc),
                    type(exc).__name__,
                )
                result = Response(
                    request_id=request_id,
                    body="",
                    ttft=0.0,
                    generation_time=0.0,
                    prompt_tokens=0,
                    generation_tokens=1,
                    launch_time=time.time(),
                    finish_time=time.time(),
                )
            finish_callback(result)

        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(
                request_id,
                prompt,
                max_tokens,
                blend_prompt_spans,
            ),
            self.loop,
        )
        future.add_done_callback(real_callback)

    def run_request_sync(self, request_id: int, prompt, max_tokens) -> Response:
        future = asyncio.run_coroutine_threadsafe(
            self._async_launch_request(request_id, prompt, max_tokens),
            self.loop,
        )
        return future.result()


def wait_for_server_ready(
    base_url: str,
    api_key: str,
    timeout_seconds: float = 300.0,
    interval_seconds: float = 2.0,
) -> None:
    base_url = base_url.rstrip("/")
    if base_url.endswith("/v1"):
        root_url = base_url[:-3]
        models_url = f"{base_url}/models"
    else:
        root_url = base_url
        models_url = f"{base_url}/v1/models"
    health_url = f"{root_url}/health"
    headers = {"Authorization": f"Bearer {api_key}"}
    start = time.time()
    last_err = None
    while time.time() - start < timeout_seconds:
        try:
            resp = requests.get(health_url, timeout=10)
            if resp.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        try:
            resp = requests.get(models_url, headers=headers, timeout=10)
            if resp.status_code == 200:
                return
        except Exception as exc:  # noqa: BLE001
            last_err = exc
        time.sleep(interval_seconds)
    raise RuntimeError(f"Server not ready. Last error: {last_err}")


def warmup_engine(executor: RequestExecutor, tokenizer_name: str) -> None:
    """Warm up the engine with Qwen3 non-thinking chat-formatted prompts."""
    logger.info("Warming up the engine")
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
    chat_prefix, chat_suffix = get_non_thinking_chat_template_parts(tokenizer)
    for i in range(10):
        content = f"WARMUP: Hi, I'm user {i}. Here are some text: {'hi ' * 100}."
        prompt = tokenize_segment_for_cache(
            tokenizer,
            chat_prefix + content + chat_suffix,
            add_special_tokens=False,
        )
        try:
            executor.run_request_sync(-1, prompt, 100)
        except Exception as exc:  # noqa: BLE001
            logger.warning("Warmup request failed: %s", exc)
    logger.info("Warm up finished.")


class RAGManager:
    def __init__(self, workload_config: WorkloadConfig):
        self.workload_config = workload_config
        eval_dataset = load_dataset(workload_config.dataset)
        start_index = workload_config.start_index
        end_index = workload_config.end_index
        if end_index < 0:
            end_index = len(eval_dataset)
        eval_dataset = eval_dataset[start_index:end_index]
        if workload_config.shuffle:
            random.shuffle(eval_dataset)
        self._prompts = []
        self._prompt_token_ids = []
        self._blend_prompt_spans: list[dict[str, list[int]]] = []
        self._answers = []
        self._build_method = workload_config.prompt_build_method
        self._generated_text: list[str | None] = []
        self._evaluated_text: list[str | None] = []
        self._reasoning_detected: list[bool] = []
        self._incomplete_reasoning: list[bool] = []
        self._generation_time: list[float | None] = []
        self._prefill_tok_cnt: list[int | None] = []
        self._generation_tok_cnt: list[int | None] = []
        self._ttft: list[float | None] = []
        self._tpot: list[float | None] = []
        self._tokenizer = AutoTokenizer.from_pretrained(workload_config.tokenizer)
        for ex in eval_dataset:
            (
                prompt_token_ids,
                _,
                blend_prompt_spans,
            ) = build_rag_prompt_token_ids_with_spans(
                self._tokenizer,
                workload_config.system_prompt,
                ex,
                workload_config.query_prompt,
                workload_config.separator,
                workload_config.prompt_build_method,
            )
            prompt = self._tokenizer.decode(
                prompt_token_ids,
                skip_special_tokens=False,
            )
            self._prompts.append(prompt)
            self._prompt_token_ids.append(prompt_token_ids)
            self._blend_prompt_spans.append(blend_prompt_spans)
            self._answers.append(ex["answers"])
            self._generated_text.append(None)
            self._evaluated_text.append(None)
            self._reasoning_detected.append(False)
            self._incomplete_reasoning.append(False)
            self._generation_time.append(None)
            self._prefill_tok_cnt.append(None)
            self._generation_tok_cnt.append(None)
            self._ttft.append(None)
            self._tpot.append(None)
        self._last_request_time = -1.0
        self._last_request_index = 0
        assert workload_config.qps > 0
        self._gap = 1.0 / workload_config.qps
        self._max_tokens = workload_config.max_tokens

    def _update_result(self, response: Response):
        processed = process_model_output(response.body)
        self._generated_text[response.request_id] = response.body
        self._evaluated_text[response.request_id] = processed.final_answer
        self._reasoning_detected[response.request_id] = processed.reasoning_detected
        self._incomplete_reasoning[response.request_id] = processed.incomplete_reasoning
        self._ttft[response.request_id] = response.ttft
        self._tpot[response.request_id] = (
            response.generation_time / response.generation_tokens
        )
        self._generation_time[response.request_id] = response.generation_time
        self._prefill_tok_cnt[response.request_id] = response.prompt_tokens
        self._generation_tok_cnt[response.request_id] = response.generation_tokens

    def step(self, timestamp: float, executor: RequestExecutor) -> bool:
        if self._last_request_index >= len(self._prompts):
            return False
        if (
            self._last_request_time < 0
            or timestamp >= self._last_request_time + self._gap
        ):
            prompt = self._prompt_token_ids[self._last_request_index]
            request_id = self._last_request_index
            self._last_request_time = timestamp
            self._last_request_index += 1
            executor.launch_request(
                request_id,
                prompt,
                self._max_tokens,
                self._update_result,
                blend_prompt_spans=self._blend_prompt_spans[request_id],
            )
        return True

    def summary(self, start_time: float, end_time: float) -> pd.DataFrame:
        cnt = len(self._ttft)
        assert cnt > 0
        avg_ttft = sum(ttft for ttft in self._ttft if ttft is not None) / cnt
        avg_tpot = sum(tpot for tpot in self._tpot if tpot is not None) / cnt
        # Create a dataframe
        quality = []
        for i in range(cnt):
            if self._build_method in {
                PromptBuildMethodType.QA,
                PromptBuildMethodType.QA_ZH,
            }:
                quality.append(
                    max(
                        [
                            compute_f1(
                                self._evaluated_text[i],
                                (
                                    answer[0]
                                    if isinstance(answer, list) and len(answer) == 1
                                    else answer
                                ),
                                self._tokenizer,
                            )
                            for answer in self._answers[i]
                        ]
                    )
                )
            elif self._build_method in {
                PromptBuildMethodType.FEW_SHOT,
                PromptBuildMethodType.DUREADER,
            }:
                quality.append(
                    max(
                        [
                            compute_rl(self._evaluated_text[i], answer)
                            for answer in self._answers[i]
                        ]
                    )
                )
            else:
                raise ValueError(f"Invalid prompt build method {self._build_method}")
        avg_quality = sum(quality) / cnt
        df = pd.DataFrame(
            {
                "quality": quality,
                "ttft": self._ttft,
                "tpot": self._tpot,
                "generation_time": self._generation_time,
                "prefill_token_cnt": self._prefill_tok_cnt,
                "generation_token_cnt": self._generation_tok_cnt,
            }
        )
        total_time = end_time - start_time
        thput = cnt / total_time
        logger.info(
            f"Summary: {cnt} requests, average_ttft={avg_ttft} (second)\n"
            f" average_tpot={avg_tpot} (second)\n"
            f"throughput={thput} (req/s)\n"
            f"average_quality={avg_quality}\n"
        )
        return df

    def debug_dump(self) -> pd.DataFrame:
        """
        Dump per-request debug data for offline inspection.

        Returns:
            A dataframe with prompts, raw generations, answers, and timings.
        """
        return pd.DataFrame(
            {
                "request_id": list(range(len(self._prompts))),
                "prompt": self._prompts,
                "prompt_token_ids": self._prompt_token_ids,
                "generated_text": self._generated_text,
                "evaluated_text": self._evaluated_text,
                "reasoning_detected": self._reasoning_detected,
                "incomplete_reasoning": self._incomplete_reasoning,
                "answers": self._answers,
                "ttft": self._ttft,
                "tpot": self._tpot,
                "generation_time": self._generation_time,
                "prefill_token_cnt": self._prefill_tok_cnt,
                "generation_token_cnt": self._generation_tok_cnt,
            }
        )


def run_rag(args):
    build_prompt_method_str = args.prompt_build_method.upper()
    build_prompt_method = None
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    elif build_prompt_method_str == "DUREADER":
        build_prompt_method = PromptBuildMethodType.DUREADER
    elif build_prompt_method_str == "QA_ZH":
        build_prompt_method = PromptBuildMethodType.QA_ZH
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")
    workload_config = WorkloadConfig(
        qps=args.qps,
        model=args.model,
        tokenizer=args.tokenizer,
        dataset=args.dataset,
        start_index=args.start_index,
        end_index=args.end_index,
        shuffle=args.shuffle,
        system_prompt=args.system_prompt,
        separator=args.separator,
        query_prompt=args.query_prompt,
        prompt_build_method=build_prompt_method,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
    )
    executor = RequestExecutor(
        base_url=args.base_url,
        api_key=args.api_key,
        prompt_build_method=build_prompt_method,
        model=args.model,
        timeout_seconds=args.request_timeout,
    )
    if args.warmup:
        wait_for_server_ready(args.base_url, args.api_key)
        warmup_engine(executor, args.tokenizer)
    manager = RAGManager(workload_config)
    step_interval = args.step_interval
    num_steps = 0
    start_time = time.time()
    try:
        while True:
            num_steps += 1
            effective = manager.step(time.time(), executor)
            if not effective:
                break
            time.sleep(step_interval)
            if args.time is not None and time.time() - start_time > args.time:
                break

    except KeyboardInterrupt:
        logger.info("Interrupted, waiting for the final result")

    AsyncLoopWrapper.StopLoop()

    logger.info(f"Finished benchmarking, dumping summary to {args.output}")
    summary = manager.summary(start_time, time.time())
    summary.to_csv(args.output, index=False)
    debug_output = args.debug_output or f"{args.output}.debug.csv"
    manager.debug_dump().to_csv(debug_output, index=False)


def main():
    args = parse_arguments()
    build_prompt_method_str = args.prompt_build_method.upper()
    build_prompt_method = None
    if build_prompt_method_str == "QA":
        build_prompt_method = PromptBuildMethodType.QA
    elif build_prompt_method_str == "FEW_SHOT":
        build_prompt_method = PromptBuildMethodType.FEW_SHOT
    elif build_prompt_method_str == "DUREADER":
        build_prompt_method = PromptBuildMethodType.DUREADER
    elif build_prompt_method_str == "QA_ZH":
        build_prompt_method = PromptBuildMethodType.QA_ZH
    else:
        raise ValueError(f"Invalid prompt build method {build_prompt_method_str}")
    if len(args.system_prompt) == 0:
        args.system_prompt = system_prompt_set[build_prompt_method]
    if len(args.query_prompt) == 0:
        args.query_prompt = query_prompt_set[build_prompt_method]
    if len(args.tokenizer) == 0:
        args.tokenizer = args.model
    args.system_prompt = args.system_prompt.encode().decode("unicode_escape")
    args.query_prompt = args.query_prompt.encode().decode("unicode_escape")
    if args.verbose:
        global logger
        logger = init_logger(__name__, logging.DEBUG)
    run_rag(args)


if __name__ == "__main__":
    main()
