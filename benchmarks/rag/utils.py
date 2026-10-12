# SPDX-License-Identifier: Apache-2.0
# Standard
from enum import Enum, auto
from logging import Logger
from typing import Any
import asyncio
import collections
import json
import logging
import re
import string
import threading

# Third Party
from output_processing import process_model_output
from rouge_score import rouge_scorer


def build_format(color):
    reset = "\x1b[0m"
    underline = "\x1b[3m"
    return (
        f"{color}[%(asctime)s] %(levelname)s:{reset} %(message)s "
        + f"{underline}(%(filename)s:%(lineno)d:%(name)s){reset}"
    )


class CustomFormatter(logging.Formatter):
    grey = "\x1b[1m"
    green = "\x1b[32;20m"
    yellow = "\x1b[33;20m"
    red = "\x1b[31;20m"
    bold_red = "\x1b[31;1m"
    reset = "\x1b[0m"

    FORMATS = {
        logging.DEBUG: build_format(grey),
        logging.INFO: build_format(green),
        logging.WARNING: build_format(yellow),
        logging.ERROR: build_format(red),
        logging.CRITICAL: build_format(bold_red),
    }

    def format(self, record):
        log_fmt = self.FORMATS.get(record.levelno)
        formatter = logging.Formatter(log_fmt)
        return formatter.format(record)


def init_logger(name: str, log_level=logging.DEBUG) -> Logger:
    logger = logging.getLogger(name)

    logger.handlers.clear()
    logger.propagate = False

    ch = logging.StreamHandler()
    ch.setLevel(log_level)
    ch.setFormatter(CustomFormatter())
    logger.addHandler(ch)
    logger.setLevel(logging.DEBUG)

    return logger


class AsyncLoopWrapper:
    _loop: asyncio.AbstractEventLoop | None = None
    _thread: threading.Thread | None = None
    _logger = init_logger("AsyncLoopWrapper")

    @classmethod
    def WaitLoop(cls):
        assert cls._loop is not None, "Loop is not started"

        async def wait_for_tasks():
            current_task = asyncio.current_task(cls._loop)
            tasks = [
                task
                for task in asyncio.all_tasks(cls._loop)
                if not task.done() and task is not current_task
            ]
            cls._logger.info(f"Waiting for {len(tasks)} tasks to finish")
            if tasks:
                results = await asyncio.gather(*tasks, return_exceptions=True)
                err_cnt = sum(1 for r in results if isinstance(r, Exception))
                if err_cnt > 0:
                    cls._logger.error(f"{err_cnt} task(s) finished with exceptions")

        # Schedule the wait_for_tasks coroutine to be executed in the loop
        future = asyncio.run_coroutine_threadsafe(wait_for_tasks(), cls._loop)
        try:
            # Wait for wait_for_tasks to complete
            future.result()
        except Exception as e:
            cls._logger.error(f"Error while waiting for tasks: {e}")

    @classmethod
    def StartLoop(cls):
        if cls._loop is not None:
            cls._logger.warning("Loop is already started")
            return

        if cls._loop is None:
            cls._loop = asyncio.new_event_loop()

        def run_loop():
            asyncio.set_event_loop(cls._loop)
            cls._logger.debug("Starting the asyncio loop")
            cls._loop.run_forever()

        cls._thread = threading.Thread(target=run_loop)
        cls._thread.start()

    @classmethod
    def StopLoop(cls):
        assert cls._loop is not None, "Loop is not started"
        assert cls._thread is not None, "Thread is not started"

        def stop_loop():
            cls._logger.debug("Stopping the loop!")
            cls._loop.stop()

        cls._logger.info("Waiting for remaining tasks to finish")
        cls.WaitLoop()

        cls._loop.call_soon_threadsafe(stop_loop)
        cls._thread.join()

    @classmethod
    def GetLoop(cls) -> asyncio.AbstractEventLoop:
        assert cls._loop is not None, "Loop is not started"
        return cls._loop

    @classmethod
    def GetOrStartLoop(cls) -> asyncio.AbstractEventLoop:
        if cls._loop is None:
            cls.StartLoop()
        assert cls._loop is not None, "Loop is not started"
        return cls._loop


class PromptBuildMethodType(Enum):
    QA = auto()
    FEW_SHOT = auto()
    DUREADER = auto()
    QA_ZH = auto()


system_prompt_set = {
    PromptBuildMethodType.QA: "You will be asked a question after reading several passages. "  # noqa: E501
    "Please directly answer the question based on the given passages. "
    "Do NOT repeat the question. "
    "The answer should be within 5 words..\nPassages:\n",
    PromptBuildMethodType.FEW_SHOT: "Summarize the dialogue into 1 to 3 short sentences. "  # noqa: E501
    "Output only the final summary. Do not show analysis or reasoning, do not repeat the dialogue, and do not add a title, preamble, or Markdown. "
    "The following are some examples.\n\n",
    PromptBuildMethodType.DUREADER: "请根据下面的多个文档直接回答问题。只输出最终答案，不要展示分析或推理过程，不要重复问题，不要输出标题或 Markdown。答案应简洁但完整，只保留与问题直接相关的信息。\n文档：\n",
    PromptBuildMethodType.QA_ZH: "请根据下面的文档直接回答问题。只输出最终答案，不要展示分析或推理过程，不要重复问题，不要输出标题或 Markdown。请使用中文回答，答案应简洁并完整覆盖问题所要求的信息。\n文档：\n",
}


query_prompt_set = {
    PromptBuildMethodType.QA: "\n\nAnswer the question directly based on the given passages."  # noqa: E501
    " Do NOT repeat the question. "
    "The answer should be within 5 words. \nQuestion:",
    PromptBuildMethodType.FEW_SHOT: "\n\nNow summarize the final dialogue below. Output only the summary:\n",
    PromptBuildMethodType.DUREADER: "\n\n请直接回答问题，不要解释。\n问题：",
    PromptBuildMethodType.QA_ZH: "\n\n请直接回答问题，不要解释。\n问题：",
}


def load_dataset(dataset_path):
    print("Loading dataset:", dataset_path)
    with open(dataset_path) as f:
        return json.load(f)


def normalize_question(question):
    if not question.endswith("?"):
        question = question + "?"

    return question[0].lower() + question[1:]


def parse_generation(s):
    s = process_model_output(s).final_answer
    s = s.lstrip("\n").split("\n")[0]
    if s.startswith("Yes") or s.startswith("yes"):
        s = "Yes"
    elif s.split() and (
        (s.split()[0]).startswith("No") or (s.split()[0]).startswith("no")
    ):
        s = "No"
    return s


def normalize_answer(s):
    def remove_articles(text):
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text):
        return " ".join(text.split())

    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text):
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def build_qa_prompt(example, query_prompt):
    q = normalize_question(example["question"])
    doc_prompts = [f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]]
    q_prompt = f"{query_prompt}{q}\nAnswer:"
    return doc_prompts, q_prompt


def build_dureader_prompt(example, query_prompt):
    """Build a Chinese multi-document QA prompt for DuReader."""
    doc_prompts = [
        f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]
    ]
    q_prompt = f"{query_prompt}{example['question']}\n答案："
    return doc_prompts, q_prompt


def build_fewshot_prompt(example, query_prompt=""):
    q = f"{query_prompt}\n\n{example['question']}"
    doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
    q_prompt = f"{q}"
    return doc_prompts, q_prompt


def compute_f1(a_pred: str | None, a_gold: str, tokenizer: Any) -> float:
    """Compute token-level F1 without assuming that the tokenizer adds a BOS.

    Args:
        a_pred: Generated answer, or ``None`` for an empty answer.
        a_gold: Reference answer.
        tokenizer: Hugging Face-compatible tokenizer.

    Returns:
        Token-level F1 between the normalized prediction and reference.
    """
    a_pred = parse_generation(a_pred)
    gold_toks = tokenizer.encode(normalize_answer(a_gold), add_special_tokens=False)
    pred_toks = tokenizer.encode(normalize_answer(a_pred), add_special_tokens=False)
    common = collections.Counter(gold_toks) & collections.Counter(pred_toks)
    num_same = sum(common.values())
    if len(gold_toks) == 0 or len(pred_toks) == 0:
        # If either is no-answer, then F1 is 1 if they agree, 0 otherwise
        return int(gold_toks == pred_toks)
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(pred_toks)
    recall = 1.0 * num_same / len(gold_toks)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_rl(pred, gold):
    pred = process_model_output(pred).final_answer
    scorer = rouge_scorer.RougeScorer(["rougeL"], use_stemmer=True)
    rougeL = scorer.score(gold, pred)["rougeL"].fmeasure
    return rougeL


def _strip_bos_token(tokenizer, token_ids: list[int]) -> list[int]:
    bos_token_id = tokenizer.bos_token_id
    if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        return token_ids[1:]
    return token_ids


def tokenize_segment_for_cache(
    tokenizer: Any,
    text: str,
    add_special_tokens: bool = False,
) -> list[int]:
    token_ids = tokenizer(text, add_special_tokens=add_special_tokens).input_ids
    if add_special_tokens:
        return token_ids
    return _strip_bos_token(tokenizer, token_ids)


_CHAT_CONTENT_MARKER = "__LMCACHE_RAG_USER_CONTENT_7F3A9C__"


def get_non_thinking_chat_template_parts(tokenizer: Any) -> tuple[str, str]:
    """Return the text before and after user content in a non-thinking prompt.

    The tokenizer's own chat template is the source of truth. Splitting the
    rendered template lets CacheBlend continue tokenizing documents as
    independent cacheable segments while adding Qwen3's required chat framing
    and ``enable_thinking=False`` assistant prefix.

    Args:
        tokenizer: Hugging Face-compatible tokenizer with a chat template.

    Returns:
        A pair containing the prompt text before and after the user content.

    Raises:
        ValueError: If the rendered template does not contain the marker exactly
            once.
    """
    rendered = tokenizer.apply_chat_template(
        [{"role": "user", "content": _CHAT_CONTENT_MARKER}],
        tokenize=False,
        add_generation_prompt=True,
        enable_thinking=False,
    )
    if not isinstance(rendered, str) or rendered.count(_CHAT_CONTENT_MARKER) != 1:
        raise ValueError(
            "The tokenizer chat template must preserve user content exactly once"
        )
    prefix, _, suffix = rendered.partition(_CHAT_CONTENT_MARKER)
    return prefix, suffix


def build_rag_prompt_token_ids(
    tokenizer: Any,
    system_prompt: str,
    example: dict[str, Any],
    query_prompt: str,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
) -> tuple[list[int], list[str]]:
    """Build cache-aligned token IDs for a non-thinking chat request.

    Args:
        tokenizer: Hugging Face-compatible tokenizer with a chat template.
        system_prompt: Task instructions placed before retrieved documents.
        example: Dataset example containing contexts and a question.
        query_prompt: Instructions placed immediately before the question.
        separator: CacheBlend separator between prompt segments.
        prompt_build_method: Dataset-specific prompt construction method.

    Returns:
        The complete chat-formatted token IDs and the document prompt strings.
    """
    doc_prompts = None
    q_prompt = None
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts, q_prompt = build_fewshot_prompt(example, query_prompt)
    elif prompt_build_method == PromptBuildMethodType.QA:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    elif prompt_build_method == PromptBuildMethodType.DUREADER:
        doc_prompts, q_prompt = build_dureader_prompt(example, query_prompt)
    elif prompt_build_method == PromptBuildMethodType.QA_ZH:
        doc_prompts, q_prompt = build_qa_prompt(example, query_prompt)
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    chat_prefix, chat_suffix = get_non_thinking_chat_template_parts(tokenizer)
    system_token_ids = tokenize_segment_for_cache(
        tokenizer,
        chat_prefix + system_prompt,
        add_special_tokens=False,
    )
    sep_token_ids = []
    if separator:
        sep_token_ids = tokenize_segment_for_cache(
            tokenizer,
            separator,
            add_special_tokens=False,
        )
    doc_token_ids = [
        tokenize_segment_for_cache(
            tokenizer,
            doc_prompt,
            add_special_tokens=False,
        )
        for doc_prompt in doc_prompts
    ]
    query_token_ids = tokenize_segment_for_cache(
        tokenizer,
        q_prompt + chat_suffix,
        add_special_tokens=False,
    )

    prompt_token_ids = list(system_token_ids)
    for doc_token_id in doc_token_ids:
        if sep_token_ids:
            prompt_token_ids.extend(sep_token_ids)
        prompt_token_ids.extend(doc_token_id)
    if sep_token_ids:
        prompt_token_ids.extend(sep_token_ids)
    prompt_token_ids.extend(query_token_ids)
    return prompt_token_ids, doc_prompts


def _query_token_regions(
    tokenizer: Any,
    instruction_text: str,
    question_text: str,
    answer_prefix_text: str,
) -> tuple[list[int], dict[str, list[int]]]:
    """Tokenize a query suffix and retain its semantic token regions."""
    query_text = instruction_text + question_text + answer_prefix_text
    encoded = tokenizer(
        query_text,
        add_special_tokens=False,
        return_offsets_mapping=True,
    )
    token_ids = list(encoded.input_ids)
    offsets = list(encoded.offset_mapping)

    bos_token_id = tokenizer.bos_token_id
    if token_ids and bos_token_id is not None and token_ids[0] == bos_token_id:
        token_ids = token_ids[1:]
        offsets = offsets[1:]

    question_start = len(instruction_text)
    question_end = question_start + len(question_text)
    regions = {
        "instruction_indices": [],
        "question_indices": [],
        "answer_prefix_indices": [],
    }
    for token_index, (char_start, char_end) in enumerate(offsets):
        overlaps_question = char_end > question_start and char_start < question_end
        if overlaps_question:
            regions["question_indices"].append(token_index)
        elif char_start >= question_end:
            regions["answer_prefix_indices"].append(token_index)
        else:
            regions["instruction_indices"].append(token_index)

    if question_text and not regions["question_indices"]:
        raise ValueError("The actual question produced no tokenizer positions")
    return token_ids, regions


def build_rag_prompt_token_ids_with_spans(
    tokenizer: Any,
    system_prompt: str,
    example: dict[str, Any],
    query_prompt: str,
    separator: str,
    prompt_build_method: PromptBuildMethodType,
) -> tuple[list[int], list[str], dict[str, list[int]]]:
    """Build a cache-aligned RAG prompt with exact semantic token regions.

    The returned indices refer to positions in the complete token sequence.
    They let query-aware CacheBlend variants distinguish reusable documents
    from separators and online-only instruction/question/answer-prefix tokens.

    Args:
        tokenizer: Hugging Face-compatible fast tokenizer.
        system_prompt: Task instructions placed before retrieved documents.
        example: Dataset example containing contexts and a question.
        query_prompt: Instructions placed immediately before the question.
        separator: CacheBlend separator between prompt segments.
        prompt_build_method: Dataset-specific prompt construction method.

    Returns:
        Complete token IDs, document prompt strings, and semantic token indices.

    Raises:
        ValueError: If the tokenizer cannot return offsets or the question has
            no token positions.
    """
    if prompt_build_method == PromptBuildMethodType.FEW_SHOT:
        doc_prompts = [f"{ctx['text']}" for ctx in example["ctxs"]]
        instruction_text = f"{query_prompt}\n\n"
        question_text = str(example["question"])
        answer_prefix_text = ""
    elif prompt_build_method == PromptBuildMethodType.DUREADER:
        doc_prompts = [
            f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]
        ]
        instruction_text = query_prompt
        question_text = str(example["question"])
        answer_prefix_text = "\n答案："
    elif prompt_build_method in {
        PromptBuildMethodType.QA,
        PromptBuildMethodType.QA_ZH,
    }:
        doc_prompts = [
            f"{ctx['title']}\n\n{ctx['text']}\n\n" for ctx in example["ctxs"]
        ]
        instruction_text = query_prompt
        question_text = normalize_question(example["question"])
        answer_prefix_text = "\nAnswer:"
    else:
        raise ValueError(f"Invalid prompt build method {prompt_build_method}")

    chat_prefix, chat_suffix = get_non_thinking_chat_template_parts(tokenizer)
    system_token_ids = tokenize_segment_for_cache(
        tokenizer,
        chat_prefix + system_prompt,
        add_special_tokens=False,
    )
    separator_token_ids = (
        tokenize_segment_for_cache(tokenizer, separator, add_special_tokens=False)
        if separator
        else []
    )
    document_token_ids = [
        tokenize_segment_for_cache(tokenizer, text, add_special_tokens=False)
        for text in doc_prompts
    ]
    query_token_ids, local_query_regions = _query_token_regions(
        tokenizer,
        instruction_text,
        question_text,
        answer_prefix_text + chat_suffix,
    )

    token_ids = list(system_token_ids)
    spans: dict[str, list[int]] = {
        "system_indices": list(range(len(system_token_ids))),
        "document_indices": [],
        "separator_indices": [],
        "instruction_indices": [],
        "question_indices": [],
        "answer_prefix_indices": [],
    }
    for document_ids in document_token_ids:
        if separator_token_ids:
            start = len(token_ids)
            token_ids.extend(separator_token_ids)
            spans["separator_indices"].extend(
                range(start, start + len(separator_token_ids))
            )
        start = len(token_ids)
        token_ids.extend(document_ids)
        spans["document_indices"].extend(range(start, start + len(document_ids)))

    if separator_token_ids:
        start = len(token_ids)
        token_ids.extend(separator_token_ids)
        spans["separator_indices"].extend(
            range(start, start + len(separator_token_ids))
        )

    query_offset = len(token_ids)
    token_ids.extend(query_token_ids)
    for region_name, local_indices in local_query_regions.items():
        spans[region_name] = [query_offset + index for index in local_indices]

    reference_ids, reference_documents = build_rag_prompt_token_ids(
        tokenizer,
        system_prompt,
        example,
        query_prompt,
        separator,
        prompt_build_method,
    )
    if token_ids != reference_ids or doc_prompts != reference_documents:
        raise ValueError("Semantic span construction changed the RAG prompt tokens")

    return token_ids, doc_prompts, spans
