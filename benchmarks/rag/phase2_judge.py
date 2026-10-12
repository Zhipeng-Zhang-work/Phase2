# SPDX-License-Identifier: Apache-2.0
"""Judge Phase 2 RAG outputs with a local Hugging Face causal language model."""

# Standard
import argparse
import ast
import csv
import json
import math
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import Any

# Third Party
import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Local
sys.path.insert(0, str(Path(__file__).resolve().parent))
from output_processing import process_model_output


_ENGLISH_PROMPT = """You are an impartial LLM judge for a RAG benchmark.
Compare the model answer against the reference answer. Reward semantic correctness,
not answer length or unsupported extra details. For summaries, assess whether key
facts are preserved. Return only this JSON object, without markdown or reasoning:
{"is_correct": true, "score": 0.0, "reason": "short reason",
 "normalized_reference": "short reference", "normalized_prediction": "short prediction"}
score must be a number from 0.0 to 1.0. Set is_correct to true iff score >= 0.70."""

_CHINESE_PROMPT = """你是公正的 RAG benchmark LLM 评审。请根据标准答案评估模型答案的
语义正确性；不要因为答案更长或带有未经标准答案支持的细节而加分。摘要任务评估关键事实是否保留。
只能返回如下 JSON，不能返回 Markdown 或推理：
{"is_correct": true, "score": 0.0, "reason": "简短原因",
 "normalized_reference": "简短标准答案", "normalized_prediction": "简短模型答案"}
score 必须在 0.0 至 1.0；仅当 score >= 0.70 时 is_correct 为 true。"""


def _parse_args() -> argparse.Namespace:
    """Parse command-line arguments for a resumable judge run."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--benchmark-dir", type=Path, required=True)
    parser.add_argument("--judge-model", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    return parser.parse_args()


def _clean_value(value: Any) -> str:
    """Convert a CSV value to a non-null string."""
    if value is None or (isinstance(value, float) and math.isnan(value)):
        return ""
    return str(value)


def _parse_answers(value: Any) -> str:
    """Normalize an answer field for the judge prompt."""
    text = _clean_value(value)
    try:
        parsed = ast.literal_eval(text)
    except (SyntaxError, ValueError):
        return text
    return json.dumps(parsed, ensure_ascii=False)


def _question_from_record(record: Any, fallback: str) -> str:
    """Extract the task question from a dataset record when available."""
    if not isinstance(record, dict):
        return fallback
    for key in ("question", "query", "input", "prompt"):
        value = record.get(key)
        if isinstance(value, str) and value.strip():
            return value
    for key in ("questions", "queries"):
        value = record.get(key)
        if isinstance(value, list) and value:
            return json.dumps(value, ensure_ascii=False)
    return fallback


def _load_dataset(path: Path) -> list[Any]:
    """Load one JSON benchmark dataset and require a top-level list."""
    with path.open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ValueError(f"Dataset must contain a JSON list: {path}")
    return records


def _source_rows(benchmark_dir: Path) -> list[dict[str, Any]]:
    """Construct per-request judge records from benchmark manifests and CSVs."""
    manifest_path = benchmark_dir / "run_manifest.csv"
    manifest = pd.read_csv(manifest_path)
    rag_dir = Path(__file__).resolve().parent
    rows: list[dict[str, Any]] = []
    datasets: dict[Path, list[Any]] = {}

    for run in manifest.to_dict(orient="records"):
        debug_path = Path(_clean_value(run["debug_csv"]))
        output_path = Path(_clean_value(run["output_csv"]))
        dataset_path = rag_dir / _clean_value(run["dataset"])
        if not debug_path.is_file() or not output_path.is_file():
            raise FileNotFoundError(f"Missing benchmark output for manifest row: {run}")
        if dataset_path not in datasets:
            datasets[dataset_path] = _load_dataset(dataset_path)

        debug = pd.read_csv(debug_path)
        summary = pd.read_csv(output_path)
        for row_index, debug_row in debug.iterrows():
            request_id = int(debug_row.get("request_id", row_index))
            record = (
                datasets[dataset_path][request_id]
                if 0 <= request_id < len(datasets[dataset_path])
                else {}
            )
            summary_row: Any = summary.iloc[row_index] if row_index < len(summary) else {}
            generated_text = _clean_value(debug_row.get("generated_text", ""))
            source_key = "|".join(
                (
                    _clean_value(run["blender"]),
                    _clean_value(run["dataset"]),
                    _clean_value(run["output_csv"]),
                    str(request_id),
                )
            )
            rows.append(
                {
                    "source_key": source_key,
                    "blender": _clean_value(run["blender"]),
                    "dataset": _clean_value(run["dataset"]),
                    "ratio": _clean_value(run["ratio"]),
                    "prompt_build_method": _clean_value(run["prompt_build_method"]),
                    "request_id": request_id,
                    "quality_metric": (
                        "f1"
                        if _clean_value(run["prompt_build_method"]).upper()
                        in {"QA", "QA_ZH"}
                        else "rouge_l"
                    ),
                    "static_quality": _clean_value(summary_row.get("quality", "")),
                    "question": _question_from_record(
                        record, _clean_value(debug_row.get("prompt", ""))
                    ),
                    "answers": _parse_answers(debug_row.get("answers", "")),
                    "generated_text": generated_text,
                    "ttft": _clean_value(debug_row.get("ttft", "")),
                    "tpot": _clean_value(debug_row.get("tpot", "")),
                    "generation_time": _clean_value(
                        debug_row.get("generation_time", "")
                    ),
                    "prefill_token_cnt": _clean_value(
                        debug_row.get("prefill_token_cnt", "")
                    ),
                    "generation_token_cnt": _clean_value(
                        debug_row.get("generation_token_cnt", "")
                    ),
                    "output_csv": _clean_value(run["output_csv"]),
                    "debug_csv": _clean_value(run["debug_csv"]),
                }
            )
    return rows


def _final_answer(generated_text: str) -> str:
    """Return the answer eligible for judging after reasoning cleanup."""
    return process_model_output(generated_text).final_answer.strip()


def _judge_prompt(row: dict[str, Any], final_answer: str) -> tuple[str, str]:
    """Build system and user content for one judge request."""
    is_chinese = "zh" in row["dataset"].lower() or "dureader" in row["dataset"].lower()
    system_prompt = _CHINESE_PROMPT if is_chinese else _ENGLISH_PROMPT
    task_type = (
        "summarization"
        if row["prompt_build_method"].upper() == "FEW_SHOT"
        else "question answering"
    )
    user_prompt = (
        f"Task type: {task_type}\nDataset: {row['dataset']}\n"
        f"Static metric: {row['quality_metric']}\n"
        f"Static score: {row['static_quality']}\n\n"
        f"Question or source task:\n{row['question']}\n\n"
        f"Reference answer(s):\n{row['answers']}\n\n"
        f"Model answer:\n{final_answer}\n"
    )
    return system_prompt, user_prompt


def _parse_judge_json(text: str) -> dict[str, Any]:
    """Parse a judge JSON response, allowing accidental Markdown fences."""
    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip())
    candidates = (cleaned,)
    match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
    if match:
        candidates = (cleaned, match.group(0))
    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return {
        "is_correct": False,
        "score": 0.0,
        "reason": "Failed to parse judge JSON.",
        "normalized_reference": "",
        "normalized_prediction": "",
    }


def _normalize_score(value: Any) -> float:
    """Clamp an untrusted judge score to the documented range."""
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _normalize_correct(value: Any, score: float) -> bool:
    """Normalize the judge boolean while enforcing the score threshold."""
    if isinstance(value, bool):
        return value and score >= 0.70
    return str(value).strip().lower() in {"true", "1", "yes"} and score >= 0.70


def _judge_one(
    row: dict[str, Any], tokenizer: Any, model: Any, device: str, max_new_tokens: int
) -> dict[str, Any]:
    """Run a judge inference or emit the required zero score for an empty answer."""
    final_answer = _final_answer(row["generated_text"])
    if not final_answer:
        return {
            "judge_is_correct": False,
            "judge_score": 0.0,
            "judge_reason": "Empty model answer; no answer was generated.",
            "judge_normalized_reference": "",
            "judge_normalized_prediction": "",
            "judge_raw_output": "",
        }
    system_prompt, user_prompt = _judge_prompt(row, final_answer)
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.inference_mode():
        output = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False,
            pad_token_id=tokenizer.eos_token_id,
        )
    raw_output = tokenizer.decode(
        output[0][inputs["input_ids"].shape[-1] :], skip_special_tokens=True
    )
    parsed = _parse_judge_json(raw_output)
    score = _normalize_score(parsed.get("score"))
    return {
        "judge_is_correct": _normalize_correct(parsed.get("is_correct"), score),
        "judge_score": score,
        "judge_reason": _clean_value(parsed.get("reason", "")),
        "judge_normalized_reference": _clean_value(
            parsed.get("normalized_reference", "")
        ),
        "judge_normalized_prediction": _clean_value(
            parsed.get("normalized_prediction", "")
        ),
        "judge_raw_output": raw_output,
    }


def _write_summary(details_path: Path, summary_path: Path) -> None:
    """Aggregate completed judge rows into per-variant dataset summary metrics."""
    details = pd.read_csv(details_path)
    if details.empty:
        return
    details["judge_is_correct"] = (
        details["judge_is_correct"].astype(str).str.lower() == "true"
    )
    details["judge_score"] = pd.to_numeric(details["judge_score"], errors="coerce")
    for column in ("static_quality", "ttft", "tpot"):
        details[column] = pd.to_numeric(details[column], errors="coerce")
    summary = (
        details.groupby(
            ["judge_model", "blender", "dataset", "ratio", "prompt_build_method"],
            dropna=False,
        )
        .agg(
            num_questions=("judge_score", "count"),
            judge_accuracy=("judge_is_correct", "mean"),
            average_judge_score=("judge_score", "mean"),
            average_static_quality=("static_quality", "mean"),
            average_ttft=("ttft", "mean"),
            average_tpot=("tpot", "mean"),
        )
        .reset_index()
    )
    summary.to_csv(summary_path, index=False)


def main() -> None:
    """Run a resumable local LLM-as-judge evaluation over benchmark results."""
    args = _parse_args()
    benchmark_dir = args.benchmark_dir.resolve()
    if not torch.cuda.is_available():
        raise SystemExit("The LLM judge requires a CUDA GPU.")
    source_rows = _source_rows(benchmark_dir)
    judge_dir = benchmark_dir / "judge"
    judge_dir.mkdir(parents=True, exist_ok=True)
    details_path = judge_dir / "judge_details.csv"
    summary_path = judge_dir / "judge_summary.csv"
    completed: set[str] = set()
    if details_path.exists():
        completed = set(pd.read_csv(details_path)["source_key"].astype(str))

    print(f"CUDA device: {torch.cuda.get_device_name(0)}")
    print(f"Loading judge model: {args.judge_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.judge_model, local_files_only=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.judge_model, local_files_only=True, torch_dtype=torch.bfloat16
    ).to("cuda")
    model.eval()

    fieldnames = [
        "timestamp", "judge_model", "source_key", "blender", "dataset", "ratio",
        "prompt_build_method", "request_id", "quality_metric", "static_quality",
        "judge_is_correct", "judge_score", "judge_reason",
        "judge_normalized_reference", "judge_normalized_prediction", "judge_raw_output",
        "question", "answers", "generated_text", "ttft", "tpot", "generation_time",
        "prefill_token_cnt", "generation_token_cnt", "output_csv", "debug_csv",
    ]
    write_header = not details_path.exists()
    processed = 0
    with details_path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for source_row in source_rows:
            if source_row["source_key"] in completed:
                continue
            result = _judge_one(
                source_row, tokenizer, model, "cuda", args.max_new_tokens
            )
            row = {
                "timestamp": datetime.now().isoformat(timespec="seconds"),
                "judge_model": args.judge_model,
                **source_row,
                **result,
            }
            writer.writerow(row)
            handle.flush()
            processed += 1
            print(
                f"[judge] blender={row['blender']} dataset={row['dataset']} "
                f"request={row['request_id']} score={row['judge_score']}"
            )
    _write_summary(details_path, summary_path)
    print(f"[judge] processed={processed} summary={summary_path}")


if __name__ == "__main__":
    main()
