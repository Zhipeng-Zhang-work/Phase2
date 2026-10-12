# SPDX-License-Identifier: Apache-2.0
"""Run standalone TurboRAG evaluation and write CSV results."""

# Standard
from datetime import datetime
from pathlib import Path
import argparse
import csv
import time

# Local
from ..lmcache.engine import TurboRAGEngine
from .metrics import compute_quality
from .prompts import build_example, load_dataset


def main() -> None:
    """Evaluate one dataset using precomputed TurboRAG caches."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-build-method", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--summary-output", required=True)
    parser.add_argument("--details-output", required=True)
    parser.add_argument("--max-new-tokens", type=int, default=2048)
    parser.add_argument("--max-documents", type=int, default=40)
    parser.add_argument("--qps", type=float, default=0.25)
    args = parser.parse_args()

    engine = TurboRAGEngine(args.model, args.cache_dir)
    examples = load_dataset(args.dataset)
    if not examples:
        raise ValueError("evaluation dataset is empty")
    detail_rows: list[dict[str, object]] = []
    benchmark_started = time.perf_counter()
    next_arrival = benchmark_started
    for index, example in enumerate(examples):
        now = time.perf_counter()
        if args.qps > 0 and now < next_arrival:
            time.sleep(next_arrival - now)
        next_arrival += 1.0 / args.qps if args.qps > 0 else 0.0
        system, documents, query, references = build_example(
            example,
            args.prompt_build_method,
        )
        result = engine.generate(
            system,
            documents[: args.max_documents],
            query,
            args.max_new_tokens,
        )
        quality = compute_quality(
            result.text,
            references,
            args.prompt_build_method,
            engine.tokenizer,
        )
        row = {
            "baseline": "turborag_reordered",
            "dataset": args.dataset,
            "prompt_build_method": args.prompt_build_method,
            "sample_index": index,
            "quality": quality,
            "ttft": result.ttft,
            "tpot": result.tpot,
            "total_time": result.total_time,
            "output_tokens": result.output_tokens,
            "question": example.get("question", ""),
            "references": " || ".join(references),
            "prediction": result.text,
        }
        detail_rows.append(row)
        print(
            f"sample={index} quality={quality:.6f} "
            f"ttft={result.ttft:.6f} tpot={result.tpot:.6f}",
            flush=True,
        )
    elapsed = time.perf_counter() - benchmark_started
    _write_details(args.details_output, detail_rows)
    _append_summary(
        args.summary_output,
        args.dataset,
        args.prompt_build_method,
        detail_rows,
        elapsed,
    )


def _write_details(path: str, rows: list[dict[str, object]]) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _append_summary(
    path: str,
    dataset: str,
    method: str,
    rows: list[dict[str, object]],
    elapsed: float,
) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    total_tokens = sum(int(row["output_tokens"]) for row in rows)
    summary = {
        "timestamp": datetime.now().isoformat(timespec="seconds"),
        "baseline": "turborag_reordered",
        "dataset": dataset,
        "prompt_build_method": method,
        "samples": len(rows),
        "average_quality": sum(float(row["quality"]) for row in rows) / len(rows),
        "average_ttft": sum(float(row["ttft"]) for row in rows) / len(rows),
        "average_tpot": sum(float(row["tpot"]) for row in rows) / len(rows),
        "throughput": total_tokens / elapsed if elapsed else 0.0,
    }
    exists = output.exists() and output.stat().st_size > 0
    with output.open("a", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(summary))
        if not exists:
            writer.writeheader()
        writer.writerow(summary)


if __name__ == "__main__":
    main()
