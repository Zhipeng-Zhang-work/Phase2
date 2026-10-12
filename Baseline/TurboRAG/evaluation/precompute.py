# SPDX-License-Identifier: Apache-2.0
"""Precompute independent document caches for TurboRAG."""

# Standard
import argparse

# Local
from ..lmcache.engine import TurboRAGEngine
from .prompts import build_example, load_dataset


def main() -> None:
    """Precompute every unique document in a benchmark dataset."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True)
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--prompt-build-method", required=True)
    parser.add_argument("--cache-dir", required=True)
    parser.add_argument("--max-documents", type=int, default=40)
    args = parser.parse_args()

    engine = TurboRAGEngine(args.model, args.cache_dir)
    created = 0
    reused = 0
    for example in load_dataset(args.dataset):
        _, documents, _, _ = build_example(
            example,
            args.prompt_build_method,
        )
        for document in documents[: args.max_documents]:
            key, was_created = engine.precompute_document(document)
            created += int(was_created)
            reused += int(not was_created)
            print(f"document={key} created={was_created}", flush=True)
    print(f"precompute_complete created={created} reused={reused}")


if __name__ == "__main__":
    main()
