# SPDX-License-Identifier: Apache-2.0

"""Build a merged teacher dataset from per-example CE trace files."""

# Standard
import argparse
from pathlib import Path

# First Party
from kvrecompute.collector import collate_teacher_results, save_teacher_dataset
from kvrecompute.hf_teacher import load_teacher_trace, trace_to_teacher_result


def parse_args() -> argparse.Namespace:
    """Parse CLI arguments."""

    parser = argparse.ArgumentParser(
        description="Merge per-example CE traces into one token-level teacher dataset."
    )
    parser.add_argument(
        "--trace-dir",
        required=True,
        help="Directory containing per-example teacher traces saved with torch.save.",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Destination .pt file for the merged teacher dataset.",
    )
    parser.add_argument(
        "--suffix",
        default=".pt",
        help="File suffix used to discover traces inside --trace-dir.",
    )
    return parser.parse_args()


def main() -> None:
    """Merge per-example trace files into one training dataset."""

    args = parse_args()
    trace_dir = Path(args.trace_dir)
    trace_paths = sorted(trace_dir.glob(f"*{args.suffix}"))
    if not trace_paths:
        raise FileNotFoundError(
            f"No teacher trace files with suffix {args.suffix!r} found in {trace_dir}"
        )

    results = [trace_to_teacher_result(load_teacher_trace(path)) for path in trace_paths]
    merged_dataset = collate_teacher_results(results)
    save_teacher_dataset(merged_dataset, args.output)
    print(f"saved merged teacher dataset to {args.output}")


if __name__ == "__main__":
    main()

