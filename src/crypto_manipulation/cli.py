"""Command-line entry points for privacy-safe demonstrations."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .causal import estimate_causal_forest, estimate_linear_dml
from .predictive import run_grouped_benchmark
from .synthetic import make_synthetic_events


def _write_or_print(payload: dict[str, object], output: str | None) -> None:
    rendered = json.dumps(payload, indent=2, sort_keys=True)
    if output:
        Path(output).write_text(f"{rendered}\n", encoding="utf-8")
        print(f"Wrote {output}")
    else:
        print(rendered)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Privacy-safe thesis demonstrations")
    subparsers = parser.add_subparsers(dest="command", required=True)

    demo = subparsers.add_parser("demo", help="Run a grouped predictive benchmark")
    demo.add_argument("--events", type=int, default=600)
    demo.add_argument("--channels", type=int, default=30)
    demo.add_argument("--seed", type=int, default=42)
    demo.add_argument("--output")

    causal = subparsers.add_parser("causal-demo", help="Run optional DML estimators")
    causal.add_argument("--events", type=int, default=600)
    causal.add_argument("--channels", type=int, default=30)
    causal.add_argument("--seed", type=int, default=42)
    causal.add_argument(
        "--treatment",
        choices=["hype_intensity", "urgency_index", "instruction_density"],
        default="urgency_index",
    )
    causal.add_argument("--output")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    events = make_synthetic_events(args.events, args.channels, args.seed)

    if args.command == "demo":
        payload = {
            "dataset": "synthetic",
            "warning": "Demo metrics do not reproduce thesis results.",
            "metrics": run_grouped_benchmark(events, seed=args.seed),
        }
    else:
        payload = {
            "dataset": "synthetic",
            "warning": "Synthetic causal estimates are interface demonstrations only.",
            "treatment": args.treatment,
            "linear_dml": estimate_linear_dml(events, args.treatment),
            "causal_forest": estimate_causal_forest(events, args.treatment),
        }
    _write_or_print(payload, args.output)


if __name__ == "__main__":
    main()

