from __future__ import annotations

import argparse
from pathlib import Path

from q1lib import Q1Pipeline, load_config
from q1lib.pipeline import configure_logging


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Question 1: quality-gated multimodal extraction and hierarchical alignment"
    )
    parser.add_argument("--config", type=Path, default=Path(__file__).with_name("config.yaml"))
    parser.add_argument(
        "--stage",
        choices=("manifest", "prepare", "quality", "align", "features", "all"),
        default="all",
        help="all runs manifest -> audio -> VAD/ASR audit -> gated MFA -> feature extraction",
    )
    parser.add_argument(
        "--sample-id",
        action="append",
        default=None,
        help="process one sample_id; may be specified multiple times",
    )
    parser.add_argument("--limit", type=int, default=None, help="process only the first N label rows")
    parser.add_argument(
        "--skip-mfa",
        action="store_true",
        help="reuse existing TextGrid files; useful when rerunning feature extraction",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    configure_logging(config.output_root / "logs")
    pipeline = Q1Pipeline(config)
    pipeline.run(
        stage=args.stage,
        sample_ids=set(args.sample_id) if args.sample_id else None,
        limit=args.limit,
        skip_mfa=args.skip_mfa,
    )


if __name__ == "__main__":
    main()
