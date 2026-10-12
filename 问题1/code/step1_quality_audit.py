"""VS Code one-click entry: audit all 100 audio/text pairs before alignment."""

from pathlib import Path

from q1lib import Q1Pipeline, load_config
from q1lib.pipeline import configure_logging


def main() -> None:
    config = load_config(Path(__file__).with_name("config.yaml"))
    configure_logging(config.output_root / "logs")
    pipeline = Q1Pipeline(config)
    pipeline.run(stage="quality")
    print("Quality audit finished. Next run calibrate_alignment_thresholds.py.")


if __name__ == "__main__":
    main()
