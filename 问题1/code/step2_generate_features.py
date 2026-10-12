"""VS Code one-click entry: use reviewed quality results to generate final features."""

from pathlib import Path

from q1lib import Q1Pipeline, load_config
from q1lib.pipeline import configure_logging


def main() -> None:
    config = load_config(Path(__file__).with_name("config.yaml"))
    configure_logging(config.output_root / "logs")
    pipeline = Q1Pipeline(config)
    pipeline.build_manifest()
    pipeline.save_run_provenance()
    pipeline.prepare_mfa_corpus(pipeline.records)
    pipeline.align_mfa(pipeline.records)
    pipeline.process_features(pipeline.records)
    print("Feature generation finished. Next run audit_outputs.py.")


if __name__ == "__main__":
    main()
