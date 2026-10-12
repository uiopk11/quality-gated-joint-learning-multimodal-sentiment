from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml


@dataclass(frozen=True)
class Q1Config:
    raw: dict[str, Any]
    config_path: Path

    @property
    def project_root(self) -> Path:
        return Path(self.raw["project_root"]).resolve()

    @property
    def data_root(self) -> Path:
        return Path(self.raw["data_root"]).resolve()

    @property
    def label_file(self) -> Path:
        return Path(self.raw["label_file"]).resolve()

    @property
    def output_root(self) -> Path:
        return Path(self.raw["output_root"]).resolve()

    @property
    def mfa_root(self) -> Path:
        return Path(self.tools["mfa_root"]).resolve()

    @property
    def mfa_work_root(self) -> Path:
        return Path(self.tools["mfa_work_root"]).resolve()

    @property
    def mfa_corpus_dir(self) -> Path:
        return self.mfa_work_root / "corpus"

    @property
    def mfa_output_dir(self) -> Path:
        return self.mfa_work_root / "gated_aligned"

    @property
    def mfa_gated_corpus_dir(self) -> Path:
        return self.mfa_work_root / "gated_corpus"

    @property
    def mfa_partial_output_dir(self) -> Path:
        return self.mfa_work_root / "gated_aligned_partial"

    @property
    def mfa_partial_gated_corpus_dir(self) -> Path:
        return self.mfa_work_root / "gated_corpus_partial"

    @property
    def tools(self) -> dict[str, Any]:
        return self.raw["tools"]

    @property
    def audio(self) -> dict[str, Any]:
        return self.raw["audio"]

    @property
    def vision(self) -> dict[str, Any]:
        return self.raw["vision"]

    @property
    def text(self) -> dict[str, Any]:
        return self.raw["text"]

    @property
    def sequence(self) -> dict[str, Any]:
        return self.raw["sequence"]

    @property
    def alignment(self) -> dict[str, Any]:
        return self.raw["alignment"]

    @property
    def quality(self) -> dict[str, Any]:
        return self.raw["quality"]

    @property
    def runtime(self) -> dict[str, Any]:
        return self.raw["runtime"]

    def ensure_output_dirs(self) -> None:
        for relative in (
            "manifest",
            "samples",
            "logs",
            "summary",
            "figures",
        ):
            (self.output_root / relative).mkdir(parents=True, exist_ok=True)
        self.mfa_corpus_dir.mkdir(parents=True, exist_ok=True)
        self.mfa_output_dir.mkdir(parents=True, exist_ok=True)
        self.mfa_gated_corpus_dir.mkdir(parents=True, exist_ok=True)
        self.mfa_partial_output_dir.mkdir(parents=True, exist_ok=True)
        self.mfa_partial_gated_corpus_dir.mkdir(parents=True, exist_ok=True)

    def validate(self) -> None:
        required_files = {
            "label_file": self.label_file,
            "ffmpeg": Path(self.tools["ffmpeg"]),
            "ffprobe": Path(self.tools["ffprobe"]),
            "openface": Path(self.tools["openface"]),
            "bert_model": Path(self.text["bert_model"]),
        }
        errors = [f"{name}: {path}" for name, path in required_files.items() if not path.exists()]
        if not self.data_root.is_dir():
            errors.append(f"data_root: {self.data_root}")
        if not Path(self.tools["mfa_env"]).is_dir():
            errors.append(f"mfa_env: {self.tools['mfa_env']}")
        acoustic_model = self.mfa_root / "pretrained_models" / "acoustic" / "english_us_arpa.zip"
        dictionary = self.mfa_root / "pretrained_models" / "dictionary" / "english_us_arpa.dict"
        if not acoustic_model.is_file():
            errors.append(f"mfa acoustic model: {acoustic_model}")
        if not dictionary.is_file():
            errors.append(f"mfa dictionary: {dictionary}")
        if errors:
            raise FileNotFoundError("Missing required paths:\n" + "\n".join(errors))
        for name, path in {
            "mfa_env": Path(self.tools["mfa_env"]),
            "mfa_root": self.mfa_root,
            "mfa_work_root": self.mfa_work_root,
        }.items():
            try:
                str(path).encode("ascii")
            except UnicodeEncodeError as exc:
                raise ValueError(
                    f"{name} must use an ASCII-only path on Windows because MFA/Kaldi "
                    f"cannot reliably process Chinese paths: {path}"
                ) from exc
        if self.output_root == self.data_root or self.output_root.is_relative_to(self.data_root):
            raise ValueError("output_root must not be inside the immutable data_root")
        if int(self.sequence["standard_length"]) <= 0:
            raise ValueError("sequence.standard_length must be positive")
        if int(self.sequence["unaligned_length"]) <= 0:
            raise ValueError("sequence.unaligned_length must be positive")
        allowed_pooling = {"mean", "std", "max", "min", "median"}
        for modality in ("audio", "vision"):
            unknown = set(self.raw[modality]["pooling"]) - allowed_pooling
            if unknown:
                raise ValueError(f"Unsupported {modality} pooling statistics: {sorted(unknown)}")
        if int(self.audio["target_dimension"]) != 74:
            raise ValueError("audio.target_dimension must be 74")
        if len(self.audio["delta_fields"]) + 65 != 74:
            raise ValueError("ComParE_2016 LLD must be extended by exactly 9 delta fields")
        if int(self.vision["target_dimension"]) != 35:
            raise ValueError("vision.target_dimension must be 35")
        if len(self.vision["feature_fields"]) != 35:
            raise ValueError("vision.feature_fields must contain exactly 35 fields")
        if len(set(self.vision["feature_fields"])) != 35:
            raise ValueError("vision.feature_fields contains duplicates")
        low = float(self.alignment["low_threshold"])
        high = float(self.alignment["high_threshold"])
        if not 0.0 <= low < high <= 1.0:
            raise ValueError("alignment thresholds must satisfy 0 <= low < high <= 1")
        padding = float(self.alignment.get("mfa_window_padding_seconds", 0.03))
        if not 0.0 <= padding <= 0.5:
            raise ValueError("alignment.mfa_window_padding_seconds must be between 0 and 0.5")


def load_config(path: str | Path) -> Q1Config:
    config_path = Path(path).resolve()
    with config_path.open("r", encoding="utf-8") as handle:
        raw = yaml.safe_load(handle)
    config = Q1Config(raw=raw, config_path=config_path)
    config.validate()
    config.ensure_output_dirs()
    return config
