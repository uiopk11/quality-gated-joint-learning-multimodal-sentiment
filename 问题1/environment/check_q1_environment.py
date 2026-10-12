# 本程序及代码是在人工智能工具辅助下完成的。
# 人工智能工具名称：DeepSeek（深度求索）
# 版本/型号：DeepSeek-V4.1-Flash
# 开发机构/公司：杭州深度求索人工智能基础技术研究有限公司
# 版本颁布日期：2026年9月10日
# 版本信息来源：https://deepseek.com/news/deepseek-v4-1-flash/

from __future__ import annotations

import importlib
import json
import shutil
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
MAIN_ENV = PROJECT_ROOT / ".envs" / "q1"
MFA_ENV = PROJECT_ROOT / ".envs" / "mfa"
OPENFACE_EXE = PROJECT_ROOT / "工具" / "OpenFace_2.2.0_win_x64" / "FeatureExtraction.exe"
BERT_DIR = PROJECT_ROOT / "模型" / "bert-base-uncased"
WHISPER_DIR = PROJECT_ROOT / "模型" / "whisper-small"
MFA_DATA = PROJECT_ROOT / "q1_work" / "mfa_data" / "pretrained_models"


def module_version(module_name: str) -> dict[str, object]:
    try:
        module = importlib.import_module(module_name)
        return {
            "ok": True,
            "version": getattr(module, "__version__", "unknown"),
        }
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"ok": False, "error": str(exc)}


def command_version(command: list[str]) -> dict[str, object]:
    try:
        completed = subprocess.run(
            command,
            check=False,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="replace",
        )
        output = (completed.stdout or completed.stderr).strip().splitlines()
        return {
            "ok": completed.returncode == 0,
            "returncode": completed.returncode,
            "version": output[0] if output else "no output",
        }
    except Exception as exc:  # pragma: no cover - diagnostic script
        return {"ok": False, "error": str(exc)}


report: dict[str, object] = {
    "python": {
        "ok": sys.version_info[:2] == (3, 11),
        "version": sys.version.split()[0],
        "executable": sys.executable,
    },
    "modules": {
        name: module_version(name)
        for name in (
            "numpy",
            "pandas",
            "torch",
            "transformers",
            "opensmile",
            "librosa",
            "cv2",
            "praatio",
            "pympi",
        )
    },
    "commands": {},
    "paths": {
        "project_root": str(PROJECT_ROOT),
        "main_environment": str(MAIN_ENV),
        "mfa_environment": str(MFA_ENV),
        "openface": str(OPENFACE_EXE),
    },
    "resources": {
        "bert_base_uncased": {
            "ok": (BERT_DIR / "model.safetensors").exists()
            and (BERT_DIR / "vocab.txt").exists(),
            "path": str(BERT_DIR),
        },
        "whisper_small": {
            "ok": (WHISPER_DIR / "model.safetensors").exists()
            and (WHISPER_DIR / "preprocessor_config.json").exists()
            and (WHISPER_DIR / "tokenizer.json").exists(),
            "path": str(WHISPER_DIR),
        },
        "mfa_english_us_arpa": {
            "ok": (MFA_DATA / "acoustic" / "english_us_arpa.zip").exists()
            and (MFA_DATA / "dictionary" / "english_us_arpa.dict").exists(),
            "path": str(MFA_DATA),
        },
    },
}

ffmpeg = shutil.which("ffmpeg") or str(MAIN_ENV / "Library" / "bin" / "ffmpeg.exe")
ffprobe = shutil.which("ffprobe") or str(MAIN_ENV / "Library" / "bin" / "ffprobe.exe")
if not Path(ffmpeg).exists():
    ffmpeg = None
if not Path(ffprobe).exists():
    ffprobe = None
report["commands"] = {
    "ffmpeg": command_version([ffmpeg, "-version"]) if ffmpeg else {"ok": False, "error": "not found"},
    "ffprobe": command_version([ffprobe, "-version"]) if ffprobe else {"ok": False, "error": "not found"},
    "mfa": command_version(["conda", "run", "--prefix", str(MFA_ENV), "mfa", "version"])
    if (MFA_ENV / "Scripts" / "mfa.exe").exists()
    else {"ok": False, "error": "not installed"},
    "openface": {"ok": OPENFACE_EXE.exists(), "version": "OpenFace 2.2.0" if OPENFACE_EXE.exists() else "not installed"},
}

try:
    import torch

    report["torch_runtime"] = {
        "cuda_available": torch.cuda.is_available(),
        "cuda_runtime": torch.version.cuda,
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
    }
except Exception as exc:  # pragma: no cover - diagnostic script
    report["torch_runtime"] = {"cuda_available": False, "error": str(exc)}

print(json.dumps(report, ensure_ascii=True, indent=2))

core_checks = [report["python"]["ok"]]
core_checks.extend(item["ok"] for item in report["modules"].values())
core_checks.extend(report["commands"][name]["ok"] for name in ("ffmpeg", "ffprobe", "mfa", "openface"))
core_checks.extend(item["ok"] for item in report["resources"].values())
raise SystemExit(0 if all(core_checks) else 1)
