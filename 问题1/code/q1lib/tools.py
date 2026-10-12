from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path
from typing import Any

from praatio import textgrid

from .data import WordInterval


def run_command(
    command: list[str],
    log_path: Path | None = None,
    env: dict[str, str] | None = None,
    cwd: Path | None = None,
) -> subprocess.CompletedProcess[str]:
    completed = subprocess.run(
        command,
        cwd=str(cwd) if cwd else None,
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=True,
        check=False,
    )
    if log_path is not None:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write("COMMAND: " + subprocess.list2cmdline(command) + "\n")
            handle.write(f"RETURN_CODE: {completed.returncode}\n")
            if completed.stdout:
                handle.write("STDOUT:\n" + completed.stdout + "\n")
            if completed.stderr:
                handle.write("STDERR:\n" + completed.stderr + "\n")
            handle.write("\n")
    if completed.returncode != 0:
        tail = (completed.stderr or completed.stdout)[-4000:]
        raise RuntimeError(f"command failed ({completed.returncode}): {command}\n{tail}")
    return completed


def probe_media(video_path: Path, ffprobe: Path, log_path: Path | None = None) -> dict[str, Any]:
    result = run_command(
        [
            str(ffprobe),
            "-v",
            "error",
            "-show_format",
            "-show_streams",
            "-of",
            "json",
            str(video_path),
        ],
        log_path=log_path,
    )
    payload = json.loads(result.stdout)
    video_stream = next((x for x in payload.get("streams", []) if x.get("codec_type") == "video"), {})
    audio_stream = next((x for x in payload.get("streams", []) if x.get("codec_type") == "audio"), {})

    def rational(value: str | None) -> float | None:
        if not value or value in {"0/0", "N/A"}:
            return None
        numerator, denominator = value.split("/")
        return float(numerator) / float(denominator) if float(denominator) else None

    duration = payload.get("format", {}).get("duration") or video_stream.get("duration")
    duration_value = float(duration) if duration not in (None, "N/A") else None
    fps = rational(video_stream.get("avg_frame_rate") or video_stream.get("r_frame_rate"))
    stored_frame_count = (
        int(video_stream["nb_frames"])
        if str(video_stream.get("nb_frames", "")).isdigit()
        else None
    )
    stored_frame_duration = (
        float(stored_frame_count / fps)
        if stored_frame_count is not None and fps is not None and fps > 0
        else None
    )
    presented_frame_count_estimate = (
        int(round(duration_value * fps))
        if duration_value is not None and fps is not None and fps > 0
        else None
    )
    return {
        "duration_seconds": duration_value,
        "video_codec": video_stream.get("codec_name"),
        "width": video_stream.get("width"),
        "height": video_stream.get("height"),
        "fps": fps,
        "video_frame_count": presented_frame_count_estimate,
        "video_stored_frame_count": stored_frame_count,
        "video_stored_frame_duration_seconds": stored_frame_duration,
        "stored_minus_presented_duration_seconds": (
            max(0.0, stored_frame_duration - duration_value)
            if stored_frame_duration is not None and duration_value is not None
            else None
        ),
        "audio_codec": audio_stream.get("codec_name"),
        "audio_sample_rate": int(audio_stream["sample_rate"])
        if str(audio_stream.get("sample_rate", "")).isdigit()
        else None,
        "audio_channels": audio_stream.get("channels"),
        "has_video": bool(video_stream),
        "has_audio": bool(audio_stream),
    }


def extract_audio(
    video_path: Path,
    wav_path: Path,
    ffmpeg: Path,
    sample_rate: int,
    channels: int,
    overwrite: bool,
    log_path: Path | None = None,
) -> None:
    if wav_path.exists() and not overwrite:
        return
    wav_path.parent.mkdir(parents=True, exist_ok=True)
    run_command(
        [
            str(ffmpeg),
            "-y" if overwrite else "-n",
            "-v",
            "error",
            "-i",
            str(video_path),
            "-vn",
            "-ac",
            str(channels),
            "-ar",
            str(sample_rate),
            "-c:a",
            "pcm_s16le",
            str(wav_path),
        ],
        log_path=log_path,
    )


def run_mfa_align(
    corpus_dir: Path,
    output_dir: Path,
    mfa_env: Path,
    mfa_root: Path,
    dictionary: str,
    acoustic_model: str,
    overwrite: bool,
    log_path: Path,
) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        "conda",
        "run",
        "--prefix",
        str(mfa_env),
        "mfa",
        "align",
        str(corpus_dir),
        dictionary,
        acoustic_model,
        str(output_dir),
        "--clean",
    ]
    if overwrite:
        command.append("--overwrite")
    environment = os.environ.copy()
    environment["MFA_ROOT_DIR"] = str(mfa_root)
    run_command(command, log_path=log_path, env=environment)


def locate_textgrid(root: Path, sample_id: str) -> Path:
    matches = list(root.rglob(f"{sample_id}.TextGrid"))
    if len(matches) != 1:
        raise FileNotFoundError(f"expected one TextGrid for {sample_id}, found {len(matches)}")
    return matches[0]


def read_word_intervals(path: Path) -> list[WordInterval]:
    grid = textgrid.openTextgrid(str(path), includeEmptyIntervals=True)
    tier_names = list(grid.tierNames)
    word_tier_name = next((name for name in tier_names if "word" in name.lower()), None)
    if word_tier_name is None:
        raise ValueError(f"no word tier in {path}; tiers={tier_names}")
    tier = grid.getTier(word_tier_name)
    intervals: list[WordInterval] = []
    ignored = {"", "<eps>", "<sil>", "sil", "sp", "spn"}
    for entry in tier.entries:
        if hasattr(entry, "start"):
            start = float(entry.start)
            end = float(entry.end)
            label = str(entry.label).strip()
        else:
            start = float(entry[0])
            end = float(entry[1])
            label = str(entry[2]).strip()
        if label.lower() in ignored or end <= start:
            continue
        intervals.append(WordInterval(len(intervals), label, start, end))
    if not intervals:
        raise ValueError(f"word tier contains no usable intervals: {path}")
    previous = -1.0
    for interval in intervals:
        if interval.start < previous or interval.end < interval.start:
            raise ValueError(f"non-monotonic word intervals in {path}")
        previous = interval.end
    return intervals


def run_openface(
    executable: Path,
    video_path: Path,
    output_dir: Path,
    overwrite: bool,
    quiet: bool,
    log_path: Path,
) -> Path:
    output_dir.mkdir(parents=True, exist_ok=True)
    expected = output_dir / f"{video_path.stem}.csv"
    if expected.exists() and not overwrite:
        return expected
    command = [
        str(executable),
        "-f",
        str(video_path),
        "-out_dir",
        str(output_dir),
        "-aus",
        "-pose",
        "-gaze",
        "-2Dfp",
    ]
    if quiet:
        command.append("-q")
    run_command(command, log_path=log_path, cwd=executable.parent)
    if expected.exists():
        return expected
    candidates = list(output_dir.glob("*.csv"))
    if len(candidates) != 1:
        raise FileNotFoundError(f"OpenFace did not create a unique CSV in {output_dir}")
    return candidates[0]
