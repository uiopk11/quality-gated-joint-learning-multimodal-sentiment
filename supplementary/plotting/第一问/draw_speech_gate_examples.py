"""Draw Fig. q1_speech_gate_examples without matplotlib (Pillow only).

Reads Attachment-1 WAV + quality-audit files (read-only outside this folder).
Writes PNG only under cursor_draw/figures/.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np
import soundfile as sf
from PIL import Image, ImageDraw, ImageFont


HERE = Path(__file__).resolve().parent
Q1_ROOT = HERE.parent
SAMPLE_ROOT = Q1_ROOT / "输出" / "samples"
OUT_DIR = HERE / "figures"
OUT_PATH = OUT_DIR / "q1_speech_gate_examples.png"

SAMPLE_SPEECH = "-3g5yACwYnA__13"
SAMPLE_SILENT = "-mJ2ud6oKI8__1"

FRAME_MS = 30.0
HOP_MS = 10.0
ENERGY_MARGIN_DB = 12.0
SILENCE_PEAK_DBFS = -50.0
MIN_ACTIVITY = 0.02
MIN_ASR_WORDS = 1
NOISE_PERCENTILE = 20

# Canvas layout (pixels). Draw and save at the same resolution so text stays sharp.
DPI_SCALE = 2
W = 1400 * DPI_SCALE
MARGIN_X = 88 * DPI_SCALE
MARGIN_TOP = 58 * DPI_SCALE
MARGIN_BOTTOM = 28 * DPI_SCALE
ROW_GAP = 12 * DPI_SCALE
PANEL_GAP = 10 * DPI_SCALE
PANEL_H = {
    "wave": 150 * DPI_SCALE,
    "energy": 170 * DPI_SCALE,
    "asr": 130 * DPI_SCALE,
}
TITLE_BAND = 36 * DPI_SCALE
TIME_BAND = 48 * DPI_SCALE


def _dbfs(value: float) -> float:
    return float(20.0 * np.log10(max(float(value), 1e-12)))


def _load_font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = []
    windir = Path(r"C:\Windows\Fonts")
    names = (
        ["msyhbd.ttc", "simhei.ttf", "msyh.ttc", "simsun.ttc"]
        if bold
        else ["msyh.ttc", "simhei.ttf", "simsun.ttc", "msyhbd.ttc"]
    )
    for name in names:
        path = windir / name
        if path.is_file():
            candidates.append(path)
    for path in candidates:
        try:
            return ImageFont.truetype(str(path), size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def analyze_frames(signal: np.ndarray, sample_rate: int) -> dict:
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    signal = np.asarray(signal, dtype=np.float32)
    if len(signal) == 0:
        return {
            "times": np.asarray([0.0]),
            "frame_db": np.asarray([-240.0]),
            "noise_floor": -240.0,
            "paper_threshold": -228.0,
            "code_threshold": SILENCE_PEAK_DBFS,
            "active_mask": np.asarray([False]),
            "activity_ratio": 0.0,
            "peak_dbfs": -240.0,
            "rms_dbfs": -240.0,
        }

    peak = float(np.max(np.abs(signal), initial=0.0))
    rms = float(np.sqrt(np.mean(np.square(signal, dtype=np.float64))))
    frame_length = max(1, int(round(FRAME_MS * 1e-3 * sample_rate)))
    hop_length = max(1, int(round(HOP_MS * 1e-3 * sample_rate)))

    if len(signal) < frame_length:
        frame_rms = np.asarray([rms], dtype=np.float64)
        centers = np.asarray([0.5 * len(signal) / sample_rate], dtype=np.float64)
    else:
        starts = np.arange(0, len(signal) - frame_length + 1, hop_length)
        frame_rms = np.asarray(
            [
                np.sqrt(np.mean(np.square(signal[s : s + frame_length], dtype=np.float64)))
                for s in starts
            ],
            dtype=np.float64,
        )
        centers = (starts + 0.5 * frame_length) / float(sample_rate)

    frame_db = 20.0 * np.log10(np.maximum(frame_rms, 1e-12))
    noise_floor = float(np.percentile(frame_db, NOISE_PERCENTILE))
    paper_threshold = noise_floor + ENERGY_MARGIN_DB
    code_threshold = max(SILENCE_PEAK_DBFS, paper_threshold)
    active_mask = frame_db >= code_threshold
    activity_ratio = float(np.mean(active_mask)) if len(active_mask) else 0.0
    return {
        "times": centers,
        "frame_db": frame_db,
        "noise_floor": noise_floor,
        "paper_threshold": paper_threshold,
        "code_threshold": code_threshold,
        "active_mask": active_mask,
        "activity_ratio": activity_ratio,
        "peak_dbfs": _dbfs(peak),
        "rms_dbfs": _dbfs(rms),
    }


def load_sample(sample_id: str) -> dict:
    sample_dir = SAMPLE_ROOT / sample_id
    metadata = json.loads((sample_dir / "metadata.json").read_text(encoding="utf-8"))
    quality = json.loads((sample_dir / "quality.json").read_text(encoding="utf-8"))
    wav_path = Path(metadata["files"]["wav"])
    if not wav_path.is_file():
        raise FileNotFoundError(f"WAV missing for {sample_id}: {wav_path}")

    signal, sample_rate = sf.read(str(wav_path), always_2d=False, dtype="float32")
    if signal.ndim > 1:
        signal = signal.mean(axis=1)
    duration = float(metadata["media"]["duration_seconds"] or (len(signal) / sample_rate))

    asr_words: list[dict] = []
    asr_csv = sample_dir / "asr_words.csv"
    if asr_csv.is_file():
        with asr_csv.open("r", encoding="utf-8", newline="") as handle:
            for row in csv.DictReader(handle):
                asr_words.append(
                    {
                        "word": row.get("word", ""),
                        "start": float(row["start"]),
                        "end": float(row["end"]),
                    }
                )
    else:
        for item in quality.get("asr_words", []):
            asr_words.append(
                {
                    "word": item["word"],
                    "start": float(item["start"]),
                    "end": float(item["end"]),
                }
            )

    frames = analyze_frames(signal, int(sample_rate))
    return {
        "sample_id": sample_id,
        "signal": np.asarray(signal, dtype=np.float32),
        "sample_rate": int(sample_rate),
        "duration": duration,
        "asr_words": asr_words,
        "n_asr": len(asr_words),
        "frames": frames,
        "speech_present_recorded": bool(quality.get("speech_present")),
        "activity_recorded": float(quality.get("signal_activity_ratio", 0.0)),
        "peak_recorded": float(quality.get("peak_dbfs", -240.0)),
        "wav_path": wav_path,
    }


def shared_limits(samples: list[dict]) -> tuple[float, tuple[float, float]]:
    wave_peak = max(float(np.max(np.abs(s["signal"]))) if len(s["signal"]) else 0.0 for s in samples)
    wave_peak = max(wave_peak, 1e-3)

    lo_candidates: list[float] = []
    hi_candidates: list[float] = []
    for s in samples:
        db = np.asarray(s["frames"]["frame_db"], dtype=np.float64)
        usable = db[db > -200.0]
        if len(usable):
            lo_candidates.append(float(np.percentile(usable, 1)))
            hi_candidates.append(float(np.percentile(usable, 99)))
        lo_candidates.extend([s["frames"]["noise_floor"], s["frames"]["paper_threshold"]])
        hi_candidates.extend([s["frames"]["noise_floor"], s["frames"]["paper_threshold"]])
    if not lo_candidates:
        energy_ylim = (-100.0, 0.0)
    else:
        lo = max(min(lo_candidates) - 8.0, -100.0)
        hi = max(hi_candidates) + 8.0
        if hi - lo < 25.0:
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 12.5, mid + 12.5
        energy_ylim = (lo, hi)
    return wave_peak, energy_ylim


def _x_map(t: float, t0: float, t1: float, x0: int, x1: int) -> int:
    if t1 <= t0:
        return x0
    ratio = (t - t0) / (t1 - t0)
    return int(round(x0 + ratio * (x1 - x0)))


def _y_map(v: float, v0: float, v1: float, y0: int, y1: int) -> int:
    if v1 <= v0:
        return y0
    ratio = (v - v0) / (v1 - v0)
    return int(round(y1 - ratio * (y1 - y0)))


def _draw_panel_frame(draw: ImageDraw.ImageDraw, box: tuple[int, int, int, int]) -> None:
    draw.rectangle(box, outline=(90, 90, 90), width=2)


def _draw_text(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    font: ImageFont.ImageFont,
    fill=(20, 20, 20),
    anchor: str = "lt",
) -> None:
    draw.text(xy, text, font=font, fill=fill, anchor=anchor)


def _downsample_wave(signal: np.ndarray, sample_rate: int, n_bins: int = 1800):
    n = len(signal)
    if n == 0:
        return np.asarray([0.0]), np.asarray([0.0]), np.asarray([0.0])
    edges = np.linspace(0, n, n_bins + 1, dtype=np.int64)
    lo = np.empty(n_bins, dtype=np.float64)
    hi = np.empty(n_bins, dtype=np.float64)
    tt = np.empty(n_bins, dtype=np.float64)
    for i in range(n_bins):
        seg = signal[edges[i] : max(edges[i] + 1, edges[i + 1])]
        lo[i] = float(np.min(seg))
        hi[i] = float(np.max(seg))
        tt[i] = 0.5 * (edges[i] + edges[i + 1]) / float(sample_rate)
    return tt, lo, hi


def draw_sample_block(
    img: Image.Image,
    draw: ImageDraw.ImageDraw,
    sample: dict,
    top: int,
    plot_left: int,
    plot_right: int,
    wave_peak: float,
    energy_ylim: tuple[float, float],
    fonts: dict,
    tag: str,
) -> int:
    duration = sample["duration"]
    y = top

    c_speech = 1 if sample["speech_present_recorded"] else 0
    title = (
        f"{tag}  {'存在可用语音' if c_speech == 1 else '无有效语音'}  |  "
        f"{sample['sample_id']}  |  c_speech = {c_speech}"
    )
    _draw_text(draw, (plot_left, y), title, fonts["title"], fill=(15, 15, 15))
    y += TITLE_BAND

    # ---- waveform panel ----
    h = PANEL_H["wave"]
    box = (plot_left, y, plot_right, y + h)
    draw.rectangle(box, fill=(250, 252, 255))
    _draw_panel_frame(draw, box)
    tt, lo, hi = _downsample_wave(sample["signal"], sample["sample_rate"])
    mid_y = y + h // 2
    draw.line([(plot_left, mid_y), (plot_right, mid_y)], fill=(180, 180, 180), width=1)
    for t, a, b in zip(tt, lo, hi):
        x = _x_map(float(t), 0.0, duration, plot_left, plot_right)
        y1 = _y_map(float(a), -wave_peak, wave_peak, y + 4, y + h - 4)
        y2 = _y_map(float(b), -wave_peak, wave_peak, y + 4, y + h - 4)
        draw.line([(x, y1), (x, y2)], fill=(31, 78, 121), width=1)
    peak = sample["peak_recorded"]
    peak_note = f"峰值 {peak:.1f} dBFS"
    if peak <= SILENCE_PEAK_DBFS + 1e-9:
        peak_note += f"  (<= {SILENCE_PEAK_DBFS:.0f} dBFS, 近似静音)"
    _draw_text(draw, (plot_right - 8, y + 8), peak_note, fonts["body"], fill=(60, 60, 60), anchor="rt")
    _draw_text(draw, (plot_left - 10, y + h // 2), "波形", fonts["ylabel"], fill=(40, 40, 40), anchor="rm")
    y += h + PANEL_GAP

    # ---- energy panel ----
    h = PANEL_H["energy"]
    box = (plot_left, y, plot_right, y + h)
    draw.rectangle(box, fill=(252, 252, 248))
    _draw_panel_frame(draw, box)
    frames = sample["frames"]
    times = frames["times"]
    frame_db = frames["frame_db"]
    e0, e1 = energy_ylim
    # active shading
    if len(times) > 1:
        dt = float(times[1] - times[0])
    else:
        dt = 0.01
    for t, active in zip(times, frames["active_mask"]):
        if not active:
            continue
        x0 = _x_map(float(t) - 0.5 * dt, 0.0, duration, plot_left, plot_right)
        x1 = _x_map(float(t) + 0.5 * dt, 0.0, duration, plot_left, plot_right)
        draw.rectangle([x0, y + 2, max(x0 + 1, x1), y + h - 2], fill=(200, 230, 180))

    # energy polyline
    pts = []
    for t, db in zip(times, frame_db):
        # clip extreme silent floors into shared window for display only
        db_plot = float(np.clip(db, e0, e1))
        pts.append(
            (
                _x_map(float(t), 0.0, duration, plot_left, plot_right),
                _y_map(db_plot, e0, e1, y + 4, y + h - 4),
            )
        )
    if len(pts) >= 2:
        draw.line(pts, fill=(42, 111, 151), width=2)

    # Reference lines: paper uses E_noise and E_noise+12 dB.
    # If the true noise floor falls below the shared axis (near-silent track),
    # annotate off-scale values and also mark the pipeline floor at -50 dBFS.
    noise = float(frames["noise_floor"])
    thr_paper = float(frames["paper_threshold"])
    thr_code = float(frames["code_threshold"])
    offscale = noise < e0 - 1e-6

    def _dashed_hline(value: float, color: tuple[int, int, int], label: str) -> None:
        yy = _y_map(float(np.clip(value, e0, e1)), e0, e1, y + 4, y + h - 4)
        x = plot_left
        while x < plot_right:
            draw.line([(x, yy), (min(x + 8, plot_right), yy)], fill=color, width=2)
            x += 14
        _draw_text(draw, (plot_right - 6, yy - 2), label, fonts["small"], fill=color, anchor="rb")

    if offscale:
        # True floors sit near -240 dB; draw the effective activity threshold instead.
        _dashed_hline(SILENCE_PEAK_DBFS, (120, 40, 140), "-50 dBFS")
        _draw_text(
            draw,
            (plot_left + 8, y + 10),
            f"E_noise≈{noise:.0f} dB, E_noise+12≈{thr_paper:.0f} dB（低于共享纵轴）; "
            f"活动判据取 max(-50, E_noise+12)={thr_code:.0f} dB",
            fonts["tiny"],
            fill=(120, 40, 140),
            anchor="lt",
        )
    else:
        _dashed_hline(noise, (193, 18, 31), "E_noise")
        _dashed_hline(thr_paper, (247, 127, 0), "E_noise+12dB")

    r_act = sample["activity_recorded"]
    note = f"r_act = {r_act:.3f}  (阈值 {MIN_ACTIVITY:.2f})"
    _draw_text(draw, (plot_left + 8, y + h - 10), note, fonts["small"], fill=(27, 67, 50), anchor="lb")
    _draw_text(draw, (plot_left - 10, y + h // 2), "短时能量", fonts["ylabel"], fill=(40, 40, 40), anchor="rm")
    for tick in np.linspace(e0, e1, 4):
        yy = _y_map(float(tick), e0, e1, y + 4, y + h - 4)
        _draw_text(draw, (plot_left - 6, yy), f"{tick:.0f}", fonts["tiny"], fill=(100, 100, 100), anchor="rm")
    y += h + PANEL_GAP

    # ---- ASR panel ----
    h = PANEL_H["asr"]
    box = (plot_left, y, plot_right, y + h)
    draw.rectangle(box, fill=(255, 255, 255))
    _draw_panel_frame(draw, box)
    words = sample["asr_words"]
    if not words:
        _draw_text(
            draw,
            ((plot_left + plot_right) // 2, y + h // 2),
            "无有效 ASR 词",
            fonts["body"],
            fill=(120, 120, 120),
            anchor="mm",
        )
    else:
        colors = [(168, 218, 220), (69, 123, 157)]
        text_colors = [(16, 42, 67), (255, 255, 255)]
        for idx, item in enumerate(words):
            start = max(0.0, float(item["start"]))
            end = min(duration, float(item["end"]))
            if end <= start:
                continue
            x0 = _x_map(start, 0.0, duration, plot_left, plot_right)
            x1 = _x_map(end, 0.0, duration, plot_left, plot_right)
            lane = idx % 2
            y0 = y + (18 if lane == 0 else 52) * DPI_SCALE // 2 + (8 * DPI_SCALE)
            # keep lanes inside panel
            y0 = y + int(0.18 * h) if lane == 0 else y + int(0.55 * h)
            bar_h = int(0.28 * h)
            x1 = max(x1, x0 + 2)
            draw.rounded_rectangle(
                [x0, y0, x1, y0 + bar_h],
                radius=4,
                fill=colors[lane],
                outline=(38, 70, 83),
                width=1,
            )
            label = str(item["word"])
            if len(label) > 10:
                label = label[:9] + "…"
            if x1 - x0 >= 28 * DPI_SCALE or len(words) <= 8:
                _draw_text(
                    draw,
                    ((x0 + x1) // 2, y0 + bar_h // 2),
                    label,
                    fonts["word"],
                    fill=text_colors[lane],
                    anchor="mm",
                )

    gate_color = (45, 106, 79) if c_speech == 1 else (155, 34, 38)
    gate_text = f"n_asr = {sample['n_asr']}   =>   c_speech = {c_speech}"
    _draw_text(draw, (plot_left + 8, y + 8), gate_text, fonts["body"], fill=gate_color)
    if c_speech == 0 and sample["n_asr"] >= 1 and sample["activity_recorded"] < MIN_ACTIVITY:
        _draw_text(
            draw,
            (plot_right - 8, y + 8),
            "仅有ASR词但活动不足，不进入词级定位",
            fonts["small"],
            fill=gate_color,
            anchor="rt",
        )
    _draw_text(draw, (plot_left - 10, y + h // 2), "ASR词段", fonts["ylabel"], fill=(40, 40, 40), anchor="rm")

    # shared time ticks
    n_ticks = 6
    for i in range(n_ticks + 1):
        t = duration * i / n_ticks
        x = _x_map(t, 0.0, duration, plot_left, plot_right)
        draw.line([(x, y + h), (x, y + h + 7)], fill=(80, 80, 80), width=2)
        _draw_text(draw, (x, y + h + 10), f"{t:.1f}", fonts["tick"], fill=(80, 80, 80), anchor="mt")
    _draw_text(
        draw,
        ((plot_left + plot_right) // 2, y + h + int(34 * DPI_SCALE)),
        "时间 (s)",
        fonts["small"],
        fill=(40, 40, 40),
        anchor="mt",
    )
    return y + h + TIME_BAND


def draw_figure(samples: list[dict], out_path: Path) -> Path:
    # Font sizes are in high-res canvas pixels; do not downscale on save.
    fonts = {
        "suptitle": _load_font(48, bold=True),
        "title": _load_font(36, bold=True),
        "body": _load_font(32),
        "small": _load_font(28),
        "tiny": _load_font(24),
        "word": _load_font(24),
        "ylabel": _load_font(30),
        "tick": _load_font(26),
    }
    wave_peak, energy_ylim = shared_limits(samples)

    row_h = (
        TITLE_BAND
        + PANEL_H["wave"]
        + PANEL_GAP
        + PANEL_H["energy"]
        + PANEL_GAP
        + PANEL_H["asr"]
        + TIME_BAND
    )
    height = MARGIN_TOP + row_h + ROW_GAP + row_h + MARGIN_BOTTOM
    img = Image.new("RGB", (W, height), (255, 255, 255))
    draw = ImageDraw.Draw(img)

    plot_left = MARGIN_X + 28 * DPI_SCALE
    plot_right = W - 36 * DPI_SCALE

    _draw_text(
        draw,
        (W // 2, int(18 * DPI_SCALE)),
        "语音可用性判断：短时信号活动与辅助转写有效词的联合门控",
        fonts["suptitle"],
        fill=(10, 10, 10),
        anchor="mt",
    )

    y = MARGIN_TOP
    tags = ["(a)", "(b)"]
    for i, sample in enumerate(samples):
        y = draw_sample_block(
            img,
            draw,
            sample,
            top=y,
            plot_left=plot_left,
            plot_right=plot_right,
            wave_peak=wave_peak,
            energy_ylim=energy_ylim,
            fonts=fonts,
            tag=tags[i],
        )
        if i == 0:
            y += ROW_GAP

    out_path.parent.mkdir(parents=True, exist_ok=True)
    img.save(out_path, format="PNG", optimize=True)
    return out_path


def main() -> None:
    speech = load_sample(SAMPLE_SPEECH)
    silent = load_sample(SAMPLE_SILENT)
    for sample in (speech, silent):
        act_rec = sample["activity_recorded"]
        act_new = sample["frames"]["activity_ratio"]
        gate = bool(act_new >= MIN_ACTIVITY and sample["n_asr"] >= MIN_ASR_WORDS)
        print(
            f"[ok] {sample['sample_id']}: speech_present={sample['speech_present_recorded']} "
            f"recomputed={gate}; r_act recorded={act_rec:.6f} recomputed={act_new:.6f}; "
            f"n_asr={sample['n_asr']}; peak={sample['peak_recorded']:.3f} dBFS"
        )
        if abs(act_rec - act_new) > 1e-6:
            print(f"[warn] activity mismatch for {sample['sample_id']}")
        if sample["speech_present_recorded"] != gate:
            print(f"[warn] speech_present mismatch for {sample['sample_id']}")

    path = draw_figure([speech, silent], OUT_PATH)
    print(f"[saved] {path}")
    print(f"[size]  {path.stat().st_size} bytes")


if __name__ == "__main__":
    main()
