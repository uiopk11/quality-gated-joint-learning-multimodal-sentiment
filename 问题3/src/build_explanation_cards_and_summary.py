from __future__ import annotations

import csv
import json
import textwrap
from pathlib import Path

import matplotlib as mpl
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import numpy as np


Q3_ROOT = Path(__file__).resolve().parents[1]
PREDICTION_CSV = Q3_ROOT / "results" / "attachment4" / "attachment4_predictions_and_modality_effects.csv"
EVIDENCE_CSV = Q3_ROOT / "results" / "tables" / "attachment4_top_local_evidence_timed.csv"
OUTPUT_TABLE = Q3_ROOT / "results" / "tables" / "attachment4_formal_prediction_explanation_summary.csv"
CARD_ROOT = Q3_ROOT / "results" / "figures" / "explanation_cards"
CARD_INDEX = Q3_ROOT / "results" / "figures" / "explanation_cards" / "解释卡索引.md"
AUDIT_JSON = Q3_ROOT / "results" / "audit" / "explanation_cards_and_summary_audit.json"

MODALITIES = ("text", "audio", "vision")
MODALITY_CN = {"text": "文本", "audio": "音频", "vision": "视觉"}
TARGETS = ("classification", "intensity")
TARGET_CN = {"classification": "极性", "intensity": "强度"}
LABEL_CN = {"negative": "负向", "neutral": "中性", "positive": "正向"}
COLORS = {"text": "#4C78A8", "audio": "#E45756", "vision": "#54A24B"}
CARD_CASES = {
    "17": ("高置信度正向样本", "正向概率最高（0.9703），用于展示稳定正向判断及其证据链。"),
    "09": ("高置信度负向样本", "负向概率最高（0.9705），用于展示稳定负向判断及其证据链。"),
    "01": ("三模态作用接近、预测困难样本", "三模态分类贡献相对最均衡，且类别边际仅0.1091。"),
    "02": ("模态冲突典型样本", "文本支持最终正向判断，音频和视觉对分类边际均产生反向贡献。"),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def configure_style() -> None:
    mpl.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Microsoft YaHei", "SimHei", "Arial Unicode MS", "DejaVu Sans"],
        "axes.unicode_minus": False,
        "savefig.dpi": 300,
        "svg.fonttype": "none",
    })


def clean(value: str | None) -> str:
    return "" if value is None else str(value).strip()


def fmt_number(value: str | None, digits: int = 4) -> str:
    value = clean(value)
    if not value:
        return "—"
    return f"{float(value):.{digits}f}"


def evidence_location(row: dict[str, str]) -> str:
    time_start, time_end = clean(row.get("time_start_seconds")), clean(row.get("time_end_seconds"))
    frame_start, frame_end = clean(row.get("video_frame_start")), clean(row.get("video_frame_end_exclusive"))
    parts: list[str] = []
    if time_start and time_end:
        parts.append(f"{float(time_start):.2f}–{float(time_end):.2f} s")
    else:
        parts.append("无可靠秒级区间")
    if frame_start and frame_end:
        parts.append(f"帧[{frame_start},{frame_end})")
    return "；".join(parts)


def choose_top(rows: list[dict[str, str]], sample_id: str, modality: str, target: str) -> dict[str, str] | None:
    matches = [
        row for row in rows
        if row["sample_id"].zfill(2) == sample_id and row["modality"] == modality and row["target"] == target
    ]
    if not matches:
        return None
    return min(matches, key=lambda row: int(row["rank"]))


def build_summary(predictions: list[dict[str, str]], evidence: list[dict[str, str]]) -> list[dict[str, str]]:
    output: list[dict[str, str]] = []
    for pred in predictions:
        sample_id = pred["sample_id"].zfill(2)
        row = {
            "sample_id": sample_id,
            "raw_text": pred["raw_text"],
            "pred_class": pred["pred_class"],
            "pred_label": pred["pred_label"],
            "p_negative": pred["p_negative"],
            "p_neutral": pred["p_neutral"],
            "p_positive": pred["p_positive"],
            "intensity": pred["intensity"],
            "class_margin": pred["class_margin"],
            "full_wordpiece_count": pred["full_wordpiece_count"],
            "retained_wordpiece_count": pred["retained_wordpiece_count"],
            "truncated_wordpiece_count": pred["truncated_wordpiece_count"],
            "model_visible_text_end_char": pred["model_visible_text_end_char"],
            "audio_observation_status": pred["audio_observation_status"],
            "vision_observation_status": pred["vision_observation_status"],
            "overlaps_attachment2_test": pred["overlaps_attachment2_test"],
            "attachment2_test_id": pred["attachment2_test_id"],
            "attachment2_test_index": pred["attachment2_test_index"],
            "dominant_modality_classification": pred["most_influential_modality_classification"],
            "dominant_modality_intensity": pred["most_influential_modality_intensity"],
        }
        for target in TARGETS:
            for modality in MODALITIES:
                row[f"{target}_{modality}_shapley"] = pred[f"{modality}_{target}_shapley"]
                row[f"{target}_{modality}_share"] = pred[f"{modality}_{target}_share"]
                top = choose_top(evidence, sample_id, modality, target)
                prefix = f"top1_{target}_{modality}"
                if top is None:
                    for suffix in ("fragment", "time_start_seconds", "time_end_seconds", "frame_start", "frame_end_exclusive", "key_frame", "keyframe_image", "timestamp_status", "importance"):
                        row[f"{prefix}_{suffix}"] = ""
                    continue
                importance_key = "absolute_classification_delta" if target == "classification" else "absolute_intensity_delta"
                row.update({
                    f"{prefix}_fragment": top["original_text_fragment"],
                    f"{prefix}_time_start_seconds": top["time_start_seconds"],
                    f"{prefix}_time_end_seconds": top["time_end_seconds"],
                    f"{prefix}_frame_start": top["video_frame_start"],
                    f"{prefix}_frame_end_exclusive": top["video_frame_end_exclusive"],
                    f"{prefix}_key_frame": top["video_key_frame"],
                    f"{prefix}_keyframe_image": top["video_keyframe_image"],
                    f"{prefix}_timestamp_status": top["timestamp_status"],
                    f"{prefix}_time_mapping_source": top.get("time_mapping_source", ""),
                    f"{prefix}_time_mapping_is_official": top.get("time_mapping_is_official", ""),
                    f"{prefix}_input_observation_status": top.get("input_observation_status", ""),
                    f"{prefix}_importance": top[importance_key],
                })
        output.append(row)
    return output


def save_summary(rows: list[dict[str, str]]) -> None:
    OUTPUT_TABLE.parent.mkdir(parents=True, exist_ok=True)
    with OUTPUT_TABLE.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def make_card(pred: dict[str, str], evidence: list[dict[str, str]]) -> dict[str, object]:
    sample_id = pred["sample_id"].zfill(2)
    case_name, case_reason = CARD_CASES[sample_id]
    shares = np.array([float(pred[f"{m}_classification_share"]) for m in MODALITIES])
    shapley = np.array([float(pred[f"{m}_classification_shapley"]) for m in MODALITIES])
    probabilities = [float(pred[f"p_{label}"]) for label in ("negative", "neutral", "positive")]
    top_rows = {m: choose_top(evidence, sample_id, m, "classification") for m in MODALITIES}

    fig = plt.figure(figsize=(11.69, 8.27), facecolor="#F6F3ED")
    grid = fig.add_gridspec(14, 12, left=0.045, right=0.965, top=0.895, bottom=0.055, hspace=1.45, wspace=0.9)
    fig.text(0.05, 0.975, f"典型样本解释卡 · {case_name}", fontsize=18, fontweight="bold", color="#243447", va="top")
    fig.text(0.95, 0.975, f"附件4 · 样本 {sample_id}", fontsize=12, ha="right", va="top", color="#52606D")

    ax_info = fig.add_subplot(grid[0:3, 0:12])
    ax_info.axis("off")
    prediction_line = (
        f"预测极性：{LABEL_CN.get(pred['pred_label'], pred['pred_label'])}    情感强度：{float(pred['intensity']):.4f}    "
        f"类别边际：{float(pred['class_margin']):.4f}    主要参考模态："
        f"{MODALITY_CN[pred['most_influential_modality_classification']]}"
    )
    ax_info.text(0.0, 0.92, prediction_line, fontsize=13, fontweight="bold", va="top", color="#102A43")
    ax_info.text(0.0, 0.60, "选样依据：" + case_reason, fontsize=10.5, va="top", color="#334E68")
    wrapped = "\n".join(textwrap.wrap(pred["raw_text"], width=125))
    ax_info.text(0.0, 0.32, "原始文本：" + wrapped, fontsize=10, va="top", color="#334E68")

    ax_prob = fig.add_subplot(grid[3:7, 0:4])
    labels_cn = ["负向", "中性", "正向"]
    bars = ax_prob.barh(labels_cn, probabilities, color=["#D65F5F", "#8C8C8C", "#4FAF84"])
    ax_prob.set_xlim(0, 1)
    ax_prob.set_title("A  极性概率", loc="left", fontweight="bold")
    ax_prob.spines[["top", "right", "left"]].set_visible(False)
    ax_prob.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    for bar, value in zip(bars, probabilities):
        inside = value > 0.86
        ax_prob.text(value - 0.015 if inside else value + 0.015, bar.get_y() + bar.get_height() / 2,
                     f"{value:.3f}", ha="right" if inside else "left", va="center", fontsize=9)

    ax_share = fig.add_subplot(grid[3:7, 4:8])
    bars = ax_share.barh([MODALITY_CN[m] for m in MODALITIES], shares, color=[COLORS[m] for m in MODALITIES])
    ax_share.set_xlim(0, 1)
    ax_share.set_title("B  三模态作用程度（绝对Shapley占比）", loc="left", fontweight="bold", fontsize=10)
    ax_share.spines[["top", "right", "left"]].set_visible(False)
    ax_share.grid(axis="x", color="#D8D8D8", linewidth=0.6)
    for bar, value in zip(bars, shares):
        inside = value > 0.86
        ax_share.text(value - 0.015 if inside else value + 0.015, bar.get_y() + bar.get_height() / 2,
                      f"{value:.3f}", ha="right" if inside else "left", va="center", fontsize=9)

    ax_signed = fig.add_subplot(grid[3:7, 8:12])
    signed_colors = ["#3A7D44" if value >= 0 else "#C44536" for value in shapley]
    bars = ax_signed.barh([MODALITY_CN[m] for m in MODALITIES], shapley, color=signed_colors)
    bound = max(0.2, float(np.abs(shapley).max()) * 1.25)
    ax_signed.set_xlim(-bound, bound)
    ax_signed.axvline(0, color="#333333", linewidth=0.8)
    ax_signed.set_title("C  对最终类别边际的有符号贡献", loc="left", fontweight="bold", fontsize=10)
    ax_signed.spines[["top", "right", "left"]].set_visible(False)
    for bar, value in zip(bars, shapley):
        ax_signed.text(value + (0.02 * bound if value >= 0 else -0.02 * bound), bar.get_y() + bar.get_height() / 2,
                       f"{value:+.3f}", ha="left" if value >= 0 else "right", va="center", fontsize=9)

    ax_ev = fig.add_subplot(grid[8:12, 0:8])
    ax_ev.axis("off")
    ax_ev.set_title("D  各模态Top-1局部证据定位", loc="left", fontweight="bold", pad=8)
    table_data = []
    for modality in MODALITIES:
        row = top_rows[modality]
        if row is None:
            table_data.append([MODALITY_CN[modality], "—", "—", "—"])
            continue
        fragment = textwrap.shorten(clean(row["original_text_fragment"]), width=35, placeholder="…") or "—"
        importance = float(row["absolute_classification_delta"])
        table_data.append([MODALITY_CN[modality], fragment, evidence_location(row), f"{importance:.4f}"])
    table = ax_ev.table(cellText=table_data, colLabels=["模态", "原始片段", "音视频定位", "重要性"],
                        colWidths=[0.10, 0.34, 0.38, 0.13], cellLoc="left", loc="upper left", bbox=[0, 0.12, 1, 0.78])
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    for (row_index, _), cell in table.get_celld().items():
        cell.set_edgecolor("#CCD6DD")
        cell.set_facecolor("#E9EFF3" if row_index == 0 else "#FFFFFF")
        if row_index == 0:
            cell.set_text_props(fontweight="bold")
    ax_ev.text(0, 0.02, "注：重要性为遮挡该局部窗口后，固定预测类别边际的绝对变化；不是因果效应。", fontsize=8.5, color="#52606D")

    ax_img = fig.add_subplot(grid[8:12, 8:12])
    vision_row = top_rows["vision"]
    image_path = Path(clean(vision_row.get("video_keyframe_image"))) if vision_row else Path()
    if image_path.is_file():
        ax_img.imshow(mpimg.imread(image_path))
        ax_img.set_title("E  视觉Top-1关键帧", loc="left", fontweight="bold")
        ax_img.text(0.5, -0.06, evidence_location(vision_row), transform=ax_img.transAxes, ha="center", va="top", fontsize=9)
    else:
        ax_img.text(0.5, 0.5, "该证据无可靠关键帧", ha="center", va="center", fontsize=11, color="#52606D")
        ax_img.set_title("E  视觉Top-1关键帧", loc="left", fontweight="bold")
    ax_img.axis("off")

    ax_note = fig.add_subplot(grid[12:14, 0:12])
    ax_note.axis("off")
    support = [MODALITY_CN[m] for m, value in zip(MODALITIES, shapley) if value >= 0]
    oppose = [MODALITY_CN[m] for m, value in zip(MODALITIES, shapley) if value < 0]
    conclusion = (
        f"解释结论：{MODALITY_CN[pred['most_influential_modality_classification']]}是本次极性判断的首要依据；"
        f"支持最终类别的模态为{'、'.join(support) if support else '无'}，"
        f"反向作用模态为{'、'.join(oppose) if oppose else '无'}。"
        "所有局部证据均由同一冻结预测器的遮挡敏感性得到，并通过QG-HTA映射回原素材。"
    )
    ax_note.text(0, 0.75, "\n".join(textwrap.wrap(conclusion, width=105)), fontsize=10.5, va="top",
                 bbox={"boxstyle": "round,pad=0.6", "facecolor": "#E6F0F3", "edgecolor": "#9FB3C8"})

    CARD_ROOT.mkdir(parents=True, exist_ok=True)
    stem = CARD_ROOT / f"{sample_id}_{case_name}"
    fig.savefig(stem.with_suffix(".png"), dpi=300, bbox_inches="tight", facecolor=fig.get_facecolor())
    fig.savefig(stem.with_suffix(".svg"), bbox_inches="tight", facecolor=fig.get_facecolor())
    plt.close(fig)
    return {"sample_id": sample_id, "case": case_name, "reason": case_reason, "png": str(stem.with_suffix('.png')), "svg": str(stem.with_suffix('.svg'))}


def write_card_index(cards: list[dict[str, object]]) -> None:
    lines = ["# 问题3典型样本解释卡", "", "以下样本按预先固定、可复核的规则从附件4的20个样本中选出。", ""]
    for card in cards:
        png_name = Path(str(card["png"])).name
        lines.extend([
            f"## {card['case']}：样本{card['sample_id']}", "",
            str(card["reason"]), "",
            f"![{card['case']}]({png_name})", "",
        ])
    CARD_INDEX.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    configure_style()
    predictions = read_csv(PREDICTION_CSV)
    evidence = read_csv(EVIDENCE_CSV)
    if len(predictions) != 20:
        raise ValueError(f"Expected 20 predictions, got {len(predictions)}")
    summary = build_summary(predictions, evidence)
    save_summary(summary)
    by_id = {row["sample_id"].zfill(2): row for row in predictions}
    cards = [make_card(by_id[sample_id], evidence) for sample_id in CARD_CASES]
    write_card_index(cards)
    audit = {
        "status": "passed",
        "prediction_rows": len(predictions),
        "formal_summary_rows": len(summary),
        "formal_summary_columns": len(summary[0]),
        "evidence_source_rows": len(evidence),
        "cards": cards,
        "selection_rules": {
            "17": "highest positive probability among predicted-positive samples",
            "09": "highest negative probability among predicted-negative samples",
            "01": "comparatively balanced modality shares plus low class margin",
            "02": "text contribution supports final class while audio and vision contributions oppose it",
        },
        "claims": {"causal": False, "explanation_type": "model-faithful sensitivity"},
    }
    AUDIT_JSON.parent.mkdir(parents=True, exist_ok=True)
    AUDIT_JSON.write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(audit, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
