"""Build editable PPTX for fig:q2_train_infer."""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.oxml import parse_xml
from pptx.util import Inches, Pt

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
FIG_DIR = ROOT_DIR / "Figure"
PPTX_PATH = ROOT_DIR / "q2_train_infer.pptx"
PNG_PATH = FIG_DIR / "q2_train_infer.png"

SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def set_font(run, *, size=12, bold=False, color="#222222", cn="Microsoft YaHei", latin="Cambria Math"):
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    run.font.name = latin
    rPr = run._r.get_or_add_rPr()
    for tag, face in (("a:latin", latin), ("a:ea", cn)):
        node = rPr.find(qn(tag))
        if node is None:
            rPr.append(
                parse_xml(
                    f'<{tag} xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{face}"/>'
                )
            )
        else:
            node.set("typeface", face)


def fill_text(shape_or_tf, text, *, size=12, bold=False, color="#222222", align=PP_ALIGN.CENTER):
    tf = shape_or_tf if hasattr(shape_or_tf, "paragraphs") else shape_or_tf.text_frame
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set("anchor", "ctr")
    except Exception:
        pass
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        # clear existing runs on first para
        if i == 0:
            for r in list(p.runs):
                r.text = ""
            if p.runs:
                run = p.runs[0]
                run.text = line
            else:
                run = p.add_run()
                run.text = line
        else:
            run = p.add_run()
            run.text = line
        set_font(run, size=size, bold=bold if i == 0 else bold, color=color)


def add_box(slide, left, top, width, height, text, *, fill="#FFFFFF", line="#333333", size=12, bold=False, color="#222222", line_pt=1.4):
    sh = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    sh.fill.solid()
    sh.fill.fore_color.rgb = rgb(fill)
    sh.line.color.rgb = rgb(line)
    sh.line.width = Pt(line_pt)
    try:
        sh.adjustments[0] = 0.12
    except Exception:
        pass
    fill_text(sh, text, size=size, bold=bold, color=color)
    return sh


def add_tb(slide, left, top, width, height, text, *, size=12, bold=False, color="#222222", align=PP_ALIGN.LEFT):
    box = slide.shapes.add_textbox(left, top, width, height)
    fill_text(box, text, size=size, bold=bold, color=color, align=align)
    return box


def add_arrow(slide, x1, y1, x2, y2, *, color="#444444", width_pt=1.6):
    c = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    c.line.color.rgb = rgb(color)
    c.line.width = Pt(width_pt)
    ln = c.line._get_or_add_ln()
    for tag in ("a:tailEnd", "a:headEnd"):
        node = ln.find(qn(tag))
        if node is not None:
            ln.remove(node)
    ln.append(
        parse_xml(
            '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'
        )
    )
    return c


def slide_notes(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_tb(
        s,
        Inches(0.35),
        Inches(0.2),
        Inches(12.6),
        Inches(0.4),
        "图 q2_train_infer｜训练与推理计算边界（编辑说明）",
        size=20,
        bold=True,
        color="#1E3A5F",
    )
    cards = [
        (
            "图要突出的计算边界",
            "训练：同一样本 clean/damaged 双视图配对；\n"
            "双任务损失 + 高层一致性 + 低层恢复；\n"
            "验证集选模。\n\n"
            "推理：单视图当前输入；冻结 μ,σ 与权重；\n"
            "只走编码→EMT→双头预测。",
        ),
        (
            "训练专属（橙色框）",
            "• projector / predictor → L_high\n"
            "• 恢复头 D_m → L_low（仅 S=P·M·O）\n"
            "• 真值 y,c；人工掩码 M；clean 参考\n"
            "• 恢复结果不回填输入、不做二阶段重预测",
        ),
        (
            "推理禁止出现",
            "虚线淡化框中的内容均不执行：\n"
            "clean 参考、真值、M/S、\n"
            "projector/predictor、恢复头、\n"
            "恢复回填再预测。",
        ),
        (
            "左右统计依赖",
            "训练集拟合 μ,σ（有效非零 A/V）\n"
            "↓ 冻结后复用\n"
            "valid/test/附件三推理不再重估",
        ),
        (
            "文件位置",
            "code/draw_q2_train_infer.py\n"
            "code/build_q2_train_infer_pptx.py\n"
            "Figure/q2_train_infer.png|.svg|.pdf\n"
            "本 PPT：第2页可编辑，第3页为 PNG 对照",
        ),
        (
            "总损失",
            "L_JPI-DLFR = L_JPI^c + L_JPI^d\n"
            "            + L_high + L_low\n\n"
            "拼接维数提醒（与编码图一致）：\n"
            "768+16+32+3×128=1200（在 f_i，不在本图主路径）",
        ),
    ]
    lefts = [0.35, 4.55, 8.75]
    tops = [0.8, 4.05]
    for i, (t, b) in enumerate(cards):
        col, row = i % 3, i // 3
        add_box(
            s,
            Inches(lefts[col]),
            Inches(tops[row]),
            Inches(3.95),
            Inches(2.95),
            "",
            fill="#F8FAFC",
            line="#94A3B8",
            size=11,
        )
        add_tb(s, Inches(lefts[col] + 0.12), Inches(tops[row] + 0.1), Inches(3.7), Inches(0.35), t, size=14, bold=True, color="#1E3A5F")
        add_tb(s, Inches(lefts[col] + 0.12), Inches(tops[row] + 0.5), Inches(3.7), Inches(2.3), b, size=12, color="#334155")


def slide_editable(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_tb(s, Inches(0.25), Inches(0.08), Inches(12.8), Inches(0.32), "训练与推理计算边界（可编辑形状页）", size=16, bold=True, color="#0F172A")

    # stats
    add_box(s, Inches(0.2), Inches(0.45), Inches(1.55), Inches(6.7), "统计依赖\n\n训练集拟合\nμ, σ\n\n↓\n\n冻结后复用\n\nvalid/test\n附件三\n不再重估", fill="#FFFBEB", line="#B45309", size=12, bold=True, color="#92400E")

    # train panel label
    add_tb(s, Inches(1.95), Inches(0.45), Inches(5), Inches(0.28), "上：训练路径", size=14, bold=True, color="#1D4ED8")
    add_box(s, Inches(1.95), Inches(0.8), Inches(1.7), Inches(0.85), "完整视图 clean\nv=c", fill="#DBEAFE", line="#1D4ED8", size=13, bold=True, color="#1D4ED8")
    add_box(s, Inches(1.95), Inches(1.8), Inches(1.7), Inches(0.85), "缺失视图 damaged\nv=d，掩码 M", fill="#FEE2E2", line="#B91C1C", size=13, bold=True, color="#B91C1C")
    add_box(s, Inches(3.9), Inches(1.0), Inches(2.5), Inches(1.5), "共享主干（训练）\nBERT/LSTM→EMT→h\n共享参数 θ", fill="#E8EEF7", line="#1E3A5F", size=13, bold=True, color="#1E3A5F")
    add_arrow(s, Inches(3.65), Inches(1.2), Inches(3.9), Inches(1.4), color="#1D4ED8")
    add_arrow(s, Inches(3.65), Inches(2.2), Inches(3.9), Inches(2.0), color="#B91C1C")

    add_box(s, Inches(6.65), Inches(0.8), Inches(1.9), Inches(0.75), "双任务头\nĉ, ŷ（两视图）", fill="#ECFDF5", line="#047857", size=13, bold=True, color="#047857")
    add_box(s, Inches(8.75), Inches(0.8), Inches(2.2), Inches(0.75), "任务损失\nL_JPI^c + L_JPI^d", fill="#ECFDF5", line="#047857", size=13, bold=True, color="#047857")
    add_box(s, Inches(6.65), Inches(1.75), Inches(1.9), Inches(1.0), "高层辅助（仅训练）\nprojector/predictor\nL_high", fill="#FEF3C7", line="#D97706", size=12, bold=True, color="#B45309")
    add_box(s, Inches(8.75), Inches(1.75), Inches(2.2), Inches(1.0), "低层恢复（仅训练）\nR̂=D_m(Z)\n仅在 S=P·M·O", fill="#FEF3C7", line="#D97706", size=12, bold=True, color="#B45309")
    add_arrow(s, Inches(6.4), Inches(1.55), Inches(6.65), Inches(1.2), color="#047857")
    add_arrow(s, Inches(6.4), Inches(1.9), Inches(6.65), Inches(2.1), color="#D97706")

    add_box(
        s,
        Inches(1.95),
        Inches(2.95),
        Inches(9.0),
        Inches(0.55),
        "L_JPI-DLFR = L_JPI^c + L_JPI^d + L_high + L_low    ｜    不回填输入、不二阶段重预测\n监督：y,c · M · S · clean参考（停梯度）   ｜   选模：Acc→Macro-F1→MAE",
        fill="#FFFFFF",
        line="#1E3A5F",
        size=12,
        color="#1E3A5F",
    )

    # infer
    add_tb(s, Inches(1.95), Inches(3.7), Inches(5), Inches(0.28), "下：推理路径", size=14, bold=True, color="#0F766E")
    add_box(s, Inches(1.95), Inches(4.1), Inches(1.7), Inches(0.95), "当前可用输入\n已有缺失原位\n不再追加 M", fill="#CCFBF1", line="#0F766E", size=12, bold=True, color="#0F766E")
    add_box(s, Inches(3.85), Inches(4.25), Inches(1.1), Inches(0.65), "冻结\nμ, σ", fill="#FFF7ED", line="#C2410C", size=13, bold=True, color="#C2410C")
    add_box(s, Inches(5.2), Inches(4.1), Inches(2.5), Inches(0.95), "共享主干（推理）\nBERT/LSTM→EMT→h\n冻结 θ", fill="#E8EEF7", line="#1E3A5F", size=13, bold=True, color="#1E3A5F")
    add_box(s, Inches(8.0), Inches(4.15), Inches(1.4), Inches(0.75), "极性\nĉ=argmax", fill="#ECFDF5", line="#047857", size=13, bold=True, color="#047857")
    add_box(s, Inches(9.55), Inches(4.15), Inches(1.4), Inches(0.75), "强度\nŷ 原值", fill="#ECFDF5", line="#047857", size=13, bold=True, color="#047857")
    add_arrow(s, Inches(3.65), Inches(4.55), Inches(3.85), Inches(4.55), color="#C2410C")
    add_arrow(s, Inches(4.95), Inches(4.55), Inches(5.2), Inches(4.55), color="#0F766E")
    add_arrow(s, Inches(7.7), Inches(4.55), Inches(8.0), Inches(4.55), color="#047857")

    add_box(
        s,
        Inches(1.95),
        Inches(5.3),
        Inches(9.0),
        Inches(1.7),
        "推理时不执行（虚线淡化）\n"
        "完整参考 clean ｜ 真值 y,c ｜ 掩码 M / 支持 S ｜ projector/predictor ｜ 恢复头 D_m ｜ 恢复回填再预测",
        fill="#F5F5F4",
        line="#A8A29E",
        size=13,
        color="#78716C",
    )


def slide_ref(prs):
    s = prs.slides.add_slide(prs.slide_layouts[6])
    add_tb(s, Inches(0.3), Inches(0.12), Inches(12.5), Inches(0.3), "PNG 参考对照｜正式编辑请回第2页", size=14, bold=True, color="#1E3A5F")
    if PNG_PATH.exists():
        s.shapes.add_picture(str(PNG_PATH), Inches(0.35), Inches(0.5), width=Inches(12.6))
    else:
        add_tb(s, Inches(1), Inches(3), Inches(11), Inches(1), f"缺少 {PNG_PATH}", size=14, color="#AA0000")


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_PATH.exists():
        from draw_q2_train_infer import draw

        draw()
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slide_notes(prs)
    slide_editable(prs)
    slide_ref(prs)
    prs.save(PPTX_PATH)
    print(f"wrote {PPTX_PATH}")


if __name__ == "__main__":
    main()
