"""Build an editable PowerPoint for fig:q2_encoder_interface.

Outputs:
  ../q2_encoder_interface.pptx

Slide 1 — structure notes (how to read / edit)
Slide 2 — fully editable native shapes (main working slide)
Slide 3 — PNG reference for visual target
"""

from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import PP_ALIGN, MSO_ANCHOR
from pptx.oxml.ns import qn
from pptx.util import Inches, Pt, Emu
from pptx.oxml import parse_xml

CODE_DIR = Path(__file__).resolve().parent
ROOT_DIR = CODE_DIR.parent
FIG_DIR = ROOT_DIR / "Figure"
PPTX_PATH = ROOT_DIR / "q2_encoder_interface.pptx"
PNG_PATH = FIG_DIR / "q2_encoder_interface.png"

# Widescreen landscape, convenient for paper figures
SLIDE_W = Inches(13.333)
SLIDE_H = Inches(7.5)


def rgb(hex_color: str) -> RGBColor:
    h = hex_color.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def set_run_font(run, *, size=13, bold=False, color="#222222", name="Microsoft YaHei", latin="Cambria Math"):
    """CN uses YaHei; Latin/math symbols prefer Cambria Math for academic look."""
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = rgb(color)
    run.font.name = latin
    rPr = run._r.get_or_add_rPr()
    # Latin
    latin_node = rPr.find(qn("a:latin"))
    if latin_node is None:
        rPr.append(
            parse_xml(
                f'<a:latin xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{latin}"/>'
            )
        )
    else:
        latin_node.set("typeface", latin)
    # East Asian
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = parse_xml(
            f'<a:ea xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" typeface="{name}"/>'
        )
        rPr.append(ea)
    else:
        ea.set("typeface", name)


def add_textbox(slide, left, top, width, height, text, *, size=11, bold=False, color="#222222", align=PP_ALIGN.CENTER, valign=MSO_ANCHOR.MIDDLE):
    box = slide.shapes.add_textbox(left, top, width, height)
    tf = box.text_frame
    tf.word_wrap = True
    tf.auto_size = None
    try:
        tf._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[valign])
    except Exception:
        pass
    # Clear default and fill lines
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        run = p.add_run()
        run.text = line
        set_run_font(run, size=size, bold=bold, color=color)
    return box


def add_round_rect(slide, left, top, width, height, text, *, fill="#FFFFFF", line="#333333", size=10, bold=False, color="#222222", line_pt=1.25):
    shape = slide.shapes.add_shape(MSO_SHAPE.ROUNDED_RECTANGLE, left, top, width, height)
    shape.fill.solid()
    shape.fill.fore_color.rgb = rgb(fill)
    shape.line.color.rgb = rgb(line)
    shape.line.width = Pt(line_pt)
    # Softer corners
    try:
        shape.adjustments[0] = 0.15
    except Exception:
        pass
    tf = shape.text_frame
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set("anchor", "ctr")
    except Exception:
        pass
    lines = text.split("\n")
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        run = p.add_run()
        run.text = line
        set_run_font(run, size=size, bold=bold if i == 0 else bold, color=color)
    return shape


def add_arrow(slide, x1, y1, x2, y2, *, color="#444444", width_pt=1.5):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width_pt)
    # Add arrow head via line end
    ln = connector.line._get_or_add_ln()
    # Remove existing tailEnd/headEnd if any, then set
    for tag in ("a:tailEnd", "a:headEnd"):
        node = ln.find(qn(tag))
        if node is not None:
            ln.remove(node)
    tail = parse_xml(
        '<a:tailEnd xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" type="triangle" w="med" len="med"/>'
    )
    ln.append(tail)
    return connector


def add_dashed_line(slide, x1, y1, x2, y2, *, color="#8A5728", width_pt=1.25):
    connector = slide.shapes.add_connector(MSO_CONNECTOR.STRAIGHT, x1, y1, x2, y2)
    connector.line.color.rgb = rgb(color)
    connector.line.width = Pt(width_pt)
    connector.line.dash_style = 3  # dash
    return connector


def build_notes_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])  # blank

    add_textbox(
        slide,
        Inches(0.4),
        Inches(0.25),
        Inches(12.5),
        Inches(0.45),
        "图 q2_encoder_interface｜三模态编码与统一投影的张量接口（编辑说明）",
        size=20,
        bold=True,
        color="#1E3A5F",
        align=PP_ALIGN.LEFT,
    )

    blocks = [
        (
            "这张图在论文里回答什么",
            "区别于总体架构图，本图只讲编码接口：\n"
            "• 三分支均输出「局部序列 H + 整体摘要 u」\n"
            "• 局部序列投影后进后续 EMT；投影后的三个摘要 stack 成 G_i^0\n"
            "• 投影前的 u^T / u^A / u^V 另走旁路，保留到预测拼接 f_i\n"
            "• 必须标有效长度 L_i：padding 不删时间步；正文内部零行仍占原位",
        ),
        (
            "五列结构（从左到右）",
            "① 输入：text_bert[3×50]、音频 50×74、视觉 50×35\n"
            "② 模态编码：BERT(微调) / LSTM_A(h=16) / LSTM_V(h=32)\n"
            "③ 分裂输出：局部矩阵 H（按 L_i）＋整体摘要 u\n"
            "④ 模态投影 φ_m：共同维 d=128；局部 +PE → Z；摘要 → g\n"
            "⑤ 接口输出：Z→EMT；g→stack→G_i^0 ∈ R^{3×d}；u 旁路→拼接",
        ),
        (
            "三行模态（从上到下）",
            "文本（蓝）：u = CLS；H = 去 CLS 正文；d_T=768\n"
            "音频（绿）：u = 最后有效状态；H = 逐位置状态；d_A=16；按 L_i pack\n"
            "视觉（橙）：u = 最后有效状态；H = 逐位置状态；d_V=32；按 L_i pack",
        ),
        (
            "关键公式与维数",
            "L_i = Σ attention_it − 2（正文 WordPiece 数，不含 SEP/PAD）\n"
            "Z_i^{m,0} = φ_m(H_i^m) + PE ， g_i^{m,0} = φ_m(u_i^m)\n"
            "G_i^0 = stack(g^T, g^A, g^V)\n"
            "f_i = [u^T; u^A; u^V; vec(G_i^K)] ，维数 768+16+32+3×128 = 1200",
        ),
        (
            "本文件夹怎么用",
            "code/     绘图脚本、本 PPT 生成脚本\n"
            "Figure/   PNG / SVG / PDF 成品图\n"
            "本文件    第2页是可编辑形状，请在第2页改文字/颜色/位置\n"
            "第3页是 PNG 参考对照，定稿后可导出第2页或直接用 Figure 里的 PNG",
        ),
        (
            "建议你怎么改",
            "• 先改第2页文字与框位置，不要直接在第3页 PNG 上硬改\n"
            "• 箭头是连接线，可拖两端对齐；虚线旁路表示 u 不进 G_i^0\n"
            "• 改完若要同步论文插图：另存/导出图片，覆盖 Figure/q2_encoder_interface.png\n"
            "• 或改完后运行 code/draw_q2_encoder_interface.py 重新出位图（会覆盖 Figure）",
        ),
    ]

    # 2x3 card grid
    lefts = [0.4, 4.55, 8.7]
    tops = [0.9, 4.1]
    card_w, card_h = 3.95, 2.95
    for idx, (title, body) in enumerate(blocks):
        col, row = idx % 3, idx // 3
        left = Inches(lefts[col])
        top = Inches(tops[row])
        add_round_rect(
            slide,
            left,
            top,
            Inches(card_w),
            Inches(card_h),
            "",
            fill="#F7FAFC",
            line="#90A4AE",
            size=10,
        )
        add_textbox(slide, left + Inches(0.12), top + Inches(0.08), Inches(card_w - 0.2), Inches(0.35), title, size=15, bold=True, color="#1E3A5F", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP)
        add_textbox(slide, left + Inches(0.12), top + Inches(0.42), Inches(card_w - 0.2), Inches(card_h - 0.55), body, size=12, color="#333333", align=PP_ALIGN.LEFT, valign=MSO_ANCHOR.TOP)


def build_editable_diagram_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    add_textbox(
        slide,
        Inches(0.25),
        Inches(0.08),
        Inches(12.8),
        Inches(0.32),
        "三模态编码接口（可编辑形状页）｜请在本页直接改框内文字与位置",
        size=16,
        bold=True,
        color="#1E3A5F",
        align=PP_ALIGN.LEFT,
    )

    # L_i banner
    add_round_rect(
        slide,
        Inches(0.25),
        Inches(0.42),
        Inches(12.8),
        Inches(0.48),
        "有效长度 L_i = Σ attention_it − 2（正文 WordPiece 数）：含内部零行/UNK 原位；不含 SEP/PAD。批次补齐只由 mask 屏蔽，不删除时间步。",
        fill="#FFF8E6",
        line="#A67C00",
        size=12,
        color="#5A4300",
    )

    # Column headers
    headers = [
        (0.25, "① 输入"),
        (1.85, "② 模态编码"),
        (3.70, "③ 局部序列 ＋ 整体摘要"),
        (6.55, "④ 投影 φ_m（d=128）"),
        (9.85, "⑤ 接口输出"),
    ]
    for x, t in headers:
        add_textbox(slide, Inches(x), Inches(0.95), Inches(2.2), Inches(0.30), t, size=13, bold=True, color="#222222", align=PP_ALIGN.LEFT)

    modalities = [
        dict(
            name="文本 Text",
            fill="#E7F0FA",
            line="#2C5F8A",
            y=1.35,
            inp="输入\ntext_bert\n[3×50]",
            enc="BERT\nθ_T 微调",
            local="H_i^T\nL_i×d_T, d_T=768\n局部=去 CLS 正文",
            summ="u_i^T\nCLS 摘要\n∈ R^768",
            z="φ_T + PE\nZ_i^{T,0}\nL_i×d",
            g="φ_T\ng_i^{T,0} ∈ R^d",
        ),
        dict(
            name="音频 Audio",
            fill="#E8F5EC",
            line="#2E6A43",
            y=3.15,
            inp="输入\nX̄_i^A\n50×74",
            enc="LSTM_A\nh=16\n按 L_i pack",
            local="H_i^A\nL_i×d_A, d_A=16\n逐位置隐状态",
            summ="u_i^A\n最后有效状态\n∈ R^16",
            z="φ_A + PE\nZ_i^{A,0}\nL_i×d",
            g="φ_A\ng_i^{A,0} ∈ R^d",
        ),
        dict(
            name="视觉 Vision",
            fill="#F8EEE6",
            line="#8A4A2D",
            y=4.95,
            inp="输入\nX̄_i^V\n50×35",
            enc="LSTM_V\nh=32\n按 L_i pack",
            local="H_i^V\nL_i×d_V, d_V=32\n逐位置隐状态",
            summ="u_i^V\n最后有效状态\n∈ R^32",
            z="φ_V + PE\nZ_i^{V,0}\nL_i×d",
            g="φ_V\ng_i^{V,0} ∈ R^d",
        ),
    ]

    box_h = 1.55
    loc_h = 0.90
    sum_h = 0.55

    for m in modalities:
        y = m["y"]
        line = m["line"]
        # input
        add_round_rect(slide, Inches(0.25), Inches(y), Inches(1.35), Inches(box_h), f"{m['name']}\n{m['inp']}", fill="#F4F4F4", line="#666666", size=12, bold=True, color=line)
        # encoder
        add_round_rect(slide, Inches(1.85), Inches(y), Inches(1.45), Inches(box_h), m["enc"], fill=m["fill"], line=line, size=14, bold=True, color=line)
        # local / summary
        add_round_rect(slide, Inches(3.70), Inches(y), Inches(2.35), Inches(loc_h), m["local"], fill="#FFFFFF", line=line, size=12, color=line)
        add_round_rect(slide, Inches(3.70), Inches(y + loc_h + 0.08), Inches(2.35), Inches(sum_h), m["summ"], fill="#F7F7F7", line=line, size=12, color=line)
        # projection
        add_round_rect(slide, Inches(6.40), Inches(y), Inches(2.55), Inches(loc_h), m["z"], fill=m["fill"], line=line, size=13, bold=True, color=line)
        add_round_rect(slide, Inches(6.40), Inches(y + loc_h + 0.08), Inches(2.55), Inches(sum_h), m["g"], fill="#FFFFFF", line=line, size=12, color=line)

        # arrows: input->enc, enc->local/sum, local/sum->proj
        cy = Inches(y + box_h / 2)
        add_arrow(slide, Inches(1.60), cy, Inches(1.85), cy, color=line)
        add_arrow(slide, Inches(3.30), Inches(y + loc_h * 0.5), Inches(3.70), Inches(y + loc_h * 0.5), color=line)
        add_arrow(slide, Inches(3.30), Inches(y + loc_h + 0.08 + sum_h * 0.5), Inches(3.70), Inches(y + loc_h + 0.08 + sum_h * 0.5), color=line)
        add_arrow(slide, Inches(6.05), Inches(y + loc_h * 0.5), Inches(6.40), Inches(y + loc_h * 0.5), color=line)
        add_arrow(slide, Inches(6.05), Inches(y + loc_h + 0.08 + sum_h * 0.5), Inches(6.40), Inches(y + loc_h + 0.08 + sum_h * 0.5), color=line)

        # dashed bypass from summary down — approximate vertical then to bypass band
        sx = Inches(3.70 + 1.175)
        sy = Inches(y + loc_h + 0.08 + sum_h)
        add_dashed_line(slide, sx, sy, sx, Inches(6.55), color=line, width_pt=1.1)

    # Right outputs
    add_round_rect(
        slide,
        Inches(9.85),
        Inches(1.35),
        Inches(3.15),
        Inches(2.85),
        "供后续 EMT 交互\n\nZ_i^{T,0}\nZ_i^{A,0}\nZ_i^{V,0}\n\n各 L_i×d\n有效位置用 mask\n保留内部零行槽位",
        fill="#EAF0F8",
        line="#1E3A5F",
        size=13,
        bold=True,
        color="#1E3A5F",
    )
    add_round_rect(
        slide,
        Inches(9.85),
        Inches(4.40),
        Inches(3.15),
        Inches(1.55),
        "初始全局上下文\n\nG_i^0 = stack(g^T, g^A, g^V)\n∈ R^{3×d}",
        fill="#F1EAF7",
        line="#5A3878",
        size=13,
        bold=True,
        color="#5A3878",
    )

    # Arrows from Z to EMT box
    for y in (1.35 + 0.45, 3.15 + 0.45, 4.95 + 0.45):
        add_arrow(slide, Inches(8.95), Inches(y), Inches(9.85), Inches(min(max(y, 1.8), 3.6)), color="#1E3A5F")

    # stack node + arrows from g
    add_round_rect(slide, Inches(9.15), Inches(4.85), Inches(0.55), Inches(0.40), "stack", fill="#5A3878", line="#5A3878", size=11, bold=True, color="#FFFFFF")
    for y in (1.35 + loc_h + 0.08 + sum_h * 0.5, 3.15 + loc_h + 0.08 + sum_h * 0.5, 4.95 + loc_h + 0.08 + sum_h * 0.5):
        add_arrow(slide, Inches(8.95), Inches(y), Inches(9.15), Inches(5.05), color="#5A3878", width_pt=1.2)
    add_arrow(slide, Inches(9.70), Inches(5.05), Inches(9.85), Inches(5.05), color="#5A3878")

    # Bypass band
    add_round_rect(
        slide,
        Inches(3.55),
        Inches(6.55),
        Inches(9.5),
        Inches(0.80),
        "投影前摘要旁路（不进入 G_i^0）：保留 u_i^T, u_i^A, u_i^V 至预测拼接\n"
        "f_i = [u^T; u^A; u^V; vec(G_i^K)]，维数 768+16+32+3×128 = 1200",
        fill="#FFF8F0",
        line="#8A5728",
        size=12,
        color="#6A3D12",
    )
    add_textbox(slide, Inches(2.85), Inches(6.70), Inches(0.65), Inches(0.35), "bypass", size=12, bold=True, color="#8A5728", align=PP_ALIGN.RIGHT)

    add_textbox(
        slide,
        Inches(0.25),
        Inches(6.60),
        Inches(2.5),
        Inches(0.70),
        "线型：实线→EMT/G^0\n虚线→u 旁路拼接",
        size=11,
        color="#444444",
        align=PP_ALIGN.LEFT,
        valign=MSO_ANCHOR.TOP,
    )


def build_reference_slide(prs: Presentation):
    slide = prs.slides.add_slide(prs.slide_layouts[6])
    add_textbox(
        slide,
        Inches(0.35),
        Inches(0.15),
        Inches(12.5),
        Inches(0.35),
        "参考对照页（PNG，整体位图，不便逐框改字）｜正式编辑请回第2页",
        size=14,
        bold=True,
        color="#1E3A5F",
        align=PP_ALIGN.LEFT,
    )
    if PNG_PATH.exists():
        # Fit image with margins
        slide.shapes.add_picture(str(PNG_PATH), Inches(0.35), Inches(0.55), width=Inches(12.6))
    else:
        add_textbox(
            slide,
            Inches(1),
            Inches(3),
            Inches(11),
            Inches(1),
            f"未找到参考图：{PNG_PATH}\n请先运行 code/draw_q2_encoder_interface.py",
            size=14,
            color="#AA0000",
        )


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    if not PNG_PATH.exists():
        # Generate reference figure first
        from draw_q2_encoder_interface import draw

        draw()

    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H

    build_notes_slide(prs)
    build_editable_diagram_slide(prs)
    build_reference_slide(prs)

    prs.save(PPTX_PATH)
    print(f"wrote {PPTX_PATH}")


if __name__ == "__main__":
    main()
