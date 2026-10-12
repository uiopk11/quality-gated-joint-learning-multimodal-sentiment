"""全文技术路线图 v2 → 可编辑竖版 PPTX（原生形状，不覆盖 v1 pptx）。"""
from __future__ import annotations

from pathlib import Path

from pptx import Presentation
from pptx.dml.color import RGBColor
from pptx.enum.shapes import MSO_CONNECTOR, MSO_SHAPE
from pptx.enum.text import MSO_ANCHOR, PP_ALIGN
from pptx.oxml.ns import qn
from pptx.util import Emu, Pt

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR / "Figure"
PPTX_PATH = FIG_DIR / "full_technical_roadmap_v2.pptx"

SLIDE_W = Emu(int(7.9 * 914400))
SLIDE_H = Emu(int(12.0 * 914400))
FONT = "微软雅黑"

C_TITLE = ("#F3F5F7", "#2F3E4E")
C_DATA = ("#EEF1F4", "#5B6770")
C_Q1 = ("#E7F0F8", "#2F5F86")
C_Q2 = ("#E8F4EE", "#2F6B4F")
C_Q2B = ("#F3F9F5", "#4C8A68")
C_Q3 = ("#F8EFE7", "#8A5530")
C_Q3B = ("#FBF6F1", "#A56B40")
C_OUT = ("#F2F2F2", "#555555")
LINK_Q1 = "#2F5F86"
LINK_Q2 = "#8A5530"


def hex_rgb(h: str) -> RGBColor:
    h = h.lstrip("#")
    return RGBColor(int(h[0:2], 16), int(h[2:4], 16), int(h[4:6], 16))


def nx(x: float) -> int:
    return int(x * SLIDE_W)


def ny(y: float, h: float = 0.0) -> int:
    return int((1.0 - y - h) * SLIDE_H)


def nh(h: float) -> int:
    return max(int(h * SLIDE_H), 1)


def nw(w: float) -> int:
    return max(int(w * SLIDE_W), 1)


def set_run(run, *, size: float, bold: bool = False, color: str = "#1F1F1F"):
    run.font.name = FONT
    run.font.size = Pt(size)
    run.font.bold = bold
    run.font.color.rgb = hex_rgb(color)
    rPr = run._r.get_or_add_rPr()
    ea = rPr.find(qn("a:ea"))
    if ea is None:
        ea = rPr.makeelement(qn("a:ea"), {})
        rPr.append(ea)
    ea.set("typeface", FONT)


def fill_line(shape, color: str, width_pt: float = 1.0):
    shape.line.color.rgb = hex_rgb(color)
    shape.line.width = Pt(width_pt)


def fill_solid(shape, color: str):
    shape.fill.solid()
    shape.fill.fore_color.rgb = hex_rgb(color)


def add_box(slide, x, y, w, h, *, fc, ec, lw=1.0):
    shape = slide.shapes.add_shape(
        MSO_SHAPE.ROUNDED_RECTANGLE, nx(x), ny(y, h), nw(w), nh(h)
    )
    fill_solid(shape, fc)
    fill_line(shape, ec, lw)
    shape.adjustments[0] = 0.12
    return shape


def set_tf(shape, text: str, *, size: float, bold: bool = False, color: str = "#1F1F1F",
           align=PP_ALIGN.CENTER, anchor=MSO_ANCHOR.MIDDLE):
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set("anchor", {MSO_ANCHOR.TOP: "t", MSO_ANCHOR.MIDDLE: "ctr", MSO_ANCHOR.BOTTOM: "b"}[anchor])
    except Exception:
        pass
    lines = text.split("\n") if text else [""]
    for i, line in enumerate(lines):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = align
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        run = p.add_run()
        run.text = line
        set_run(run, size=size, bold=bold, color=color)


def add_labeled_box(slide, x, y, w, h, title: str, body: str, *, fc, ec,
                    title_size=8.5, body_size=7.0, lw=1.0):
    shape = add_box(slide, x, y, w, h, fc=fc, ec=ec, lw=lw)
    tf = shape.text_frame
    tf.clear()
    tf.word_wrap = True
    try:
        tf._txBody.bodyPr.set("anchor", "t")
    except Exception:
        pass
    p0 = tf.paragraphs[0]
    p0.alignment = PP_ALIGN.CENTER
    p0.space_before = Pt(1)
    p0.space_after = Pt(0)
    r0 = p0.add_run()
    r0.text = title
    set_run(r0, size=title_size, bold=True)
    for line in (body.split("\n") if body else []):
        p = tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = line
        set_run(r, size=body_size, color="#333333")
    return shape


def add_plain_box(slide, x, y, w, h, text: str, *, fc, ec, size=8.0, bold=False, color="#1F1F1F", lw=1.0):
    shape = add_box(slide, x, y, w, h, fc=fc, ec=ec, lw=lw)
    set_tf(shape, text, size=size, bold=bold, color=color)
    return shape


def add_side_stack(slide, x, y, w, h, cells, *, ec, title_fc):
    n = len(cells)
    gap = 0.003
    if n == 1:
        add_plain_box(slide, x, y, w, h, cells[0], fc=title_fc, ec=ec, size=7.5, bold=True, lw=0.9)
        return
    title_h = min(0.018, h * 0.38)
    rest = n - 1
    body_h = (h - title_h - (rest - 1) * gap) / rest
    add_plain_box(slide, x, y + h - title_h, w, title_h, cells[0], fc=title_fc, ec=ec, size=7.5, bold=True, lw=0.9)
    for i, line in enumerate(cells[1:]):
        yy = y + (rest - 1 - i) * (body_h + gap)
        add_plain_box(slide, x, yy, w, body_h, line, fc="#FFFFFF", ec=ec, size=6.5, color="#444444", lw=0.75)


def add_h_arrow(slide, x0, y_c, x1, color: str):
    y = y_c - 0.002
    h = 0.004
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, nx(x0), ny(y, h), nx(x1), ny(y, h)
    )
    conn.line.color.rgb = hex_rgb(color)
    conn.line.width = Pt(1.15)
    ln = conn.line._ln
    if ln.find(qn("a:tailEnd")) is None:
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def add_v_arrow(slide, x_c, y_hi, y_lo, color: str):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, nx(x_c), ny(max(y_hi, y_lo), 0), nx(x_c), ny(min(y_hi, y_lo), 0)
    )
    conn.line.color.rgb = hex_rgb(color)
    conn.line.width = Pt(1.15)
    ln = conn.line._ln
    if ln.find(qn("a:tailEnd")) is None:
        ln.append(ln.makeelement(qn("a:tailEnd"), {"type": "triangle", "w": "med", "len": "med"}))


def add_hline(slide, x0, x1, y, color: str, width_pt: float = 1.0):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, nx(x0), ny(y, 0), nx(x1), ny(y, 0)
    )
    conn.line.color.rgb = hex_rgb(color)
    conn.line.width = Pt(width_pt)
    return conn


def add_vline(slide, x, y0, y1, color: str, width_pt: float = 1.0):
    conn = slide.shapes.add_connector(
        MSO_CONNECTOR.STRAIGHT, nx(x), ny(max(y0, y1), 0), nx(x), ny(min(y0, y1), 0)
    )
    conn.line.color.rgb = hex_rgb(color)
    conn.line.width = Pt(width_pt)
    return conn


def add_rotated_label(slide, x, y, w, h, text: str, color: str, size: float = 7.5):
    shape = slide.shapes.add_textbox(nx(x), ny(y, h), nw(w), nh(h))
    tf = shape.text_frame
    tf.clear()
    for i, ch in enumerate(text):
        p = tf.paragraphs[0] if i == 0 else tf.add_paragraph()
        p.alignment = PP_ALIGN.CENTER
        p.space_before = Pt(0)
        p.space_after = Pt(0)
        r = p.add_run()
        r.text = ch
        set_run(r, size=size, bold=True, color=color)


def build():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    prs = Presentation()
    prs.slide_width = SLIDE_W
    prs.slide_height = SLIDE_H
    slide = prs.slides.add_slide(prs.slide_layouts[6])

    Lx, Lw = 0.018, 0.088
    Rx, Rw = 0.894, 0.088
    left, width = 0.158, 0.684
    mid = left + width / 2
    seam_l = (Lx + Lw + left) / 2
    seam_r = (left + width + Rx) / 2
    x0 = left + 0.012
    sw, g = 0.158, 0.012
    sides = []

    def link_sides(y_c, color):
        add_h_arrow(slide, Lx + Lw, y_c, left, color)
        add_h_arrow(slide, Rx, y_c, left + width, color)

    # 标题
    title = add_plain_box(
        slide, left, 0.962, width, 0.026,
        "复杂场景下多模态情感预测：全文技术路线\n数据 → 特征对齐 → 鲁棒预测 → 可解释预测",
        fc=C_TITLE[0], ec=C_TITLE[1], size=12.0, bold=True, lw=1.25,
    )
    if len(title.text_frame.paragraphs) > 1:
        for run in title.text_frame.paragraphs[0].runs:
            set_run(run, size=12.5, bold=True)
        for run in title.text_frame.paragraphs[1].runs:
            set_run(run, size=8.5, bold=False, color="#444444")
    add_side_stack(slide, Lx, 0.962, Lw, 0.026, ["题目 E"], ec=C_TITLE[1], title_fc=C_TITLE[0])
    add_side_stack(slide, Rx, 0.962, Rw, 0.026, ["指标"], ec=C_TITLE[1], title_fc=C_TITLE[0])
    link_sides(0.975, C_TITLE[1])
    sides.append((0.962, 0.026))

    # 数据
    add_plain_box(slide, left, 0.936, width, 0.016, "题面数据与任务链", fc=C_DATA[0], ec=C_DATA[1], size=10.5, bold=True, lw=1.2)
    y, h = 0.872, 0.070
    add_side_stack(slide, Lx, y, Lw, h, ["数据", "附件1–4"], ec=C_DATA[1], title_fc=C_DATA[0])
    add_side_stack(slide, Rx, y, Rw, h, ["规则", "valid选模", "专项推理"], ec=C_DATA[1], title_fc=C_DATA[0])
    link_sides(y + h / 2, C_DATA[1])
    sides.append((y, h))
    for i, (name, body) in enumerate([
        ("附件1", "视频+文本\n→问题一"),
        ("附件2", "aligned_50\n→问题二/三"),
        ("附件3", "局部缺失\n→问题二测"),
        ("附件4", "完整模态\n→问题三测"),
    ]):
        add_labeled_box(slide, x0 + i * (sw + g), 0.880, sw, 0.044, name, body, fc="#FFFFFF", ec=C_DATA[1], title_size=8.5, body_size=7.0)
    add_plain_box(slide, left, 0.864, width, 0.010, "Acc / Macro-F1；MAE / Pearson", fc="#FFFFFF", ec=C_DATA[1], size=7.5, color="#333333", lw=0.8)
    add_v_arrow(slide, mid, 0.862, 0.852, "#4A4A4A")

    # 问题一
    add_plain_box(slide, left, 0.836, width, 0.014, "问题一　QG-HTA：特征提取与时序对齐", fc=C_Q1[0], ec=C_Q1[1], size=10.5, bold=True, lw=1.2)
    y, h = 0.640, 0.180
    add_side_stack(slide, Lx, y, Lw, h, ["对齐", "特征提取", "时序对齐", "双视图", "掩码"], ec=C_Q1[1], title_fc=C_Q1[0])
    add_side_stack(slide, Rx, y, Rw, h, ["边界", "文本不变", "ASR不替换", "显式掩码"], ec=C_Q1[1], title_fc=C_Q1[0])
    link_sides(y + h / 2, C_Q1[1])
    sides.append((y, h))

    feat_x, feat_w = x0, 0.148
    align_x = feat_x + feat_w + 0.010
    align_w = left + width - 0.012 - align_x
    block_top, block_bot = 0.800, 0.656
    block_h = block_top - block_bot

    add_box(slide, feat_x - 0.002, block_bot - 0.002, feat_w + 0.004, block_h + 0.016, fc="#F7FAFC", ec=C_Q1[1], lw=0.6)
    add_box(slide, align_x - 0.002, block_bot - 0.002, align_w + 0.004, block_h + 0.016, fc="#F7FAFC", ec=C_Q1[1], lw=0.6)
    tb1 = slide.shapes.add_textbox(nx(feat_x), ny(block_top + 0.010, 0.010), nw(feat_w), nh(0.010))
    set_tf(tb1, "特征提取", size=8.0, bold=True, color=C_Q1[1])
    tb2 = slide.shapes.add_textbox(nx(align_x), ny(block_top + 0.010, 0.010), nw(align_w), nh(0.010))
    set_tf(tb2, "时序对齐", size=8.0, bold=True, color=C_Q1[1])

    moda_items = [
        ("BERT 文本", "官方文本主轴\nbert-base末层\nWordPiece·768"),
        ("OpenSMILE", "真实音轨帧级\nComParE+Δ\n声学·74维"),
        ("OpenFace", "真实画面帧级\n视线·头姿·AU\n视觉·35维"),
    ]
    moda_gap = 0.004
    moda_h = (block_h - 2 * moda_gap) / 3
    moda_ys = []
    for i, (title, body) in enumerate(moda_items):
        yy = block_top - (i + 1) * moda_h - i * moda_gap
        moda_ys.append(yy + moda_h / 2)
        add_labeled_box(slide, feat_x, yy, feat_w, moda_h, title, body, fc="#FFFFFF", ec=C_Q1[1], title_size=7.5, body_size=6.2)

    ag = 0.008
    gate_w = br_w = agg_w = std_w = 0.118
    gate_x = align_x
    br_x = gate_x + gate_w + ag
    agg_x = br_x + br_w + ag
    std_x = agg_x + agg_w + ag
    gate_yc = block_bot + block_h / 2

    add_labeled_box(
        slide, gate_x, block_bot, gate_w, block_h, "①质量门控",
        "Whisper核验\nS=词F1+字符似\n阈值τL/τH\n首尾锚点检查\n文本永不替换",
        fc="#FFFFFF", ec=C_Q1[1], title_size=7.5, body_size=6.0,
    )

    bus_in = feat_x + feat_w + 0.005
    for yc in moda_ys:
        add_hline(slide, feat_x + feat_w, bus_in, yc, C_Q1[1], 1.0)
    add_vline(slide, bus_in, moda_ys[0], moda_ys[-1], C_Q1[1], 1.0)
    add_hline(slide, bus_in, gate_x, gate_yc, C_Q1[1], 1.0)
    add_h_arrow(slide, gate_x - 0.012, gate_yc, gate_x, C_Q1[1])

    br_gap = 0.005
    br_h = (block_h - br_gap) / 2
    br_top_y = block_bot + br_h + br_gap
    br_bot_y = block_bot
    br_yc_top = br_top_y + br_h / 2
    br_yc_bot = br_bot_y + br_h / 2
    add_labeled_box(slide, br_x, br_top_y, br_w, br_h, "②a 词级MFA", "窗口W内MFA\n官方文本强制\n时间戳回移", fc="#FFFFFF", ec=C_Q1[1], title_size=7.2, body_size=6.0)
    add_labeled_box(slide, br_x, br_bot_y, br_w, br_h, "②b ASR回退", "锚点不足降级\n可靠词段对齐\n或无词级对齐", fc="#FFFFFF", ec=C_Q1[1], title_size=7.2, body_size=6.0)

    bus_out = gate_x + gate_w + ag * 0.45
    add_hline(slide, gate_x + gate_w, bus_out, gate_yc, C_Q1[1], 1.0)
    add_vline(slide, bus_out, br_yc_top, br_yc_bot, C_Q1[1], 1.0)
    add_hline(slide, bus_out, br_x, br_yc_top, C_Q1[1], 1.0)
    add_hline(slide, bus_out, br_x, br_yc_bot, C_Q1[1], 1.0)
    add_h_arrow(slide, br_x - 0.012, br_yc_top, br_x, C_Q1[1])
    add_h_arrow(slide, br_x - 0.012, br_yc_bot, br_x, C_Q1[1])

    add_labeled_box(
        slide, agg_x, block_bot, agg_w, block_h, "③区间汇聚",
        "自然词区间 I\n区间内帧汇聚\n子词继承 π(j)\n不可靠处置零",
        fc="#FFFFFF", ec=C_Q1[1], title_size=7.5, body_size=6.0,
    )
    bus_m = br_x + br_w + ag * 0.45
    add_hline(slide, br_x + br_w, bus_m, br_yc_top, C_Q1[1], 1.0)
    add_hline(slide, br_x + br_w, bus_m, br_yc_bot, C_Q1[1], 1.0)
    add_vline(slide, bus_m, br_yc_top, br_yc_bot, C_Q1[1], 1.0)
    add_hline(slide, bus_m, agg_x, gate_yc, C_Q1[1], 1.0)
    add_h_arrow(slide, agg_x - 0.012, gate_yc, agg_x, C_Q1[1])

    add_labeled_box(slide, std_x, br_top_y, std_w, br_h, "aligned_50", "T/A/V·长度50\n词元对齐主视图", fc="#FFFFFF", ec=C_Q1[1], title_size=7.2, body_size=6.0)
    add_labeled_box(slide, std_x, br_bot_y, std_w, br_h, "unaligned_500", "A/V·长度500\n掩码可追溯", fc="#FFFFFF", ec=C_Q1[1], title_size=7.2, body_size=6.0)
    bus_std = agg_x + agg_w + ag * 0.45
    add_hline(slide, agg_x + agg_w, bus_std, gate_yc, C_Q1[1], 1.0)
    add_vline(slide, bus_std, br_yc_top, br_yc_bot, C_Q1[1], 1.0)
    add_hline(slide, bus_std, std_x, br_yc_top, C_Q1[1], 1.0)
    add_hline(slide, bus_std, std_x, br_yc_bot, C_Q1[1], 1.0)
    add_h_arrow(slide, std_x - 0.012, br_yc_top, std_x, C_Q1[1])
    add_h_arrow(slide, std_x - 0.012, br_yc_bot, std_x, C_Q1[1])

    add_plain_box(slide, left, 0.640, width, 0.012, "不训练情感分类器；对齐不确定处用零填充+显式掩码共同表达", fc="#FFFFFF", ec=C_Q1[1], size=7.5, color=C_Q1[1], lw=0.8)
    add_v_arrow(slide, mid, 0.638, 0.628, "#4A4A4A")

    # 问题二
    add_plain_box(slide, left, 0.610, width, 0.014, "问题二　局部缺失鲁棒预测（主模型 M3 / JPI-DLFR）", fc=C_Q2[0], ec=C_Q2[1], size=10.0, bold=True, lw=1.2)
    y, h = 0.430, 0.164
    add_side_stack(slide, Lx, y, Lw, h, ["鲁棒", "双视图", "EMT", "DLFR", "四模型"], ec=C_Q2[1], title_fc=C_Q2[0])
    add_side_stack(slide, Rx, y, Rw, h, ["消融", "M0对M1", "M2对M3", "附件3推理"], ec=C_Q2[1], title_fc=C_Q2[0])
    link_sides(y + h / 2, C_Q2[1])
    sides.append((y, h))

    pipe = [
        ("编码骨干", "BERT正文编码\n音频/视觉LSTM"),
        ("EMT融合", "全局—局部交互\n2层·4头"),
        ("JPI双头", "极性分类头\n强度回归头"),
        ("DLFR双层", "低层恢复头\n高层对比约束"),
    ]
    pw, pg = 0.152, 0.010
    py, ph = 0.528, 0.062
    for i, (title, body) in enumerate(pipe):
        x = x0 + i * (pw + pg)
        add_labeled_box(slide, x, py, pw, ph, title, body, fc=C_Q2B[0], ec=C_Q2[1], title_size=8.0, body_size=6.5)
        if i < 3:
            add_h_arrow(slide, x + pw, py + ph / 2, x + pw + pg, C_Q2[1])

    q2b_y, q2b_h = 0.446, 0.072
    proto_w, model_w, note_w = 0.200, 0.268, 0.174
    gap2 = 0.012
    proto_x = x0
    model_x = proto_x + proto_w + gap2
    note_x = model_x + model_w + gap2
    add_labeled_box(slide, proto_x, q2b_y, proto_w, q2b_h, "缺失协议", "mix_v1冻结掩码\nclean/missing双视图\n类型·率·位配对面板", fc=C_Q2B[0], ec=C_Q2[1], title_size=8.0, body_size=6.5)
    add_box(slide, model_x, q2b_y, model_w, q2b_h, fc="#FFFFFF", ec=C_Q2[1], lw=0.9)
    tbox = slide.shapes.add_textbox(nx(model_x), ny(q2b_y + q2b_h - 0.014, 0.012), nw(model_w), nh(0.012))
    set_tf(tbox, "四模型对照", size=8.0, bold=True, color=C_Q2[1])

    models = [
        ("M0", "完整·仅回归", "#FFFFFF", C_Q2[1]),
        ("M1", "完整·联合→三", "#F7EFE8", LINK_Q2),
        ("M2", "缺失·仅回归", "#FFFFFF", C_Q2[1]),
        ("M3", "缺失·联合·主", "#DCEFE4", C_Q2[1]),
    ]
    mw, mh = 0.126, 0.026
    mx_gap, my_gap = 0.008, 0.005
    my0 = q2b_y + q2b_h - 0.012 - mh
    m1_yc = None
    for i, (name, role, fc, ec) in enumerate(models):
        r, c = divmod(i, 2)
        x = model_x + 0.004 + c * (mw + mx_gap)
        yy = my0 - r * (mh + my_gap)
        if name == "M1":
            m1_yc = yy + mh / 2
        add_plain_box(slide, x, yy, mw, mh, f"{name}  {role}", fc=fc, ec=ec, size=7.5, bold=True, color=ec, lw=1.1 if name in {"M1", "M3"} else 0.8)

    add_labeled_box(slide, note_x, q2b_y, note_w, q2b_h, "训练/推理", "联合损失选模\n训练辅恢复\n推理不回填", fc=C_Q2B[0], ec=C_Q2[1], title_size=8.0, body_size=6.5)
    add_plain_box(slide, left, 0.434, width, 0.010, "JPI-DLFR：双视图监督 + 双层恢复；附件3冻结M3单视图推理", fc="#FFFFFF", ec=C_Q2[1], size=7.5, color=C_Q2[1], lw=0.8)
    add_v_arrow(slide, mid, 0.432, 0.422, "#4A4A4A")

    # 问题三
    add_plain_box(slide, left, 0.412, width, 0.016, "问题三　完整输入可解释预测（冻结M1 + 分层解释）", fc=C_Q3[0], ec=C_Q3[1], size=10.0, bold=True, lw=1.2)
    y, h = 0.210, 0.186
    add_side_stack(slide, Lx, y, Lw, h, ["解释", "对象sc/sr", "8联盟", "局部窗", "忠实度"], ec=C_Q3[1], title_fc=C_Q3[0])
    add_side_stack(slide, Rx, y, Rw, h, ["边界", "非因果", "不计分", "无新参数"], ec=C_Q3[1], title_fc=C_Q3[0])
    link_sides(y + h / 2, C_Q3[1])
    sides.append((y, h))

    q3_top = [
        ("冻结M1-EMT", "完整输入只读预测\n分类头 + 回归头\n参数不回训"),
        ("解释对象", "固定预测类 k*\nlogit 边际 sc\n原始强度 sr"),
        ("精确Shapley", "8联盟全前向\n分类/强度两套φ\n主导·支持·反对"),
        ("局部多尺度", "窗宽1/3/5连续蔽\n逐位置重要性\nTop互不重叠"),
    ]
    pw3, pg3 = 0.152, 0.010
    py3, ph3 = 0.322, 0.070
    for i, (title, body) in enumerate(q3_top):
        x = x0 + i * (pw3 + pg3)
        add_labeled_box(slide, x, py3, pw3, ph3, title, body, fc=C_Q3B[0], ec=C_Q3[1], title_size=8.5, body_size=7.0)
        if i < 3:
            add_h_arrow(slide, x + pw3, py3 + ph3 / 2, x + pw3 + pg3, C_Q3[1])

    q3b_y, q3b_h = 0.240, 0.072
    bw, bg = 0.212, 0.012
    for i, (title, body) in enumerate([
        ("遮蔽基线", "文本正文→[UNK]\n音/视有效行置零\n长度与mask不变"),
        ("忠实度验证", "Top删10/20/30%\n对比等量随机删\n效率与排序稳定"),
        ("原素材回溯", "QG-HTA秒/帧映射\n文本offset定位\n异常显式标记"),
    ]):
        add_labeled_box(slide, x0 + i * (bw + bg), q3b_y, bw, q3b_h, title, body, fc=C_Q3B[0], ec=C_Q3[1], title_size=8.5, body_size=7.0)

    add_plain_box(slide, left, 0.218, width, 0.012, "解释层无新增参数；归因=遮蔽敏感性，不作因果；附件4不计分", fc="#F7EFE8", ec=C_Q3[1], size=7.5, color=C_Q3[1], lw=0.85)
    add_v_arrow(slide, mid, 0.216, 0.206, "#4A4A4A")

    # 交付
    add_plain_box(slide, left, 0.186, width, 0.014, "验证闭环与论文交付", fc=C_OUT[0], ec=C_OUT[1], size=10.5, bold=True, lw=1.2)
    y, h = 0.078, 0.090
    add_side_stack(slide, Lx, y, Lw, h, ["交付", "代码", "结果", "追溯"], ec=C_OUT[1], title_fc=C_OUT[0])
    add_side_stack(slide, Rx, y, Rw, h, ["提交", "专项CSV", "≤50MB"], ec=C_OUT[1], title_fc=C_OUT[0])
    link_sides(y + h / 2, C_OUT[1])
    sides.append((y, h))
    for i, (title, body) in enumerate([
        ("问题一", "特征与图"),
        ("问题二", "四模型+附件3"),
        ("问题三", "解释卡+附件4"),
        ("证据链", "配置可追溯"),
    ]):
        add_labeled_box(slide, x0 + i * (sw + g), 0.088, sw, 0.078, title, body, fc="#FFFFFF", ec=C_OUT[1], title_size=8.5, body_size=7.0)

    foot = add_plain_box(
        slide, left, 0.030, width, 0.034,
        "主线：数据 → QG-HTA → M3/JPI-DLFR → 冻结M1解释 → 交付\n旁路：问题一对齐复用；问题二选取完整输入M1作问题三预测器",
        fc="#FAFAFA", ec="#666666", size=8.0, color="#222222", lw=0.95,
    )
    if len(foot.text_frame.paragraphs) > 1:
        for run in foot.text_frame.paragraphs[1].runs:
            set_run(run, size=7.5, color="#555555")

    for i in range(len(sides) - 1):
        y0, _ = sides[i]
        y1, h1 = sides[i + 1]
        y_from = y0 - 0.002
        y_to = y1 + h1 + 0.002
        if y_from <= y_to + 0.004:
            continue
        add_v_arrow(slide, Lx + Lw / 2, y_from, y_to, "#888888")
        add_v_arrow(slide, Rx + Rw / 2, y_from, y_to, "#888888")

    y_q1, y_q3_bar = 0.728, 0.224
    add_vline(slide, seam_r, y_q1, y_q3_bar, LINK_Q1, 1.3)
    add_h_arrow(slide, left + width, y_q1, seam_r, LINK_Q1)
    add_h_arrow(slide, seam_r, y_q3_bar, left + width, LINK_Q1)
    add_rotated_label(slide, seam_r - 0.012, (y_q1 + y_q3_bar) / 2 - 0.06, 0.024, 0.12, "对齐复用", LINK_Q1, size=7.5)

    assert m1_yc is not None
    y_q3_top = 0.420
    add_vline(slide, seam_l, m1_yc, y_q3_top, LINK_Q2, 1.3)
    add_h_arrow(slide, left, m1_yc, seam_l, LINK_Q2)
    add_h_arrow(slide, seam_l, y_q3_top, left, LINK_Q2)
    add_rotated_label(slide, seam_l - 0.012, (m1_yc + y_q3_top) / 2 - 0.04, 0.024, 0.08, "选取M1", LINK_Q2, size=7.5)

    prs.save(PPTX_PATH)
    print(f"[saved] {PPTX_PATH}")


if __name__ == "__main__":
    build()
