"""竖向全文技术路线图（论文单页，仅 PNG）。

中间主体分区色块；左右竖列小格 + 横向连到主体 + 纵向串联；
跨问旁路走窄缝，避免遮挡。
"""
from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib import font_manager as fm
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

CODE_DIR = Path(__file__).resolve().parent
FIG_DIR = CODE_DIR / "Figure"
PNG_PATH = FIG_DIR / "full_technical_roadmap.png"


def setup_fonts() -> str:
    preferred = ["Microsoft YaHei", "SimHei", "SimSun"]
    available = {f.name for f in fm.fontManager.ttflist}
    cn = next((name for name in preferred if name in available), None)
    if cn is None:
        raise RuntimeError("No Chinese-capable font found.")
    plt.rcParams.update({"font.family": cn, "font.size": 9, "axes.unicode_minus": False})
    return cn


def box(ax, xy, w, h, *, fc, ec, lw=1.05, r=0.009, z=2):
    ax.add_patch(
        FancyBboxPatch(
            xy, w, h,
            boxstyle=f"round,pad=0.004,rounding_size={r}",
            facecolor=fc, edgecolor=ec, linewidth=lw,
            mutation_aspect=0.35, zorder=z,
        )
    )


def arrow(ax, p0, p1, *, color="#555555", lw=1.0, ms=8, z=7, rad=0.0):
    ax.add_patch(
        FancyArrowPatch(
            p0, p1, arrowstyle="-|>", mutation_scale=ms, linewidth=lw,
            color=color, connectionstyle=f"arc3,rad={rad}",
            zorder=z, shrinkA=1.0, shrinkB=1.0,
        )
    )


def txt(ax, x, y, s, *, size=7.8, weight="normal", color="#1f1f1f", ha="center", va="center", **kw):
    return ax.text(x, y, s, fontsize=size, fontweight=weight, color=color, ha=ha, va=va, zorder=8, **kw)


def section_bar(ax, x, y, w, h, title, *, fc, ec):
    box(ax, (x, y), w, h, fc=fc, ec=ec, lw=1.2)
    txt(ax, x + w / 2, y + h / 2, title, size=8.0, weight="bold")


def mini(ax, x, y, w, h, lines, *, fc, ec, title, title_size=6.8, body_size=5.6, pack_top=True):
    box(ax, (x, y), w, h, fc=fc, ec=ec, lw=0.9)
    th = min(0.018, h * 0.28)
    txt(ax, x + w / 2, y + h - th / 2, title, size=title_size, weight="bold")
    body = "\n".join(lines)
    if pack_top:
        # 顶对齐，避免大框里字悬在中间
        n = max(len(lines), 1)
        body_top = y + h - th - 0.004
        line_h = min(0.016, (h - th - 0.008) / n)
        for i, line in enumerate(lines):
            txt(ax, x + w / 2, body_top - (i + 0.5) * line_h, line, size=body_size, color="#333333")
    else:
        txt(ax, x + w / 2, y + (h - th) / 2 - 0.001, body, size=body_size, color="#333333", linespacing=1.15)


def side_stack(ax, x, y, w, h, cells, *, ec, title_fc):
    n = len(cells)
    gap = 0.003
    title_h = min(0.018, h * 0.38) if n > 1 else h
    if n == 1:
        box(ax, (x, y), w, h, fc=title_fc, ec=ec, lw=0.9, r=0.004)
        txt(ax, x + w / 2, y + h / 2, cells[0], size=6.2, weight="bold")
        return
    rest = n - 1
    body_h = (h - title_h - (rest - 1) * gap) / rest
    box(ax, (x, y + h - title_h), w, title_h, fc=title_fc, ec=ec, lw=0.9, r=0.004)
    txt(ax, x + w / 2, y + h - title_h / 2, cells[0], size=6.2, weight="bold")
    for i, line in enumerate(cells[1:]):
        yy = y + (rest - 1 - i) * (body_h + gap)
        box(ax, (x, yy), w, body_h, fc="#FFFFFF", ec=ec, lw=0.75, r=0.003)
        txt(ax, x + w / 2, yy + body_h / 2, line, size=5.2, color="#444444")


def draw():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    setup_fonts()

    fig, ax = plt.subplots(figsize=(7.9, 11.2), dpi=320)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    C_TITLE = ("#F3F5F7", "#2F3E4E")
    C_DATA = ("#EEF1F4", "#5B6770")
    C_Q1 = ("#E7F0F8", "#2F5F86")
    C_Q1B = ("#F4F8FC", "#4A7CA8")
    C_Q2 = ("#E8F4EE", "#2F6B4F")
    C_Q2B = ("#F3F9F5", "#4C8A68")
    C_Q3 = ("#F8EFE7", "#8A5530")
    C_Q3B = ("#FBF6F1", "#A56B40")
    C_OUT = ("#F2F2F2", "#555555")
    LINK_Q1 = "#2F5F86"
    LINK_Q2 = "#8A5530"

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
        arrow(ax, (Lx + Lw, y_c), (left, y_c), color=color, lw=0.95, ms=8)
        arrow(ax, (Rx, y_c), (left + width, y_c), color=color, lw=0.95, ms=8)

    # 标题
    box(ax, (left, 0.958), width, 0.030, fc=C_TITLE[0], ec=C_TITLE[1], lw=1.25)
    txt(ax, mid, 0.978, "复杂场景下多模态情感预测：全文技术路线", size=10.0, weight="bold")
    txt(ax, mid, 0.963, "数据 → 特征对齐 → 鲁棒预测 → 可解释预测", size=6.6, color="#444444")
    side_stack(ax, Lx, 0.958, Lw, 0.030, ["题目 E"], ec=C_TITLE[1], title_fc=C_TITLE[0])
    side_stack(ax, Rx, 0.958, Rw, 0.030, ["指标"], ec=C_TITLE[1], title_fc=C_TITLE[0])
    link_sides(0.973, C_TITLE[1])
    sides.append((0.958, 0.030, C_TITLE[1]))

    # 数据
    section_bar(ax, left, 0.928, width, 0.018, "题面数据与任务链", fc=C_DATA[0], ec=C_DATA[1])
    y, h = 0.858, 0.078
    side_stack(ax, Lx, y, Lw, h, ["数据", "附件1–4"], ec=C_DATA[1], title_fc=C_DATA[0])
    side_stack(ax, Rx, y, Rw, h, ["规则", "valid选模", "专项推理"], ec=C_DATA[1], title_fc=C_DATA[0])
    link_sides(y + h / 2, C_DATA[1])
    sides.append((y, h, C_DATA[1]))

    for i, (name, body) in enumerate([
        ("附件1", "视频+文本\n→问题一"),
        ("附件2", "aligned_50\n→问题二/三"),
        ("附件3", "局部缺失\n→问题二测"),
        ("附件4", "完整模态\n→问题三测"),
    ]):
        mini(ax, x0 + i * (sw + g), 0.868, sw, 0.048, body.split("\n"), fc="#FFFFFF", ec=C_DATA[1], title=name)
    box(ax, (left, 0.848), width, 0.012, fc="#FFFFFF", ec=C_DATA[1], lw=0.8)
    txt(ax, mid, 0.854, "Acc / Macro-F1；MAE / Pearson", size=5.8, color="#333333")
    arrow(ax, (mid, 0.846), (mid, 0.834), color="#4A4A4A", lw=1.35, ms=10)

    # 问题一（略压缩）
    section_bar(ax, left, 0.812, width, 0.018, "问题一　QG-HTA 分层词元对齐", fc=C_Q1[0], ec=C_Q1[1])
    y, h = 0.728, 0.090
    side_stack(ax, Lx, y, Lw, h, ["对齐", "表征", "门控", "分层"], ec=C_Q1[1], title_fc=C_Q1[0])
    side_stack(ax, Rx, y, Rw, h, ["边界", "文本不变", "ASR核验"], ec=C_Q1[1], title_fc=C_Q1[0])
    link_sides(y + h / 2, C_Q1[1])
    sides.append((y, h, C_Q1[1]))

    for i, (title, body) in enumerate([
        ("三模态表征", "BERT·声学·视觉"),
        ("质量测量", "一致性 S"),
        ("分层对齐", "MFA / 局部ASR"),
        ("标准化", "双视图输出"),
    ]):
        x = x0 + i * (sw + g)
        mini(ax, x, 0.738, sw, 0.060, body.split("\n"), fc=C_Q1B[0], ec=C_Q1[1], title=title)
        if i < 3:
            arrow(ax, (x + sw, 0.768), (x + sw + g, 0.768), color=C_Q1[1], lw=0.9, ms=7)
    box(ax, (left, 0.720), width, 0.012, fc="#FFFFFF", ec=C_Q1[1], lw=0.8)
    txt(ax, mid, 0.726, "特征工程；不训练情感分类器", size=5.8, color=C_Q1[1])
    arrow(ax, (mid, 0.718), (mid, 0.706), color="#4A4A4A", lw=1.35, ms=10)

    # 问题二：主干管线 + 协议/四模型
    section_bar(ax, left, 0.684, width, 0.018, "问题二　局部缺失鲁棒预测（主模型 M3 / JPI-DLFR）", fc=C_Q2[0], ec=C_Q2[1])
    y, h = 0.478, 0.190
    side_stack(ax, Lx, y, Lw, h, ["鲁棒", "双视图", "EMT", "DLFR", "四模型"], ec=C_Q2[1], title_fc=C_Q2[0])
    side_stack(ax, Rx, y, Rw, h, ["消融", "M0对M1", "M2对M3", "附件3推理"], ec=C_Q2[1], title_fc=C_Q2[0])
    link_sides(y + h / 2, C_Q2[1])
    sides.append((y, h, C_Q2[1]))

    pipe = [
        ("编码骨干", "BERT正文编码\n音频/视觉LSTM"),
        ("EMT融合", "全局—局部交互\n2层·4头"),
        ("JPI双头", "极性分类头\n强度回归头"),
        ("DLFR双层", "低层恢复头\n高层对比约束"),
    ]
    pw, pg = 0.152, 0.010
    py, ph = 0.590, 0.072
    for i, (title, body) in enumerate(pipe):
        x = x0 + i * (pw + pg)
        mini(ax, x, py, pw, ph, body.split("\n"), fc=C_Q2B[0], ec=C_Q2[1], title=title, title_size=6.5, body_size=5.3)
        if i < 3:
            arrow(ax, (x + pw, py + ph / 2), (x + pw + pg, py + ph / 2), color=C_Q2[1], lw=0.9, ms=7)

    q2b_y, q2b_h = 0.498, 0.082
    proto_w, model_w, note_w = 0.200, 0.268, 0.174
    gap2 = 0.012
    proto_x = x0
    model_x = proto_x + proto_w + gap2
    note_x = model_x + model_w + gap2

    mini(
        ax, proto_x, q2b_y, proto_w, q2b_h,
        ["mix_v1冻结掩码", "clean/missing双视图", "类型·率·位配对面板"],
        fc=C_Q2B[0], ec=C_Q2[1], title="缺失协议", title_size=6.5, body_size=5.1,
    )

    models = [
        ("M0", "完整·仅回归"),
        ("M1", "完整·联合→三"),
        ("M2", "缺失·仅回归"),
        ("M3", "缺失·联合·主"),
    ]
    mw, mh = 0.126, 0.030
    mx_gap, my_gap = 0.008, 0.006
    my0 = q2b_y + q2b_h - 0.014 - mh
    txt(ax, model_x + model_w / 2, q2b_y + q2b_h - 0.006, "四模型对照", size=6.2, weight="bold", color=C_Q2[1], va="top")
    m1_yc = None
    for i, (name, role) in enumerate(models):
        r, c = divmod(i, 2)
        x = model_x + 0.004 + c * (mw + mx_gap)
        yy = my0 - r * (mh + my_gap)
        if name == "M1":
            fc, ec = "#F7EFE8", LINK_Q2
            m1_yc = yy + mh / 2
        elif name == "M3":
            fc, ec = "#DCEFE4", C_Q2[1]
        else:
            fc, ec = "#FFFFFF", C_Q2[1]
        box(ax, (x, yy), mw, mh, fc=fc, ec=ec, lw=1.1 if name in {"M1", "M3"} else 0.8, r=0.004)
        txt(ax, x + 0.018, yy + mh / 2, name, size=6.2, weight="bold", color=ec, ha="left")
        txt(ax, x + 0.048, yy + mh / 2, role, size=5.0, color="#333333", ha="left")

    mini(
        ax, note_x, q2b_y, note_w, q2b_h,
        ["联合损失选模", "训练辅恢复", "推理不回填"],
        fc=C_Q2B[0], ec=C_Q2[1], title="训练/推理", title_size=6.5, body_size=5.1,
    )

    box(ax, (left, 0.484), width, 0.012, fc="#FFFFFF", ec=C_Q2[1], lw=0.8)
    txt(ax, mid, 0.490, "JPI-DLFR：双视图监督 + 双层恢复；附件3冻结M3单视图推理", size=5.6, color=C_Q2[1])
    arrow(ax, (mid, 0.482), (mid, 0.470), color="#4A4A4A", lw=1.35, ms=10)

    # 问题三：双行建模链（框更贴文字，字号加大）
    section_bar(ax, left, 0.448, width, 0.018, "问题三　完整输入可解释预测（冻结M1 + 分层解释）", fc=C_Q3[0], ec=C_Q3[1])
    y, h = 0.228, 0.202
    side_stack(ax, Lx, y, Lw, h, ["解释", "对象sc/sr", "8联盟", "局部窗", "忠实度"], ec=C_Q3[1], title_fc=C_Q3[0])
    side_stack(ax, Rx, y, Rw, h, ["边界", "非因果", "不计分", "无新参数"], ec=C_Q3[1], title_fc=C_Q3[0])
    link_sides(y + h / 2, C_Q3[1])
    sides.append((y, h, C_Q3[1]))

    # 上行：解释建模主链（三行正文 + 更大字号）
    q3_top = [
        ("冻结M1-EMT", "完整输入只读预测\n分类头 + 回归头\n参数不回训"),
        ("解释对象", "固定预测类 k*\nlogit 边际 sc\n原始强度 sr"),
        ("精确Shapley", "8联盟全前向\n分类/强度两套φ\n主导·支持·反对"),
        ("局部多尺度", "窗宽1/3/5连续蔽\n逐位置重要性\nTop互不重叠"),
    ]
    pw3, pg3 = 0.152, 0.010
    py3, ph3 = 0.352, 0.076
    for i, (title, body) in enumerate(q3_top):
        x = x0 + i * (pw3 + pg3)
        mini(ax, x, py3, pw3, ph3, body.split("\n"), fc=C_Q3B[0], ec=C_Q3[1], title=title, title_size=7.2, body_size=6.0)
        if i < 3:
            arrow(ax, (x + pw3, py3 + ph3 / 2), (x + pw3 + pg3, py3 + ph3 / 2), color=C_Q3[1], lw=0.9, ms=7)

    # 下行：遮蔽 / 验证 / 回溯
    q3b_y, q3b_h = 0.262, 0.080
    bw, bg = 0.212, 0.012
    q3_bot = [
        ("遮蔽基线", "文本正文→[UNK]\n音/视有效行置零\n长度与mask不变"),
        ("忠实度验证", "Top删10/20/30%\n对比等量随机删\n效率与排序稳定"),
        ("原素材回溯", "QG-HTA秒/帧映射\n文本offset定位\n异常显式标记"),
    ]
    for i, (title, body) in enumerate(q3_bot):
        x = x0 + i * (bw + bg)
        mini(ax, x, q3b_y, bw, q3b_h, body.split("\n"), fc=C_Q3B[0], ec=C_Q3[1], title=title, title_size=7.2, body_size=6.0)

    box(ax, (left, 0.236), width, 0.014, fc="#F7EFE8", ec=C_Q3[1], lw=0.85)
    txt(ax, mid, 0.243, "解释层无新增参数；归因=遮蔽敏感性，不作因果；附件4不计分", size=5.8, color=C_Q3[1])
    arrow(ax, (mid, 0.234), (mid, 0.222), color="#4A4A4A", lw=1.35, ms=10)

    # 交付（略压缩）
    section_bar(ax, left, 0.200, width, 0.016, "验证闭环与论文交付", fc=C_OUT[0], ec=C_OUT[1])
    y, h = 0.082, 0.098
    side_stack(ax, Lx, y, Lw, h, ["交付", "代码", "结果", "追溯"], ec=C_OUT[1], title_fc=C_OUT[0])
    side_stack(ax, Rx, y, Rw, h, ["提交", "专项CSV", "≤50MB"], ec=C_OUT[1], title_fc=C_OUT[0])
    link_sides(y + h / 2, C_OUT[1])
    sides.append((y, h, C_OUT[1]))

    for i, (title, body) in enumerate([
        ("问题一", "特征与图"),
        ("问题二", "四模型+附件3"),
        ("问题三", "解释卡+附件4"),
        ("证据链", "配置可追溯"),
    ]):
        mini(ax, x0 + i * (sw + g), 0.092, sw, 0.086, body.split("\n"), fc="#FFFFFF", ec=C_OUT[1], title=title, title_size=6.6, body_size=5.5)

    box(ax, (left, 0.032), width, 0.036, fc="#FAFAFA", ec="#666666", lw=0.95)
    txt(ax, mid, 0.054, "主线：数据 → QG-HTA → M3/JPI-DLFR → 冻结M1解释 → 交付", size=5.9, color="#222222")
    txt(ax, mid, 0.040, "旁路：问题一对齐复用；问题二选取完整输入M1作问题三预测器", size=5.6, color="#555555")

    for i in range(len(sides) - 1):
        y0, _, _ = sides[i]
        y1, h1, _ = sides[i + 1]
        y_from = y0 - 0.002
        y_to = y1 + h1 + 0.002
        if y_from <= y_to + 0.004:
            continue
        arrow(ax, (Lx + Lw / 2, y_from), (Lx + Lw / 2, y_to), color="#888888", lw=0.85, ms=7)
        arrow(ax, (Rx + Rw / 2, y_from), (Rx + Rw / 2, y_to), color="#888888", lw=0.85, ms=7)

    y_q1 = 0.768
    y_q3_bar = 0.243
    ax.plot([seam_r, seam_r], [y_q1, y_q3_bar], color=LINK_Q1, lw=1.3, zorder=5, solid_capstyle="round")
    arrow(ax, (left + width, y_q1), (seam_r, y_q1), color=LINK_Q1, lw=1.05, ms=7)
    arrow(ax, (seam_r, y_q3_bar), (left + width, y_q3_bar), color=LINK_Q1, lw=1.05, ms=7)
    txt(ax, seam_r, (y_q1 + y_q3_bar) / 2, "对齐复用", size=5.2, color=LINK_Q1, weight="bold", rotation=90)

    assert m1_yc is not None
    y_m1 = m1_yc
    y_q3_top = 0.457
    ax.plot([seam_l, seam_l], [y_m1, y_q3_top], color=LINK_Q2, lw=1.3, zorder=5, solid_capstyle="round")
    arrow(ax, (left, y_m1), (seam_l, y_m1), color=LINK_Q2, lw=1.05, ms=7)
    arrow(ax, (seam_l, y_q3_top), (left, y_q3_top), color=LINK_Q2, lw=1.05, ms=7)
    txt(ax, seam_l, (y_m1 + y_q3_top) / 2, "选取M1", size=5.2, color=LINK_Q2, weight="bold", rotation=90)

    fig.savefig(PNG_PATH, dpi=320, bbox_inches="tight", pad_inches=0.04, facecolor="white")
    plt.close(fig)
    print(f"[saved] {PNG_PATH}")


if __name__ == "__main__":
    draw()
