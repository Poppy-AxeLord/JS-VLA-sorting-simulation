# -*- coding: utf-8 -*-
"""
src/dashboard/charts.py —— 看板可复用 Plotly 图表构造器

设计目标（产品决策）：
- 把所有图表的「配色 / 字体 / 中文标签 / hover 模板」收口到一个文件，
  保证三个页面（总览 / 失败分析 / 版本对比）视觉风格统一、专业、B 端数据看板气质。
- 所有图表只接收「已经聚合好的数据」（list/dict），不在这里做任何重计算或仿真，
  符合 SPEC §12「看板仅从 storage 读已落库数据」的约定。
- 失败相关图表统一使用 SPEC §3 的 5 大类配色；非失败图表统一使用主蓝 #2563EB 色系。

所有函数返回 plotly.graph_objects.Figure，由页面用 st.plotly_chart 渲染。
"""

from __future__ import annotations

from typing import Any

import plotly.graph_objects as go


# ============================================================
# 一、全局配色与样式常量（唯一事实来源，改这里即可全局换肤）
# ============================================================

#: 产品主色（主蓝），SPEC §0 / §12 约定
PRIMARY = "#2563EB"
#: 主色的浅色/辅助梯度，用于柱状图渐变、对比辅助系列
PRIMARY_LIGHT = "#60A5FA"
PRIMARY_DARK = "#1E40AF"
#: 正向（成功 / 提升）与负向（失败 / 下降）语义色（统一设计规范 v1）
POSITIVE = "#10B981"
NEGATIVE = "#EF4444"
#: 中性灰，用于网格线、次要文字
GRID = "#E5E7EB"
TEXT_MUTED = "#6B7280"

#: SPEC §3 —— 5 大类失败分类配色（key=大类中文，与存储字段 failure_category 一致）
FAILURE_COLORS: dict[str, str] = {
    "感知类失败": "#5B8FF9",
    "理解类失败": "#5AD8A6",
    "规划类失败": "#F6BD16",
    "执行类失败": "#E8684A",
    "环境类失败": "#9270CA",
}

#: 大类英文 key → 中文（与 SPEC §3 对齐，供需要英文 key 的场景反查）
FAILURE_EN2CN: dict[str, str] = {
    "perception": "感知类失败",
    "understanding": "理解类失败",
    "planning": "规划类失败",
    "execution": "执行类失败",
    "environment": "环境类失败",
}

#: 难度配色（简单→中等→困难，颜色由浅到深，传达「越深越难」的直觉）
DIFFICULTY_COLORS: dict[str, str] = {
    "简单": "#60A5FA",
    "中等": "#2563EB",
    "困难": "#1E3A8A",
}

#: 统一字体（系统栈：SF Pro / 苹方优先，保证中文不乱码；退化到通用无衬线）
FONT_FAMILY = '-apple-system, "SF Pro", "PingFang SC", "Helvetica Neue", "Microsoft YaHei", sans-serif'

#: st.plotly_chart 统一渲染配置：隐藏英文 modebar（悬停工具条），演示更干净
PLOTLY_CONFIG: dict[str, Any] = {"displayModeBar": False}


def _base_layout(**overrides: Any) -> dict[str, Any]:
    """
    返回所有图表共享的基础 layout 字典。

    设计：透明背景以贴合 Streamlit 主题；统一字体与边距；
    图例横排置顶，符合数据看板阅读习惯。overrides 可覆盖任意字段。
    """
    layout: dict[str, Any] = dict(
        font=dict(family=FONT_FAMILY, size=13, color="#1F2937"),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=48, r=24, t=48, b=40),
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.02,
            xanchor="right",
            x=1,
            font=dict(size=12),
        ),
        hoverlabel=dict(
            bgcolor="#FFFFFF",
            bordercolor=GRID,
            font=dict(family=FONT_FAMILY, size=12, color="#1F2937"),
        ),
        colorway=[PRIMARY, PRIMARY_LIGHT, "#5AD8A6", "#F6BD16", "#9270CA", "#E8684A"],
    )
    layout.update(overrides)
    return layout


def _empty_figure(message: str = "暂无数据") -> go.Figure:
    """
    数据为空时返回一张带中文提示的占位图。

    产品考量：看板第一次打开 / 某筛选无结果时，绝不能白屏或报错，
    要给用户「这里本应有图、但当前没数据」的明确反馈。
    """
    fig = go.Figure()
    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color=TEXT_MUTED, family=FONT_FAMILY),
    )
    fig.update_layout(
        **_base_layout(
            xaxis=dict(visible=False),
            yaxis=dict(visible=False),
            height=260,
        )
    )
    return fig


def _style_axes(fig: go.Figure, x_title: str = "", y_title: str = "") -> go.Figure:
    """统一坐标轴样式：淡灰网格、无多余轴线、刻度文字次要灰、中文轴标题。"""
    fig.update_xaxes(
        title_text=x_title,
        showgrid=False,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED, size=12),
        title_font=dict(color=TEXT_MUTED, size=12),
    )
    fig.update_yaxes(
        title_text=y_title,
        showgrid=True,
        gridcolor=GRID,
        zeroline=False,
        linecolor=GRID,
        tickfont=dict(color=TEXT_MUTED, size=12),
        title_font=dict(color=TEXT_MUTED, size=12),
    )
    return fig


# ============================================================
# 二、柱状图：分组表现对比（按难度 / 按指令类型）
# ============================================================

def bar_chart(
    categories: list[str],
    values: list[float],
    *,
    title: str = "",
    y_title: str = "",
    x_title: str = "",
    is_percent: bool = True,
    colors: list[str] | None = None,
    text_suffix: str | None = None,
) -> go.Figure:
    """
    通用单系列竖向柱状图。

    参数
    ----
    categories : 横轴分类（如 ["简单","中等","困难"] 或 ["基础分拣",...]）
    values     : 对应数值（百分比传 0~1 的小数，is_percent=True 时自动 ×100）
    is_percent : 是否按百分比展示（影响柱顶文字与 hover、y 轴范围）
    colors     : 每个柱子的颜色列表；None 则全部用主蓝
    text_suffix: 柱顶文字单位后缀（覆盖 is_percent 的默认 "%"）
    """
    if not categories or not values:
        return _empty_figure()

    if is_percent:
        display_vals = [round(v * 100, 1) for v in values]
        suffix = text_suffix if text_suffix is not None else "%"
        hover = "%{x}<br>%{y:.1f}" + suffix + "<extra></extra>"
        y_range = [0, 100]
    else:
        display_vals = [round(v, 1) for v in values]
        suffix = text_suffix if text_suffix is not None else ""
        hover = "%{x}<br>%{y:.1f}" + suffix + "<extra></extra>"
        y_range = None

    bar_colors = colors if colors else PRIMARY

    fig = go.Figure(
        go.Bar(
            x=categories,
            y=display_vals,
            marker=dict(color=bar_colors, line=dict(width=0)),
            text=[f"{v}{suffix}" for v in display_vals],
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            hovertemplate=hover,
        )
    )
    layout = _base_layout(title=dict(text=title, font=dict(size=15)), height=340, showlegend=False)
    if y_range:
        layout["yaxis"] = dict(range=[y_range[0], y_range[1] + 8])
    fig.update_layout(**layout)
    return _style_axes(fig, x_title=x_title, y_title=y_title)


def grouped_bar_chart(
    categories: list[str],
    series: list[dict[str, Any]],
    *,
    title: str = "",
    y_title: str = "",
    x_title: str = "",
    is_percent: bool = True,
) -> go.Figure:
    """
    多系列分组柱状图（如同一难度下「成功率 vs 准确率」两根柱）。

    series : [{"name":"成功率","values":[...],"color":"#..."}, ...]
    """
    if not categories or not series:
        return _empty_figure()

    fig = go.Figure()
    default_colors = [PRIMARY, "#5AD8A6", "#F6BD16", "#9270CA"]
    for idx, s in enumerate(series):
        raw = s.get("values", [])
        vals = [round(v * 100, 1) for v in raw] if is_percent else [round(v, 1) for v in raw]
        suffix = "%" if is_percent else ""
        fig.add_trace(
            go.Bar(
                name=s.get("name", f"系列{idx+1}"),
                x=categories,
                y=vals,
                marker=dict(color=s.get("color", default_colors[idx % len(default_colors)])),
                hovertemplate="%{x}<br>" + s.get("name", "") + ": %{y:.1f}" + suffix + "<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=360,
            barmode="group",
            bargap=0.25,
            bargroupgap=0.08,
        )
    )
    return _style_axes(fig, x_title=x_title, y_title=y_title)


def horizontal_bar_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    x_title: str = "次数",
    colors: list[str] | None = None,
) -> go.Figure:
    """
    横向柱状图，用于「Top10 失败子类排行」。

    自动按数值升序排列（最大值在顶部，符合排行榜阅读习惯）。
    colors 可逐条着色（按子类所属大类配色）。
    """
    if not labels or not values:
        return _empty_figure()

    # 升序排列：Plotly 横向柱第一项在底部，故升序后最大值落在顶部
    order = sorted(range(len(values)), key=lambda i: values[i])
    labels = [labels[i] for i in order]
    values = [values[i] for i in order]
    bar_colors = [colors[i] for i in order] if colors else PRIMARY

    fig = go.Figure(
        go.Bar(
            x=values,
            y=labels,
            orientation="h",
            marker=dict(color=bar_colors),
            text=values,
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            hovertemplate="%{y}<br>%{x} 次<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=max(320, 36 * len(labels) + 80),
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text=x_title, showgrid=True, gridcolor=GRID, zeroline=False)
    fig.update_yaxes(showgrid=False, zeroline=False, automargin=True)
    return fig


# ============================================================
# 三、饼图：失败原因分布（5 类配色）
# ============================================================

def pie_chart(
    labels: list[str],
    values: list[float],
    *,
    title: str = "",
    color_map: dict[str, str] | None = None,
) -> go.Figure:
    """
    环形饼图（甜甜圈），用于失败原因 5 大类分布。

    color_map : 标签→颜色映射；失败分类传 FAILURE_COLORS。
    """
    if not labels or not values or sum(values) == 0:
        return _empty_figure("暂无失败数据（恭喜，全部成功）")

    cmap = color_map or FAILURE_COLORS
    colors = [cmap.get(lb, PRIMARY) for lb in labels]

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=values,
            hole=0.5,
            marker=dict(colors=colors, line=dict(color="#FFFFFF", width=2)),
            textinfo="label+percent",
            textfont=dict(size=12, family=FONT_FAMILY),
            hovertemplate="%{label}<br>%{value} 次（%{percent}）<extra></extra>",
            sort=False,
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=360,
            legend=dict(orientation="v", x=1.0, y=0.5, yanchor="middle"),
        )
    )
    # 环心总数标注
    fig.add_annotation(
        text=f"共<br>{int(sum(values))} 次",
        x=0.5,
        y=0.5,
        showarrow=False,
        font=dict(size=15, color="#1F2937", family=FONT_FAMILY),
    )
    return fig


# ============================================================
# 四、折线图：历史趋势（多次评测）
# ============================================================

def line_chart(
    x_labels: list[str],
    series: list[dict[str, Any]],
    *,
    title: str = "",
    y_title: str = "",
    is_percent: bool = True,
) -> go.Figure:
    """
    多系列折线图，用于历史趋势（v1→v2→v3 的成功率 / 准确率）
    或各类失败随版本的变化趋势。

    series : [{"name":"成功率","values":[...],"color":"#..."}, ...]
    """
    if not x_labels or not series:
        return _empty_figure()

    fig = go.Figure()
    default_colors = [PRIMARY, "#5AD8A6", "#F6BD16", "#E8684A", "#9270CA", "#5B8FF9"]
    for idx, s in enumerate(series):
        raw = s.get("values", [])
        vals = [round(v * 100, 1) for v in raw] if is_percent else [round(v, 1) for v in raw]
        suffix = "%" if is_percent else ""
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=vals,
                name=s.get("name", f"系列{idx+1}"),
                mode="lines+markers",
                line=dict(color=s.get("color", default_colors[idx % len(default_colors)]), width=3),
                marker=dict(size=8),
                hovertemplate="%{x}<br>" + s.get("name", "") + ": %{y:.1f}" + suffix + "<extra></extra>",
            )
        )
    layout = _base_layout(title=dict(text=title, font=dict(size=15)), height=360)
    if is_percent:
        layout["yaxis"] = dict(range=[0, 105], showgrid=True, gridcolor=GRID)
    fig.update_layout(**layout)
    return _style_axes(fig, y_title=y_title)


def stacked_area_chart(
    x_labels: list[str],
    series: list[dict[str, Any]],
    *,
    title: str = "",
    y_title: str = "失败次数",
) -> go.Figure:
    """
    堆叠面积图，用于「各类失败随版本的趋势」（5 类堆叠，配色用 §3）。
    series : [{"name":"感知类失败","values":[...],"color":"#5B8FF9"}, ...]
    """
    if not x_labels or not series:
        return _empty_figure()

    fig = go.Figure()
    for s in series:
        fig.add_trace(
            go.Scatter(
                x=x_labels,
                y=s.get("values", []),
                name=s.get("name", ""),
                mode="lines",
                stackgroup="one",
                line=dict(width=0.5, color=s.get("color", PRIMARY)),
                fillcolor=s.get("color", PRIMARY),
                hovertemplate="%{x}<br>" + s.get("name", "") + ": %{y} 次<extra></extra>",
            )
        )
    fig.update_layout(
        **_base_layout(title=dict(text=title, font=dict(size=15)), height=360)
    )
    return _style_axes(fig, y_title=y_title)


# ============================================================
# 五、雷达图：多版本综合能力对比（comparison 页核心）
# ============================================================

def radar_chart(
    dimensions: list[str],
    series: list[dict[str, Any]],
    *,
    title: str = "",
) -> go.Figure:
    """
    多版本雷达图。每个版本一条闭合多边形。

    dimensions : 维度中文名（成功率/准确率/速度/稳定性/低成本/恢复率）
    series     : [{"name":"v1","values":[0~1 归一化],"color":"#..."}, ...]
                 values 必须已归一化到 0~1（数值越大越好），由调用方按
                 metrics.yaml 的 radar_weight 维度计算并归一化。
    """
    if not dimensions or not series:
        return _empty_figure()

    fig = go.Figure()
    default_colors = [PRIMARY, "#5AD8A6", "#E8684A", "#9270CA"]
    # 版本超过 3 个时叠加填充会糊成一团：只画描边线，保证可读性
    many_series = len(series) > 3
    for idx, s in enumerate(series):
        vals = list(s.get("values", []))
        color = s.get("color", default_colors[idx % len(default_colors)])
        # 闭合多边形：首尾相接
        closed_dims = dimensions + [dimensions[0]]
        closed_vals = vals + [vals[0]] if vals else vals
        fig.add_trace(
            go.Scatterpolar(
                r=[round(v * 100, 1) for v in closed_vals],
                theta=closed_dims,
                name=s.get("name", f"版本{idx+1}"),
                fill=None if many_series else "toself",
                line=dict(color=color, width=2),
                opacity=0.9 if many_series else 0.55,
                hovertemplate="%{theta}: %{r:.0f}/100<extra>" + s.get("name", "") + "</extra>",
            )
        )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=440,
            polar=dict(
                radialaxis=dict(visible=True, range=[0, 100], gridcolor=GRID, tickfont=dict(size=10)),
                angularaxis=dict(tickfont=dict(size=12)),
                bgcolor="rgba(0,0,0,0)",
            ),
        )
    )
    return fig


# ============================================================
# 六、Delta 瀑布 / 提升对比（版本间各指标 delta）
# ============================================================

def delta_bar_chart(
    metric_names: list[str],
    deltas: list[float],
    *,
    title: str = "",
    is_percent: bool = True,
) -> go.Figure:
    """
    版本间各指标提升 / 下降的横向 delta 柱状图。
    正向（提升）用绿色，负向（下降）用红色，直观体现迭代收益。

    deltas : 差值（百分比传 0~1 小数，is_percent=True 自动 ×100）
    """
    if not metric_names or not deltas:
        return _empty_figure()

    display = [round(d * 100, 1) for d in deltas] if is_percent else [round(d, 1) for d in deltas]
    suffix = "pct" if is_percent else ""
    colors = [POSITIVE if d >= 0 else NEGATIVE for d in display]
    texts = [f"{'+' if d >= 0 else ''}{d}{suffix}" for d in display]

    fig = go.Figure(
        go.Bar(
            x=display,
            y=metric_names,
            orientation="h",
            marker=dict(color=colors),
            text=texts,
            textposition="outside",
            textfont=dict(size=12),
            cliponaxis=False,
            hovertemplate="%{y}<br>变化: %{text}<extra></extra>",
        )
    )
    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=15)),
            height=max(300, 42 * len(metric_names) + 80),
            showlegend=False,
        )
    )
    fig.update_xaxes(title_text=f"变化幅度（{suffix or '绝对值'}）", showgrid=True, gridcolor=GRID, zeroline=True, zerolinecolor=TEXT_MUTED)
    fig.update_yaxes(showgrid=False, automargin=True)
    return fig


# ============================================================
# 七、合成俯视场景图（由 scene_json 现场用 Plotly 绘制）
# ============================================================

#: 零件颜色中文名 → 绘图用 HEX（与 §2 零件库颜色字段对齐）
_COLOR_HEX: dict[str, str] = {
    "银色": "#C0C0C0",
    "蓝色": "#2563EB",
    "棕色": "#92400E",
    "黑色": "#1F2937",
    "白色": "#F3F4F6",
    "绿色": "#16A34A",
    "红色": "#DC2626",
}

#: 料盒区域配色（A/B/C 区淡色背景块）
_BIN_FILL: dict[str, str] = {
    "A": "rgba(91,143,249,0.12)",
    "B": "rgba(90,216,166,0.12)",
    "C": "rgba(246,189,22,0.12)",
}
_BIN_LINE: dict[str, str] = {
    "A": "#5B8FF9",
    "B": "#5AD8A6",
    "C": "#F6BD16",
}


def synthetic_scene_figure(scene: dict[str, Any] | None, *, title: str = "合成俯视场景图") -> go.Figure:
    """
    根据 TaskResult.scene（SceneState）现场绘制一张「合成俯视场景图」。

    用途：失败分析页选中某失败案例时，把当时的场景可视化出来——
    让 PM / 评审一眼看懂「这一帧机械臂面对的是什么布局」，
    遮挡零件用虚线描边标注，是定位失败原因的关键证据图。

    scene 结构（SPEC §5 SceneState）：
        { "parts": [{"part_id","code","name","color","size","pos":[x,y],"occluded"}...],
          "bins": {"A":..,"B":..,"C":..} }
    坐标系约定：x 向右、y 向上，取值大致 0~1（仿真侧归一化）；
    本函数对缺字段做了健壮处理，scene 为空时返回占位图。
    """
    if not scene or not isinstance(scene, dict):
        return _empty_figure("无场景数据")

    parts = scene.get("parts", []) or []
    bins = scene.get("bins", {}) or {}

    fig = go.Figure()

    # —— 1) 画三个料盒区域（俯视图右侧三条竖向条带 A/B/C）——
    #    布局产品决策：工作台主体在左 2/3，料盒在右 1/3，贴近真实分拣线俯视。
    bin_keys = ["A", "B", "C"]
    bin_x0 = 0.70
    for i, bk in enumerate(bin_keys):
        y0 = i / 3.0
        y1 = (i + 1) / 3.0
        fig.add_shape(
            type="rect",
            x0=bin_x0, x1=1.0, y0=y0 + 0.01, y1=y1 - 0.01,
            line=dict(color=_BIN_LINE.get(bk, PRIMARY), width=2, dash="dot"),
            fillcolor=_BIN_FILL.get(bk, "rgba(37,99,235,0.10)"),
            layer="below",
        )
        fig.add_annotation(
            x=(bin_x0 + 1.0) / 2, y=(y0 + y1) / 2,
            text=f"{bk}区", showarrow=False,
            font=dict(size=14, color=_BIN_LINE.get(bk, PRIMARY), family=FONT_FAMILY),
        )

    # —— 2) 工作台边界（左侧主区域）——
    fig.add_shape(
        type="rect",
        x0=0.02, x1=bin_x0 - 0.03, y0=0.02, y1=0.98,
        line=dict(color=GRID, width=1.5),
        fillcolor="rgba(243,244,246,0.5)",
        layer="below",
    )

    # —— 3) 画零件（色块 + 中文名标签；遮挡件加虚线红描边）——
    size_radius = {"小": 14, "中": 20, "大": 28}
    xs, ys, texts, marker_colors, marker_sizes, marker_lines = [], [], [], [], [], []
    line_colors, line_widths = [], []
    for p in parts:
        pos = p.get("pos", [0.4, 0.5]) or [0.4, 0.5]
        try:
            px, py = float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            px, py = 0.4, 0.5
        # 把可能超出 [0,1] 的坐标夹到工作台范围，避免画到料盒上
        px = min(max(px, 0.05), bin_x0 - 0.06)
        py = min(max(py, 0.05), 0.95)
        xs.append(px)
        ys.append(py)
        color_cn = p.get("color", "")
        marker_colors.append(_COLOR_HEX.get(color_cn, PRIMARY))
        marker_sizes.append(size_radius.get(p.get("size", "中"), 18))
        occluded = bool(p.get("occluded", False))
        line_colors.append(NEGATIVE if occluded else "#FFFFFF")
        line_widths.append(3 if occluded else 1.5)
        name = p.get("name") or p.get("code") or "零件"
        tag = "（遮挡）" if occluded else ""
        texts.append(f"{name}{tag}")

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs,
                y=ys,
                mode="markers+text",
                marker=dict(
                    size=marker_sizes,
                    color=marker_colors,
                    line=dict(color=line_colors, width=line_widths),
                    symbol="square",
                ),
                text=texts,
                textposition="top center",
                textfont=dict(size=10, family=FONT_FAMILY),
                hovertemplate="%{text}<br>位置: (%{x:.2f}, %{y:.2f})<extra></extra>",
                showlegend=False,
            )
        )

    fig.update_layout(
        **_base_layout(
            title=dict(text=title, font=dict(size=14)),
            height=320,  # 稍降高度：窄列下 1:1 锁比留白更少，仍保留 scaleanchor 防零件变形
            showlegend=False,
        )
    )
    fig.update_xaxes(range=[0, 1], visible=False, scaleanchor="y", scaleratio=1)
    fig.update_yaxes(range=[0, 1], visible=False)
    return fig


# ============================================================
# 八、优先级彩色标签 / 徽标（HTML 片段，供 st.markdown 渲染）
# ============================================================

#: 优化建议优先级配色（高/中/低，语义色对齐设计规范 v1）
PRIORITY_COLORS: dict[str, str] = {
    "高": "#EF4444",
    "中": "#F59E0B",
    "低": "#10B981",
}


def priority_badge_html(priority: str) -> str:
    """返回一个优先级彩色标签的 HTML 片段（用于 st.markdown(unsafe_allow_html=True)）。"""
    color = PRIORITY_COLORS.get(priority, TEXT_MUTED)
    return (
        f'<span style="background:{color};color:#fff;padding:2px 10px;'
        f'border-radius:6px;font-size:12px;font-weight:600;">{priority}优先级</span>'
    )


def badge_html(text: str, color: str = PRIMARY, *, filled: bool = True) -> str:
    """通用彩色徽标 HTML（顶部 VLA 后端 / Mock 仿真状态徽标用）。标签圆角统一 6px。"""
    if filled:
        return (
            f'<span style="background:{color};color:#fff;padding:3px 12px;'
            f'border-radius:6px;font-size:12px;font-weight:600;">{text}</span>'
        )
    return (
        f'<span style="border:1px solid {color};color:{color};padding:3px 12px;'
        f'border-radius:6px;font-size:12px;font-weight:600;">{text}</span>'
    )


# ============================================================
# 九、闭环 / 架构流程条（横向 HTML 流程图，供 3 分钟讲解“一图看懂”）
# ============================================================

#: 端到端闭环的 8 个环节：中文名 + 一句职责 + 配色（与 §3 失败大类色系呼应）。
#: 用于「总览页顶部」的横向流程条，帮助使用者一眼看清系统全貌与数据闭环。
_CLOSED_LOOP_STEPS: list[dict[str, str]] = [
    {"name": "指令", "desc": "自然语言分拣指令", "color": "#2563EB"},
    {"name": "感知", "desc": "相机/视觉看零件", "color": "#5B8FF9"},
    {"name": "理解", "desc": "VLA 解析意图", "color": "#5AD8A6"},
    {"name": "规划", "desc": "顺序/路径/优先级", "color": "#F6BD16"},
    {"name": "执行", "desc": "抓取→放置", "color": "#E8684A"},
    {"name": "评测", "desc": "四组指标量化", "color": "#2563EB"},
    {"name": "失败归因", "desc": "5 类 MECE 定位", "color": "#9270CA"},
    {"name": "优化建议", "desc": "按 ROI 排序", "color": "#16A34A"},
]


def closed_loop_flow_html() -> str:
    """
    返回「指令→感知→理解→规划→执行→评测→失败归因→优化建议」横向流程条 HTML。

    产品用途：这是 3 分钟讲解的「地图」——一眼看清系统端到端链路 + 数据闭环
    （最后一环「优化建议」用回流箭头指回「指令」，强调迭代闭环）。用纯内联样式的
    flex 布局，随容器宽度自适应，窄屏自动换行，不依赖任何外部资源。
    """
    # 「箭头 + 下一个节点」打包进同一个不换行的子容器：换行时箭头随节点成组移动，
    # 永远不会孤立落在行首 / 行尾（窄窗自动换行也保持视觉完整）。
    groups: list[str] = []
    for i, step in enumerate(_CLOSED_LOOP_STEPS):
        color = step["color"]
        chip = (
            '<div style="display:flex;flex-direction:column;align-items:center;'
            'min-width:88px;flex:1 1 88px;">'
            f'<div style="background:{color};color:#fff;padding:6px 10px;border-radius:10px;'
            f'font-size:13px;font-weight:700;white-space:nowrap;">{step["name"]}</div>'
            f'<div style="color:{TEXT_MUTED};font-size:11px;margin-top:4px;'
            f'text-align:center;line-height:1.25;">{step["desc"]}</div>'
            '</div>'
        )
        arrow = (
            f'<div style="color:{TEXT_MUTED};font-size:16px;align-self:flex-start;'
            'margin-top:6px;flex:0 0 auto;padding:0 2px;">→</div>'
            if i > 0 else ''
        )
        groups.append(
            '<div style="display:flex;flex-wrap:nowrap;align-items:flex-start;'
            f'flex:1 1 auto;">{arrow}{chip}</div>'
        )

    inner = "".join(groups)
    return (
        '<div style="border:1px solid #E5E7EB;border-radius:12px;padding:14px 16px;'
        'background:linear-gradient(90deg,#F8FAFF,#FFFFFF);margin-bottom:6px;">'
        '<div style="display:flex;flex-wrap:wrap;align-items:flex-start;gap:6px 4px;'
        f'justify-content:space-between;">{inner}</div>'
        f'<div style="margin-top:10px;color:{POSITIVE};font-size:12px;font-weight:600;">'
        '闭环：优化建议 → 回到指令重跑评测，涨了证明归因对、没涨就回炉——迭代闭环即本产品核心方法论。'
        '</div>'
        '</div>'
    )
