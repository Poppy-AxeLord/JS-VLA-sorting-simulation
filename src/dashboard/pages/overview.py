# -*- coding: utf-8 -*-
"""
src/dashboard/pages/overview.py —— 「总览」页

回答的核心问题：当前这次评测整体表现如何？哪类难度 / 哪类指令是短板？历史在变好吗？

页面结构（SPEC §12）：
1) 核心指标卡片：北极星「任务成功率」+ 核心「分拣准确率」+ 过程「平均任务耗时」「吞吐」
2) 按难度表现对比（柱）：简单 / 中等 / 困难 的成功率 + 准确率
3) 按指令类型表现对比（柱）：5 类任务（基础/条件/优先级/批量/异常）的成功率
4) 历史趋势（折线）：多次评测的成功率 / 准确率随版本变化

数据全部来自 storage 已落库结果（benchmark_runs + task_results），不在看板内重算仿真。
所有读取均 try/except 守卫，缺数据时给中文降级提示而非崩溃。
"""

from __future__ import annotations

from typing import Any

import streamlit as st

from src.dashboard import charts


# ============================================================
# 一、辅助：从 task_results 行做分组聚合（看板侧的轻量统计）
# ============================================================
def _safe_get_task_results(storage, run_id: Any) -> list[dict]:
    """安全读取某次运行的全部任务结果；异常降级为空列表。"""
    try:
        rows = storage.get_task_results(run_id)
        return [dict(r) for r in rows] if rows else []
    except Exception as exc:  # pragma: no cover
        st.warning(f"读取任务结果失败，已降级：{exc}")
        return []


def _group_metrics(rows: list[dict], key: str) -> dict[str, dict[str, float]]:
    """
    按某字段（difficulty / type）分组，计算每组的成功率与平均分拣准确率。

    这是看板侧的「轻量再聚合」：benchmark_runs 存的是整体指标，而分组维度
    需要从 task_results 明细现算。逻辑刻意简单透明，便于评审核对口径。

    返回：{ 分组名: {"success_rate":.., "sort_accuracy":.., "count":..} }
    """
    groups: dict[str, dict[str, float]] = {}
    for r in rows:
        g = r.get(key) or "未知"
        bucket = groups.setdefault(g, {"_succ": 0.0, "_acc": 0.0, "count": 0.0})
        bucket["_succ"] += 1.0 if r.get("success") else 0.0
        bucket["_acc"] += float(r.get("sort_accuracy") or 0.0)
        bucket["count"] += 1.0
    out: dict[str, dict[str, float]] = {}
    for g, b in groups.items():
        n = b["count"] or 1.0
        out[g] = {
            "success_rate": b["_succ"] / n,
            "sort_accuracy": b["_acc"] / n,
            "count": int(b["count"]),
        }
    return out


def _order_keys(keys: list[str], preferred: list[str]) -> list[str]:
    """把分组键按业务习惯排序（难度：简单→困难；类型：按 preferred 顺序）。"""
    ordered = [k for k in preferred if k in keys]
    ordered += [k for k in keys if k not in preferred]
    return ordered


# ============================================================
# 二、主渲染函数
# ============================================================
def render(storage, ctx: dict) -> None:
    """总览页入口。storage：数据源；ctx：app.py 传入的上下文。"""
    current_run = ctx.get("current_run")
    runs = ctx.get("runs", [])

    st.subheader("评测总览")

    # ---------- 0) 端到端闭环流程条（3 分钟讲解的“地图”，一眼看清系统全貌）----------
    # 无论是否有数据都先展示，帮助使用者快速建立系统心智模型。
    try:
        st.markdown(charts.closed_loop_flow_html(), unsafe_allow_html=True)
    except Exception:  # pragma: no cover —— 流程条纯装饰，异常绝不阻断主内容
        pass

    if not current_run:
        st.info("暂无评测数据。请在项目根目录运行 `bash run_benchmark.sh` 生成评测结果。")
        return

    run_id = current_run.get("id")
    rows = _safe_get_task_results(storage, run_id)

    # ---------- 1) 核心指标卡片 ----------
    st.markdown("#### 核心指标")
    _render_metric_cards(current_run, rows, runs)

    st.markdown("")  # 间隔

    # ---------- 2) & 3) 分组表现对比（双列）----------
    col_left, col_right = st.columns(2)

    with col_left:
        st.markdown("#### 按难度表现对比")
        diff_stats = _group_metrics(rows, "difficulty")
        diff_keys = _order_keys(list(diff_stats.keys()), ["简单", "中等", "困难"])
        if diff_keys:
            fig = charts.grouped_bar_chart(
                categories=diff_keys,
                series=[
                    {"name": "任务成功率", "values": [diff_stats[k]["success_rate"] for k in diff_keys], "color": charts.PRIMARY},
                    {"name": "分拣准确率", "values": [diff_stats[k]["sort_accuracy"] for k in diff_keys], "color": "#5AD8A6"},
                ],
                title="不同难度下的成功率与准确率",
                y_title="百分比",
            )
            with st.container(border=True):  # 白卡容器：图表浮于浅灰页面
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
                st.caption("越难的场景（遮挡 / 相似件增多）成功率越低，是优化的主战场。")
        else:
            st.info("无难度分组数据。")

    with col_right:
        st.markdown("#### 按指令类型表现对比")
        type_stats = _group_metrics(rows, "type")
        type_pref = ["基础分拣", "条件分拣", "优先级分拣", "批量分拣", "异常场景"]
        type_keys = _order_keys(list(type_stats.keys()), type_pref)
        if type_keys:
            fig = charts.bar_chart(
                categories=type_keys,
                values=[type_stats[k]["success_rate"] for k in type_keys],
                title="不同指令类型的任务成功率",
                y_title="任务成功率",
                is_percent=True,
            )
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
                st.caption("模糊 / 批量 / 异常类指令通常更难，反映理解与规划能力短板。")
        else:
            st.info("无指令类型分组数据。")

    st.divider()

    # ---------- 4) 历史趋势 ----------
    st.markdown("#### 历史趋势（多次评测）")
    _render_trend(runs)


# ============================================================
# 三、子组件
# ============================================================
def _find_previous_run(run: dict, runs: list[dict]) -> dict | None:
    """
    在时间倒序的 runs 里找到「当前运行的上一次评测」（时间上更早的相邻一条）。

    用于指标卡 delta：找不到（首个版本 / runs 缺失）返回 None，卡片不显示 delta。
    """
    try:
        ids = [r.get("id") for r in (runs or [])]
        i = ids.index(run.get("id"))
        if i + 1 < len(runs):
            return runs[i + 1]
    except (ValueError, AttributeError):
        pass
    return None


def _render_metric_cards(run: dict, rows: list[dict], runs: list[dict] | None = None) -> None:
    """
    四张核心指标卡（用 st.metric）：
    - 任务成功率（北极星，效果）
    - 分拣准确率（核心，效果）
    - 平均任务耗时（过程，效率）
    - 吞吐 件/分钟（过程，效率）

    优先用 benchmark_runs 已落库的汇总值；缺失则从 task_results 现算兜底。
    delta 展示相对上一次评测的变化（耗时用 inverse 配色：降低才是好），
    增强「好 / 坏」的直觉；无上一次评测时不显示 delta。
    """
    success_rate = run.get("success_rate")
    sort_accuracy = run.get("sort_accuracy")
    avg_ms = run.get("avg_duration_ms")
    throughput = run.get("throughput")

    # 兜底：汇总值缺失时从明细现算
    if rows:
        n = len(rows)
        if success_rate is None:
            success_rate = sum(1 for r in rows if r.get("success")) / n
        if sort_accuracy is None:
            sort_accuracy = sum(float(r.get("sort_accuracy") or 0) for r in rows) / n
        if avg_ms is None:
            avg_ms = sum(int(r.get("duration_ms") or 0) for r in rows) / n

    # 相对上一次评测的 delta（找不到上一次则全部为 None，st.metric 不渲染 delta）
    prev = _find_previous_run(run, runs or [])
    d_sr = d_acc = d_ms = d_tp = None
    if prev:
        if prev.get("success_rate") is not None and success_rate is not None:
            d_sr = f"{(success_rate - float(prev['success_rate'])) * 100:+.1f} pct"
        if prev.get("sort_accuracy") is not None and sort_accuracy is not None:
            d_acc = f"{(sort_accuracy - float(prev['sort_accuracy'])) * 100:+.1f} pct"
        if prev.get("avg_duration_ms") is not None and avg_ms is not None:
            d_ms = f"{(float(avg_ms) - float(prev['avg_duration_ms'])) / 1000:+.2f} s"
        if prev.get("throughput") is not None and throughput is not None:
            d_tp = f"{float(throughput) - float(prev['throughput']):+.1f} 件/分"

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        st.metric(
            "任务成功率（北极星）",
            f"{(success_rate or 0) * 100:.1f}%",
            delta=d_sr,
            help="成功任务数 / 总任务数。衡量整体可用性的北极星指标。delta 为相对上一次评测的变化。",
        )
    with c2:
        st.metric(
            "分拣准确率",
            f"{(sort_accuracy or 0) * 100:.1f}%",
            delta=d_acc,
            help="正确分拣零件数 / 目标零件总数。效果类核心指标。delta 为相对上一次评测的变化。",
        )
    with c3:
        st.metric(
            "平均任务耗时",
            f"{(avg_ms or 0) / 1000:.2f} s",
            delta=d_ms,
            delta_color="inverse",
            help="单个分拣任务的平均完成时间（毫秒换算为秒）。效率类过程指标；耗时下降（绿色）代表变好。",
        )
    with c4:
        st.metric(
            "吞吐量",
            f"{(throughput or 0):.1f} 件/分",
            delta=d_tp,
            help="单位时间分拣件数。效率类过程指标。delta 为相对上一次评测的变化。",
        )


def _render_trend(runs: list[dict]) -> None:
    """
    历史趋势折线：横轴为各版本（按时间正序），纵轴为成功率 / 准确率。

    runs 由 app.py 传入（list_runs 约定时间倒序），这里反转为正序，
    让趋势从左到右呈现「随迭代演进」的方向。
    """
    if not runs or len(runs) < 1:
        st.info("历史评测不足，暂无趋势可展示。运行多次 `run_benchmark.sh` 后即可看到迭代曲线。")
        return

    ordered = list(reversed(runs))  # 时间正序
    x_labels = []
    for r in ordered:
        ver = r.get("version", "?")
        wm = "+WM" if r.get("world_model_on") else ""
        x_labels.append(f"{ver}{wm}")

    success_series = [float(r.get("success_rate") or 0) for r in ordered]
    accuracy_series = [float(r.get("sort_accuracy") or 0) for r in ordered]

    fig = charts.line_chart(
        x_labels=x_labels,
        series=[
            {"name": "任务成功率", "values": success_series, "color": charts.PRIMARY},
            {"name": "分拣准确率", "values": accuracy_series, "color": "#5AD8A6"},
        ],
        title="历次评测：成功率 / 准确率走势",
        y_title="百分比",
        is_percent=True,
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)

        # 简短的趋势文字结论（产品视角）
        if len(success_series) >= 2:
            delta = (success_series[-1] - success_series[0]) * 100
            arrow = "提升" if delta >= 0 else "下降"
            st.caption(
                f"自首版以来任务成功率累计{arrow} {abs(delta):.1f} 个百分点"
                f"（{success_series[0]*100:.0f}% → {success_series[-1]*100:.0f}%），迭代方向正确。"
            )
