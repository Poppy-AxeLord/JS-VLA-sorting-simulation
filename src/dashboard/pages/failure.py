# -*- coding: utf-8 -*-
"""
src/dashboard/pages/failure.py —— 「失败分析」页（项目核心亮点之一）

回答的核心问题：任务为什么失败？失败集中在哪一类？典型案例长什么样？

页面结构（SPEC §12）：
1) 失败原因分布饼图（5 大类，统一 §3 配色）
2) Top10 失败子类排行（横向柱，逐条按所属大类着色）
3) 各类失败随版本的趋势（多版本堆叠面积，看优化是否消灭了某类失败）
4) 选择某一大类 → 失败案例列表：指令 / 难度 / 模型输出 / 失败原因
   + 由 scene_json 现场用 Plotly 绘制的「合成俯视场景图」（关键证据图）

数据来源：
- 失败明细优先用 storage.get_failure_cases()（failure_cases 表，含 scene_json / model_output）；
- 分布 / 趋势用 task_results 的 failure_category / failure_subtype 聚合（覆盖所有失败任务）。
全部 try/except 守卫，缺数据降级为中文提示。
"""

from __future__ import annotations

import json
from typing import Any

import streamlit as st

from src.dashboard import charts

#: 5 大类固定顺序（保证饼图 / 图例 / 下拉一致）
_CATEGORY_ORDER = ["感知类失败", "理解类失败", "规划类失败", "执行类失败", "环境类失败"]


# ============================================================
# 一、数据读取与聚合辅助
# ============================================================
def _safe_task_results(storage, run_id: Any) -> list[dict]:
    try:
        rows = storage.get_task_results(run_id)
        return [dict(r) for r in rows] if rows else []
    except Exception as exc:  # pragma: no cover
        st.warning(f"读取任务结果失败，已降级：{exc}")
        return []


def _safe_failure_cases(storage, filters: dict) -> list[dict]:
    """
    安全读取失败案例（failure_cases 表）。

    storage.get_failure_cases(filters) 的 filters 约定支持按 run_id / failure_category
    过滤；不同实现可能签名略有差异，这里对两种常见调用方式都做了兜底。
    """
    try:
        rows = storage.get_failure_cases(filters)
        return [dict(r) for r in rows] if rows else []
    except TypeError:
        # 兼容无参或关键字参数实现
        try:
            rows = storage.get_failure_cases(**filters)
            return [dict(r) for r in rows] if rows else []
        except Exception as exc:  # pragma: no cover
            st.warning(f"读取失败案例失败，已降级：{exc}")
            return []
    except Exception as exc:  # pragma: no cover
        st.warning(f"读取失败案例失败，已降级：{exc}")
        return []


def _category_distribution(rows: list[dict]) -> dict[str, int]:
    """统计每个失败大类的次数（只计入 success=0 且有 failure_category 的行）。"""
    dist: dict[str, int] = {}
    for r in rows:
        if r.get("success"):
            continue
        cat = r.get("failure_category")
        if not cat:
            continue
        dist[cat] = dist.get(cat, 0) + 1
    return dist


def _subtype_top10(rows: list[dict]) -> list[dict]:
    """统计失败子类 Top10，并带回其所属大类（用于按大类配色）。"""
    counter: dict[tuple[str, str], int] = {}
    for r in rows:
        if r.get("success"):
            continue
        sub = r.get("failure_subtype")
        cat = r.get("failure_category") or ""
        if not sub:
            continue
        counter[(sub, cat)] = counter.get((sub, cat), 0) + 1
    items = [
        {"subtype": k[0], "category": k[1], "count": v}
        for k, v in counter.items()
    ]
    items.sort(key=lambda x: x["count"], reverse=True)
    return items[:10]


# ============================================================
# 二、主渲染函数
# ============================================================
def render(storage, ctx: dict) -> None:
    current_run = ctx.get("current_run")
    runs = ctx.get("runs", [])

    st.subheader("失败分析")

    if not current_run:
        st.info("暂无评测数据。请运行 `bash run_benchmark.sh` 生成评测结果。")
        return

    run_id = current_run.get("id")
    rows = _safe_task_results(storage, run_id)
    dist = _category_distribution(rows)

    total_fail = sum(dist.values())
    total_task = len(rows)
    if total_task:
        st.caption(
            f"当前运行共 {total_task} 个任务，其中失败 {total_fail} 个"
            f"（失败率 {total_fail / total_task * 100:.1f}%）。下方为失败的归因诊断。"
        )

    # ---------- 1) & 2) 分布饼 + Top10 子类（双列）----------
    col_left, col_right = st.columns([1, 1.2])

    with col_left:
        st.markdown("#### 失败原因分布（5 大类）")
        cats = [c for c in _CATEGORY_ORDER if c in dist]
        vals = [dist[c] for c in cats]
        if cats:
            fig = charts.pie_chart(cats, vals, title="按失败大类占比", color_map=charts.FAILURE_COLORS)
            with st.container(border=True):  # 白卡容器：图表浮于浅灰页面
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
        else:
            st.success("该运行没有失败任务，无需归因。")

    with col_right:
        st.markdown("#### Top10 失败子类排行")
        top10 = _subtype_top10(rows)
        if top10:
            labels = [t["subtype"] for t in top10]
            counts = [t["count"] for t in top10]
            colors = [charts.FAILURE_COLORS.get(t["category"], charts.PRIMARY) for t in top10]
            fig = charts.horizontal_bar_chart(labels, counts, title="出现次数最多的失败子类", colors=colors)
            with st.container(border=True):
                st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
                st.caption("颜色对应所属大类；排在前面的子类是优先攻坚对象。")
        else:
            st.info("无失败子类数据。")

    st.divider()

    # ---------- 3) 各类失败趋势（多版本堆叠面积）----------
    st.markdown("#### 各类失败趋势（随版本）")
    _render_failure_trend(storage, runs)

    st.divider()

    # ---------- 4) 失败案例列表（选择某大类 → 列表 + 场景图）----------
    st.markdown("#### 失败案例详查")
    _render_case_explorer(storage, run_id, rows, dist)


# ============================================================
# 三、子组件：失败趋势
# ============================================================
def _render_failure_trend(storage, runs: list[dict]) -> None:
    """
    跨版本统计每个大类的失败次数，画成堆叠面积图。

    产品价值：能直观看到「某类失败是否随迭代被消灭 / 转移」，
    例如 v2 优化感知后感知类失败明显收窄，是迭代有效性的证据。
    """
    if not runs:
        st.info("无历史评测，无法展示失败趋势。")
        return

    ordered = list(reversed(runs))  # 时间正序
    x_labels = []
    # cat -> 各版本次数列表
    series_map: dict[str, list[int]] = {c: [] for c in _CATEGORY_ORDER}

    # 逐版本查询 task_results 并聚合，版本较多时避免无反馈卡顿
    with st.spinner("正在聚合各版本失败数据……"):
        for r in ordered:
            ver = r.get("version", "?")
            wm = "+WM" if r.get("world_model_on") else ""
            x_labels.append(f"{ver}{wm}")
            rows = _safe_task_results(storage, r.get("id"))
            dist = _category_distribution(rows)
            for c in _CATEGORY_ORDER:
                series_map[c].append(dist.get(c, 0))

    # 过滤掉全 0 的类，避免图例噪音
    series = [
        {"name": c, "values": series_map[c], "color": charts.FAILURE_COLORS[c]}
        for c in _CATEGORY_ORDER
        if any(series_map[c])
    ]
    if not series:
        st.success("各版本均无失败记录。")
        return

    fig = charts.stacked_area_chart(x_labels, series, title="各类失败次数随版本变化")
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)


# ============================================================
# 四、子组件：失败案例浏览器（含合成俯视场景图）
# ============================================================
def _parse_scene(case: dict) -> dict | None:
    """
    从失败案例记录里解析出 scene（SceneState）。

    优先取 failure_cases.scene_json（字符串），其次取 task_results.detail_json 里的 scene。
    解析失败返回 None，由场景图函数显示「无场景数据」占位。
    """
    raw = case.get("scene_json") or case.get("scene")
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, str) and raw.strip():
        try:
            return json.loads(raw)
        except (json.JSONDecodeError, ValueError):
            return None
    # 兜底：从 detail_json 里掏 scene
    detail = case.get("detail_json")
    if isinstance(detail, str) and detail.strip():
        try:
            d = json.loads(detail)
            if isinstance(d, dict):
                return d.get("scene")
        except (json.JSONDecodeError, ValueError):
            return None
    return None


def _render_case_explorer(storage, run_id: Any, rows: list[dict], dist: dict[str, int]) -> None:
    """
    选择一个失败大类 → 列出该类失败案例（指令 / 难度 / 模型输出 / 失败原因）
    → 选中某个案例 → 现场用 Plotly 画出当时的合成俯视场景图。
    """
    cats_present = [c for c in _CATEGORY_ORDER if dist.get(c)]
    if not cats_present:
        st.success("当前运行无失败案例可查。")
        return

    selected_cat = st.selectbox(
        "选择失败大类查看案例",
        options=cats_present,
        format_func=lambda c: f"{c}（{dist.get(c, 0)} 例）",
        help="切换大类，下方列出该类的具体失败案例与场景图",
    )

    # 先尝试从 failure_cases 表取（含 scene_json / model_output），失败再从 task_results 兜底
    cases = _safe_failure_cases(storage, {"run_id": run_id, "failure_category": selected_cat})
    if not cases:
        cases = [
            r for r in rows
            if (not r.get("success")) and r.get("failure_category") == selected_cat
        ]

    if not cases:
        st.info(f"「{selected_cat}」暂无可展示的案例明细。")
        return

    # 案例下拉：用「任务ID + 指令摘要」作标签
    def _case_label(i: int) -> str:
        c = cases[i]
        instr = (c.get("instruction") or "")[:24]
        tid = c.get("task_id", f"#{i}")
        diff = c.get("difficulty", "")
        return f"{tid}｜{diff}｜{instr}"

    idx = st.selectbox(
        f"选择「{selected_cat}」中的案例",
        options=list(range(len(cases))),
        format_func=_case_label,
    )
    case = cases[idx]

    # —— 左：案例文字信息；右：合成俯视场景图 ——
    info_col, scene_col = st.columns([1, 1])

    with info_col:
        st.markdown("**案例详情**")
        st.markdown(
            f"- **任务指令**：{case.get('instruction', '—')}\n"
            f"- **难度**：{case.get('difficulty', '—')}\n"
            f"- **失败大类**：{case.get('failure_category', '—')}\n"
            f"- **失败子类**：{case.get('failure_subtype', '—')}"
        )
        # 失败原因高亮
        reason = case.get("failure_reason") or "（无详细原因）"
        st.markdown(
            charts.badge_html("失败原因", charts.NEGATIVE) + f"  {reason}",
            unsafe_allow_html=True,
        )
        # 模型输出（VLA 的 ActionPlan / 文字推理）
        model_output = case.get("model_output")
        if model_output:
            with st.expander("查看模型输出（VLA 推理 / 动作计划）", expanded=False):
                if isinstance(model_output, str):
                    # 尝试美化 JSON；失败则原样展示
                    try:
                        st.json(json.loads(model_output))
                    except (json.JSONDecodeError, ValueError):
                        st.code(model_output, language="text")
                else:
                    st.json(model_output)

    with scene_col:
        st.markdown("**合成俯视场景图（失败时刻）**")
        scene = _parse_scene(case)
        fig = charts.synthetic_scene_figure(scene, title="俯视布局（红框=遮挡件）")
        with st.container(border=True):
            st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
            st.caption("方块=零件（颜色对应材质色），右侧 A/B/C 为分拣料盒；遮挡件以红色描边标注。")

    # 全部案例的紧凑表格，便于横向扫读
    with st.expander(f"「{selected_cat}」全部 {len(cases)} 例（表格）", expanded=False):
        table_rows = [
            {
                "任务ID": c.get("task_id", ""),
                "难度": c.get("difficulty", ""),
                "失败子类": c.get("failure_subtype", ""),
                "指令": c.get("instruction", ""),
                "失败原因": c.get("failure_reason", ""),
            }
            for c in cases
        ]
        # column_config 控宽：长文本列（指令/失败原因）给足宽度，短列收紧，窄屏不挤压
        st.dataframe(
            table_rows,
            use_container_width=True,
            hide_index=True,
            column_config={
                "任务ID": st.column_config.TextColumn("任务ID", width="small"),
                "难度": st.column_config.TextColumn("难度", width="small"),
                "失败子类": st.column_config.TextColumn("失败子类", width="medium"),
                "指令": st.column_config.TextColumn("指令", width="large"),
                "失败原因": st.column_config.TextColumn("失败原因", width="large"),
            },
        )
