# -*- coding: utf-8 -*-
"""
src/dashboard/pages/comparison.py —— 「版本对比」页（最体现 PM 能力的页面）

回答的核心问题：v1→v2→v3 到底好了多少？好在哪、差在哪？下一步该投哪？

页面结构（SPEC §12）：
1) 多版本雷达图：维度取自 metrics.yaml 的 radar_weight 维度
   （成功率 / 准确率 / 速度 / 稳定性 / 低成本 / 恢复率），全部归一化到 0~1。
2) 版本间提升 / 下降分析：相邻版本（v1→v2、v2→v3）各指标 delta 横向柱。
3) 优化建议列表：调用 evaluation.recommendation.generate() 生成，
   带优先级彩色标签 + 预期收益；并提供「运行新评测」说明（指向 run_benchmark.sh）。

数据来源：benchmark_runs（metrics_json 含完整指标）+ task_results（失败聚合给建议引擎）。
metrics.yaml / recommendation 模块均 try/except 守卫，缺失时用内置降级方案，绝不崩溃。
"""

from __future__ import annotations

import json
import os
from typing import Any

import streamlit as st

from src.dashboard import charts


# ============================================================
# 一、雷达维度定义（单一事实来源 = config/metrics.yaml 的 radar_weight）
# ============================================================
# 【P0 修复｜消除 spec/code 漂移】
# 历史问题：本文件曾硬编码一份雷达维度列表，但文档/metrics.yaml 声称「雷达维度取自
# metrics.yaml 的 radar_weight」——号称单一来源、实为硬编码，两处一旦不一致（例如
# 低成本维度到底用 avg_retry 还是 human_intervention_rate）就产生 spec/code 漂移。
#
# 现做法（Option A｜兑现承诺）：启动时真正读取 metrics.yaml，凡 radar_weight > 0 的
# 指标即为一个雷达维度；方向由 group / key 语义推断（成本、效率类为逆向 down）。
# metrics.yaml 缺失 / 无 pyyaml 时，退化到下方 _FALLBACK_RADAR_DIMENSIONS——该降级表
# 已与 metrics.yaml 的当前声明严格对齐（低成本 = avg_retry），保证「代码=文档」。

#: 逆向指标（越小越好，归一化时取反）——按 metrics.yaml 的 key 显式列出，
#: 避免用 group 字符串做模糊判断，读起来一目了然、便于评审核对口径。
_RADAR_DOWN_METRICS: set[str] = {"avg_task_ms", "avg_retry", "human_intervention_rate"}

#: metrics key → 雷达维度中文短标签（仅覆盖会进雷达的 6 个指标）。
_RADAR_LABELS: dict[str, str] = {
    "task_success_rate": "成功率",
    "sort_accuracy": "准确率",
    "avg_task_ms": "速度",
    "grasp_success_rate": "稳定性",
    "avg_retry": "低成本",
    "human_intervention_rate": "低成本",
    "recovery_rate": "恢复率",
}

#: 降级雷达维度：仅当读取 metrics.yaml 失败时使用。
#: 【关键】此表须与 config/metrics.yaml 中 radar_weight>0 的指标严格一致——
#: 低成本维度取 avg_retry（与 metrics.yaml 第 141~148 行、docs/metrics_system.md 第四节一致）。
_FALLBACK_RADAR_DIMENSIONS: list[dict[str, Any]] = [
    {"label": "成功率", "metric": "task_success_rate", "direction": "up"},
    {"label": "准确率", "metric": "sort_accuracy", "direction": "up"},
    {"label": "速度", "metric": "avg_task_ms", "direction": "down"},
    {"label": "稳定性", "metric": "grasp_success_rate", "direction": "up"},
    {"label": "低成本", "metric": "avg_retry", "direction": "down"},
    {"label": "恢复率", "metric": "recovery_rate", "direction": "up"},
]

#: 版本对比里展示 delta 的核心指标（中文名 → key + 是否百分比 + 方向）
_DELTA_METRICS: list[dict[str, Any]] = [
    {"label": "任务成功率", "metric": "task_success_rate", "percent": True, "good": "up"},
    {"label": "分拣准确率", "metric": "sort_accuracy", "percent": True, "good": "up"},
    {"label": "抓取成功率", "metric": "grasp_success_rate", "percent": True, "good": "up"},
    {"label": "异常恢复率", "metric": "recovery_rate", "percent": True, "good": "up"},
    {"label": "误拣率", "metric": "mis_pick_rate", "percent": True, "good": "down"},
    {"label": "人工介入率", "metric": "human_intervention_rate", "percent": True, "good": "down"},
]


def _project_root() -> str:
    """本文件位于 <root>/src/dashboard/pages/comparison.py，上溯三级得项目根。"""
    here = os.path.abspath(os.path.dirname(__file__))
    return os.path.abspath(os.path.join(here, os.pardir, os.pardir, os.pardir))


@st.cache_data(show_spinner=False)
def _load_metric_items() -> list[dict[str, Any]]:
    """
    读取 config/metrics.yaml，返回指标定义 list（每项含 key/target/radar_weight 等）。

    这是雷达维度与归一化上界的「单一事实来源」入口：本文件所有对 metrics.yaml 的
    读取都收口到这里，避免多处打开文件、结构判断不一致。读取失败（无 pyyaml /
    文件缺失）时返回空 list，上层各 loader 自行退化为内置降级方案，绝不崩溃。
    """
    path = os.path.join(_project_root(), "config", "metrics.yaml")
    try:
        import yaml  # 可选，但核心依赖里有 pyyaml
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    except Exception:
        return []

    # metrics.yaml 结构兼容两种：顶层 list 或 {"metrics":[...]}
    items = data.get("metrics", data) if isinstance(data, dict) else data
    if isinstance(items, dict):
        items = list(items.values())
    if isinstance(items, list):
        return [m for m in items if isinstance(m, dict) and m.get("key")]
    return []


@st.cache_data(show_spinner=False)
def _load_metric_targets() -> dict[str, float]:
    """
    取每个指标的 target 作为归一化参考上界。

    用途：雷达图把原始指标按「目标值」归一化（达成目标≈满分），
    比简单 min-max 更有业务含义。缺失时返回空 dict，
    归一化逻辑会退化为按经验范围处理，不影响出图。
    """
    targets: dict[str, float] = {}
    for m in _load_metric_items():
        if m.get("target") is not None:
            try:
                targets[m["key"]] = float(m["target"])
            except (TypeError, ValueError):
                continue
    return targets


@st.cache_data(show_spinner=False)
def _load_radar_dimensions() -> list[dict[str, Any]]:
    """
    【P0 单一事实来源】从 metrics.yaml 的 radar_weight 构造雷达维度列表。

    规则：凡 radar_weight > 0 的指标即为一个雷达维度；维度顺序 = 指标在 yaml 中的
    声明顺序（成功率/准确率/速度/稳定性/低成本/恢复率）。方向由 _RADAR_DOWN_METRICS
    判定（成本/耗时类逆向）。中文短标签取自 _RADAR_LABELS，缺失则回退指标 name / key。

    读取不到任何维度（无 pyyaml / 文件缺失 / 全部 radar_weight=0）时，
    退化到 _FALLBACK_RADAR_DIMENSIONS——该降级表已与 metrics.yaml 声明严格对齐，
    保证「代码 = 文档」，彻底消除号称单一来源实为硬编码的漂移。
    """
    dims: list[dict[str, Any]] = []
    for m in _load_metric_items():
        try:
            weight = float(m.get("radar_weight", 0) or 0)
        except (TypeError, ValueError):
            weight = 0.0
        if weight <= 0:
            continue
        key = m["key"]
        label = _RADAR_LABELS.get(key) or m.get("name") or key
        direction = "down" if key in _RADAR_DOWN_METRICS else "up"
        dims.append({"label": label, "metric": key, "direction": direction})

    return dims or list(_FALLBACK_RADAR_DIMENSIONS)


# ============================================================
# 二、从 run 记录提取指标 dict
# ============================================================
def _run_metrics(run: dict) -> dict[str, float]:
    """
    从一条 benchmark_run 记录解析出完整指标 dict。

    优先解析 metrics_json（benchmark 落库时写入的 §4 全部指标）；
    缺失则用记录里的汇总字段拼一个最小集合兜底。
    """
    raw = run.get("metrics_json")
    if isinstance(raw, str) and raw.strip():
        try:
            d = json.loads(raw)
            if isinstance(d, dict):
                return d
        except (json.JSONDecodeError, ValueError):
            pass
    # 兜底：用顶层字段拼最小指标集
    return {
        "task_success_rate": float(run.get("success_rate") or 0),
        "sort_accuracy": float(run.get("sort_accuracy") or 0),
        "avg_task_ms": float(run.get("avg_duration_ms") or 0),
        "throughput_per_min": float(run.get("throughput") or 0),
    }


# 小量级指标的「经验满分上界」：这些指标绝对值通常远小于 1（如人工介入率 <0.1、
# 恢复率 <0.5），若直接按 0~1 比例归一化，所有版本都挤在满分/零分附近，雷达维度
# 失去区分度。用经验上界做 min-max 缩放，放大版本间的真实差异（产品决策：
# 雷达图的使命是"看清版本差异"，而非精确复读绝对值——绝对值在 delta 分析区呈现）。
_METRIC_CEILING: dict[str, float] = {
    "avg_retry": 2.50,                # 低成本雷达维度：平均重试经验上界 2.5 次/任务（逆向）
    "human_intervention_rate": 0.20,  # 介入率经验上界 20%：达到即视为"成本拉满"（逆向）
    "recovery_rate": 0.50,            # 恢复率经验上界 50%：达到即视为"恢复能力满分"
}


def _normalize(metric_key: str, value: float, direction: str, targets: dict[str, float]) -> float:
    """
    把单个指标值归一化到 0~1（越大越好）。

    - 百分比类（成功率等）天然在 0~1，direction=down 时取 1-value。
    - 耗时类（avg_task_ms）用目标值或经验上界做归一化后取反。
    - 小量级指标（人工介入率/恢复率）用 _METRIC_CEILING 的经验上界缩放，
      避免所有版本挤在满分附近、维度失去区分度。
    结果夹到 [0,1]，保证雷达不溢出。
    """
    v = float(value or 0)

    if metric_key == "avg_task_ms":
        # 速度：耗时越短越好。参考上界取 target 或经验 8000ms。
        ceiling = targets.get(metric_key) or 8000.0
        score = 1.0 - min(v / ceiling, 1.0)
        return max(0.0, min(1.0, score))

    # 小量级指标：先按经验上界缩放到 0~1，再按方向取值/取反
    ceiling = _METRIC_CEILING.get(metric_key)
    if ceiling:
        scaled = min(v / ceiling, 1.0)
        score = 1.0 - scaled if direction == "down" else scaled
        return max(0.0, min(1.0, score))

    # 其余默认按 0~1 比例类处理
    if direction == "down":
        score = 1.0 - v
    else:
        score = v
    return max(0.0, min(1.0, score))


# ============================================================
# 三、主渲染函数
# ============================================================
def render(storage, ctx: dict) -> None:
    runs = ctx.get("runs", [])

    st.subheader("版本对比")

    if not runs:
        st.info("暂无评测数据。请运行 `bash run_benchmark.sh` 生成多个版本后再来对比。")
        return

    ordered = list(reversed(runs))  # 时间正序，便于 v1→v2→v3 阅读
    if len(ordered) < 2:
        st.warning("当前仅有 1 次评测，无法做版本对比。多跑几次 `bash run_benchmark.sh` 即可解锁完整对比。")
        # 仍展示单版本雷达，给用户一点东西看
        _render_radar(ordered, _load_metric_targets())
        _render_run_benchmark_hint()
        return

    targets = _load_metric_targets()

    # ---------- 1) 多版本雷达图 ----------
    st.markdown("#### 多版本综合能力雷达")
    _render_radar(ordered, targets)
    st.caption("六维均已归一化到 0~100（越靠外越好；速度=耗时取反，低成本=人工介入取反）。")

    st.divider()

    # ---------- 2) 版本间 delta 分析 ----------
    st.markdown("#### 版本间提升 / 下降分析")
    _render_deltas(ordered)

    st.divider()

    # ---------- 3) 优化建议 ----------
    st.markdown("#### 优化建议（数据驱动）")
    _render_recommendations(storage, ordered)

    st.divider()
    _render_run_benchmark_hint()


# ============================================================
# 四、子组件
# ============================================================
def _render_radar(ordered_runs: list[dict], targets: dict[str, float]) -> None:
    """构造多版本雷达图：每个版本一条多边形，维度取自 metrics.yaml 的 radar_weight。"""
    radar_dims = _load_radar_dimensions()  # 单一事实来源：metrics.yaml
    dims = [d["label"] for d in radar_dims]
    series = []
    palette = [charts.PRIMARY, "#5AD8A6", "#E8684A", "#9270CA", "#F6BD16"]
    for i, run in enumerate(ordered_runs):
        m = _run_metrics(run)
        vals = [
            _normalize(d["metric"], m.get(d["metric"], 0.0), d["direction"], targets)
            for d in radar_dims
        ]
        ver = run.get("version", f"v{i+1}")
        wm = "+世界模型" if run.get("world_model_on") else ""
        series.append({"name": f"{ver}{wm}", "values": vals, "color": palette[i % len(palette)]})

    fig = charts.radar_chart(dims, series, title="各版本六维能力对比")
    with st.container(border=True):  # 白卡容器：图表浮于浅灰页面
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)


def _render_deltas(ordered_runs: list[dict]) -> None:
    """
    展示相邻版本之间各指标的 delta。

    用一个下拉选择「对比哪两个版本」（默认首版 vs 末版，体现整体迭代收益），
    再画横向 delta 柱（正向绿 / 负向红）。对「越小越好」的指标，会把 delta
    的符号语义对齐到「是否变好」——即误拣率下降显示为正向收益。
    """
    labels = []
    for i, r in enumerate(ordered_runs):
        ver = r.get("version", f"v{i+1}")
        wm = "+WM" if r.get("world_model_on") else ""
        labels.append(f"{ver}{wm}")

    c1, c2 = st.columns(2)
    with c1:
        base_idx = st.selectbox("基准版本", options=list(range(len(ordered_runs))),
                                index=0, format_func=lambda i: labels[i], key="cmp_base")
    with c2:
        target_idx = st.selectbox("对比版本", options=list(range(len(ordered_runs))),
                                  index=len(ordered_runs) - 1, format_func=lambda i: labels[i], key="cmp_target")

    if base_idx == target_idx:
        st.info("请选择两个不同的版本进行对比。")
        return

    base_m = _run_metrics(ordered_runs[base_idx])
    tgt_m = _run_metrics(ordered_runs[target_idx])

    metric_names, deltas = [], []
    for spec in _DELTA_METRICS:
        key = spec["metric"]
        bv = float(base_m.get(key, 0) or 0)
        tv = float(tgt_m.get(key, 0) or 0)
        raw_delta = tv - bv
        # 把「越小越好」指标的 delta 符号翻转，使「正=变好」语义统一
        signed = raw_delta if spec["good"] == "up" else -raw_delta
        metric_names.append(spec["label"])
        deltas.append(signed)

    fig = charts.delta_bar_chart(
        metric_names, deltas,
        title=f"{labels[base_idx]} → {labels[target_idx]} 各指标变化（绿=改善 / 红=退化）",
        is_percent=True,
    )
    with st.container(border=True):
        st.plotly_chart(fig, use_container_width=True, config=charts.PLOTLY_CONFIG)
        st.caption("对「误拣率 / 人工介入率」这类越小越好的指标，已将符号对齐为「正=改善」。")

    # 一句话结论：涨用成功绿、跌用危险红，语义色直接传达「变好 / 变差」
    sr_base = float(base_m.get("task_success_rate", 0) or 0)
    sr_tgt = float(tgt_m.get("task_success_rate", 0) or 0)
    sr_delta = (sr_tgt - sr_base) * 100
    conclusion = (
        f"**结论**：相比 {labels[base_idx]}，{labels[target_idx]} 的任务成功率"
        f" {sr_delta:+.1f} 个百分点"
        f"（{sr_base*100:.0f}% → {sr_tgt*100:.0f}%）。"
    )
    if sr_delta >= 0:
        st.success(conclusion)
    else:
        st.error(conclusion)


def _render_recommendations(storage, ordered_runs: list[dict]) -> None:
    """
    生成并展示优化建议。

    优先调用 evaluation.recommendation.generate(failure_analysis, metrics)；
    需要先用 failure_analysis.analyze(task_results) 聚合最新一次运行的失败分布。
    任一环节缺失（模块未就绪 / 报错）时，退化为基于失败分布的内置建议，绝不空屏。
    """
    latest = ordered_runs[-1]
    run_id = latest.get("id")

    # 读最新运行的任务结果
    try:
        rows = [dict(r) for r in (storage.get_task_results(run_id) or [])]
    except Exception:
        rows = []

    recs: list[dict] = []
    # —— 路径 A：调用项目内 recommendation + failure_analysis ——
    try:
        from src.evaluation import failure_analysis as fa_mod
        from src.evaluation import recommendation as rec_mod

        fa = fa_mod.analyze(rows)
        metrics = _run_metrics(latest)
        recs = list(rec_mod.generate(fa, metrics) or [])
    except Exception as exc:
        # —— 路径 B：内置降级建议（基于失败分布的简单规则）——
        st.caption(f"（建议引擎未就绪，已使用内置降级建议：{type(exc).__name__}）")
        recs = _fallback_recommendations(rows)

    if not recs:
        recs = _fallback_recommendations(rows)

    if not recs:
        st.success("当前未发现显著短板，建议保持现有策略并扩大评测样本以验证稳定性。")
        return

    # 渲染建议卡片：优先级彩签 + 预期收益
    for i, rec in enumerate(recs, start=1):
        priority = rec.get("priority", "中")
        title = rec.get("title", f"建议 {i}")
        with st.container(border=True):
            head_c1, head_c2 = st.columns([4, 1])
            with head_c1:
                st.markdown(f"**{i}. {title}**")
            with head_c2:
                st.markdown(charts.priority_badge_html(priority), unsafe_allow_html=True)

            if rec.get("problem"):
                st.markdown(f"- **问题**：{rec['problem']}")
            if rec.get("evidence"):
                st.markdown(f"- **数据支撑**：{rec['evidence']}")
            if rec.get("solution"):
                st.markdown(f"- **方案**：{rec['solution']}")

            gain = rec.get("expected_gain")
            impact = rec.get("impact")
            cost = rec.get("cost")
            chips = []
            if gain:
                chips.append(charts.badge_html(f"预期收益：{gain}", charts.POSITIVE, filled=False))
            if impact:
                chips.append(charts.badge_html(f"影响：{impact}", charts.PRIMARY, filled=False))
            if cost:
                chips.append(charts.badge_html(f"成本：{cost}", charts.TEXT_MUTED, filled=False))
            if chips:
                st.markdown("  ".join(chips), unsafe_allow_html=True)


def _fallback_recommendations(rows: list[dict]) -> list[dict]:
    """
    内置降级建议：当 recommendation 模块不可用时，直接按失败大类分布给 3 条建议。

    逻辑刻意简单：找出占比最高的失败大类，给出对应的标准化优化方向与预期收益，
    保证版本对比页在任何依赖状态下都能给出「数据驱动」的可执行建议。
    """
    dist: dict[str, int] = {}
    total_fail = 0
    for r in rows:
        if r.get("success"):
            continue
        cat = r.get("failure_category")
        if cat:
            dist[cat] = dist.get(cat, 0) + 1
            total_fail += 1
    if not dist:
        return []

    # 标准化方案库：大类 → 建议模板
    playbook = {
        "感知类失败": {
            "title": "引入多视角融合 / 提升遮挡场景感知",
            "solution": "增加侧视相机融合与遮挡补全，困难场景优先重检测。",
            "expected_gain": "成功率 +6~8pct",
        },
        "理解类失败": {
            "title": "强化模糊 / 条件指令的语义解析",
            "solution": "扩充指令解析规则与约束校验，对歧义指令触发二次确认。",
            "expected_gain": "理解类失败 -50%",
        },
        "规划类失败": {
            "title": "优化分拣顺序与优先级调度",
            "solution": "引入最近邻路径 + 优先级约束求解，减少顺序错误。",
            "expected_gain": "规划类失败 -40%",
        },
        "执行类失败": {
            "title": "易碎 / 大件抓取姿态优化",
            "solution": "对易碎大件降低抓取力度、增加放置前姿态校验与重试。",
            "expected_gain": "放置失败 -35%",
        },
        "环境类失败": {
            "title": "增强异常恢复与物理鲁棒性",
            "solution": "完善碰撞 / 意外移动的恢复策略，提升异常恢复率。",
            "expected_gain": "异常恢复率 +15pct",
        },
    }

    ranked = sorted(dist.items(), key=lambda kv: kv[1], reverse=True)
    recs = []
    for idx, (cat, cnt) in enumerate(ranked[:3]):
        tpl = playbook.get(cat, {})
        pct = cnt / total_fail * 100 if total_fail else 0
        recs.append({
            "title": tpl.get("title", f"优化{cat}"),
            "problem": f"{cat}是当前主要失败来源。",
            "evidence": f"{cat}占全部失败的 {pct:.0f}%（{cnt} 例）。",
            "solution": tpl.get("solution", "针对该类失败做专项优化。"),
            "expected_gain": tpl.get("expected_gain", "—"),
            "priority": "高" if idx == 0 else ("中" if idx == 1 else "低"),
            "impact": "高" if idx == 0 else "中",
            "cost": "中",
        })
    return recs


def _render_run_benchmark_hint() -> None:
    """「运行新评测」说明，指向 run_benchmark.sh（看板不在内部跑重评测）。"""
    with st.expander("如何运行一次新评测以加入对比？", expanded=False):
        st.markdown(
            "看板只读取已落库的评测结果，不在网页内跑重仿真。\n\n"
            "在**项目根目录**执行以下命令生成新评测，跑完刷新本页即可看到新版本：\n"
        )
        st.code("bash run_benchmark.sh", language="bash")
        st.markdown(
            "或直接调用模块（可指定版本号 / VLA 后端 / 是否启用世界模型 / 随机种子）："
        )
        st.code(
            "python -m src.evaluation.benchmark --version v4 --vla rule_based "
            "--strategy optimized --world-model on --seed 42",
            language="bash",
        )
