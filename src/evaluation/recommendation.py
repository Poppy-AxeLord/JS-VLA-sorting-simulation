"""优化建议生成（src/evaluation/recommendation.py）—— SPEC §9。

对外接口：
    generate(failure_analysis, metrics) -> [
        {title, problem, evidence, solution, expected_gain,
         priority("高/中/低"), impact, cost}
    ]

设计目标：把"失败分析 + 指标"转化为 3~6 条**具体可执行、带数据支撑与预期收益**
的优化建议，按"高影响低成本优先"排序——这是失败分析驱动迭代的产品闭环核心。

排序规则（impact 越高、cost 越低越靠前）：
    score = impact * 权重 - cost * 权重；impact/cost 取 {高=3,中=2,低=1}。
"""

from __future__ import annotations

from typing import Any, Dict, List

# 影响/成本等级到分值的映射（用于排序）
_LEVEL_SCORE = {"高": 3, "中": 2, "低": 1}


def _level_to_priority(impact: str, cost: str) -> str:
    """由影响与成本推导优先级标签（高影响低成本 → 高优先）。"""
    score = _LEVEL_SCORE.get(impact, 2) * 2 - _LEVEL_SCORE.get(cost, 2)
    if score >= 4:
        return "高"
    if score >= 2:
        return "中"
    return "低"


def _pct(x: float) -> str:
    """0-1 浮点 -> 百分比字符串。"""
    return f"{x * 100:.1f}%"


def generate(failure_analysis: Dict[str, Any],
             metrics: Dict[str, Any]) -> List[Dict[str, Any]]:
    """根据失败分析与指标生成优化建议列表（3~6 条，已排序）。

    参数
    ----
    failure_analysis : dict
        failure_analysis.analyze(...) 的返回。
    metrics : dict
        metrics.compute_metrics(...) 的返回。
    """
    failure_analysis = failure_analysis or {}
    metrics = metrics or {}

    dist = failure_analysis.get("category_distribution", []) or []
    total_failures = failure_analysis.get("total_failures", 0) or sum(
        d.get("count", 0) for d in dist)
    subtypes = {s["subtype"]: s for s in failure_analysis.get("subtype_top10", [])}

    # 各大类占比（占总失败数）
    def cat_count(name: str) -> int:
        for d in dist:
            if d.get("category") == name:
                return int(d.get("count", 0))
        return 0

    def cat_ratio(name: str) -> float:
        return (cat_count(name) / total_failures) if total_failures else 0.0

    def sub_count(name: str) -> int:
        s = subtypes.get(name)
        return int(s["count"]) if s else 0

    recs: List[Dict[str, Any]] = []

    # ---------------- 候选建议（按失败大类逐条评估是否触发） ---------------- #

    # 1) 感知类失败 —— 遮挡/识别 → 多视角融合
    perc_ratio = cat_ratio("感知类失败")
    if cat_count("感知类失败") > 0:
        occ = sub_count("遮挡看不见")
        recog = sub_count("识别错误")
        impact = "高" if perc_ratio >= 0.25 else "中"
        recs.append({
            "title": "引入多视角融合与遮挡感知增强",
            "problem": "困难场景下遮挡与相似零件导致感知失败占比偏高。",
            "evidence": f"感知类失败占总失败 {_pct(perc_ratio)}"
                        f"（遮挡看不见 {occ} 例、识别错误 {recog} 例）。",
            "solution": "增加侧视相机做多视角融合；对相似银色件（螺丝/螺母）"
                        "引入形状先验与尺寸约束二次确认；困难场景启用主动视角调整。",
            "expected_gain": "预期感知失败下降 50~60%，整体任务成功率 +6~8pct。",
            "impact": impact,
            "cost": "中",
        })

    # 2) 理解类失败 —— 模糊/约束 → 指令解析增强
    und_ratio = cat_ratio("理解类失败")
    if cat_count("理解类失败") > 0:
        impact = "高" if und_ratio >= 0.20 else "中"
        recs.append({
            "title": "强化模糊指令与约束条件的语义解析",
            "problem": "模糊指令（大左小右）与条件约束（仅金属件）解析易出错。",
            "evidence": f"理解类失败占总失败 {_pct(und_ratio)}"
                        f"（含指令理解错误 {sub_count('指令理解错误')} 例、"
                        f"歧义处理失败 {sub_count('歧义处理失败')} 例）。",
            "solution": "为模糊指令建立可量化的边界规则（按尺寸阈值二分）；"
                        "条件分拣前先做属性过滤校验；必要时接入 VLM 做歧义消解。",
            "expected_gain": "预期理解失败下降 40~50%，条件/模糊类任务成功率 +5~7pct。",
            "impact": impact,
            "cost": "低",
        })

    # 3) 执行类失败 —— 抓取/放置/碰撞 → 世界模型 + 自适应抓取
    exe_ratio = cat_ratio("执行类失败")
    if cat_count("执行类失败") > 0:
        drop = sub_count("放置失败(掉落)") + sub_count("抓取失败(滑落)")
        impact = "高" if exe_ratio >= 0.25 else "中"
        recs.append({
            "title": "启用世界模型风险评估 + 易碎件自适应抓取",
            "problem": "易碎大件（PCB/显示屏）抓取滑落与放置掉落是执行失败主因。",
            "evidence": f"执行类失败占总失败 {_pct(exe_ratio)}"
                        f"（抓取/放置掉落合计 {drop} 例，碰撞 "
                        f"{sub_count('碰撞导致失败')} 例）。",
            "solution": "抓取前用世界模型评估掉落风险，对易碎/平板件降低速度、"
                        "增大夹持接触面与放置缓冲；高风险动作自动二次规划。",
            "expected_gain": "预期执行失败下降 45~55%，整体成功率 +5~9pct。",
            "impact": impact,
            "cost": "中",
        })

    # 4) 规划类失败 —— 顺序/优先级 → 路径与优先级优化
    plan_ratio = cat_ratio("规划类失败")
    if cat_count("规划类失败") > 0:
        impact = "中" if plan_ratio < 0.25 else "高"
        recs.append({
            "title": "优化分拣路径与优先级调度",
            "problem": "批量/优先级任务中路径绕行与优先级处理错误导致超时失败。",
            "evidence": f"规划类失败占总失败 {_pct(plan_ratio)}"
                        f"（分拣顺序错误 {sub_count('分拣顺序错误')} 例、"
                        f"优先级处理错误 {sub_count('优先级处理错误')} 例）。",
            "solution": "用最近邻 + 2-opt 改进路径；显式建模优先级约束并在规划层校验；"
                        "批量任务分批提交降低单次复杂度。",
            "expected_gain": "预期规划失败下降 35~45%，批量任务平均耗时 -15%。",
            "impact": impact,
            "cost": "低",
        })

    # 5) 环境类失败 —— 物理异常/障碍 → 鲁棒性与异常恢复
    env_ratio = cat_ratio("环境类失败")
    if cat_count("环境类失败") > 0:
        recs.append({
            "title": "增强环境扰动下的异常检测与恢复",
            "problem": "物体意外移动、临时障碍与物理异常导致执行中断。",
            "evidence": f"环境类失败占总失败 {_pct(env_ratio)}"
                        f"（物体意外移动 {sub_count('物体意外移动')} 例、"
                        f"仿真物理异常 {sub_count('仿真物理异常')} 例）。",
            "solution": "抓取前重新校验目标位姿；增加碰撞后安全停-重试-换序的恢复流程；"
                        "对物理异常做接触参数自适应。",
            "expected_gain": "预期环境失败下降 30~40%，异常恢复率 +20pct。",
            "impact": "中",
            "cost": "中",
        })

    # ---------------- 基于全局指标的补充建议（成本/介入维度） ---------------- #

    # 6) 人工介入率偏高 → 提高置信度自动化处理
    hir = float(metrics.get("human_intervention_rate", 0.0) or 0.0)
    if hir >= 0.08:
        recs.append({
            "title": "降低人工介入率，提升自动化闭环",
            "problem": "低置信度动作频繁触发人工介入，影响无人化与吞吐。",
            "evidence": f"当前人工介入率 {_pct(hir)}，"
                        f"平均重试 {metrics.get('avg_retry', 0)} 次。",
            "solution": "对低置信度场景引入自动二次确认（多视角/重定位）"
                        "替代人工，仅保留安全相关的人工兜底。",
            "expected_gain": "预期人工介入率下降一半以上，吞吐 +10%。",
            "impact": "中",
            "cost": "低",
        })

    # 兜底：若失败极少（如高版本），仍给出一条"巩固优势"建议，保证 ≥3 条
    if len(recs) < 3:
        recs.append({
            "title": "保持优化策略并扩大评测覆盖",
            "problem": "当前版本表现良好，需防止回归并验证泛化。",
            "evidence": f"任务成功率 {_pct(float(metrics.get('task_success_rate', 0.0)))}，"
                        f"抓取成功率 {_pct(float(metrics.get('grasp_success_rate', 0.0)))}。",
            "solution": "扩充困难场景与长尾指令的评测集；建立回归基线，"
                        "对每次策略变更做 A/B 验证。",
            "expected_gain": "保障迭代不回归，泛化成功率稳定。",
            "impact": "中",
            "cost": "低",
        })

    # ---------------- 排序：高影响低成本优先，截断到 3~6 条 ---------------- #
    for r in recs:
        r["priority"] = _level_to_priority(r["impact"], r["cost"])
        r["_score"] = _LEVEL_SCORE.get(r["impact"], 2) * 2 - \
            _LEVEL_SCORE.get(r["cost"], 2)

    recs.sort(key=lambda r: r["_score"], reverse=True)
    recs = recs[:6]
    if len(recs) < 3:
        # 极端兜底（理论不触发）：补足到 3 条
        while len(recs) < 3:
            recs.append({
                "title": "完善评测与监控体系",
                "problem": "需要更全面的指标监控以支撑持续迭代。",
                "evidence": "当前样本失败较少，建议扩大评测规模。",
                "solution": "增加任务多样性与运行次数，建立指标看板告警。",
                "expected_gain": "提升评测可信度。",
                "impact": "低", "cost": "低", "priority": "低", "_score": 1,
            })

    # 去掉内部排序字段
    for r in recs:
        r.pop("_score", None)
    return recs
