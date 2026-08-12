"""指标计算（src/evaluation/metrics.py）—— SPEC §4 全部指标。

分层 × 功能四组：
- 北极星：任务成功率（task_success_rate）
- 核心(效果)：分拣准确率/误拣率/漏拣率
- 过程(效率)：平均任务耗时/平均单步耗时/单位时间分拣数
- 过程(稳定性)：抓取成功率/识别准确率/碰撞次数/异常恢复率
- 辅助(成本)：平均重试次数/人工介入率

对外接口：
- compute_metrics(task_results) -> dict（英文 key，与 SPEC §4/§11 一致）
- compute_by_difficulty(task_results) -> {难度: metrics}
- compute_by_type(task_results) -> {类型: metrics}

【口径说明】所有比率类指标在分母为 0 时返回 0.0，避免除零崩溃；
耗时类返回整数毫秒；吞吐为"件/分钟"。task_results 为 TaskResult dict 列表
（字段见 SPEC §7），同时兼容来自 DB 的扁平行（含相同英文字段）。
"""

from __future__ import annotations

from typing import Any, Dict, List


def _safe_div(numerator: float, denominator: float) -> float:
    """安全除法：分母为 0 返回 0.0。"""
    return float(numerator) / float(denominator) if denominator else 0.0


def _clamp01(x: float) -> float:
    """把比率类指标夹到 [0, 1]。

    【防御性设计】上游（引擎/仿真）偶尔可能产出统计口径外的数据（例如某任务
    recognition_correct 大于其 target_count，导致聚合后识别准确率 > 100%）。
    比率类指标在概念上恒处于 [0,1]，这里统一夹紧，保证看板不会显示 >100% 的
    异常值；同时不掩盖问题——分子分母仍按原值参与其它计算。
    """
    return max(0.0, min(1.0, float(x)))


def _g(tr: Dict[str, Any], key: str, default: Any = 0) -> Any:
    """从 TaskResult 取值，缺失返回默认（兼容 DB 行与内存 dict）。"""
    val = tr.get(key, default)
    return default if val is None else val


def compute_metrics(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """计算一批任务结果的全部指标，返回英文 key 的 dict（SPEC §4）。

    返回键：
        task_success_rate, sort_accuracy, mis_pick_rate, miss_pick_rate,
        avg_task_ms, avg_step_ms, throughput_per_min,
        grasp_success_rate, recognition_accuracy, collision_count,
        recovery_rate, avg_retry, human_intervention_rate
    另附若干计数辅助字段（total_tasks/success_count 等）便于看板展示。
    """
    n = len(task_results)
    if n == 0:
        # 空集合返回全 0，保证看板与 benchmark 不崩溃
        return _empty_metrics()

    # —— 北极星：任务成功率 ——
    success_count = sum(1 for t in task_results if _g(t, "success", False))
    task_success_rate = _safe_div(success_count, n)

    # —— 核心(效果)：分拣准确率/误拣率/漏拣率 ——
    # 分拣准确率：取各任务 sort_accuracy 的均值（每任务已按"正确/目标"归一）
    sort_accuracy = _safe_div(sum(float(_g(t, "sort_accuracy", 0.0))
                                  for t in task_results), n)
    # 误拣率 = 误拣总数 / 已拣总数；漏拣率 = 漏拣总数 / 目标总数
    total_mis = sum(int(_g(t, "mis_pick", 0)) for t in task_results)
    total_miss = sum(int(_g(t, "miss_pick", 0)) for t in task_results)
    total_target = sum(int(_g(t, "target_count",
                              _g(t, "target_count", 1))) for t in task_results)
    total_grasp_success = sum(int(_g(t, "grasp_success", 0)) for t in task_results)
    # 已拣数近似为成功抓取数 + 误拣数（误拣也属"已拣出但放错"）
    total_picked = total_grasp_success + total_mis
    mis_pick_rate = _safe_div(total_mis, total_picked if total_picked else total_target)
    miss_pick_rate = _safe_div(total_miss, total_target)

    # —— 过程(效率)：耗时与吞吐 ——
    total_duration_ms = sum(int(_g(t, "duration_ms", 0)) for t in task_results)
    avg_task_ms = int(round(_safe_div(total_duration_ms, n)))
    total_steps = sum(int(_g(t, "step_count", _g(t, "step_count", 0)))
                      for t in task_results)
    avg_step_ms = int(round(_safe_div(total_duration_ms, total_steps))) \
        if total_steps else 0
    # 单位时间分拣数（件/分钟）：成功抓取的目标件数 / 总耗时（分钟）
    total_minutes = _safe_div(total_duration_ms, 60_000.0)  # ms -> min
    throughput_per_min = round(_safe_div(total_grasp_success, total_minutes), 2) \
        if total_minutes else 0.0

    # —— 过程(稳定性)：抓取/识别/碰撞/恢复 ——
    total_grasp_attempts = sum(int(_g(t, "grasp_attempts", 0)) for t in task_results)
    grasp_success_rate = _safe_div(total_grasp_success, total_grasp_attempts)
    total_recog_correct = sum(int(_g(t, "recognition_correct", 0)) for t in task_results)
    recognition_accuracy = _safe_div(total_recog_correct, total_target)
    collision_count = sum(int(_g(t, "collisions", 0)) for t in task_results)
    # 异常恢复率 = 成功恢复次数 / 触发异常次数（碰撞次数近似为异常次数）
    total_recovered = sum(int(_g(t, "recovered", 0)) for t in task_results)
    total_anomalies = collision_count + total_recovered  # 恢复也意味着发生过异常
    recovery_rate = _safe_div(total_recovered, total_anomalies)

    # —— 辅助(成本)：重试与人工介入 ——
    total_retry = sum(int(_g(t, "retry_count", 0)) for t in task_results)
    avg_retry = round(_safe_div(total_retry, n), 2)
    intervention_count = sum(int(_g(t, "human_intervention", 0)) for t in task_results)
    human_intervention_rate = _safe_div(intervention_count, n)

    return {
        # 北极星
        "task_success_rate": round(_clamp01(task_success_rate), 4),
        # 核心(效果)
        "sort_accuracy": round(_clamp01(sort_accuracy), 4),
        "mis_pick_rate": round(_clamp01(mis_pick_rate), 4),
        "miss_pick_rate": round(_clamp01(miss_pick_rate), 4),
        # 过程(效率)
        "avg_task_ms": avg_task_ms,
        "avg_step_ms": avg_step_ms,
        "throughput_per_min": throughput_per_min,
        # 过程(稳定性)
        "grasp_success_rate": round(_clamp01(grasp_success_rate), 4),
        "recognition_accuracy": round(_clamp01(recognition_accuracy), 4),
        "collision_count": int(collision_count),
        "recovery_rate": round(_clamp01(recovery_rate), 4),
        # 辅助(成本)
        "avg_retry": avg_retry,
        "human_intervention_rate": round(_clamp01(human_intervention_rate), 4),
        # 辅助计数（看板展示用，非 §4 必需但便于呈现）
        "total_tasks": n,
        "success_count": success_count,
        "failure_count": n - success_count,
    }


def _empty_metrics() -> Dict[str, Any]:
    """空集合的零值指标（保持键齐全）。"""
    return {
        "task_success_rate": 0.0,
        "sort_accuracy": 0.0,
        "mis_pick_rate": 0.0,
        "miss_pick_rate": 0.0,
        "avg_task_ms": 0,
        "avg_step_ms": 0,
        "throughput_per_min": 0.0,
        "grasp_success_rate": 0.0,
        "recognition_accuracy": 0.0,
        "collision_count": 0,
        "recovery_rate": 0.0,
        "avg_retry": 0.0,
        "human_intervention_rate": 0.0,
        "total_tasks": 0,
        "success_count": 0,
        "failure_count": 0,
    }


def _group_by(task_results: List[Dict[str, Any]], field: str) -> Dict[str, List]:
    """按指定字段对任务分组（缺失值归入 '未知'）。"""
    groups: Dict[str, List] = {}
    for t in task_results:
        key = t.get(field) or "未知"
        groups.setdefault(key, []).append(t)
    return groups


def compute_by_difficulty(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按难度（简单/中等/困难）分组计算指标。

    返回 {难度: metrics_dict}，难度顺序固定便于看板呈现。
    """
    groups = _group_by(task_results, "difficulty")
    # 固定顺序输出（仅包含实际出现的难度）
    order = ["简单", "中等", "困难"]
    ordered = {d: groups[d] for d in order if d in groups}
    # 追加未在标准顺序中的其它键（兜底）
    for k, v in groups.items():
        if k not in ordered:
            ordered[k] = v
    return {diff: compute_metrics(items) for diff, items in ordered.items()}


def compute_by_type(task_results: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """按指令类型分组计算指标。

    返回 {类型: metrics_dict}，类型顺序按 5 大任务类固定。
    """
    groups = _group_by(task_results, "type")
    order = ["基础分拣", "条件分拣", "优先级分拣", "批量分拣", "异常场景"]
    ordered = {t: groups[t] for t in order if t in groups}
    for k, v in groups.items():
        if k not in ordered:
            ordered[k] = v
    return {ttype: compute_metrics(items) for ttype, items in ordered.items()}
