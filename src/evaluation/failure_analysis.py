"""失败分析（src/evaluation/failure_analysis.py）—— SPEC §3 五大类配色。

对外接口：
    analyze(task_results) -> {
        category_distribution: [{category, count, color}],   # 5 大类分布(含配色)
        subtype_top10:         [{subtype, category, count}],  # 子类 Top10
        by_difficulty:         {难度: {大类: count}},          # 难度 × 大类交叉
        cases:                 [失败案例摘要],                 # 用于看板钻取
    }

配色严格使用 §3 的固定值；存储中 failure_category 存"大类中文"，
failure_subtype 存"子类中文"。本模块兼容内存 TaskResult 与 DB 行。
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any, Dict, List

# §3 失败分类配色：大类中文 -> 颜色（全项目统一，看板饼图/柱图共用）
CATEGORY_COLORS: Dict[str, str] = {
    "感知类失败": "#5B8FF9",
    "理解类失败": "#5AD8A6",
    "规划类失败": "#F6BD16",
    "执行类失败": "#E8684A",
    "环境类失败": "#9270CA",
}

# 固定的大类展示顺序（保证饼图/图例稳定）
CATEGORY_ORDER: List[str] = [
    "感知类失败", "理解类失败", "规划类失败", "执行类失败", "环境类失败",
]

# 未归类失败的兜底配色（理论上不出现，防御性）
_UNKNOWN_COLOR = "#9CA3AF"


def category_color(category: str) -> str:
    """大类中文 -> 配色（未知大类返回灰色兜底）。"""
    return CATEGORY_COLORS.get(category, _UNKNOWN_COLOR)


def analyze(task_results: List[Dict[str, Any]]) -> Dict[str, Any]:
    """聚合失败任务，产出分布/Top10/难度交叉/案例摘要。

    仅统计 success=False 且带 failure_category 的任务。
    """
    failures = [
        t for t in task_results
        if not _truthy(t.get("success")) and t.get("failure_category")
    ]

    category_distribution = _category_distribution(failures)
    subtype_top10 = _subtype_top10(failures)
    by_difficulty = _by_difficulty(failures)
    cases = _cases(failures)

    return {
        "category_distribution": category_distribution,
        "subtype_top10": subtype_top10,
        "by_difficulty": by_difficulty,
        "cases": cases,
        "total_failures": len(failures),
    }


def _truthy(val: Any) -> bool:
    """兼容 DB 的 0/1 与内存的 bool。"""
    return bool(val)


def _category_distribution(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """5 大类分布（含 count 与配色），按固定顺序输出，count=0 也保留便于图例完整。"""
    counter = Counter(t.get("failure_category") for t in failures)
    dist: List[Dict[str, Any]] = []
    for cat in CATEGORY_ORDER:
        dist.append({
            "category": cat,
            "count": int(counter.get(cat, 0)),
            "color": category_color(cat),
        })
    # 处理可能出现的未知大类（防御性，正常不触发）
    for cat, cnt in counter.items():
        if cat not in CATEGORY_ORDER:
            dist.append({"category": cat, "count": int(cnt),
                         "color": category_color(cat)})
    return dist


def _subtype_top10(failures: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """子类 Top10 排行（含所属大类与计数），按计数降序。"""
    counter: Counter = Counter()
    subtype_to_cat: Dict[str, str] = {}
    for t in failures:
        sub = t.get("failure_subtype")
        if not sub:
            continue
        counter[sub] += 1
        subtype_to_cat[sub] = t.get("failure_category", "")
    top = counter.most_common(10)
    return [
        {
            "subtype": sub,
            "category": subtype_to_cat.get(sub, ""),
            "count": int(cnt),
            "color": category_color(subtype_to_cat.get(sub, "")),
        }
        for sub, cnt in top
    ]


def _by_difficulty(failures: List[Dict[str, Any]]) -> Dict[str, Dict[str, int]]:
    """难度 × 大类的失败计数交叉表：{难度: {大类: count}}。"""
    table: Dict[str, Dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for t in failures:
        diff = t.get("difficulty") or "未知"
        cat = t.get("failure_category") or "未知"
        table[diff][cat] += 1
    # 转为普通 dict，难度按固定顺序
    order = ["简单", "中等", "困难"]
    result: Dict[str, Dict[str, int]] = {}
    for diff in order:
        if diff in table:
            result[diff] = dict(table[diff])
    for diff, row in table.items():
        if diff not in result:
            result[diff] = dict(row)
    return result


def _cases(failures: List[Dict[str, Any]], limit: int = 200) -> List[Dict[str, Any]]:
    """失败案例摘要列表（供看板列表展示与钻取，含场景快照引用）。"""
    cases: List[Dict[str, Any]] = []
    for t in failures[:limit]:
        cases.append({
            "task_id": t.get("task_id"),
            "instruction": t.get("instruction"),
            "type": t.get("type"),
            "difficulty": t.get("difficulty"),
            "failure_category": t.get("failure_category"),
            "failure_subtype": t.get("failure_subtype"),
            "failure_reason": t.get("failure_reason"),
            "color": category_color(t.get("failure_category", "")),
            # scene 可能来自内存 TaskResult；DB 行则在 failure_cases.scene_json
            "scene": t.get("scene"),
        })
    return cases
