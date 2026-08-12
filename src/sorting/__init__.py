"""分拣引擎子包（src.sorting）

把“VLA 输出的动作计划”落地为“可评测的分拣执行流水”，并内置失败注入模型，
使得在没有真实机器人 / 真实 CV / 真实 VLA 的情况下，评测与失败分析依然真实可分析。

对外暴露：
- SortingEngine：分拣执行引擎（reset→感知→预测→规划→执行→汇总）。
- plan_order：动作排序（最近邻路径优化 + 优先级）。
- get_strategy / SortingStrategy：执行策略（置信度阈值/重试/异常恢复，baseline/optimized）。
"""

from src.sorting.engine import SortingEngine
from src.sorting.planner import plan_order
from src.sorting.strategies import SortingStrategy, get_strategy

__all__ = [
    "SortingEngine",
    "plan_order",
    "SortingStrategy",
    "get_strategy",
]
