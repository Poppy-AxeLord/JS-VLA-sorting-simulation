"""评测体系包（src/evaluation）。

包含：
- metrics.py          指标计算（§4 全部指标，分层 + 四组）
- failure_analysis.py 失败聚合分析（§3 五大类配色）
- recommendation.py   优化建议生成（高影响低成本优先）
- benchmark.py        基准评测运行器（跑 30 任务 → 落库）

统一以 src 为根做绝对导入，运行入口从项目根目录执行：
    python -m src.evaluation.benchmark
"""

from src.evaluation.metrics import (
    compute_metrics,
    compute_by_difficulty,
    compute_by_type,
)
from src.evaluation.failure_analysis import analyze
from src.evaluation.recommendation import generate

__all__ = [
    "compute_metrics",
    "compute_by_difficulty",
    "compute_by_type",
    "analyze",
    "generate",
]
