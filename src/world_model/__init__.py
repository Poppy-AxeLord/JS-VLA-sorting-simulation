"""世界模型子包（src.world_model）

作用：在“真正抓取/放置之前”，用启发式（或未来可替换为学习式）对动作做风险预判，
供分拣引擎（src.sorting.engine）在抓取前调整姿态 / 改顺序，从而提升任务成功率。

设计动机（产品视角）：
- 真实工业分拣里，最贵的不是“失败”，而是“本可预见却没预见的失败”（易碎件掉落、遮挡误抓）。
- 世界模型把“事后失败分析”前移成“事前风险评估”，是 v3 相对 v2 的核心增量价值点。
- 这里用纯启发式（SimpleGraspPredictor），无任何重依赖，永远可运行；
  接口（BaseWorldModel）保持稳定，未来可无缝替换为 MuJoCo rollout 或学习式世界模型。

对外暴露：
- BaseWorldModel：抽象基类（接口契约，见 SPEC §8）。
- SimpleGraspPredictor：启发式实现（默认）。
- get_world_model(name, config)：工厂，构造失败时统一降级为 SimpleGraspPredictor。
"""

from src.world_model.base import BaseWorldModel
from src.world_model.simple_predictor import SimpleGraspPredictor


def get_world_model(name: str = "simple", config: dict | None = None) -> BaseWorldModel:
    """世界模型工厂。

    参数：
        name: 世界模型名称，目前支持 "simple"（启发式）。
        config: 配置 dict（可含 enabled 开关、风险阈值等）。

    返回：
        BaseWorldModel 实例。任何构造失败都降级为 SimpleGraspPredictor，绝不抛错。

    容错原则：世界模型属“可选增强模块”，构造失败不应影响主流程，
    因此这里捕获所有异常并回退到最简实现。
    """
    config = config or {}
    try:
        if name in (None, "", "simple", "simple_predictor", "heuristic"):
            return SimpleGraspPredictor(config)
        # 未知名称：降级为 simple，并打印中文告警
        print(f"[world_model] 未知世界模型 '{name}'，已降级为 SimpleGraspPredictor")
        return SimpleGraspPredictor(config)
    except Exception as exc:  # noqa: BLE001  —— 故意捕获一切，保证不崩
        print(f"[world_model] 构造世界模型失败（{exc}），降级为 SimpleGraspPredictor")
        return SimpleGraspPredictor(config)


__all__ = ["BaseWorldModel", "SimpleGraspPredictor", "get_world_model"]
