"""世界模型抽象基类（SPEC §8）

定义“抓取风险评估”与“计划级 rollout 预测”的统一接口契约。
任何世界模型实现都遵循此契约，便于在分拣引擎中即插即换、做 A/B 对比。

接口（与 SPEC §8 严格一致）：
- assess_grasp_risk(part, grasp_pose) -> {risk: 0-1, reason}
    针对“单个零件 + 抓取位姿”给出掉落/识别/碰撞风险评分与中文原因。
- simulate_rollout(scene, action_plan) -> {predicted_failures: [...], risk_score}
    针对“整个场景 + 整份动作计划”做前瞻预测，输出预测失败列表与综合风险。
"""

from abc import ABC, abstractmethod


class BaseWorldModel(ABC):
    """世界模型抽象基类。

    子类必须实现 assess_grasp_risk 与 simulate_rollout。
    通过 `enabled` 属性提供 A/B 开关：关闭时引擎应跳过风险评估（等价 world_model off）。
    """

    #: 人类可读名称（中文/英文均可），用于日志与看板展示
    name: str = "base_world_model"

    def __init__(self, config: dict | None = None):
        """初始化。

        参数：
            config: 配置 dict。约定字段：
                - enabled(bool): 是否启用世界模型（A/B 开关），默认 True。
                - high_risk_threshold(float): 判定“高风险”的阈值，默认 0.6。
        """
        self.config = config or {}
        # A/B 开关：用于对比 world_model on/off 对成功率的提升
        self.enabled: bool = bool(self.config.get("enabled", True))
        # 高风险阈值：引擎据此决定是否触发姿态调整 / 改顺序
        self.high_risk_threshold: float = float(self.config.get("high_risk_threshold", 0.6))

    # ------------------------------------------------------------------ #
    # 抽象接口
    # ------------------------------------------------------------------ #
    @abstractmethod
    def assess_grasp_risk(self, part: dict, grasp_pose: dict | None = None) -> dict:
        """评估单个零件在给定抓取位姿下的风险。

        参数：
            part: 零件信息 dict（来自 SceneState.parts 的元素），
                  至少含 code/name/material/size/shape/fragile/occluded 等字段。
            grasp_pose: 抓取位姿 dict（可选），如 {"approach": "top", "width": 0.02}。

        返回：
            {"risk": float(0-1), "reason": str(中文原因)}
            risk 越大越危险；reason 给出可读解释，用于失败注入与看板。
        """
        raise NotImplementedError

    @abstractmethod
    def simulate_rollout(self, scene: dict, action_plan: dict) -> dict:
        """对整份动作计划做前瞻 rollout 预测。

        参数：
            scene: SceneState dict（含 parts/bins）。
            action_plan: ActionPlan dict（含 actions/parsed_intent 等）。

        返回：
            {
              "predicted_failures": [ {"part_code","category","reason","risk"} , ... ],
              "risk_score": float(0-1)   # 整体综合风险
            }
        """
        raise NotImplementedError

    # ------------------------------------------------------------------ #
    # 便捷方法
    # ------------------------------------------------------------------ #
    def is_high_risk(self, risk: float) -> bool:
        """风险是否达到“高风险”阈值。"""
        try:
            return float(risk) >= self.high_risk_threshold
        except (TypeError, ValueError):
            return False
