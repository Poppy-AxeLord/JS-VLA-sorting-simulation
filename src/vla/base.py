# -*- coding: utf-8 -*-
"""
VLA 抽象基类 base.py
====================

定义所有 VLA（Vision-Language-Action）后端的统一接口。无论底层是纯规则、
VLM+规则、还是真正的 SmolVLA 神经网络模型，对外都暴露同一个 ``predict``
方法，从而让分拣引擎（src.sorting.engine）无需关心具体实现，可以随意替换
后端做 A/B 对比。

【设计要点 / 产品决策】
- 接口稳定是“可替换性”的前提：本 POC 的核心卖点之一就是“今天用规则跑通
  全流程，明天换上更强的 VLA 模型只需改一行工厂参数”。因此 BaseVLA 把
  observation / ActionPlan 的形状固定下来，作为各模块之间的契约。
- observation 同时携带 SceneState 真值（仿真给出）与相机图像：
    * 规则后端主要消费 SceneState 真值；
    * 视觉/神经后端主要消费 images；
  两者用同一个 observation 是为了让评测在“是否使用真值”这件事上保持公平、
  可切换。
- 所有后端的 ``predict`` 必须返回符合 ActionPlan 形状的 dict，绝不抛异常到
  引擎层（具体后端内部要 try/except 兜底），这是“优雅降级第一原则”的体现。

【observation 形状】（与 SPEC §5 一致）
    observation = {
        "instruction": str,            # 自然语言指令，例如“把红色的螺丝放到A区”
        "scene": SceneState,           # 仿真给出的场景真值，见下方 SceneState
        "images": {                    # 相机图像，可能为 None（无渲染时）
            "top":  ndarray | None,    # 俯视图
            "side": ndarray | None,    # 侧视图
        },
    }

【SceneState 形状】（仿真真值，供规则与评分使用）
    SceneState = {
        "parts": [
            {
                "part_id": int,        # 场景内唯一 id
                "code": str,           # 零件 code，如 "screw"
                "name": str,           # 中文名，如 "螺丝"
                "material": str,       # 材质，如 "金属"
                "color": str,          # 颜色，如 "银色"
                "size": str,           # 大小：小/中/大
                "shape": str,          # 形状，如 "圆柱"
                "fragile": bool,       # 是否易碎
                "weight": float,       # 重量（objects.build_scene_state 回填，rule_based 会读取）
                "color_hex": str,      # 颜色十六进制（objects.build_scene_state 回填，供可视化）
                "pos": [x, y],         # 平面坐标（工作台坐标系）
                "occluded": bool,      # 是否被遮挡（困难场景）
            },
            ...
        ],
        "bins": {"A": {...}, "B": {...}, "C": {...}},  # 三个料盒区域
    }

【action 形状】（ActionPlan.actions 中的单个动作，与 SPEC §5 一致）
    action = {
        "type": "move" | "grasp" | "place" | "return",  # 动作类型
        "part_code": str | None,       # 目标零件 code
        "part_id": int | None,         # 目标零件场景 id
        "target_bin": "A" | "B" | "C" | None,           # 目标料盒
        "confidence": float,           # 0.0~1.0 置信度
        "params": dict,                # 附加参数（姿态/速度等）
        "note": str,                   # 中文说明
    }

【ActionPlan 形状】（predict 的返回值，与 SPEC §5 一致）
    ActionPlan = {
        "instruction": str,            # 原始指令
        "parsed_intent": {             # 解析出的意图
            "mode": "基础" | "条件" | "优先级" | "批量" | "模糊",
            "rules": [...],            # 结构化规则列表（各后端自定义内部结构）
        },
        "actions": [action, ...],      # 有序动作序列
        "reasoning": str,              # 中文推理说明，便于看板展示与失败复盘
    }
"""

from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# 三个分拣料盒的标准 key（'A'/'B'/'C'），中文名为 "A区"/"B区"/"C区"。
# 各后端在产出 target_bin 时统一用 'A'/'B'/'C'，展示时再映射中文。
# ---------------------------------------------------------------------------
BIN_KEYS: List[str] = ["A", "B", "C"]
BIN_NAMES: Dict[str, str] = {"A": "A区", "B": "B区", "C": "C区"}

# parsed_intent.mode 的合法取值（与 SPEC §5 / tasks.yaml 的 type 对应）
INTENT_MODES = ["基础", "条件", "优先级", "批量", "模糊"]


def make_action(
    action_type: str,
    part_code: Optional[str] = None,
    part_id: Optional[int] = None,
    target_bin: Optional[str] = None,
    confidence: float = 1.0,
    params: Optional[Dict[str, Any]] = None,
    note: str = "",
) -> Dict[str, Any]:
    """构造一个符合 SPEC §5 形状的 action dict 的小工具。

    统一在这里构造，避免各后端字段写漏或拼错。confidence 会被裁剪到 [0,1]。
    """
    conf = float(confidence)
    if conf < 0.0:
        conf = 0.0
    elif conf > 1.0:
        conf = 1.0
    return {
        "type": action_type,
        "part_code": part_code,
        "part_id": part_id,
        "target_bin": target_bin,
        "confidence": conf,
        "params": params if params is not None else {},
        "note": note,
    }


def empty_action_plan(instruction: str, reasoning: str = "") -> Dict[str, Any]:
    """构造一个空的（无动作的）ActionPlan，作为各后端的兜底返回值。

    当指令完全无法解析、或场景中无可分拣零件时返回它，保证下游引擎拿到的
    始终是合法结构而非 None。
    """
    return {
        "instruction": instruction,
        "parsed_intent": {"mode": "基础", "rules": []},
        "actions": [],
        "reasoning": reasoning or "未能解析出可执行动作，返回空计划。",
    }


class BaseVLA(ABC):
    """所有 VLA 后端的抽象基类。

    子类必须实现 ``predict``。子类应在 ``__init__`` 中设置好 ``name`` 与
    ``backend`` 两个属性（默认在此处给出占位值）。

    属性
    ----
    name : str
        后端的人类可读名称（用于日志/看板），如 "规则基线"、"VLM+规则"。
    backend : str
        后端的机器标识，取值 ∈ {"rule_based", "vlm_rule", "smolvla"}。
        注意：当高级后端降级回规则时，backend 会反映“实际生效”的后端，
        以便评测如实记录真正跑的是什么（避免把规则结果误记为 smolvla）。
    config : dict
        构造时传入的配置（通常来自 model_config.yaml 的相关片段）。
    """

    #: 后端的机器标识，子类覆盖
    backend: str = "base"
    #: 后端的人类可读名称，子类覆盖
    name: str = "BaseVLA"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        # 统一保存配置，子类按需读取；允许传 None（用空 dict 兜底）。
        self.config: Dict[str, Any] = dict(config) if config else {}

    @abstractmethod
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """根据 observation 产出 ActionPlan（dict）。

        参数
        ----
        observation : dict
            形状见模块文档：``{instruction, scene, images}``。

        返回
        ----
        dict
            符合 ActionPlan 形状的字典。**实现必须保证不抛异常**，
            无法处理时返回 ``empty_action_plan(...)``。
        """
        raise NotImplementedError

    # --------------------------------------------------------------
    # 一些对所有后端通用的小工具，放基类里供子类复用。
    # --------------------------------------------------------------
    @staticmethod
    def _safe_scene_parts(observation: Dict[str, Any]) -> List[Dict[str, Any]]:
        """从 observation 中安全取出零件列表，缺失则返回 []。"""
        scene = observation.get("scene") or {}
        parts = scene.get("parts") if isinstance(scene, dict) else None
        return list(parts) if parts else []

    @staticmethod
    def _safe_instruction(observation: Dict[str, Any]) -> str:
        """从 observation 中安全取出指令字符串。"""
        return str(observation.get("instruction") or "")

    def __repr__(self) -> str:  # pragma: no cover - 仅调试用
        return f"<{self.__class__.__name__} name={self.name!r} backend={self.backend!r}>"
