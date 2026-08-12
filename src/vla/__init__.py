# -*- coding: utf-8 -*-
"""
VLA 模块包入口 —— src/vla/__init__.py
=====================================

对外暴露统一工厂 ``get_vla(name, config) -> BaseVLA``，以及三个后端类与基类。

【工厂的容错承诺（核心）】
- name ∈ {"rule_based", "vlm_rule", "smolvla"}。
- 任何后端的**构造失败**（缺包、配置异常、初始化报错等）都会被捕获，
  统一回退到 RuleBasedVLA，并 log 一条中文告警。
- 未知 name 也回退 rule_based 并告警。
- 因此调用方拿到的永远是一个**可用的** BaseVLA 实例，绝不会因为选了重后端而崩溃。
  这正是“只装核心依赖即可跑通完整演示”的入口保障。

用法：
    from src.vla import get_vla
    vla = get_vla("smolvla", model_config)   # 跑不动会自动给你 RuleBasedVLA
    plan = vla.predict(observation)
"""

import logging
from typing import Any, Dict, Optional

from src.vla.base import BaseVLA
from src.vla.rule_based import RuleBasedVLA

logger = logging.getLogger("vla.factory")

# 支持的后端名称集合（对外可见，便于配置校验/看板下拉）。
SUPPORTED_BACKENDS = ("rule_based", "vlm_rule", "smolvla")

__all__ = [
    "BaseVLA",
    "RuleBasedVLA",
    "get_vla",
    "SUPPORTED_BACKENDS",
]


def get_vla(name: str, config: Optional[Dict[str, Any]] = None) -> BaseVLA:
    """VLA 后端工厂：按 name 创建后端实例，失败统一回退 rule_based。

    参数
    ----
    name : str
        后端名称，取值 ∈ {"rule_based", "vlm_rule", "smolvla"}。
        大小写不敏感；空/None/未知值都回退 rule_based。
    config : dict | None
        传给后端的配置（通常是 model_config.yaml 解析后的 dict 或其子片段）。

    返回
    ----
    BaseVLA
        一个保证可用的后端实例。
    """
    cfg = config or {}
    key = (name or "").strip().lower()

    # rule_based：直接构造（理论上不会失败，仍兜底）。
    if key in ("", "rule_based", "rule", "baseline"):
        if key == "":
            logger.info("未指定 VLA 后端，使用默认 rule_based（规则基线）。")
        return _safe_construct(RuleBasedVLA, cfg, "rule_based")

    if key in ("vlm_rule", "vlm", "vlm-rule"):
        # 延迟导入：vlm_rule 仅在被选用时才 import，避免无谓加载。
        try:
            from src.vla.vlm_rule import VLMRuleVLA
        except Exception as exc:
            logger.warning("导入 VLMRuleVLA 失败（%s），回退 rule_based 规则基线。", exc)
            return RuleBasedVLA(cfg)
        return _safe_construct(VLMRuleVLA, cfg, "vlm_rule")

    if key in ("smolvla", "smol", "smolvla_500m", "smolvla-500m"):
        try:
            from src.vla.smolvla import SmolVLAModel
        except Exception as exc:
            logger.warning("导入 SmolVLAModel 失败（%s），回退 rule_based 规则基线。", exc)
            return RuleBasedVLA(cfg)
        return _safe_construct(SmolVLAModel, cfg, "smolvla")

    # 未知名称：告警并回退。
    logger.warning(
        "未知的 VLA 后端名称：%r（支持 %s），回退 rule_based 规则基线。",
        name,
        "/".join(SUPPORTED_BACKENDS),
    )
    return RuleBasedVLA(cfg)


def _safe_construct(cls, config: Dict[str, Any], label: str) -> BaseVLA:
    """安全构造一个后端实例：构造异常时回退 RuleBasedVLA 并 log 中文告警。

    注意：各重后端（vlm_rule/smolvla）内部还有自己的“运行期降级”（无 key/
    加载失败时 predict 走规则），此处覆盖的是“构造阶段”就失败的极端情况，
    形成双重保险。
    """
    try:
        instance = cls(config)
        # 兜底校验：必须是 BaseVLA 子类实例，否则视为构造失败。
        if not isinstance(instance, BaseVLA):
            raise TypeError(f"{cls.__name__} 不是 BaseVLA 实例")
        return instance
    except Exception as exc:
        logger.warning(
            "构造 VLA 后端 [%s] 失败（%s: %s），回退 rule_based 规则基线。",
            label,
            type(exc).__name__,
            exc,
        )
        try:
            return RuleBasedVLA(config)
        except Exception as exc2:  # pragma: no cover - 理论上不可能
            # 连规则基线都构造失败属严重异常，但仍不抛出，给一个最小可用占位。
            logger.error("严重：规则基线也构造失败（%s）。", exc2)
            return _NullVLA(config)


class _NullVLA(BaseVLA):
    """最终极兜底后端：永远返回空计划，确保 get_vla 绝不返回 None / 抛异常。

    仅在“连 RuleBasedVLA 都无法构造”的不可能情形下出现，存在意义是把
    “系统永不崩溃”的承诺贯彻到底。
    """

    backend = "rule_based"
    name = "空兜底"

    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        from src.vla.base import empty_action_plan

        return empty_action_plan(
            self._safe_instruction(observation),
            reasoning="VLA 全部不可用，返回空计划（系统仍未崩溃）。",
        )
