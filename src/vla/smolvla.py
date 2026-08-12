# -*- coding: utf-8 -*-
"""
SmolVLA 神经网络 VLA —— smolvla.py
==================================

SmolVLAModel 尝试加载真正的 SmolVLA-500M（一个轻量级 Vision-Language-Action
模型），用 PyTorch + transformers 在 Apple Silicon 的 MPS 上推理。它代表“系统
可以无缝接入真实 VLA 大模型”这一技术实锤。

【优雅降级是第一原则（务必读懂）】
本后端把所有重依赖与重操作都包在 try/except 中，任一环节失败都**记录中文原因
并回退到 RuleBasedVLA**，绝不让缺包/缺模型/推理报错把整个系统拖垮：
  1) import torch / transformers 失败        → 回退规则（最常见：演示机只装核心依赖）
  2) 设备选择（MPS/CPU）异常                  → 回退规则
  3) 模型/处理器下载或加载失败（无网络/无权重）→ 回退规则
  4) 单次推理异常                             → 该次回退规则（其余次数仍可正常）
这样“只装核心依赖即可跑通完整演示”，而装了 torch+transformers 且能联网拉到权重
时，自动获得真实模型能力。

【Apple Silicon / 性能相关说明（注释即文档，不强依赖）】
- device 由 src.utils.mps_utils.get_device() 决定：torch 可用且 MPS 可用→'mps'，
  否则 'cpu'；torch 缺失也安全返回 'cpu'。本项目禁止 CUDA-only 包。
- HF 镜像：国内拉权重慢/失败时，可设环境变量 HF_ENDPOINT=https://hf-mirror.com
  （或在 model_config.yaml 配置 hf_endpoint，本类会在加载前注入到环境）。
- 量化：500M 模型在 16G+ 统一内存的 Mac 上 fp16 即可跑；显存吃紧时可启用 8bit/4bit
  （bitsandbytes 在 Apple Silicon 支持有限，故仅作说明，默认不启用，避免引入
  CUDA-only 依赖）。相关开关从 config 读取（quantization: none/8bit/4bit）。
- 类级缓存：模型加载昂贵，用 *类变量* 缓存“模型+处理器+设备”，同一进程内多次
  实例化 SmolVLAModel 不会重复加载（评测里会反复 new 后端，缓存很关键）。

【为什么动作仍由规则收尾】
SmolVLA 这类策略模型原生输出的是**连续控制动作**（关节/末端位姿），与本 POC
所需的“离散分拣计划（grasp/place + 目标料盒）”不在同一抽象层。把低层控制映射到
高层分拣分配本身是一个研究问题。本 POC 的工程取舍是：用模型证明“可加载、可推理、
可在 MPS 上跑”，而最终的离散动作分配仍交由经过验证的规则后端产出，保证结果可用、
可复现、可评测。模型推理的产物（特征/置信度）写进 reasoning 供审计。
"""

import os
from typing import Any, Dict, Optional, Tuple

from src.vla.base import BaseVLA
from src.vla.rule_based import RuleBasedVLA

# ---------------------------------------------------------------------------
# 重依赖的安全导入：torch / transformers 均为可选。缺失则 _TORCH_OK=False，
# 本后端在构造时即决定降级，不会在 import 阶段让进程崩溃。
# ---------------------------------------------------------------------------
try:
    import torch  # type: ignore

    _TORCH_OK = True
except Exception:  # pragma: no cover - 未装 torch 时触发
    torch = None  # type: ignore
    _TORCH_OK = False

try:
    import transformers  # type: ignore  # noqa: F401

    _TRANSFORMERS_OK = True
except Exception:  # pragma: no cover - 未装 transformers 时触发
    transformers = None  # type: ignore
    _TRANSFORMERS_OK = False

# mps_utils 也用安全导入：若 utils 尚未生成或导入异常，提供一个本地兜底 get_device，
# 保证本模块单独可用、不会因依赖顺序问题而 import 失败（容错第一原则）。
try:
    from src.utils.mps_utils import get_device as _get_device  # type: ignore
except Exception:  # pragma: no cover - utils 不可用时的兜底

    def _get_device() -> str:
        """兜底设备选择：torch 可用且 MPS 可用→'mps'，否则 'cpu'。"""
        if not _TORCH_OK:
            return "cpu"
        try:
            if torch.backends.mps.is_available():  # type: ignore
                return "mps"
        except Exception:
            pass
        return "cpu"


import logging

logger = logging.getLogger("vla.smolvla")


class SmolVLAModel(BaseVLA):
    """SmolVLA-500M 后端（可选；加载/推理失败自动回退规则）。"""

    backend = "smolvla"
    name = "SmolVLA-500M"

    # --- 类级缓存：进程内只加载一次模型，避免评测反复实例化时重复加载 ---
    _cached_model: Any = None
    _cached_processor: Any = None
    _cached_device: Optional[str] = None
    _cached_model_id: Optional[str] = None
    #: 一旦确定加载失败，记下原因；后续实例直接降级，不再重复尝试拉权重。
    _load_failed_reason: Optional[str] = None

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 规则后端：既作降级目标，又负责最终离散动作的产出（见模块文档）。
        self._rule = RuleBasedVLA(config)

        smol_cfg = (self.config.get("smolvla") or {}) if isinstance(self.config, dict) else {}
        self._model_id = smol_cfg.get("model_id", "lerobot/smolvla_base")
        self._quantization = str(smol_cfg.get("quantization", "none")).lower()
        # HF 镜像：若配置了 hf_endpoint，则注入环境变量，便于国内拉权重。
        hf_endpoint = smol_cfg.get("hf_endpoint") or self.config.get("hf_endpoint")
        if hf_endpoint:
            os.environ.setdefault("HF_ENDPOINT", str(hf_endpoint))

        # 是否真正加载成功；失败则 self._degraded=True，predict 时纯走规则。
        self._degraded = False
        self._degrade_reason = ""
        self._device = "cpu"

        self._try_load()

    # ==================================================================
    # 加载（带类级缓存 + 全程容错）
    # ==================================================================
    def _try_load(self) -> None:
        """尝试加载模型；任何失败都设置降级标志并记录中文原因，不抛异常。"""
        # 0) 之前已确认加载失败过 → 直接降级，避免重复联网拉权重拖慢评测。
        if SmolVLAModel._load_failed_reason is not None:
            self._enter_degraded(SmolVLAModel._load_failed_reason, log=False)
            return

        # 1) 缺重依赖 → 降级
        if not _TORCH_OK:
            self._fail_load("未安装 torch，无法加载 SmolVLA，回退规则基线。")
            return
        if not _TRANSFORMERS_OK:
            self._fail_load("未安装 transformers，无法加载 SmolVLA，回退规则基线。")
            return

        # 2) 命中类级缓存（同一 model_id）→ 直接复用，不重复加载
        if (
            SmolVLAModel._cached_model is not None
            and SmolVLAModel._cached_model_id == self._model_id
        ):
            self._device = SmolVLAModel._cached_device or "cpu"
            logger.info("SmolVLA 命中类级缓存，复用已加载模型（device=%s）。", self._device)
            return

        # 3) 选设备（MPS 优先，CPU 兜底）
        try:
            self._device = _get_device()
        except Exception as exc:
            self._fail_load(f"设备选择失败（{exc}），回退规则基线。")
            return

        # 4) 实际加载模型与处理器（最易失败的一步：无网络/无权重/版本不兼容）
        try:
            model, processor = self._load_model_and_processor()
        except Exception as exc:
            # 记到类级，后续实例不再重试（评测里很关键，避免每个任务都卡一次超时）。
            self._fail_load(
                f"SmolVLA 模型加载失败（{type(exc).__name__}: {exc}），回退规则基线。"
            )
            return

        # 5) 写入类级缓存
        SmolVLAModel._cached_model = model
        SmolVLAModel._cached_processor = processor
        SmolVLAModel._cached_device = self._device
        SmolVLAModel._cached_model_id = self._model_id
        logger.info("SmolVLA 加载成功：model_id=%s，device=%s。", self._model_id, self._device)

    def _load_model_and_processor(self) -> Tuple[Any, Any]:
        """真正去拉/加载权重与处理器。

        说明：SmolVLA 的官方实现位于 lerobot 生态，标准 transformers 不一定直接
        提供其专用类。这里采用“尽力而为”的通用加载路径：
          - 优先尝试 transformers.AutoModel / AutoProcessor 通用接口；
          - 量化开关仅在注释层面给出（8bit/4bit 在 Apple Silicon 支持有限，
            默认 none，避免引入 CUDA-only 的 bitsandbytes）。
        若该 model_id 不被通用接口支持，会抛异常 → 上层统一降级到规则。
        这完全符合“跑不动就降规则”的承诺，且不阻塞演示。
        """
        from transformers import AutoModel, AutoProcessor  # type: ignore

        # dtype：MPS 上用 fp16 更省内存更快；CPU 上用 fp32 更稳。
        dtype = torch.float16 if self._device == "mps" else torch.float32  # type: ignore

        # 量化说明（不强依赖）：如需 8bit/4bit，可在此传 load_in_8bit/4bit，
        # 但 bitsandbytes 主要面向 CUDA，Apple Silicon 不建议，故默认不启用。
        load_kwargs: Dict[str, Any] = {"trust_remote_code": True}
        try:
            load_kwargs["torch_dtype"] = dtype
        except Exception:
            pass

        processor = AutoProcessor.from_pretrained(self._model_id, trust_remote_code=True)
        model = AutoModel.from_pretrained(self._model_id, **load_kwargs)
        try:
            model = model.to(self._device)
            model.eval()
        except Exception:
            # 放置到目标设备失败（如 MPS 不支持某算子）→ 退到 CPU 再试一次。
            self._device = "cpu"
            model = model.to("cpu")
            model.eval()
        return model, processor

    def _fail_load(self, reason: str) -> None:
        """记录一次加载失败：写类级原因（防重试）+ 进入降级。"""
        SmolVLAModel._load_failed_reason = reason
        self._enter_degraded(reason, log=True)

    def _enter_degraded(self, reason: str, log: bool = True) -> None:
        self._degraded = True
        self._degrade_reason = reason
        if log:
            logger.warning("SmolVLA 降级：%s", reason)

    # ==================================================================
    # 主接口
    # ==================================================================
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        # 降级态：直接走规则，并在 reasoning 注明原因，保持评测如实记录。
        if self._degraded or SmolVLAModel._cached_model is None:
            reason = self._degrade_reason or SmolVLAModel._load_failed_reason or \
                "SmolVLA 不可用"
            plan = self._rule.predict(observation)
            plan["reasoning"] = f"[SmolVLA 降级→规则：{reason}] " + plan.get("reasoning", "")
            return plan

        # 模型可用：跑一次推理（仅用于产出特征/置信度证据），动作仍由规则收尾。
        infer_note = ""
        try:
            infer_note = self._run_inference(observation)
        except Exception as exc:
            # 单次推理失败：本次降级到规则，但不污染类级缓存（下次可能成功）。
            plan = self._rule.predict(observation)
            plan["reasoning"] = (
                f"[SmolVLA 推理异常→本次回退规则：{exc}] " + plan.get("reasoning", "")
            )
            return plan

        plan = self._rule.predict(observation)
        plan["reasoning"] = (
            f"[SmolVLA 已推理（device={self._device}）。{infer_note}"
            f" 离散分拣动作由规则引擎产出以保证可执行] " + plan.get("reasoning", "")
        )
        return plan

    def _run_inference(self, observation: Dict[str, Any]) -> str:
        """跑一次前向，返回一句中文证据说明。

        这里做最小可用的推理：把指令文本（及可选俯视图）交给 processor 编码、
        模型前向，捕获输出张量形状作为“模型确实在跑”的证据。真实把低层动作
        映射到分拣分配超出本 POC 范围（见模块文档）。
        """
        model = SmolVLAModel._cached_model
        processor = SmolVLAModel._cached_processor
        instruction = self._safe_instruction(observation)
        images = observation.get("images") or {}
        top_img = images.get("top")

        # 用 no_grad 省内存；MPS/CPU 均适用。
        with torch.no_grad():  # type: ignore
            try:
                if top_img is not None:
                    inputs = processor(
                        text=instruction, images=top_img, return_tensors="pt"
                    )
                else:
                    inputs = processor(text=instruction, return_tensors="pt")
            except Exception:
                # 某些 processor 仅接受 text 关键字或位置参数差异，做一次兜底。
                inputs = processor(instruction, return_tensors="pt")

            inputs = {
                k: (v.to(self._device) if hasattr(v, "to") else v)
                for k, v in inputs.items()
            }
            outputs = model(**inputs)

        # 提取一个可读的形状作为证据
        shape_desc = "输出已生成"
        try:
            if hasattr(outputs, "last_hidden_state"):
                shape_desc = f"last_hidden_state 形状={tuple(outputs.last_hidden_state.shape)}"
            elif hasattr(outputs, "logits"):
                shape_desc = f"logits 形状={tuple(outputs.logits.shape)}"
        except Exception:
            pass
        return f"前向完成（{shape_desc}）。"
