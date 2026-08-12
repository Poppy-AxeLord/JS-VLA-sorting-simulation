# -*- coding: utf-8 -*-
"""
VLM + 规则混合 VLA —— vlm_rule.py
=================================

VLMRuleVLA 在规则基线之上，叠加一个**可选的视觉语言模型（VLM）**做指令解析
与感知增强。它的定位是“规则保底、VLM 加分”：

【设计哲学（产品决策）】
- 纯规则后端在“歧义指令/复杂条件”上能力有限（例如把口语化、多约束、含指代
  的指令解析准确）。VLM 擅长语言理解，正好补这块短板。
- 但 VLM 依赖外部 API（OpenAI / Qwen-VL 等），有 key/网络/配额/超时等不确定性。
  因此本后端遵循铁律：**VLM 的任何失败都不影响系统可用性**——无 key、网络错、
  解析超时、返回非法 JSON，统统回退到内置 RuleBasedVLA 的结果。
- 这样在演示机上“不配 key 也能跑”（纯规则），配了 key 则自动获得更强的指令理解，
  完美契合“只装核心依赖即可跑通完整演示”的总原则。

【与真实 VLM 的对接（轻量、可选）】
- 通过标准库无法发 HTTPS POST，故用 httpx（可选依赖）。httpx 缺失 → 回退规则。
- provider/base_url/model/api_key 等从 config 读取（来自 model_config.yaml 的
  vlm 片段）。api_key 也支持从环境变量名读取，避免把密钥写进配置。
- 这里只用 VLM 做“指令 → 结构化意图”的解析（chat/JSON），不强依赖图像，
  保证即便相机图为 None 也能工作；图像增强留作扩展点（见 _build_messages 注释）。
"""

import json
import os
import re
from typing import Any, Dict, List, Optional

from src.vla.base import BaseVLA, empty_action_plan
from src.vla.rule_based import (
    RuleBasedVLA,
    CATEGORY_WORDS,
    COLOR_WORDS,
    MATERIAL_WORDS,
)

# httpx 为可选依赖：用 try/except 守卫，缺失时 _HTTPX_OK=False，本后端纯走规则。
try:
    import httpx  # type: ignore

    _HTTPX_OK = True
except Exception:  # pragma: no cover - 仅在未装 httpx 时触发
    httpx = None  # type: ignore
    _HTTPX_OK = False


# 让 VLM 输出结构化意图的系统提示词（中文，约束输出为 JSON）。
_SYSTEM_PROMPT = """你是工业分拣指令解析器。请把用户的中文分拣指令解析为 JSON，字段如下：
{
  "mode": "基础|条件|优先级|批量|模糊",
  "color": 颜色或 null,
  "material": 材质或 null,
  "size": "大|中|小"或 null,
  "categories": [零件code...],         // 取值范围: screw,nut,capacitor,resistor,chip,connector,heatsink,pcb,button,display
  "target_bin": "A|B|C"或 null,
  "priority_order": [零件code...],      // 仅优先级指令需要，按先后
  "batch_by": "color|material|size"或 null  // 仅批量指令
}
只输出 JSON，不要任何解释。"""


class VLMRuleVLA(BaseVLA):
    """VLM + 规则混合后端。"""

    backend = "vlm_rule"
    name = "VLM+规则"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 内嵌一个规则后端实例，既做最终动作生成，又做 VLM 失败时的兜底。
        self._rule = RuleBasedVLA(config)

        # 从 config 读取 VLM 连接参数（来自 model_config.yaml 的 vlm 片段）。
        vlm_cfg = (self.config.get("vlm") or {}) if isinstance(self.config, dict) else {}
        self._provider = str(vlm_cfg.get("provider", "openai")).lower()
        self._model = vlm_cfg.get("model", "gpt-4o-mini")
        self._base_url = vlm_cfg.get(
            "base_url", "https://api.openai.com/v1/chat/completions"
        )
        self._timeout = float(vlm_cfg.get("timeout", 15.0))
        # api_key 优先取配置里的明文；否则取 api_key_env 指定的环境变量。
        self._api_key = vlm_cfg.get("api_key") or os.environ.get(
            str(vlm_cfg.get("api_key_env", "OPENAI_API_KEY")), ""
        )

        # 是否真正具备调用 VLM 的条件：装了 httpx + 有 key。否则纯规则。
        self._vlm_ready = bool(_HTTPX_OK and self._api_key)
        if not _HTTPX_OK:
            self._unavailable_reason = "未安装 httpx，VLM 不可用，回退纯规则。"
        elif not self._api_key:
            self._unavailable_reason = "未配置 VLM api_key，回退纯规则。"
        else:
            self._unavailable_reason = ""

    # ==================================================================
    # 主接口
    # ==================================================================
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        instruction = self._safe_instruction(observation)
        parts = self._safe_scene_parts(observation)

        # 无 VLM 条件：直接走规则，但在 reasoning 里注明原因，便于看板/复盘。
        if not self._vlm_ready:
            plan = self._rule.predict(observation)
            plan["reasoning"] = (
                f"[VLM 未启用：{self._unavailable_reason}] " + plan.get("reasoning", "")
            )
            return plan

        # 有 VLM 条件：先用 VLM 解析意图，失败则整体回退规则。
        try:
            intent = self._call_vlm(instruction)
        except Exception as exc:
            # 任何 VLM 侧异常都安全降级，不让系统崩溃。
            plan = self._rule.predict(observation)
            plan["reasoning"] = (
                f"[VLM 调用失败，已回退规则：{exc}] " + plan.get("reasoning", "")
            )
            return plan

        if not intent:
            plan = self._rule.predict(observation)
            plan["reasoning"] = "[VLM 返回为空，已回退规则] " + plan.get("reasoning", "")
            return plan

        # VLM 解析成功：用 VLM 的结构化意图“重写指令约束”，再交给规则后端做
        # 确定性的零件分配与动作生成（VLM 负责理解，规则负责落地，各司其职）。
        try:
            return self._compose_with_intent(instruction, intent, parts, observation)
        except Exception as exc:  # pragma: no cover - 防御
            plan = self._rule.predict(observation)
            plan["reasoning"] = (
                f"[VLM 意图融合异常，已回退规则：{exc}] " + plan.get("reasoning", "")
            )
            return plan

    # ==================================================================
    # VLM 调用
    # ==================================================================
    def _build_messages(self, instruction: str) -> List[Dict[str, Any]]:
        """构造 chat messages。

        当前只用文本指令做解析。若要做“感知增强”，可在此把相机图像
        编码为 base64 data-url 加进 user content（OpenAI/Qwen-VL 多模态格式），
        这里保留为扩展点，不强依赖图像，保证 images=None 时也能工作。
        """
        return [
            {"role": "system", "content": _SYSTEM_PROMPT},
            {"role": "user", "content": instruction},
        ]

    def _call_vlm(self, instruction: str) -> Optional[Dict[str, Any]]:
        """调用 VLM 的 chat completions 接口，返回解析出的意图 dict。

        兼容 OpenAI 风格的 /chat/completions（Qwen-VL DashScope 兼容模式同形）。
        失败时抛异常，由上层统一捕获并降级。
        """
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
        }
        payload = {
            "model": self._model,
            "messages": self._build_messages(instruction),
            "temperature": 0.0,  # 解析任务要确定性
        }
        # 用 httpx 同步客户端，带超时；上下文管理器确保连接释放。
        with httpx.Client(timeout=self._timeout) as client:  # type: ignore
            resp = client.post(self._base_url, headers=headers, json=payload)
            resp.raise_for_status()
            data = resp.json()
        content = (
            data.get("choices", [{}])[0].get("message", {}).get("content", "")
        )
        return self._parse_intent_json(content)

    @staticmethod
    def _parse_intent_json(content: str) -> Optional[Dict[str, Any]]:
        """从 VLM 文本输出中稳健地抽取 JSON 意图。

        VLM 有时会在 JSON 外包裹 ```json ... ``` 或多余文字，这里用正则提取
        第一个花括号块再 json.loads，失败返回 None（触发上层降级）。
        """
        if not content:
            return None
        # 去掉可能的代码块围栏
        text = content.strip()
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()
        # 抓取第一个 {...}
        m = re.search(r"\{.*\}", text, re.DOTALL)
        if not m:
            return None
        try:
            obj = json.loads(m.group(0))
            return obj if isinstance(obj, dict) else None
        except Exception:
            return None

    # ==================================================================
    # 把 VLM 意图融合进规则后端的确定性分配
    # ==================================================================
    def _compose_with_intent(
        self,
        instruction: str,
        intent: Dict[str, Any],
        parts: List[Dict[str, Any]],
        observation: Dict[str, Any],
    ) -> Dict[str, Any]:
        """根据 VLM 给出的结构化意图，构造一条“规范化指令”重新喂给规则后端，
        从而复用规则后端经过测试的零件过滤/分配/动作生成逻辑。

        这是“VLM 理解 + 规则执行”的桥接：好处是动作生成逻辑只有一份、可控、
        可复现，VLM 只改变“理解”这一层。
        """
        mode = intent.get("mode")
        # 用 VLM 意图反向拼出一条无歧义的中文指令，交给规则后端解析。
        canon = self._intent_to_instruction(intent)
        canon_obs = dict(observation)
        canon_obs["instruction"] = canon

        plan = self._rule.predict(canon_obs)

        # 用原始指令覆盖回 instruction 字段（保持对外契约：展示原话）。
        plan["instruction"] = instruction
        # 标注这是 VLM 增强的结果，并把 VLM 原始意图并入 parsed_intent.rules 便于审计。
        if isinstance(plan.get("parsed_intent"), dict):
            if mode in ("基础", "条件", "优先级", "批量", "模糊"):
                plan["parsed_intent"]["mode"] = mode
            plan["parsed_intent"].setdefault("rules", [])
            plan["parsed_intent"]["rules"].append({"vlm_intent": intent})
        plan["reasoning"] = (
            f"[VLM 增强] 由 {self._provider}:{self._model} 解析意图后，"
            f"规范化为指令“{canon}”交规则引擎执行。 " + plan.get("reasoning", "")
        )
        return plan

    @staticmethod
    def _intent_to_instruction(intent: Dict[str, Any]) -> str:
        """把结构化意图回译为一条规则后端能解析的规范中文指令。

        这层回译保证：无论 VLM 用什么措辞，最终落到规则引擎的都是标准说法，
        从而稳定、可复现。
        """
        mode = intent.get("mode")
        target_bin = intent.get("target_bin")
        bin_cn = {"A": "A区", "B": "B区", "C": "C区"}.get(target_bin, "A区")

        # 反查 code → 中文名，便于拼成自然指令
        def code_cn(code: str) -> str:
            for w, c in CATEGORY_WORDS.items():
                if c == code:
                    return w
            return code

        if mode == "优先级":
            order = intent.get("priority_order") or intent.get("categories") or []
            names = "，然后是".join(code_cn(c) for c in order) if order else "芯片"
            return f"优先分拣{names}"

        if mode == "批量":
            by = intent.get("batch_by") or "color"
            by_cn = {"color": "颜色", "material": "材质", "size": "大小"}.get(by, "颜色")
            return f"把所有零件按{by_cn}分类"

        if mode == "模糊":
            return "把大的零件放左边，小的放右边"

        # 基础/条件：拼出 颜色+材质+大小+类别 + 目标料盒
        seg = []
        if intent.get("color"):
            seg.append(str(intent["color"]))
        if intent.get("material"):
            seg.append(str(intent["material"]))
        if intent.get("size"):
            seg.append(str(intent["size"]) + "的")
        cats = intent.get("categories") or []
        if cats:
            seg.append(code_cn(cats[0]))
        scope = "所有" if mode == "条件" else ""
        subject = "".join(seg) if seg else "零件"
        return f"把{scope}{subject}放到{bin_cn}"
