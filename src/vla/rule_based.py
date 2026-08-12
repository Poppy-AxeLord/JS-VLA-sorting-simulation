# -*- coding: utf-8 -*-
"""
规则基线 VLA —— rule_based.py
=============================

RuleBasedVLA 是整个系统的**默认后端**：零重依赖（只用标准库 + 项目内模块），
永远可用、永远不崩溃。它通过对中文指令做关键词/模式匹配，结合仿真给出的
SceneState 真值，产出一份可执行的 ActionPlan。

【为什么以规则基线为默认（产品决策）】
- 本项目是“产品验证 POC”，不是刷算法分。我们要先用最低成本把“感知→理解→
  规划→执行→评测→失败分析”的完整闭环跑通，证明系统价值；真正的 VLA 大模型
  随时可以通过工厂热插拔进来对比。
- 规则基线天然给出一个**诚实的下界**：评测里它在简单/中等/困难的成功率约
  0.85/0.70/0.55（失败注入在 sorting 引擎里完成，本模块只负责“理解+规划”）。
  上层换更优策略或世界模型时，提升才有可信的参照系。
- 规则基线也是所有高级后端的**统一降级目标**：VLM 不可用、SmolVLA 加载失败时
  都回退到它，保证系统“无论缺什么包都能完整演示”。

【它能解析哪些指令类型】（对应 tasks.yaml 的 type）
- 基础：  “把红色的螺丝放到A区”           → 颜色/类别约束 + 指定料盒
- 条件：  “把所有金属零件分拣到B区”       → 材质/属性条件 + 指定料盒
- 优先级：“优先分拣芯片，然后是电容”      → 处理顺序约束
- 批量：  “把所有零件按颜色分类”          → 按某维度自动归类到 A/B/C
- 模糊：  “把大的零件放左边，小的放右边”  → 模糊空间映射（左=A、右=C）

解析结果落在 parsed_intent.mode（基础/条件/优先级/批量/模糊）与 rules 上，
再结合 SceneState 把每个零件分配到具体料盒，生成 grasp→place 动作对。
"""

import re
from typing import Any, Dict, List, Optional, Tuple

from src.vla.base import (
    BaseVLA,
    BIN_KEYS,
    BIN_NAMES,
    make_action,
    empty_action_plan,
)

# ---------------------------------------------------------------------------
# 中文关键词词表：把自然语言里的说法映射到 SceneState 中零件的标准属性值。
# 这些值必须与 SPEC §2 的零件库严格一致（颜色/材质/大小/形状/中文名/code）。
# ---------------------------------------------------------------------------

# 颜色词 → 标准颜色值
COLOR_WORDS: Dict[str, str] = {
    "银色": "银色", "银": "银色",
    "蓝色": "蓝色", "蓝": "蓝色",
    "棕色": "棕色", "棕": "棕色", "褐色": "棕色",
    "黑色": "黑色", "黑": "黑色",
    "白色": "白色", "白": "白色",
    "绿色": "绿色", "绿": "绿色",
    "红色": "红色", "红": "红色",
}

# 材质词 → 标准材质值
MATERIAL_WORDS: Dict[str, str] = {
    "金属": "金属",
    "陶瓷": "陶瓷",
    "塑料": "塑料",
    "复合": "复合", "复合材料": "复合",
    "玻璃": "玻璃",
}

# 大小词 → 标准大小值
SIZE_WORDS: Dict[str, str] = {
    "大": "大", "大的": "大", "大件": "大",
    "中": "中", "中等": "中",
    "小": "小", "小的": "小", "小件": "小", "微": "小",
}

# 形状词 → 标准形状值
SHAPE_WORDS: Dict[str, str] = {
    "圆柱": "圆柱", "圆柱形": "圆柱",
    "六边形": "六边形", "六角": "六边形",
    "方形": "方形", "方": "方形",
    "块状": "块状",
    "平板": "平板", "板状": "平板",
}

# 类别词（零件中文名 / 同义说法）→ code。注意把“PCB板/电路板”都映射到 pcb。
CATEGORY_WORDS: Dict[str, str] = {
    "螺丝": "screw", "螺钉": "screw",
    "螺母": "nut",
    "电容": "capacitor",
    "电阻": "resistor",
    "芯片": "chip", "ic": "chip",
    "连接器": "connector", "接插件": "connector",
    "散热器": "heatsink", "散热片": "heatsink",
    "pcb": "pcb", "PCB": "pcb", "电路板": "pcb", "pcb板": "pcb",
    "按键": "button", "按钮": "button",
    "显示屏": "display", "屏幕": "display", "显示器": "display",
}

# 料盒词 → 标准料盒 key。覆盖 "A区/A/甲" 等说法。
BIN_WORDS: Dict[str, str] = {
    "a区": "A", "a": "A", "甲": "A",
    "b区": "B", "b": "B", "乙": "B",
    "c区": "C", "c": "C", "丙": "C",
}

# “易碎”相关词
FRAGILE_WORDS = ["易碎", "脆弱", "怕摔", "易碎品"]

# 颜色 → 默认料盒：用于“按颜色分类”批量指令时给每种颜色一个稳定的归属。
# 仅 3 个料盒，故多种颜色会复用；这里给一个确定性的映射，保证可复现与可解释。
COLOR_TO_BIN_DEFAULT: Dict[str, str] = {
    "银色": "A", "蓝色": "A",          # 冷/金属系 → A
    "棕色": "B", "绿色": "B", "白色": "B",  # 中性/复合系 → B
    "黑色": "C", "红色": "C",          # 深色/警示系 → C
}

# 材质 → 默认料盒：用于“按材质分类”等批量指令。
MATERIAL_TO_BIN_DEFAULT: Dict[str, str] = {
    "金属": "A",
    "陶瓷": "B", "塑料": "B", "复合": "B",
    "玻璃": "C",
}

# 大小 → 默认料盒：用于“按大小分类”/模糊“大左小右”等。
SIZE_TO_BIN_DEFAULT: Dict[str, str] = {
    "大": "A",   # 左
    "中": "B",   # 中
    "小": "C",   # 右
}


class RuleBasedVLA(BaseVLA):
    """纯规则 VLA 后端（默认 / 永远可用）。"""

    backend = "rule_based"
    name = "规则基线"

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        # 规则后端给出的动作置信度：解析到明确约束时高，模糊/兜底时低。
        # 这个置信度会被 strategies 用来决定是否二次确认/人工介入，因此有意义。
        self._conf_high = float(self.config.get("conf_high", 0.95))
        self._conf_mid = float(self.config.get("conf_mid", 0.8))
        self._conf_low = float(self.config.get("conf_low", 0.55))

    # ==================================================================
    # 对外主接口
    # ==================================================================
    def predict(self, observation: Dict[str, Any]) -> Dict[str, Any]:
        """解析指令 + 结合 SceneState 真值 → ActionPlan。

        本方法整体包在 try/except 中：任何意外都不会冒泡到引擎层，
        最坏情况返回空计划。这是“优雅降级”原则在最底层后端的体现。
        """
        instruction = self._safe_instruction(observation)
        parts = self._safe_scene_parts(observation)
        try:
            return self._predict_impl(instruction, parts)
        except Exception as exc:  # pragma: no cover - 防御性兜底
            return empty_action_plan(
                instruction,
                reasoning=f"规则解析发生异常，已返回空计划：{exc}",
            )

    # ==================================================================
    # 解析主流程
    # ==================================================================
    def _predict_impl(
        self, instruction: str, parts: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        text = instruction.strip()
        low = text.lower()

        # 1) 判定指令模式（优先级 > 批量 > 模糊 > 条件 > 基础）。
        #    顺序很重要：先识别强信号（“优先”“按…分类”），再退到一般情形。
        if self._is_priority(low):
            mode = "优先级"
            plan = self._build_priority(text, low, parts)
        elif self._is_fuzzy(low):
            mode = "模糊"
            plan = self._build_fuzzy(text, low, parts)
        elif self._is_batch(low):
            mode = "批量"
            plan = self._build_batch(text, low, parts)
        elif self._is_conditional(low, text):
            mode = "条件"
            plan = self._build_conditional(text, low, parts)
        else:
            mode = "基础"
            plan = self._build_basic(text, low, parts)

        rules, assignments, actions, reasoning = plan

        # 2) 组装 ActionPlan（形状严格对齐 SPEC §5）。
        return {
            "instruction": instruction,
            "parsed_intent": {"mode": mode, "rules": rules},
            "actions": actions,
            "reasoning": reasoning,
        }

    # ==================================================================
    # 模式判定
    # ==================================================================
    @staticmethod
    def _is_priority(low: str) -> bool:
        # “优先”“先…后/再/然后…”是优先级指令的强信号。
        if "优先" in low:
            return True
        if ("先" in low) and ("后" in low or "再" in low or "然后" in low):
            return True
        return False

    @staticmethod
    def _is_fuzzy(low: str) -> bool:
        # 模糊空间指令：出现“左/右”或“大…小…”这种相对空间映射。
        if ("左" in low or "右" in low) and ("放" in low or "边" in low):
            return True
        return False

    @staticmethod
    def _is_batch(low: str) -> bool:
        # 批量分类指令：“所有…按…分类/归类”。
        if "按" in low and ("分类" in low or "归类" in low or "分" in low):
            return True
        if ("所有" in low or "全部" in low) and ("分类" in low or "归类" in low):
            return True
        return False

    def _is_conditional(self, low: str, text: str) -> bool:
        # 条件指令：含“所有/全部”+ 某个属性条件（材质/颜色/大小/易碎），
        # 且指向单一料盒。例如“把所有金属零件分拣到B区”。
        has_scope = ("所有" in low) or ("全部" in low)
        has_attr = (
            self._match_material(text) is not None
            or self._match_size(text) is not None
            or any(w in text for w in FRAGILE_WORDS)
        )
        return has_scope and has_attr

    # ==================================================================
    # 各模式的计划构建
    # 返回统一为 (rules, assignments, actions, reasoning)
    #   rules       : List[dict]  结构化规则，落进 parsed_intent.rules
    #   assignments : Dict[part_id, bin]  仅用于推理展示，引擎以 actions 为准
    #   actions     : List[action]
    #   reasoning   : str
    # ==================================================================

    def _build_basic(
        self, text: str, low: str, parts: List[Dict[str, Any]]
    ) -> Tuple[list, dict, list, str]:
        """基础分拣：按指令中的（颜色/材质/大小/类别）过滤出目标零件，
        放到指令指定的料盒（若未指定则默认 A 区）。"""
        color = self._match_color(text)
        material = self._match_material(text)
        size = self._match_size(text)
        code = self._match_category(text)
        target_bin = self._match_bin(low) or "A"

        rule = {
            "filter": {
                "color": color,
                "material": material,
                "size": size,
                "code": code,
            },
            "target_bin": target_bin,
        }
        targets = self._filter_parts(parts, color, material, size, code)

        actions: List[Dict[str, Any]] = []
        assignments: Dict[int, str] = {}
        for p in targets:
            assignments[p["part_id"]] = target_bin
            actions.extend(self._pick_and_place(p, target_bin, self._conf_high))

        desc = self._describe_filter(color, material, size, code)
        if not targets:
            reasoning = (
                f"基础分拣：未在场景中找到匹配【{desc}】的零件，"
                f"目标料盒为 {BIN_NAMES[target_bin]}，本次无可执行动作。"
            )
        else:
            reasoning = (
                f"基础分拣：识别到 {len(targets)} 个匹配【{desc}】的零件，"
                f"全部分配到 {BIN_NAMES[target_bin]}。"
            )
        return [rule], assignments, actions, reasoning

    def _build_conditional(
        self, text: str, low: str, parts: List[Dict[str, Any]]
    ) -> Tuple[list, dict, list, str]:
        """条件分拣：按某一属性条件（材质/大小/易碎）筛选所有符合的零件到单一料盒。"""
        material = self._match_material(text)
        size = self._match_size(text)
        fragile = any(w in text for w in FRAGILE_WORDS)
        target_bin = self._match_bin(low) or "B"

        rule = {
            "scope": "all",
            "condition": {"material": material, "size": size, "fragile": fragile},
            "target_bin": target_bin,
        }
        targets = []
        for p in parts:
            if material is not None and p.get("material") != material:
                continue
            if size is not None and p.get("size") != size:
                continue
            if fragile and not p.get("fragile", False):
                continue
            targets.append(p)

        actions: List[Dict[str, Any]] = []
        assignments: Dict[int, str] = {}
        for p in targets:
            assignments[p["part_id"]] = target_bin
            actions.extend(self._pick_and_place(p, target_bin, self._conf_high))

        cond_parts = []
        if material:
            cond_parts.append(f"材质={material}")
        if size:
            cond_parts.append(f"大小={size}")
        if fragile:
            cond_parts.append("易碎=是")
        cond_desc = "、".join(cond_parts) if cond_parts else "全部零件"
        reasoning = (
            f"条件分拣：按条件【{cond_desc}】筛选出 {len(targets)} 个零件，"
            f"统一分配到 {BIN_NAMES[target_bin]}。"
        )
        return [rule], assignments, actions, reasoning

    def _build_priority(
        self, text: str, low: str, parts: List[Dict[str, Any]]
    ) -> Tuple[list, dict, list, str]:
        """优先级分拣：按指令中出现的类别先后顺序排出处理优先级，
        优先类别先抓取；不同优先级落不同料盒（A>B>C 表示优先级递减）。"""
        order_codes = self._extract_priority_order(text)
        rules = [{"priority_order": order_codes}]

        # 料盒按优先级分层：第 1 优先 → A，第 2 → B，其余 → C。
        bin_by_rank = {0: "A", 1: "B"}
        actions: List[Dict[str, Any]] = []
        assignments: Dict[int, str] = {}

        # 先把零件按“是否在优先级列表 + 优先级名次”排序。
        def rank(p: Dict[str, Any]) -> int:
            c = p.get("code")
            return order_codes.index(c) if c in order_codes else len(order_codes)

        ordered_parts = sorted(parts, key=rank)
        for p in ordered_parts:
            r = rank(p)
            target_bin = bin_by_rank.get(r, "C")
            assignments[p["part_id"]] = target_bin
            # 优先级靠前的置信度更高（指令明确点名）。
            conf = self._conf_high if p.get("code") in order_codes else self._conf_mid
            actions.extend(self._pick_and_place(p, target_bin, conf))

        names = "、".join(self._code_to_name(c) for c in order_codes) or "（未识别到具体类别）"
        reasoning = (
            f"优先级分拣：解析出处理顺序【{names}】，"
            f"按优先级先后抓取并分层放入 A→B→C 料盒，共 {len(ordered_parts)} 个零件。"
        )
        return rules, assignments, actions, reasoning

    def _build_batch(
        self, text: str, low: str, parts: List[Dict[str, Any]]
    ) -> Tuple[list, dict, list, str]:
        """批量分类：把所有零件按某个维度（颜色/材质/大小）自动归类到 A/B/C。"""
        # 判定按哪个维度分类（默认按颜色）。
        if "材质" in text:
            dim = "material"
            mapping = MATERIAL_TO_BIN_DEFAULT
            getter = lambda p: p.get("material")
        elif "大小" in text or "尺寸" in text:
            dim = "size"
            mapping = SIZE_TO_BIN_DEFAULT
            getter = lambda p: p.get("size")
        else:
            dim = "color"
            mapping = COLOR_TO_BIN_DEFAULT
            getter = lambda p: p.get("color")

        rules = [{"batch_by": dim, "mapping": mapping}]
        actions: List[Dict[str, Any]] = []
        assignments: Dict[int, str] = {}
        for p in parts:
            key = getter(p)
            target_bin = mapping.get(key, "C")  # 未覆盖到的值兜底放 C
            assignments[p["part_id"]] = target_bin
            actions.extend(self._pick_and_place(p, target_bin, self._conf_mid))

        dim_cn = {"color": "颜色", "material": "材质", "size": "大小"}[dim]
        reasoning = (
            f"批量分类：把场景中全部 {len(parts)} 个零件按【{dim_cn}】自动归类，"
            f"依据固定映射分配到 A/B/C 料盒（未覆盖到的归 C 区）。"
        )
        return rules, assignments, actions, reasoning

    def _build_fuzzy(
        self, text: str, low: str, parts: List[Dict[str, Any]]
    ) -> Tuple[list, dict, list, str]:
        """模糊空间指令：如“把大的零件放左边，小的放右边”。
        左=A、中=B、右=C；按大小映射。这是规则后端置信度较低的分支。"""
        # 解析“大→左、小→右”这类映射；缺省用 SIZE_TO_BIN_DEFAULT（大A 中B 小C）。
        size_to_bin = dict(SIZE_TO_BIN_DEFAULT)
        # 若指令明说“小的放左/右”，则覆盖默认。
        if "小" in text and "左" in text:
            size_to_bin["小"] = "A"
        if "小" in text and "右" in text:
            size_to_bin["小"] = "C"
        if "大" in text and "右" in text:
            size_to_bin["大"] = "C"
        if "大" in text and "左" in text:
            size_to_bin["大"] = "A"

        rules = [{"fuzzy_spatial": True, "size_to_bin": size_to_bin}]
        actions: List[Dict[str, Any]] = []
        assignments: Dict[int, str] = {}
        for p in parts:
            target_bin = size_to_bin.get(p.get("size"), "B")
            assignments[p["part_id"]] = target_bin
            # 模糊指令置信度最低：left/right 与具体料盒的对应是“我们替它做的假设”，
            # 这恰好是“理解类失败”最容易发生的地方，低置信度让 strategies 更易触发复核。
            actions.extend(self._pick_and_place(p, target_bin, self._conf_low))

        reasoning = (
            "模糊空间分拣：将“左/中/右”近似映射为 A/B/C 料盒，"
            f"按零件大小放置共 {len(parts)} 个零件。该类指令存在歧义，置信度偏低。"
        )
        return rules, assignments, actions, reasoning

    # ==================================================================
    # 关键词匹配辅助
    # ==================================================================
    @staticmethod
    def _match_color(text: str) -> Optional[str]:
        for w, v in COLOR_WORDS.items():
            if w in text:
                return v
        return None

    @staticmethod
    def _match_material(text: str) -> Optional[str]:
        for w, v in MATERIAL_WORDS.items():
            if w in text:
                return v
        return None

    @staticmethod
    def _match_size(text: str) -> Optional[str]:
        # 注意：避免“大小”一词被误命中——若同时出现“大”和“小”，交给上层模糊分支。
        if "大" in text and "小" in text:
            return None
        for w, v in SIZE_WORDS.items():
            if w in text:
                return v
        return None

    @staticmethod
    def _match_category(text: str) -> Optional[str]:
        low = text.lower()
        for w, code in CATEGORY_WORDS.items():
            if w.lower() in low:
                return code
        return None

    @staticmethod
    def _match_bin(low: str) -> Optional[str]:
        # 优先匹配 "x区"，再匹配单字母，避免把无关字母误判。
        for w in ("a区", "b区", "c区", "甲", "乙", "丙"):
            if w in low:
                return BIN_WORDS[w]
        # 单独的 a/b/c：用正则要求其前后不是字母，降低误命中。
        m = re.search(r"(?<![a-z])([abc])(?![a-z])", low)
        if m:
            return BIN_WORDS[m.group(1)]
        return None

    def _extract_priority_order(self, text: str) -> List[str]:
        """从优先级指令中按出现先后抽取类别 code，去重保序。

        例：“优先分拣芯片，然后是电容” → ["chip", "capacitor"]
        """
        low = text.lower()
        found: List[Tuple[int, str]] = []
        for w, code in CATEGORY_WORDS.items():
            idx = low.find(w.lower())
            if idx >= 0:
                found.append((idx, code))
        found.sort(key=lambda x: x[0])
        order: List[str] = []
        for _, code in found:
            if code not in order:
                order.append(code)
        return order

    # ==================================================================
    # 零件过滤与动作生成
    # ==================================================================
    @staticmethod
    def _filter_parts(
        parts: List[Dict[str, Any]],
        color: Optional[str],
        material: Optional[str],
        size: Optional[str],
        code: Optional[str],
    ) -> List[Dict[str, Any]]:
        out = []
        for p in parts:
            if color is not None and p.get("color") != color:
                continue
            if material is not None and p.get("material") != material:
                continue
            if size is not None and p.get("size") != size:
                continue
            if code is not None and p.get("code") != code:
                continue
            out.append(p)
        return out

    def _pick_and_place(
        self, part: Dict[str, Any], target_bin: str, confidence: float
    ) -> List[Dict[str, Any]]:
        """为单个零件生成 grasp→place 动作对（move 由引擎/planner 视情况补充）。

        易碎件附加 gentle 参数，提示执行层用更轻柔的放置姿态——这同时也是
        失败注入模型里“易碎大件↑放置失败”的可观测抓手。
        """
        pid = part.get("part_id")
        code = part.get("code")
        fragile = bool(part.get("fragile", False))
        params = {"gentle": True} if fragile else {}
        note_g = f"抓取 {part.get('name', code)}（id={pid}）"
        note_p = f"放置到 {BIN_NAMES.get(target_bin, target_bin)}"
        return [
            make_action(
                "grasp",
                part_code=code,
                part_id=pid,
                target_bin=target_bin,
                confidence=confidence,
                params=params,
                note=note_g,
            ),
            make_action(
                "place",
                part_code=code,
                part_id=pid,
                target_bin=target_bin,
                confidence=confidence,
                params=params,
                note=note_p,
            ),
        ]

    # ==================================================================
    # 文案辅助
    # ==================================================================
    @staticmethod
    def _code_to_name(code: str) -> str:
        for w, c in CATEGORY_WORDS.items():
            if c == code:
                return w
        return code

    @staticmethod
    def _describe_filter(
        color: Optional[str],
        material: Optional[str],
        size: Optional[str],
        code: Optional[str],
    ) -> str:
        parts = []
        if color:
            parts.append(f"颜色={color}")
        if material:
            parts.append(f"材质={material}")
        if size:
            parts.append(f"大小={size}")
        if code:
            parts.append(f"类别={code}")
        return "、".join(parts) if parts else "全部零件"
