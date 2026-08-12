"""启发式抓取风险预测器（SPEC §8）

SimpleGraspPredictor —— 纯启发式、零重依赖、永远可运行的世界模型实现。

核心启发规则（产品/工程权衡，均来自工业分拣的常识与失败分析）：
1) 掉落风险（execution/放置失败）：
   - 易碎(fragile) 件本身夹持窗口窄、夹太紧会碎、夹太松会滑 → 基础风险高。
   - 大件(size=大) 重心高、力矩大、二指夹爪接触面有限 → 掉落概率上升。
   - 平板/玻璃形状（pcb/display，shape=平板 或 material=玻璃）→ 表面光滑、接触面薄，
     既容易滑脱又容易碎，叠加后风险最高。
2) 识别风险（perception/遮挡看不见 或 识别错误）：
   - 被遮挡(occluded=True) → 感知置信度低、定位不准 → 识别/定位风险高。
   - 小件(size=小) + 银色金属（screw/nut/capacitor）相互相似 → 识别错误风险上升。

这些风险分数有两个用途：
- 供 engine 在“抓取前”决定是否调整姿态 / 改顺序（world_model on 的增益来源）。
- 作为失败注入模型的“先验”，让评测/失败分析更贴近真实物理直觉。

开关（A/B）：通过 config["enabled"] 或实例属性 enabled 控制；
关闭时 engine 不应调用本模块（或调用也应返回零增益），用于对比世界模型的提升。
"""

from src.world_model.base import BaseWorldModel


# 失败大类中文名（与 SPEC §3 严格一致，供 predicted_failures 使用）
_CAT_PERCEPTION = "感知类失败"
_CAT_EXECUTION = "执行类失败"

# 与 §2 一致：易被误识别的“小件银色金属”集合（相似物）
_SIMILAR_SMALL_METAL = {"screw", "nut", "capacitor"}
# 平板/玻璃类（最易“滑脱 + 易碎”叠加）
_FLAT_FRAGILE = {"pcb", "display"}


class SimpleGraspPredictor(BaseWorldModel):
    """启发式抓取风险预测器（默认世界模型）。"""

    name = "启发式世界模型(SimpleGraspPredictor)"

    def __init__(self, config: dict | None = None):
        super().__init__(config)
        # 各启发项的权重（可在 config 覆盖；默认值经手调，使风险落在合理区间）
        w = (config or {}).get("risk_weights", {})
        self.w_fragile = float(w.get("fragile", 0.30))       # 易碎基础风险
        self.w_large = float(w.get("large", 0.20))           # 大件附加风险
        self.w_flat = float(w.get("flat_glass", 0.25))       # 平板/玻璃附加风险
        self.w_occluded = float(w.get("occluded", 0.45))     # 遮挡识别风险
        self.w_similar = float(w.get("similar", 0.25))       # 相似小金属识别风险

    # ------------------------------------------------------------------ #
    # 单件抓取风险
    # ------------------------------------------------------------------ #
    def assess_grasp_risk(self, part: dict, grasp_pose: dict | None = None) -> dict:
        """评估单个零件的抓取风险（掉落 + 识别两类，取主导项作为 reason）。

        返回 {"risk": 0-1, "reason": 中文}。
        """
        part = part or {}
        code = part.get("code", "")
        size = part.get("size", "")
        shape = part.get("shape", "")
        material = part.get("material", "")
        fragile = bool(part.get("fragile", False))
        occluded = bool(part.get("occluded", False))

        # ---- 掉落风险（执行类） ----
        drop_risk = 0.0
        drop_reasons = []
        if fragile:
            drop_risk += self.w_fragile
            drop_reasons.append("易碎件夹持窗口窄")
        if size in ("大", "large", "L"):
            drop_risk += self.w_large
            drop_reasons.append("大件重心高力矩大")
        if code in _FLAT_FRAGILE or shape in ("平板", "flat") or material in ("玻璃", "glass"):
            drop_risk += self.w_flat
            drop_reasons.append("平板/玻璃表面光滑易滑脱")

        # ---- 识别风险（感知类） ----
        percept_risk = 0.0
        percept_reasons = []
        if occluded:
            percept_risk += self.w_occluded
            percept_reasons.append("被遮挡导致定位不准")
        if code in _SIMILAR_SMALL_METAL and size in ("小", "small", "S"):
            percept_risk += self.w_similar
            percept_reasons.append("小件银色金属相似易混淆")

        # 综合风险取两类的“软最大”（避免简单相加溢出），并裁剪到 [0,1]
        risk = max(drop_risk, percept_risk) + 0.3 * min(drop_risk, percept_risk)
        risk = max(0.0, min(1.0, risk))

        # 主导原因：哪类风险更大就以哪类解释为主
        if drop_risk >= percept_risk and drop_reasons:
            reason = "掉落风险偏高：" + "、".join(drop_reasons)
            dominant = _CAT_EXECUTION
        elif percept_reasons:
            reason = "识别风险偏高：" + "、".join(percept_reasons)
            dominant = _CAT_PERCEPTION
        else:
            reason = "无明显风险（常规金属小件）"
            dominant = None

        return {
            "risk": round(risk, 3),
            "reason": reason,
            # 附加字段（非 SPEC 强制，但便于 engine 决策与失败注入复用）
            "dominant_category": dominant,
            "drop_risk": round(min(1.0, drop_risk), 3),
            "perception_risk": round(min(1.0, percept_risk), 3),
        }

    # ------------------------------------------------------------------ #
    # 计划级 rollout 预测
    # ------------------------------------------------------------------ #
    def simulate_rollout(self, scene: dict, action_plan: dict) -> dict:
        """对整份动作计划做前瞻预测。

        逻辑：对场景内每个零件评估抓取风险，挑出“高风险”项作为 predicted_failures，
        整体 risk_score 取所有零件风险的均值（再叠加“零件数量”带来的规划复杂度）。
        """
        scene = scene or {}
        parts = scene.get("parts", []) or []

        predicted_failures = []
        risks = []
        for part in parts:
            assessment = self.assess_grasp_risk(part)
            r = assessment.get("risk", 0.0)
            risks.append(r)
            if self.is_high_risk(r):
                predicted_failures.append({
                    "part_code": part.get("code"),
                    "part_id": part.get("part_id"),
                    "category": assessment.get("dominant_category") or _CAT_EXECUTION,
                    "reason": assessment.get("reason"),
                    "risk": r,
                })

        # 综合风险：平均风险 + 规划复杂度（零件越多，规划/路径出错概率越高）
        avg_risk = sum(risks) / len(risks) if risks else 0.0
        n = len(parts)
        # 每超过 5 个零件，叠加少量规划复杂度风险（上限 0.2）
        planning_complexity = min(0.20, max(0, (n - 5)) * 0.04)
        risk_score = max(0.0, min(1.0, avg_risk + planning_complexity))

        return {
            "predicted_failures": predicted_failures,
            "risk_score": round(risk_score, 3),
        }
