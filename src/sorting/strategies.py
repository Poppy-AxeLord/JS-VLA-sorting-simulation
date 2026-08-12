"""分拣执行策略（SPEC §7 strategies.py）

策略层把“一次动作执行”的决策逻辑从引擎主流程里抽离出来，便于做版本化 A/B：
- baseline（基线）：阈值高、不做二次确认、重试少、恢复弱 —— 对应 v1 朴素策略。
- optimized（优化）：合理阈值 + 低置信二次确认 + 抓取失败微调姿态重试(≤3) + 异常恢复 —— 对应 v2+。

三类核心能力（均会回写评测指标）：
1) 置信度阈值（confidence threshold）：
   - 置信度低于阈值时，optimized 触发“二次确认”（小幅提升等效成功概率，计 human_intervention），
     极低时直接“人工介入”（计 human_intervention），baseline 则直接按原样执行（更易失败）。
2) 失败重试（retry）：
   - 抓取/执行失败后，optimized 微调姿态重试，最多 3 次，每次成功概率回升一点；
     每次重试计入 retry_count。baseline 不重试（或仅 1 次）。
3) 异常恢复（recovery）：
   - 碰撞/掉落/识别失败等异常发生后，optimized 有一定概率“恢复”（计 recovered），
     恢复成功则该步最终算成功；baseline 恢复能力弱。

重要：策略不直接决定“最终成功率数值”，它只调节概率与重试/恢复行为；
真正的失败注入与判定在 engine 里完成（见 engine._inject_failure）。这样职责清晰、可组合。
"""

import random


class SortingStrategy:
    """分拣执行策略（可命名版本）。

    通过若干数值参数刻画策略“激进/保守”，引擎在执行每个动作时查询这些参数。
    """

    def __init__(
        self,
        name: str = "baseline",
        *,
        confidence_threshold: float = 0.5,
        enable_second_check: bool = False,
        second_check_threshold: float = 0.35,
        human_intervention_threshold: float = 0.2,
        max_retry: int = 1,
        retry_recovery_gain: float = 0.15,
        recovery_prob: float = 0.2,
        success_bonus: float = 0.0,
    ):
        """
        参数：
            name: 策略名（baseline / optimized 等）。
            confidence_threshold: 置信度阈值，低于它视为“低置信”。
            enable_second_check: 低置信时是否做二次确认。
            second_check_threshold: 低于此值才触发二次确认（介于人工介入与正常之间）。
            human_intervention_threshold: 低于此值直接人工介入。
            max_retry: 失败最大重试次数（SPEC 上限 3）。
            retry_recovery_gain: 每次微调姿态重试，成功概率回升的幅度。
            recovery_prob: 异常发生后成功恢复的基础概率。
            success_bonus: 策略给“每步成功概率”的整体加成（optimized>baseline，体现策略优劣）。
        """
        self.name = name
        self.confidence_threshold = float(confidence_threshold)
        self.enable_second_check = bool(enable_second_check)
        self.second_check_threshold = float(second_check_threshold)
        self.human_intervention_threshold = float(human_intervention_threshold)
        # SPEC 约束：重试最多 3 次
        self.max_retry = max(0, min(3, int(max_retry)))
        self.retry_recovery_gain = float(retry_recovery_gain)
        self.recovery_prob = float(recovery_prob)
        self.success_bonus = float(success_bonus)

    def bridge_gap(self) -> float:
        """“首次成功率”与“最终有效成功率”之间的差额（由重试/恢复桥接）。

        gap 越大，说明越多的成功来自“失败后救回”（产生更多 retry/recovered 计数）。
        baseline 的救回能力弱 → gap 小；optimized 强 → gap 大。
        该值不改变最终 p_eff（p_eff 已含策略增益），只决定“多少成功是被救回的”，
        从而让 retry_count / recovered 等过程指标在不同策略下有可区分的真实分布。
        """
        if self.max_retry <= 0:
            return 0.0
        # 用重试增益与恢复概率综合刻画“救回强度”
        strength = 0.5 * self.retry_recovery_gain * self.max_retry + 0.5 * self.recovery_prob
        return max(0.0, min(0.25, strength))

    # ------------------------------------------------------------------ #
    # 置信度处理
    # ------------------------------------------------------------------ #
    def handle_confidence(self, confidence: float, rng: random.Random) -> dict:
        """根据动作置信度决定执行前的干预。

        返回：
            {
              "action": "execute"|"second_check"|"human"|"skip",
              "human_intervention": 0/1,   # 是否计入人工介入
              "confidence_bonus": float,   # 对本步成功概率的修正（二次确认会小幅提升）
              "note": 中文说明
            }
        """
        conf = float(confidence if confidence is not None else 1.0)

        # 极低置信 → 人工介入（无论什么策略都应介入，但 baseline 阈值更低=更少介入）
        if conf < self.human_intervention_threshold:
            return {
                "action": "human",
                "human_intervention": 1,
                "confidence_bonus": 0.30,  # 人工介入后该步基本能成
                "note": f"置信度{conf:.2f}过低，触发人工介入复核",
            }

        # 低置信 → 二次确认（仅 optimized 开启）
        if self.enable_second_check and conf < self.second_check_threshold:
            return {
                "action": "second_check",
                "human_intervention": 1,  # 二次确认也视为一次轻量人工/系统介入
                "confidence_bonus": 0.18,
                "note": f"置信度{conf:.2f}较低，触发二次确认",
            }

        # 低于阈值但未到二次确认线：optimized 略加成（更谨慎），baseline 不处理
        if conf < self.confidence_threshold:
            return {
                "action": "execute",
                "human_intervention": 0,
                "confidence_bonus": 0.05 if self.enable_second_check else 0.0,
                "note": f"置信度{conf:.2f}偏低，按策略谨慎执行",
            }

        # 正常置信
        return {
            "action": "execute",
            "human_intervention": 0,
            "confidence_bonus": 0.0,
            "note": "置信度正常，直接执行",
        }

    # ------------------------------------------------------------------ #
    # 重试（微调姿态）
    # ------------------------------------------------------------------ #
    def retry_grasp(self, base_success_prob: float, rng: random.Random) -> dict:
        """失败后微调姿态重试，最多 max_retry 次。

        每次重试成功概率在 base 基础上叠加 retry_recovery_gain（递增、封顶 0.95）。
        返回：
            {"success": bool, "retries": int, "note": 中文}
        retries 计入 retry_count（无论最终成功与否，重试次数都已发生）。
        """
        retries = 0
        prob = base_success_prob
        for i in range(self.max_retry):
            retries += 1
            prob = min(0.95, prob + self.retry_recovery_gain)
            if rng.random() < prob:
                return {
                    "success": True,
                    "retries": retries,
                    "note": f"第{retries}次微调姿态后抓取成功",
                }
        return {
            "success": False,
            "retries": retries,
            "note": f"微调姿态重试{retries}次仍失败",
        }

    # ------------------------------------------------------------------ #
    # 异常恢复
    # ------------------------------------------------------------------ #
    def try_recover(self, failure_category: str, rng: random.Random) -> dict:
        """异常（碰撞/掉落/识别失败等）发生后尝试恢复。

        不同类别恢复难度不同：执行类（重抓）较易恢复，理解/规划类较难恢复。
        返回：
            {"recovered": bool, "note": 中文}
        recovered=True 计入 recovered，并使该步最终算成功。
        """
        # 按类别给恢复概率乘一个难度系数
        category_factor = {
            "执行类失败": 1.0,    # 滑落/掉落，重抓相对容易
            "环境类失败": 0.8,    # 物体移动/物理异常，重定位后可补救
            "感知类失败": 0.6,    # 识别错误，重感知有一定机会
            "规划类失败": 0.4,    # 顺序/路径错误，运行中较难纠正
            "理解类失败": 0.2,    # 指令理解错了，运行中几乎无法自纠
        }.get(failure_category, 0.5)

        prob = self.recovery_prob * category_factor
        if rng.random() < prob:
            return {"recovered": True, "note": f"针对[{failure_category}]执行异常恢复成功"}
        return {"recovered": False, "note": f"针对[{failure_category}]异常恢复失败"}


# ---------------------------------------------------------------------- #
# 预置命名策略 + 工厂
# ---------------------------------------------------------------------- #
def _baseline() -> SortingStrategy:
    """基线策略（对应 v1）：保守、少干预、几乎不重试、弱恢复。

    标定意图：baseline 的“有效每步成功率”应≈失败注入给出的基础概率本身，
    因此重试/恢复设得很弱（仅 1 次小幅重试、低恢复概率），不显著抬升成功率。
    这样 v1 才能落在 0.85/0.70/0.55 区间，给 v2/v3 留出提升空间。
    """
    return SortingStrategy(
        name="baseline",
        confidence_threshold=0.45,
        enable_second_check=False,
        human_intervention_threshold=0.12,
        max_retry=1,
        retry_recovery_gain=0.05,
        recovery_prob=0.08,
        success_bonus=0.0,
    )


def _optimized() -> SortingStrategy:
    """优化策略（对应 v2+）：合理阈值 + 二次确认 + 多次微调重试 + 较强恢复 + 成功加成。

    success_bonus 体现“更好的策略整体把每步成功率抬高”，是 v2 相对 v1 提升的来源之一。
    """
    return SortingStrategy(
        name="optimized",
        confidence_threshold=0.55,
        enable_second_check=True,
        second_check_threshold=0.40,
        human_intervention_threshold=0.20,
        max_retry=3,
        retry_recovery_gain=0.10,
        recovery_prob=0.30,
        success_bonus=0.14,
    )


# 策略注册表（名称 → 构造函数）
_REGISTRY = {
    "baseline": _baseline,
    "optimized": _optimized,
}


def get_strategy(name: str = "baseline") -> SortingStrategy:
    """策略工厂。未知名称降级为 baseline，并打印中文告警。"""
    factory = _REGISTRY.get(name)
    if factory is None:
        print(f"[strategies] 未知策略 '{name}'，已降级为 baseline")
        return _baseline()
    return factory()


def list_strategies() -> list:
    """返回所有已注册策略名。"""
    return list(_REGISTRY.keys())
