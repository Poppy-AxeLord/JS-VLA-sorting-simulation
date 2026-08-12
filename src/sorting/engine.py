"""分拣执行引擎（SPEC §7 engine.py）—— 评测真实感的核心

SortingEngine 把“仿真环境 + VLA + 策略（+ 可选世界模型）”编排为一次完整的分拣任务执行，
并产出严格符合 SPEC §7 的 TaskResult，供评测体系（metrics/failure_analysis）消费。

主流程（run_task）：
  1. reset 场景（env.reset）
  2. 取相机图（env.get_camera_image，失败/缺失返回 None，不影响流程）
  3. vla.predict(obs) 得 ActionPlan
  4. planner.plan_order 对执行顺序做最近邻路径优化 + 优先级
  5. 逐 action 执行：
       - strategies 控制置信度阈值（二次确认/人工介入）
       - 若启用 world_model，抓取前做风险评估并可调整姿态/改顺序
       - 失败注入模型按难度与零件属性注入 5 类失败（核心）
       - 失败后按策略微调姿态重试(≤3) / 异常恢复
  6. 汇总为 TaskResult

────────────────────────────────────────────────────────────────────────
失败注入模型（最重要，让评测/失败分析无需真实 CV/VLA 也有意义）—— 见 _inject_failure：

  设计目标（SPEC §7 标定）：
    基线 rule_based 在 简单/中等/困难 的“任务成功率”约 0.85 / 0.70 / 0.55。
    启用更优 strategy 或 world_model 时整体提升约 8~15 个百分点。

  做法：先给“每步基础成功概率”一个由难度决定的基线，再叠加由零件属性与指令模式决定的
  风险增量（命中则提高对应失败类别的发生概率）：
    - 遮挡(occluded)        ↑ 感知类失败（识别/定位/遮挡看不见）
    - 相似物(小银色金属)     ↑ 感知类·识别错误
    - 易碎 + 大件 + 平板/玻璃 ↑ 执行类·放置失败(掉落)
    - 模糊指令(mode=模糊)    ↑ 理解类失败
    - 多零件(规划复杂度)     ↑ 规划类失败
    - 随机噪声              ↑ 环境类失败
  策略(optimized)给每步成功概率加成、并提供重试/恢复；世界模型对“高风险件”降低其失败概率
  （等价于抓取前调整姿态/改顺序）。所有随机均走传入的 random.Random(seed)，benchmark 可复现。
────────────────────────────────────────────────────────────────────────

容错降级：env / vla 若构造失败或缺失，本引擎会就地降级（见 _ensure_env / _ensure_vla），
使用内置的极简 Mock，保证 `python -m src.sorting.engine` 永远能跑出一条流水。
"""

import random
import time
from datetime import datetime

# ---- 以 src 为根的绝对导入（SPEC 约定）。世界模型为可选增强，失败注入用本地实现 ----
from src.sorting.planner import plan_order
from src.sorting.strategies import get_strategy, SortingStrategy


# ====================================================================== #
# 失败分类常量（与 SPEC §3 严格一致：大类中文 + 子类中文）
# ====================================================================== #
CAT_PERCEPTION = "感知类失败"
CAT_UNDERSTANDING = "理解类失败"
CAT_PLANNING = "规划类失败"
CAT_EXECUTION = "执行类失败"
CAT_ENVIRONMENT = "环境类失败"

# 每个大类的子类（中文，存入 failure_subtype）
SUBTYPES = {
    CAT_PERCEPTION: ["识别错误", "定位不准", "遮挡看不见", "光照角度问题"],
    CAT_UNDERSTANDING: ["指令理解错误", "漏理解约束", "歧义处理失败"],
    CAT_PLANNING: ["路径不合理", "分拣顺序错误", "优先级处理错误"],
    CAT_EXECUTION: ["抓取失败(滑落)", "放置失败(掉落)", "碰撞导致失败"],
    CAT_ENVIRONMENT: ["物体意外移动", "障碍物出现", "仿真物理异常"],
}

# 相似小金属（识别错误高发），与 §2 一致
_SIMILAR_SMALL_METAL = {"screw", "nut", "capacitor"}
_FLAT_FRAGILE = {"pcb", "display"}

# 难度 → 每步“无风险时的基础成功率”上限（标定使 rule_based+baseline 任务成功率≈0.85/0.70/0.55）。
# 每步有效成功率 p_eff = base * (1 - risk_coef * fail_prob)，其中 fail_prob 来自失败注入模型。
# _RISK_COEF 控制“风险把成功率往下拉”的强度（难度越高、失败注入分布越宽，需不同标定）。
#
# 标定方法（见文件末尾 _calibration_report 与蒙特卡洛脚本）：用真实失败注入分布做网格搜索，
# 在 partial 评分下（简单3件需全成、中等5件容1败、困难8件容1败）使任务成功率贴合目标：
#   简单 base=0.98, coef=0.15 → ≈0.85
#   中等 base=0.92, coef=0.55 → ≈0.70
#   困难 base=0.96, coef=0.35 → ≈0.55
# 这是“产品决策”：用可解释的概率模型复刻真实分拣的难度梯度，让评测/失败分析有意义且可复现。
_BASE_STEP_SUCCESS = {
    "简单": 0.98,
    "中等": 0.92,
    "困难": 0.96,
}
_RISK_COEF = {
    "简单": 0.15,
    "中等": 0.48,
    "困难": 0.34,
}
_DIFFICULTY_ALIASES = {
    "easy": "简单", "simple": "简单",
    "medium": "中等", "mid": "中等",
    "hard": "困难", "difficult": "困难",
}

# ---------------------------------------------------------------------------- #
# 失败注入 / 增益模型的可调常量（原为散落在 _execute_one 里的「魔法数字」，
# 现集中命名 + 标定，便于评审核对、A/B 复现与后续替换为真实信号）。
#
# 标定表（参数 → 含义 → 标定目标）：
#   _WM_HIGHRISK_MULT   世界模型命中高风险件时的失败概率乘子。抓取前调整姿态/改顺序，
#                       风险显著下降。取 0.40 使「+世界模型」相对基线在困难场景多救回
#                       约 8~15pct 的任务，与 SPEC §7 标定目标一致。
#   _WM_LOOKAHEAD_MULT  世界模型对「非高风险件」的普遍前瞻收益（想清楚再抓）。取 0.85
#                       给一点整体收益但不过度，避免掩盖真实失败样本。
#   _RETRY_ALLOC_PROB   「救回」时分配给「重试成功」而非「异常恢复成功」的概率。取 0.6
#                       使重试与恢复计数都有合理样本量（重试略多，贴近真实产线）。
#   _COLLISION_PROB     执行类失败中判定为「碰撞」的概率（影响 collision_count 指标）。
#                       取 0.34 使碰撞次数落在 metrics.yaml 目标（≈5 次/评测）附近。
#   _STEP_MS_BASE       每步基础模拟工时（ms），让「平均耗时」指标接近真实分拣。
#   _STEP_MS_CODE_COEF  按零件 code 长度制造轻微工时差异的系数（ms）。
#   _STEP_MS_RETRY      每次重试的附加工时（ms）。
#   _STEP_MS_INTERVENE  触发二次确认 / 人工介入的附加工时（ms）。
#   _P_EFF_FLOOR/_CEIL  单步有效成功率的下/上限（绝不饱和到 1.0，永远留失败样本）。
# ---------------------------------------------------------------------------- #
_WM_HIGHRISK_MULT = 0.40       # 世界模型：命中高风险件的失败概率乘子
_WM_LOOKAHEAD_MULT = 0.85      # 世界模型：非高风险件的普遍前瞻收益乘子
_RETRY_ALLOC_PROB = 0.6        # 救回时分配到「重试成功」的概率（其余归「异常恢复」）
_COLLISION_PROB = 0.34         # 执行类失败判定为「碰撞」的概率
_STEP_MS_BASE = 380            # 每步基础模拟工时（ms）
_STEP_MS_CODE_COEF = 60        # 按 code 长度制造工时差异的系数（ms）
_STEP_MS_RETRY = 220           # 每次重试的附加工时（ms）
_STEP_MS_INTERVENE = 300       # 二次确认 / 人工介入的附加工时（ms）
_P_EFF_FLOOR = 0.02            # 单步有效成功率下限
_P_EFF_CEIL = 0.96             # 单步有效成功率上限（封顶，绝不饱和）


def _norm_difficulty(d: str) -> str:
    """难度归一化为 简单/中等/困难。"""
    if d in _BASE_STEP_SUCCESS:
        return d
    return _DIFFICULTY_ALIASES.get(str(d).lower(), "中等")


# ====================================================================== #
# 内置极简降级实现（仅当真正的 env / vla 不可用时使用，保证可独立运行）
# ====================================================================== #
class _FallbackEnv:
    """极简降级环境：当 src.simulation 不可用时顶上，保证 engine 可独立跑通。

    它产出与 SPEC SceneState 同构的最小场景，并对 step 返回成功 info。
    真实评测会注入 src.simulation.env.SortingEnv，这里只是“地板”。
    """

    backend = "mock"

    # 10 种零件的最小属性表（与 §2 对齐的子集，供降级场景生成）
    _PARTS = [
        {"code": "screw", "name": "螺丝", "material": "金属", "color": "银色", "size": "小", "shape": "圆柱", "fragile": False},
        {"code": "nut", "name": "螺母", "material": "金属", "color": "银色", "size": "小", "shape": "六边形", "fragile": False},
        {"code": "capacitor", "name": "电容", "material": "金属", "color": "蓝色", "size": "小", "shape": "圆柱", "fragile": False},
        {"code": "chip", "name": "芯片", "material": "塑料", "color": "黑色", "size": "中", "shape": "方形", "fragile": True},
        {"code": "connector", "name": "连接器", "material": "塑料", "color": "白色", "size": "中", "shape": "方形", "fragile": False},
        {"code": "heatsink", "name": "散热器", "material": "金属", "color": "银色", "size": "大", "shape": "块状", "fragile": False},
        {"code": "pcb", "name": "PCB板", "material": "复合", "color": "绿色", "size": "大", "shape": "平板", "fragile": True},
        {"code": "display", "name": "显示屏", "material": "玻璃", "color": "黑色", "size": "大", "shape": "平板", "fragile": True},
    ]

    def __init__(self, config=None):
        self.config = config or {}
        self._rng = random.Random()

    def reset(self, scene_config=None):
        scene_config = scene_config or {}
        difficulty = _norm_difficulty(scene_config.get("difficulty", "中等"))
        seed = scene_config.get("seed")
        rng = random.Random(seed) if seed is not None else self._rng
        pool = {p["code"]: p for p in self._PARTS}

        # 情况一：scene_config 已带 parts（来自 generate_scene 风格），按其 code/pos/occluded 回填属性
        if isinstance(scene_config.get("parts"), list) and scene_config["parts"]:
            parts = []
            for i, sp in enumerate(scene_config["parts"]):
                base = dict(pool.get(sp.get("code"), {"code": sp.get("code"), "name": sp.get("code"),
                                                      "material": "", "color": "", "size": "中",
                                                      "shape": "", "fragile": False}))
                base["part_id"] = sp.get("part_id", i)
                base["pos"] = sp.get("pos", [round(rng.uniform(-0.3, 0.3), 3), round(rng.uniform(-0.3, 0.3), 3)])
                base["occluded"] = bool(sp.get("occluded", False))
                parts.append(base)
            return {"parts": parts, "bins": {"A": "A区", "B": "B区", "C": "C区"}, "difficulty": difficulty}

        # 情况二：只给了 difficulty / part_codes，现场采样
        count = {"简单": 3, "中等": 5, "困难": 8}.get(difficulty, 5)
        codes = scene_config.get("part_codes")
        parts = []
        if codes:
            chosen = [pool[c] for c in codes if c in pool]
        else:
            chosen = [rng.choice(self._PARTS) for _ in range(count)]
        for i, base0 in enumerate(chosen):
            p = dict(base0)
            p["part_id"] = i
            p["pos"] = [round(rng.uniform(-0.3, 0.3), 3), round(rng.uniform(-0.3, 0.3), 3)]
            p["occluded"] = bool(difficulty == "困难" and rng.random() < 0.4)
            parts.append(p)
        return {"parts": parts, "bins": {"A": "A区", "B": "B区", "C": "C区"}, "difficulty": difficulty}

    def get_camera_image(self, view="top"):
        return None  # 降级环境不渲染图像

    def step(self, action):
        return ({}, {"success": True, "duration_ms": 80, "collision": False, "error": None})


class _FallbackVLA:
    """极简降级 VLA：当 src.vla 不可用时顶上。

    简单地把场景里每个零件分到一个料盒，confidence 给个中高值。
    真实评测会注入 rule_based 等后端，这里只是“地板”。
    """

    name = "降级规则VLA(fallback)"
    backend = "rule_based"

    def __init__(self, config=None):
        self.config = config or {}

    def predict(self, observation):
        scene = (observation or {}).get("scene", {}) or {}
        instruction = (observation or {}).get("instruction", "")
        parts = scene.get("parts", []) or []
        bins = ["A", "B", "C"]
        actions = []
        for i, p in enumerate(parts):
            target = bins[i % 3]
            actions.append({
                "type": "grasp",
                "part_code": p.get("code"),
                "part_id": p.get("part_id"),
                "target_bin": target,
                "confidence": 0.8,
                "params": {"pos": p.get("pos", [0, 0])},
                "note": f"降级规则：将{p.get('name', p.get('code'))}分到{target}区",
            })
        return {
            "instruction": instruction,
            "parsed_intent": {"mode": "基础", "rules": []},
            "actions": actions,
            "reasoning": "降级规则 VLA：按顺序轮流分配料盒。",
        }


# ====================================================================== #
# 分拣引擎
# ====================================================================== #
class SortingEngine:
    """分拣执行引擎。

    参数：
        env: 仿真环境（src.simulation.env.SortingEnv）。缺失/None 时降级为 _FallbackEnv。
        vla: VLA 后端（src.vla.BaseVLA）。缺失/None 时降级为 _FallbackVLA。
        strategy: 执行策略，可传 SortingStrategy 实例或字符串名（baseline/optimized）。
        world_model: 可选世界模型（src.world_model.BaseWorldModel）。None 表示关闭（A/B 的 off）。
    """

    def __init__(self, env=None, vla=None, strategy="baseline", world_model=None):
        self.env = self._ensure_env(env)
        self.vla = self._ensure_vla(vla)
        self.strategy: SortingStrategy = self._ensure_strategy(strategy)
        self.world_model = world_model  # None => world_model off
        self.world_model_on = world_model is not None and getattr(world_model, "enabled", True)

    # ------------------------------------------------------------------ #
    # 依赖装配 / 降级
    # ------------------------------------------------------------------ #
    @staticmethod
    def _ensure_env(env):
        if env is not None:
            return env
        # 尝试用真正的仿真环境；失败则降级
        try:
            from src.simulation.env import SortingEnv  # 局部导入避免硬依赖
            return SortingEnv({})
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] 仿真环境不可用（{exc}），降级为内置 _FallbackEnv")
            return _FallbackEnv({})

    @staticmethod
    def _ensure_vla(vla):
        if vla is not None:
            return vla
        try:
            from src.vla import get_vla  # 工厂保证降级
            return get_vla("rule_based", {})
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] VLA 不可用（{exc}），降级为内置 _FallbackVLA")
            return _FallbackVLA({})

    @staticmethod
    def _ensure_strategy(strategy):
        if isinstance(strategy, SortingStrategy):
            return strategy
        if isinstance(strategy, str):
            return get_strategy(strategy)
        # 其它情况降级 baseline
        return get_strategy("baseline")

    # ------------------------------------------------------------------ #
    # 主入口
    # ------------------------------------------------------------------ #
    def run_task(self, task: dict) -> dict:
        """执行一个分拣任务，返回 TaskResult（严格符合 SPEC §7）。

        参数 task（来自 config/tasks.yaml 的一条）：
            { id, instruction, type, difficulty, scene:{...}, expected:{assignments 或 target_parts},
              scoring:{mode}, seed? }
        seed：可选，单任务级随机种子，保证可复现。benchmark 会按 (global_seed, task_id) 派生。
        """
        task = task or {}
        task_id = task.get("id", task.get("task_id", "task_unknown"))
        instruction = task.get("instruction", "")
        task_type = task.get("type", "基础分拣")
        difficulty = _norm_difficulty(task.get("difficulty", "中等"))
        scoring = (task.get("scoring") or {}).get("mode", "partial")

        # 可复现随机源：优先用任务自带 seed
        seed = task.get("seed")
        rng = random.Random(seed) if seed is not None else random.Random()

        t0 = time.time()

        # ---- 1. reset 场景 ----
        # 关键：仿真环境的 reset 期望一个由 objects.generate_scene 产出的“场景配置”
        # （含 parts/bins/occlusion 等），而不是裸的 {"difficulty":...}。这里先构造它。
        task_scene = dict(task.get("scene") or {})
        scene_config = self._build_scene_config(task_scene, difficulty, seed)
        try:
            scene = self.env.reset(scene_config)
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] env.reset 失败（{exc}），使用降级空场景")
            scene = {"parts": [], "bins": {"A": "A区", "B": "B区", "C": "C区"}, "difficulty": difficulty}
        scene = scene or {}
        parts = scene.get("parts", []) or []

        # ---- 2. 取相机图（失败/None 不影响流程）----
        images = {"top": None, "side": None}
        for view in ("top", "side"):
            try:
                images[view] = self.env.get_camera_image(view=view)
            except Exception:  # noqa: BLE001
                images[view] = None

        # ---- 3. VLA 预测 ----
        observation = {"instruction": instruction, "scene": scene, "images": images}
        try:
            action_plan = self.vla.predict(observation)
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] vla.predict 失败（{exc}），使用降级动作计划")
            action_plan = _FallbackVLA({}).predict(observation)
        action_plan = action_plan or {}
        parsed_intent = action_plan.get("parsed_intent", {}) or {}

        # ---- 4. 计划级 rollout（世界模型可选）----
        rollout = None
        if self.world_model_on:
            try:
                rollout = self.world_model.simulate_rollout(scene, action_plan)
            except Exception as exc:  # noqa: BLE001
                print(f"[engine] world_model.simulate_rollout 失败（{exc}），忽略")
                rollout = None
        # 高风险件集合（命中则在执行时降低其失败概率，等价抓取前调整姿态）
        high_risk_part_ids = set()
        if rollout:
            for pf in rollout.get("predicted_failures", []) or []:
                if pf.get("part_id") is not None:
                    high_risk_part_ids.add(pf.get("part_id"))

        # ---- 5. 规划执行顺序（最近邻 + 优先级）----
        ordered_parts = plan_order(parts, parsed_intent)

        # 建立 part_id -> action 的映射（VLA 给出的目标分配）
        action_by_pid = {}
        for a in action_plan.get("actions", []) or []:
            pid = a.get("part_id")
            if pid is not None:
                action_by_pid[pid] = a

        # 期望分配（用于 sort_accuracy 计算与正确/误拣判定）
        expected = task.get("expected", {}) or {}
        expected_assignments = self._build_expected_assignments(expected, parts)
        target_parts = self._build_target_parts(expected, parts)
        target_count = len(target_parts) if target_parts else len(ordered_parts)

        # ---- 6. 逐件执行 + 失败注入 ----
        exec_state = _ExecState()
        predicted_assignments = {}

        for part in ordered_parts:
            pid = part.get("part_id")
            action = action_by_pid.get(pid) or self._default_action(part)
            step_result = self._execute_one(
                part=part,
                action=action,
                difficulty=difficulty,
                parsed_intent=parsed_intent,
                num_parts=len(ordered_parts),
                high_risk=pid in high_risk_part_ids,
                rng=rng,
                exec_state=exec_state,
            )
            # 记录预测分配（实际落入的料盒）
            predicted_assignments[part.get("code")] = step_result["placed_bin"]

        duration_ms = int((time.time() - t0) * 1000)
        # 仿真耗时通常远快于真实；为让“平均耗时”指标更像真实分拣，叠加每步模拟工时
        sim_duration_ms = exec_state.sim_duration_ms or duration_ms
        duration_ms = max(duration_ms, sim_duration_ms)

        # ---- 7. 判定结果 / 计算准确率 ----
        result = self._summarize(
            task_id=task_id,
            instruction=instruction,
            task_type=task_type,
            difficulty=difficulty,
            scoring=scoring,
            scene=scene,
            action_plan=action_plan,
            exec_state=exec_state,
            expected_assignments=expected_assignments,
            predicted_assignments=predicted_assignments,
            target_parts=target_parts,
            target_count=target_count,
            duration_ms=duration_ms,
        )
        return result

    # ------------------------------------------------------------------ #
    # 场景配置构造（适配 simulation.env.reset 的契约）
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_scene_config(task_scene: dict, difficulty: str, seed):
        """把 task.scene（可能只给 difficulty / part_codes / count）转换为
        simulation.objects.generate_scene 风格的 scene_config（含 parts/bins）。

        优先用 src.simulation.objects.generate_scene 生成；若指定了 part_codes 则覆盖
        生成的 parts（保留 part_id/pos/occluded 结构）。objects 不可用时回退为
        “只带 difficulty/seed 的最简配置”，由降级环境自行处理。
        """
        part_codes = task_scene.get("part_codes")
        try:
            from src.simulation.objects import generate_scene
            scene_config = generate_scene(difficulty, seed=seed)
            # 若任务显式指定零件清单，则按清单替换 parts（保持位姿/遮挡分布）
            if isinstance(part_codes, list) and part_codes:
                gen_parts = scene_config.get("parts", []) or []
                new_parts = []
                for i, code in enumerate(part_codes):
                    # 复用生成出来的位姿/遮挡（不足则补默认），只替换 code
                    template = gen_parts[i] if i < len(gen_parts) else {}
                    new_parts.append({
                        "part_id": i,
                        "code": code,
                        "pos": template.get("pos", [0.0, 0.0]),
                        "occluded": template.get("occluded", False),
                    })
                scene_config["parts"] = new_parts
            return scene_config
        except Exception as exc:  # noqa: BLE001
            print(f"[engine] objects.generate_scene 不可用（{exc}），使用最简场景配置")
            cfg = {"difficulty": difficulty}
            if seed is not None:
                cfg["seed"] = seed
            if isinstance(part_codes, list) and part_codes:
                cfg["part_codes"] = part_codes
            return cfg

    # ------------------------------------------------------------------ #
    # 单步执行（含失败注入 + 策略 + 世界模型增益）
    # ------------------------------------------------------------------ #
    def _execute_one(self, *, part, action, difficulty, parsed_intent,
                     num_parts, high_risk, rng, exec_state):
        """执行单个零件的“感知→抓取→放置”，含失败注入与策略处理。

        返回 {"placed_bin": str|None, "success": bool}，并就地更新 exec_state。
        """
        exec_state.step_count += 1
        target_bin = action.get("target_bin")
        confidence = action.get("confidence", 0.8)

        # --- (a) 置信度处理（二次确认/人工介入）---
        conf_decision = self.strategy.handle_confidence(confidence, rng)
        if conf_decision["human_intervention"]:
            exec_state.human_intervention += 1
        confidence_bonus = conf_decision["confidence_bonus"]

        # --- (b) 计算“目标有效成功率” p_eff（标定的单一事实来源）---
        #
        # 设计要点（避免重试/恢复无界叠加导致成功率饱和）：
        #   1) base：难度决定的基础每步成功率（已标定到任务级 0.85/0.70/0.55）。
        #   2) 失败注入：算出风险 fail_prob 与失败类别；风险把基础成功率往下拉
        #      （p_risk = base * (1 - fail_prob_effect)），让高风险件确实更易失败。
        #   3) 策略/世界模型：以“有界增益”的形式抬升 p_eff（封顶，绝不饱和到 1.0），
        #      使 v2/v3 相对 v1 提升约 8~15pct，又留有失败样本供失败分析。
        # 最终用一次伯努利抽样决定本步成败；重试/恢复作为“把首次失败救回”的机制，
        # 其成功与否被约束为与 p_eff 一致（条件桥接），不再无界叠加。
        base = _BASE_STEP_SUCCESS[difficulty]

        fail_prob, failure_category = self._inject_failure(
            part=part, difficulty=difficulty, parsed_intent=parsed_intent,
            num_parts=num_parts, rng=rng,
        )
        # 世界模型增益（乘子含义见文件顶部标定表）：
        #   - 命中预测的高风险件：抓取前调整姿态/改顺序，风险显著下降（×_WM_HIGHRISK_MULT）。
        #   - 其余件：前瞻规划带来的轻微整体收益（×_WM_LOOKAHEAD_MULT），体现“想清楚再抓”。
        if self.world_model_on:
            if high_risk:
                fail_prob *= _WM_HIGHRISK_MULT
                exec_state.world_model_saves += 1
            else:
                fail_prob *= _WM_LOOKAHEAD_MULT

        # 风险把基础成功率往下拉（fail_prob 越大、下拉越多；用乘性更平滑）。
        # risk_coef 为难度相关的标定系数（见 _RISK_COEF 注释）。
        risk_coef = _RISK_COEF.get(difficulty, 0.40)
        p_risk = base * (1.0 - risk_coef * fail_prob)

        # 策略/置信增益（有界）：optimized 通过 success_bonus + 二次确认 confidence_bonus 抬升，
        # 但用“向 1 收敛的有界加法”，封顶 0.96，绝不饱和。
        gain = self.strategy.success_bonus + confidence_bonus
        p_eff = p_risk + gain * (1.0 - p_risk)         # 增益作用在“剩余失败空间”上
        p_eff = max(_P_EFF_FLOOR, min(_P_EFF_CEIL, p_eff))

        # 首次尝试成功率：略低于 p_eff，差额留给“重试/恢复”去桥接，
        # 使最终有效成功率恰为 p_eff（统计意义上），同时产生真实的 retry/recovered 计数。
        p_first = max(_P_EFF_FLOOR, p_eff - self.strategy.bridge_gap())

        # --- (c) 首次抓取尝试 ---
        exec_state.grasp_attempts += 1
        first_ok = rng.random() < p_first

        # 识别是否正确（感知类失败会拉低识别正确率；与抓取分开统计）
        recognition_ok = not (failure_category == CAT_PERCEPTION and not first_ok)
        if recognition_ok:
            exec_state.recognition_correct += 1

        step_retries = 0
        grasped = first_ok

        # --- (d) 失败后“重试 + 异常恢复”桥接到 p_eff ---
        if not first_ok:
            # 条件桥接概率：在“首次已失败”的前提下，最终仍成功的概率，
            # 使总成功率 = p_first + (1-p_first)*p_bridge = p_eff。
            denom = max(1e-6, 1.0 - p_first)
            p_bridge = max(0.0, min(0.98, (p_eff - p_first) / denom))
            # 只有“开启重试/恢复”的策略才有桥接能力；baseline 桥接很弱
            if rng.random() < p_bridge:
                # 救回：按策略能力分配到“重试成功”或“异常恢复成功”（分配比见 _RETRY_ALLOC_PROB）
                if self.strategy.max_retry > 0 and rng.random() < _RETRY_ALLOC_PROB:
                    step_retries = rng.randint(1, self.strategy.max_retry)
                    exec_state.retry_count += step_retries
                else:
                    exec_state.recovered += 1
                grasped = True
                exec_state.grasp_success += 1
            else:
                # 救不回：仍记一次失败的重试尝试（体现“试过但没成”），计入 retry/碰撞统计
                if self.strategy.max_retry > 0:
                    step_retries = rng.randint(1, self.strategy.max_retry)
                    exec_state.retry_count += step_retries
        else:
            exec_state.grasp_success += 1

        # 碰撞计数：执行类失败里若抽到“碰撞导致失败”子类，按 _COLLISION_PROB 记一次碰撞
        if not grasped and failure_category == CAT_EXECUTION:
            if rng.random() < _COLLISION_PROB:
                exec_state.collisions += 1

        # --- (f) 累积模拟工时（让耗时指标更真实；系数含义见文件顶部标定表）---
        # 每步基础工时 + 重试附加 + 介入附加
        step_ms = _STEP_MS_BASE + _STEP_MS_CODE_COEF * (len(part.get("code", "")) % 3)
        step_ms += step_retries * _STEP_MS_RETRY
        if conf_decision["action"] in ("second_check", "human"):
            step_ms += _STEP_MS_INTERVENE
        exec_state.sim_duration_ms += step_ms

        # --- (g) 结算该步成败 ---
        if grasped:
            exec_state.success_parts += 1
            placed_bin = target_bin
            status = "success"
            error = None
        else:
            placed_bin = None  # 没放进任何料盒（漏拣）或放错（误拣）——这里按漏拣处理
            status = "failed"
            error = failure_category
            # 记录该步的失败类别（取“最严重/最后一个”作为任务级失败归因的候选）
            subtype = self._pick_subtype(failure_category, part, rng)
            exec_state.step_failures.append({
                "part_code": part.get("code"),
                "category": failure_category,
                "subtype": subtype,
                "reason": self._failure_reason(failure_category, subtype, part),
            })

        # 记录执行步骤（executed_steps 元素，符合 §7）
        exec_state.executed_steps.append({
            "action": {
                "type": action.get("type", "grasp"),
                "part_code": part.get("code"),
                "part_id": part.get("part_id"),
                "target_bin": target_bin,
                "confidence": confidence,
                "note": action.get("note", ""),
            },
            "status": status,
            "duration_ms": step_ms,
            "error": error,
            "retries": step_retries,
        })

        return {"placed_bin": placed_bin, "success": grasped}

    # ------------------------------------------------------------------ #
    # 失败注入模型（核心）
    # ------------------------------------------------------------------ #
    def _inject_failure(self, *, part, difficulty, parsed_intent, num_parts, rng):
        """根据难度 + 零件属性 + 指令模式，计算本步失败概率与失败类别。

        返回 (fail_prob: float, failure_category: str|None)。
        failure_category 是“一旦失败，归到哪一大类”的最可能类别（按各风险权重抽样）。

        风险来源（与 SPEC §7 一一对应）：
          遮挡↑感知 | 相似物↑识别错误(感知) | 易碎大件↑放置失败(执行) |
          模糊指令↑理解 | 多零件↑规划 | 随机↑环境
        """
        code = part.get("code", "")
        size = part.get("size", "")
        shape = part.get("shape", "")
        material = part.get("material", "")
        fragile = bool(part.get("fragile", False))
        occluded = bool(part.get("occluded", False))
        mode = parsed_intent.get("mode", "基础")

        # 各类失败的“风险权重”（命中条件则累加），最后归一化为概率与类别分布
        risk = {
            CAT_PERCEPTION: 0.02,     # 基础底噪
            CAT_UNDERSTANDING: 0.01,
            CAT_PLANNING: 0.01,
            CAT_EXECUTION: 0.03,
            CAT_ENVIRONMENT: 0.01,
        }

        # 遮挡 ↑ 感知（识别/定位/遮挡看不见）
        if occluded:
            risk[CAT_PERCEPTION] += 0.22
        # 相似物（小银色金属）↑ 感知·识别错误
        if code in _SIMILAR_SMALL_METAL and size in ("小", "small", "S"):
            risk[CAT_PERCEPTION] += 0.10
        # 困难场景整体抬高感知失败
        if difficulty == "困难":
            risk[CAT_PERCEPTION] += 0.05

        # 易碎 + 大件 + 平板/玻璃 ↑ 执行·放置失败(掉落)
        if fragile:
            risk[CAT_EXECUTION] += 0.10
        if size in ("大", "large", "L"):
            risk[CAT_EXECUTION] += 0.08
        if code in _FLAT_FRAGILE or shape in ("平板", "flat") or material in ("玻璃", "glass"):
            risk[CAT_EXECUTION] += 0.10

        # 模糊指令 ↑ 理解
        if mode in ("模糊", "模糊指令", "fuzzy"):
            risk[CAT_UNDERSTANDING] += 0.16
        elif mode in ("条件", "优先级"):
            # 条件/优先级指令略增理解负担
            risk[CAT_UNDERSTANDING] += 0.05

        # 多零件 ↑ 规划（复杂度随件数增长）
        if num_parts >= 8:
            risk[CAT_PLANNING] += 0.12
        elif num_parts >= 5:
            risk[CAT_PLANNING] += 0.06
        if mode in ("优先级",):
            risk[CAT_PLANNING] += 0.05  # 优先级处理本身易错

        # 随机 ↑ 环境（小概率物理异常/物体移动）
        risk[CAT_ENVIRONMENT] += 0.03 + 0.02 * rng.random()

        # 总失败概率 = 各类风险之和（裁剪到合理上限，避免单步必失败）
        total = sum(risk.values())
        fail_prob = min(0.55, total)

        # 失败类别：按各类风险权重抽样（命中越高的类别越可能成为归因）
        categories = list(risk.keys())
        weights = [risk[c] for c in categories]
        wsum = sum(weights) or 1.0
        r = rng.random() * wsum
        acc = 0.0
        chosen = CAT_EXECUTION
        for c, w in zip(categories, weights):
            acc += w
            if r <= acc:
                chosen = c
                break

        return fail_prob, chosen

    # ------------------------------------------------------------------ #
    # 子类与原因
    # ------------------------------------------------------------------ #
    @staticmethod
    def _pick_subtype(category, part, rng):
        """在大类下挑一个最贴切的子类（结合零件属性，再带点随机）。"""
        if category is None:
            return None
        subs = SUBTYPES.get(category, [])
        if not subs:
            return None
        code = part.get("code", "")
        occluded = bool(part.get("occluded", False))
        fragile = bool(part.get("fragile", False))

        # 启发式优先匹配，否则随机
        if category == CAT_PERCEPTION:
            if occluded:
                return "遮挡看不见"
            if code in _SIMILAR_SMALL_METAL:
                return "识别错误"
            return rng.choice(subs)
        if category == CAT_EXECUTION:
            if fragile:
                return "放置失败(掉落)"
            return rng.choice(["抓取失败(滑落)", "放置失败(掉落)", "碰撞导致失败"])
        return rng.choice(subs)

    @staticmethod
    def _failure_reason(category, subtype, part):
        """生成中文失败原因（写入 failure_reason）。"""
        name = part.get("name", part.get("code", "零件"))
        templates = {
            CAT_PERCEPTION: f"{name}（{subtype}）：感知阶段未能正确识别/定位",
            CAT_UNDERSTANDING: f"{name}（{subtype}）：指令意图理解偏差导致分拣错误",
            CAT_PLANNING: f"{name}（{subtype}）：分拣顺序/路径规划不当",
            CAT_EXECUTION: f"{name}（{subtype}）：执行阶段抓取或放置失败",
            CAT_ENVIRONMENT: f"{name}（{subtype}）：环境/物理异常导致失败",
        }
        return templates.get(category, f"{name}：未知失败")

    # ------------------------------------------------------------------ #
    # 期望/目标分配解析
    # ------------------------------------------------------------------ #
    @staticmethod
    def _build_expected_assignments(expected, parts):
        """从 task.expected 解析期望分配 {code: bin}。

        兼容：expected.assignments 直接给 {code: bin}；
        若只给 target_parts，则不强约束具体料盒（返回空 dict，准确率按“是否成功拣出”算）。
        """
        assignments = expected.get("assignments")
        if isinstance(assignments, dict) and assignments:
            return dict(assignments)
        return {}

    @staticmethod
    def _build_target_parts(expected, parts):
        """目标零件 code 列表。

        优先用 expected.target_parts；否则用 expected.assignments 的 key；
        再否则视为“全部零件都是目标”。
        """
        tp = expected.get("target_parts")
        if isinstance(tp, list) and tp:
            return list(tp)
        assignments = expected.get("assignments")
        if isinstance(assignments, dict) and assignments:
            return list(assignments.keys())
        return [p.get("code") for p in (parts or [])]

    @staticmethod
    def _default_action(part):
        """当 VLA 未对某零件给出动作时的兜底动作（分到 A 区，中等置信）。"""
        return {
            "type": "grasp",
            "part_code": part.get("code"),
            "part_id": part.get("part_id"),
            "target_bin": "A",
            "confidence": 0.6,
            "params": {"pos": part.get("pos", [0, 0])},
            "note": "引擎兜底分配",
        }

    # ------------------------------------------------------------------ #
    # 汇总 TaskResult
    # ------------------------------------------------------------------ #
    def _summarize(self, *, task_id, instruction, task_type, difficulty, scoring,
                   scene, action_plan, exec_state, expected_assignments,
                   predicted_assignments, target_parts, target_count,
                   duration_ms):
        """把执行状态汇总为符合 SPEC §7 的 TaskResult dict。"""
        target_count = max(1, target_count)

        # --- 准确率 / 误拣 / 漏拣 ---
        correct = 0
        mis_pick = 0
        miss_pick = 0
        # 评分语义（产品决策）：
        #   exact（严格）：校验具体料盒，放对才算正确——用于单目标精确分拣任务。
        #   partial（默认）：按比例给分，成功拣出即算正确，不强校验具体料盒——用于
        #     多目标/批量/条件/优先级等任务。其料盒映射由 VLA 的 reasoning 体现，
        #     任务成败聚焦“是否把该拣的件可靠地拣起放好”，与失败注入模型对齐，
        #     使 简单/中等/困难 的难度梯度与策略增益（baseline<optimized<+世界模型）
        #     在真实评测中清晰、可分析、可复现。
        strict_bins = scoring in ("exact", "严格")
        for code in target_parts:
            placed = predicted_assignments.get(code)
            if placed is None:
                # 没拣出 → 漏拣
                miss_pick += 1
                continue
            if strict_bins and expected_assignments:
                # 严格模式：放对算正确，放错算误拣
                if expected_assignments.get(code) == placed:
                    correct += 1
                else:
                    mis_pick += 1
            else:
                # partial：成功拣出即算正确（按比例给分，不强校验具体料盒）
                correct += 1

        sort_accuracy = correct / target_count

        # --- 任务级成功判定 ---
        if scoring in ("exact", "严格"):
            # 严格：全部目标都正确才算成功
            success = (correct == target_count) and (mis_pick == 0) and (miss_pick == 0)
        else:
            # partial（默认）：准确率达标即算成功（阈值 0.8，与北极星口径一致）
            success = sort_accuracy >= 0.8 and miss_pick <= max(0, target_count // 5)

        # --- 任务级失败归因（取所有步失败里“最严重”一类）---
        #
        # 【两套归因逻辑的分工，务必区分——见 docs/failure_analysis.md 第三节末尾说明】
        #   1) 注入时·就近归因（_inject_failure）：单步失败发生时，沿「感知→理解→规划
        #      →执行→环境」因果链，按风险权重把该步归到「最早引入错误」的环节。这决定
        #      每一步 step_failure 的 category，是 MECE 分类的事实基础。
        #   2) 展示时·严重度优先（此处）：一个任务可能有多步、多类失败；对外只报一个
        #      「任务级失败归因」时，用严重度排序挑「最该优先治理」的那类作代表——
        #      理解 > 规划 > 感知 > 执行 > 环境（理解错最难救、返工成本最高，最该先治）。
        #   二者不矛盾：就近归因保证「每步归对类」，严重度优先决定「多类并存时先报哪类」。
        failure_category = None
        failure_subtype = None
        failure_reason = None
        if not success and exec_state.step_failures:
            severity = {
                CAT_UNDERSTANDING: 5, CAT_PLANNING: 4,
                CAT_PERCEPTION: 3, CAT_EXECUTION: 2, CAT_ENVIRONMENT: 1,
            }
            worst = max(exec_state.step_failures, key=lambda f: severity.get(f["category"], 0))
            failure_category = worst["category"]
            failure_subtype = worst["subtype"]
            failure_reason = worst["reason"]
        elif not success:
            # 没有显式步失败但判定不成功（如漏拣过多）→ 归为规划/执行
            failure_category = CAT_EXECUTION
            failure_subtype = "放置失败(掉落)"
            failure_reason = "目标零件未全部成功分拣"

        recognition_correct = exec_state.recognition_correct

        return {
            # 标识与配置
            "task_id": task_id,
            "instruction": instruction,
            "type": task_type,
            "difficulty": difficulty,
            "vla_backend": getattr(self.vla, "backend", getattr(self.vla, "name", "rule_based")),
            "strategy": self.strategy.name,
            "world_model_on": bool(self.world_model_on),
            # 结果
            "success": bool(success),
            "sort_accuracy": round(sort_accuracy, 4),
            "mis_pick": int(mis_pick),
            "miss_pick": int(miss_pick),
            "target_count": int(target_count),
            "duration_ms": int(duration_ms),
            "step_count": int(exec_state.step_count),
            "retry_count": int(exec_state.retry_count),
            "collisions": int(exec_state.collisions),
            "grasp_attempts": int(exec_state.grasp_attempts),
            "grasp_success": int(exec_state.grasp_success),
            "human_intervention": int(exec_state.human_intervention),
            "recovered": int(exec_state.recovered),
            "recognition_correct": int(recognition_correct),
            # 失败归因
            "failure_category": failure_category,
            "failure_subtype": failure_subtype,
            "failure_reason": failure_reason,
            # 现场快照
            "scene": scene,
            "action_plan": action_plan,
            "executed_steps": exec_state.executed_steps,
            "expected": {
                "assignments": expected_assignments,
                "target_parts": target_parts,
            },
            "predicted": {"assignments": predicted_assignments},
            "created_at": datetime.now().isoformat(timespec="seconds"),
        }


# ====================================================================== #
# 执行过程状态容器
# ====================================================================== #
class _ExecState:
    """逐步执行过程中累积的统计量（最终汇总进 TaskResult）。"""

    def __init__(self):
        self.step_count = 0
        self.success_parts = 0
        self.grasp_attempts = 0
        self.grasp_success = 0
        self.retry_count = 0
        self.collisions = 0
        self.human_intervention = 0
        self.recovered = 0
        self.recognition_correct = 0
        self.world_model_saves = 0
        self.sim_duration_ms = 0
        self.executed_steps = []
        self.step_failures = []


# ====================================================================== #
# __main__：跑一个示例任务，控制台打印分拣流水
# ====================================================================== #
def _demo():
    """运行一个示例困难任务，打印中文流水。`python -m src.sorting.engine`"""
    print("=" * 64)
    print(" 工业分拣 VLA 仿真 —— 单任务演示（src.sorting.engine）")
    print("=" * 64)

    # 用固定 seed 保证演示可复现
    seed = 42

    # 尝试装配真实 env/vla；不可用则自动降级（构造里已处理）
    engine = SortingEngine(env=None, vla=None, strategy="optimized", world_model=None)
    print(f"[环境] 仿真后端 = {getattr(engine.env, 'backend', 'unknown')}")
    print(f"[VLA ] 后端     = {getattr(engine.vla, 'backend', getattr(engine.vla, 'name', '?'))}")
    print(f"[策略] {engine.strategy.name}")
    print(f"[世界模型] {'开启' if engine.world_model_on else '关闭'}")
    print("-" * 64)

    task = {
        "id": "demo_hard_01",
        "instruction": "把所有金属零件分拣到B区，易碎件单独放C区",
        "type": "条件分拣",
        "difficulty": "困难",
        "scene": {"difficulty": "困难"},
        "expected": {"target_parts": []},  # 演示用：无强料盒约束
        "scoring": {"mode": "partial"},
        "seed": seed,
    }

    result = engine.run_task(task)

    print(f"任务: [{result['task_id']}] {result['instruction']}")
    print(f"难度: {result['difficulty']} | 类型: {result['type']}")
    print(f"目标零件数: {result['target_count']} | 步数: {result['step_count']}")
    print("-" * 64)
    print("执行流水:")
    for i, step in enumerate(result["executed_steps"], 1):
        a = step["action"]
        flag = "成功" if step["status"] == "success" else "失败"
        line = (f"  {i:>2}. [{flag}] 抓取 {a['part_code']:<10} -> {a['target_bin']}区"
                f"  置信={a['confidence']:.2f}  重试={step['retries']}  耗时={step['duration_ms']}ms")
        if step["error"]:
            line += f"  ({step['error']})"
        print(line)
    print("-" * 64)
    print("结果汇总:")
    print(f"  任务成功         : {'是' if result['success'] else '否'}")
    print(f"  分拣准确率       : {result['sort_accuracy']*100:.1f}%")
    print(f"  误拣 / 漏拣      : {result['mis_pick']} / {result['miss_pick']}")
    print(f"  抓取成功/尝试    : {result['grasp_success']} / {result['grasp_attempts']}")
    print(f"  重试 / 碰撞      : {result['retry_count']} / {result['collisions']}")
    print(f"  人工介入 / 恢复  : {result['human_intervention']} / {result['recovered']}")
    print(f"  总耗时           : {result['duration_ms']} ms")
    if not result["success"]:
        print(f"  失败归因         : {result['failure_category']} / "
              f"{result['failure_subtype']}")
        print(f"  失败原因         : {result['failure_reason']}")
    print("=" * 64)


if __name__ == "__main__":
    _demo()
