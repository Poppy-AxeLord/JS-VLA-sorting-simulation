# -*- coding: utf-8 -*-
"""
机械臂模块（src/simulation/robot.py）
=====================================

提供一个**简化的 UR5e + 二指夹爪**抽象（``SimpleArm``）。它不依赖 MuJoCo：
在 mock 后端下用"简化 IK 近似 + 直接状态更新"来模拟动作；在 mujoco 后端下
env.py 可把这里算出的目标关节角喂给物理引擎（POC 阶段以运动学近似为主）。

设计目标（POC 取舍）：
- 提供和真实机械臂一致的"动作语义"接口：move_to / grasp / place / home，
  让上层分拣引擎的代码无需关心后端差异（mujoco 还是 mock）。
- mock 下不做真实动力学：抓取/放置的"成败"由分拣引擎的**失败注入模型**统一
  决定（见 src/sorting）。本文件只负责"几何上是否可达、是否碰撞、夹爪开合
  状态、持有哪个零件"这类运动学/状态信息，并估算动作耗时（供效率指标）。
- 含一个**基础碰撞检测开关**（collision_check）：开启时用简单的平面距离判据
  判断末端轨迹是否撞到料盒/其它零件（粗略），关闭时永不报碰撞。

为什么把"成败"交给上层而不是这里？因为评测体系需要按零件属性/难度/指令
注入 5 类失败并保证可复现；若机械臂自己随机判定成败，会与上层的种子和概率
模型割裂，难以复现与分析。这里只给"几何事实"，决策权上交，职责清晰。
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Tuple

import numpy as np


# UR5e 简化连杆长度（米），仅用于 mock 下的 2 连杆平面 IK 近似与可达性判断。
# 真实 UR5e 是 6 自由度空间臂，这里降维到平面 2R + 升降，足够 POC 几何用途。
# 连杆长度需保证**整个抓取区 + 三个料盒都落在可达范围内**：基座在 y=-0.50，
# 抓取区最远点约 (0.30, 0.22) 距基座 ~0.78m，故单连杆取 0.42m，使
# reach_max = 0.42+0.42-0.02 = 0.82m 覆盖全工作台；reach_min ≈ 0.02m 保证
# 近处料盒(距基座约 0.32m)也可达。这样"几何不可达"几乎不会误伤正常分拣点，
# 真正的成败交由上层失败注入模型决定（见模块说明的职责边界）。
_L1 = 0.42
_L2 = 0.42
# 机械臂基座在工作台坐标系中的位置（逻辑坐标，单位米）。
_BASE = np.array([0.0, -0.50])
# 可达半径（米）：末端到基座的距离需落在 [reach_min, reach_max] 内。
_REACH_MAX = _L1 + _L2 - 0.02
_REACH_MIN = abs(_L1 - _L2) + 0.02

# 二指夹爪开度（米）：home 时张开，抓取时按零件大小闭合到合适开度。
_GRIPPER_OPEN = 0.085
_GRIPPER_SIZE_GAP = {"小": 0.020, "中": 0.045, "大": 0.075}

# 动作耗时基准（毫秒）—— 供效率指标（平均单步耗时/任务耗时）使用。
# 这些是"标称"耗时，分拣引擎在重试/异常时会在此基础上叠加。
_MOVE_BASE_MS = 600.0       # 移动基准
_MOVE_PER_METER_MS = 1200.0  # 每米额外耗时
_GRASP_MS = 450.0
_PLACE_MS = 500.0
_HOME_MS = 700.0


class SimpleArm:
    """简化机械臂（UR5e + 二指夹爪）。

    参数
    ----
    config : 机械臂/物理配置 dict（可来自 env_config.yaml），可选键：
        - collision_check (bool) : 是否启用基础碰撞检测，默认 True。
        - speed_scale (float)    : 速度缩放，>1 更快（耗时更短），默认 1.0。

    状态属性
    --------
    - ``ee_pos``        : 末端执行器当前 2D 位置（np.ndarray）。
    - ``gripper_open``  : 夹爪当前开度（米），>0 视为张开。
    - ``holding``       : 当前持有的零件 dict（None 表示空手）。
    - ``home_pos``      : home 位姿坐标。
    - ``last_collision``: 最近一次动作是否检测到碰撞。
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # 兼容嵌套：env_config.yaml 里可能放在 robot: {...} 或 physics: {...}。
        rob = cfg.get("robot", cfg)
        self.collision_check: bool = bool(rob.get("collision_check", True))
        self.speed_scale: float = float(rob.get("speed_scale", 1.0)) or 1.0

        self.home_pos = np.array([0.0, -0.05])
        self.ee_pos = self.home_pos.copy()
        self.gripper_open = _GRIPPER_OPEN
        self.holding: Optional[Dict] = None
        self.last_collision = False
        # 场景中其它零件的位置缓存（供碰撞检测）；由 env.reset 时通过
        # set_scene_obstacles 注入。未注入时碰撞检测仅考虑料盒。
        self._obstacles: List[Tuple[float, float]] = []
        self._bin_positions: List[Tuple[float, float]] = []

    # ------------------------------------------------------------------
    # 场景信息注入（供碰撞检测）
    # ------------------------------------------------------------------
    def set_scene(self, scene_state: Dict) -> None:
        """从 SceneState 提取障碍物（零件）与料盒位置，供碰撞检测使用。

        碰撞检测是"基础/粗略"的：只在末端目标点附近做平面距离判据，不做完整
        扫掠体相交。够 POC 用于"偶发碰撞"语义，且开销极小。
        """
        self._obstacles = [tuple(p.get("pos", [0.0, 0.0]))
                           for p in scene_state.get("parts", [])]
        self._bin_positions = [tuple(b.get("pos", [0.0, 0.0]))
                              for b in (scene_state.get("bins", {}) or {}).values()]

    # ------------------------------------------------------------------
    # 运动学：可达性与简化 IK
    # ------------------------------------------------------------------
    def is_reachable(self, pos) -> bool:
        """判断目标点是否在机械臂可达范围内（平面距基座的距离判据）。"""
        d = float(np.linalg.norm(np.asarray(pos, dtype=float) - _BASE))
        return _REACH_MIN <= d <= _REACH_MAX

    def inverse_kinematics(self, pos) -> Optional[Tuple[float, float]]:
        """平面 2R 简化逆运动学，返回 (theta1, theta2) 弧度；不可达返回 None。

        这是教科书式的 2 连杆 IK（用余弦定理）。POC 中我们其实不消费关节角
        本身，但提供它能让"几何可达性"判断更真实，也方便日后接 MuJoCo 时把
        关节角直接喂给执行器。返回的是"肘部向上"的一组解。
        """
        p = np.asarray(pos, dtype=float) - _BASE
        x, y = float(p[0]), float(p[1])
        r2 = x * x + y * y
        # 余弦定理求第二关节角。
        cos_t2 = (r2 - _L1 * _L1 - _L2 * _L2) / (2 * _L1 * _L2)
        if cos_t2 < -1.0 or cos_t2 > 1.0:
            return None  # 不可达
        theta2 = math.acos(cos_t2)  # 肘上解
        k1 = _L1 + _L2 * math.cos(theta2)
        k2 = _L2 * math.sin(theta2)
        theta1 = math.atan2(y, x) - math.atan2(k2, k1)
        return (theta1, theta2)

    # ------------------------------------------------------------------
    # 碰撞检测（基础/粗略）
    # ------------------------------------------------------------------
    def _check_collision(self, target, ignore_part_pos=None) -> bool:
        """粗略碰撞检测：末端移动到 target 时是否会撞到非目标零件/料盒边缘。

        判据：若 target 与某个"非目标零件"或某个"料盒中心"过近（小于阈值），
        视为潜在碰撞。ignore_part_pos 用于排除"正在抓取的目标零件本身"。
        关闭 collision_check 时恒返回 False。
        """
        if not self.collision_check:
            return False
        t = np.asarray(target, dtype=float)
        ignore = None if ignore_part_pos is None else np.asarray(ignore_part_pos, float)

        # 与其它零件过近（阈值 3.5cm）。
        for ob in self._obstacles:
            obp = np.asarray(ob, dtype=float)
            if ignore is not None and np.linalg.norm(obp - ignore) < 1e-6:
                continue  # 跳过目标零件自身
            if np.linalg.norm(t - obp) < 0.035:
                return True
        return False

    # ------------------------------------------------------------------
    # 动作接口：move_to / grasp / place / home
    # ------------------------------------------------------------------
    def _move_cost_ms(self, frm, to) -> float:
        """根据移动距离估算耗时（毫秒），受 speed_scale 缩放。"""
        dist = float(np.linalg.norm(np.asarray(to, float) - np.asarray(frm, float)))
        ms = _MOVE_BASE_MS + dist * _MOVE_PER_METER_MS
        return ms / self.speed_scale

    def move_to(self, pos, ignore_part_pos=None) -> Dict:
        """移动末端到目标 2D 位置。

        返回 info dict::
            {"ok": bool, "reachable": bool, "collision": bool,
             "duration_ms": int, "error": str|None}

        几何上：不可达 → ok=False 且 error 说明；可达则更新 ee_pos。
        碰撞：若检测到碰撞，info.collision=True（但仍移动到位，碰撞的"后果"
        由上层失败注入决定，这里只报告几何事实）。
        """
        target = np.asarray(pos, dtype=float)
        reachable = self.is_reachable(target)
        duration = self._move_cost_ms(self.ee_pos, target)
        if not reachable:
            self.last_collision = False
            return {"ok": False, "reachable": False, "collision": False,
                    "duration_ms": int(duration),
                    "error": "目标超出机械臂可达范围"}
        collision = self._check_collision(target, ignore_part_pos)
        self.last_collision = collision
        # 运动学近似：直接把末端"瞬移"到目标（mock 物理）。
        self.ee_pos = target.copy()
        # 持有零件时，零件随末端一起移动。
        if self.holding is not None:
            self.holding["pos"] = [float(target[0]), float(target[1])]
        return {"ok": True, "reachable": True, "collision": collision,
                "duration_ms": int(duration), "error": None}

    def grasp(self, part: Dict) -> Dict:
        """抓取一个零件（先移动到其位置，再闭合夹爪到合适开度）。

        返回 info dict（含 reached/collision/duration_ms/error）。这里只表达
        "几何上把夹爪开合到位、把零件标记为持有"；抓取是否真的成功（滑落与否）
        由上层失败注入模型判定后再调用 release/confirm。
        """
        pos = part.get("pos", [0.0, 0.0])
        move_info = self.move_to(pos, ignore_part_pos=pos)
        if not move_info["ok"]:
            return {"ok": False, "reached": False,
                    "collision": move_info["collision"],
                    "duration_ms": move_info["duration_ms"],
                    "error": move_info["error"]}
        # 按零件大小闭合夹爪。
        gap = _GRIPPER_SIZE_GAP.get(part.get("size", "中"), 0.045)
        self.gripper_open = gap
        self.holding = part
        total = move_info["duration_ms"] + _GRASP_MS / self.speed_scale
        return {"ok": True, "reached": True,
                "collision": move_info["collision"],
                "duration_ms": int(total), "error": None}

    def place(self, bin_key: str, bins: Dict) -> Dict:
        """把当前持有的零件放到指定料盒（先移动到料盒，再张开夹爪）。

        参数
        ----
        bin_key : 料盒 key（'A'/'B'/'C'）。
        bins    : 料盒字典（来自 SceneState['bins']），用于取料盒坐标。

        返回 info dict。空手放置会返回 ok=False。
        """
        if self.holding is None:
            return {"ok": False, "reached": False, "collision": False,
                    "duration_ms": 0, "error": "夹爪未持有零件，放置无效"}
        b = (bins or {}).get(bin_key)
        if b is None:
            return {"ok": False, "reached": False, "collision": False,
                    "duration_ms": 0, "error": f"未知料盒：{bin_key}"}
        move_info = self.move_to(b.get("pos", [0.0, -0.34]))
        # 张开夹爪释放零件。
        released = self.holding
        released["pos"] = list(b.get("pos", [0.0, -0.34]))
        self.gripper_open = _GRIPPER_OPEN
        self.holding = None
        total = move_info["duration_ms"] + _PLACE_MS / self.speed_scale
        return {"ok": move_info["ok"], "reached": move_info["ok"],
                "collision": move_info["collision"],
                "duration_ms": int(total),
                "error": move_info["error"]}

    def home(self) -> Dict:
        """机械臂回到 home 位姿、张开夹爪、释放持有（异常恢复常用）。"""
        duration = self._move_cost_ms(self.ee_pos, self.home_pos) + \
            _HOME_MS / self.speed_scale
        self.ee_pos = self.home_pos.copy()
        self.gripper_open = _GRIPPER_OPEN
        self.holding = None
        self.last_collision = False
        return {"ok": True, "reached": True, "collision": False,
                "duration_ms": int(duration), "error": None}

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------
    def state(self) -> Dict:
        """返回当前机械臂状态快照（调试/可视化用）。"""
        return {
            "ee_pos": [float(self.ee_pos[0]), float(self.ee_pos[1])],
            "gripper_open": float(self.gripper_open),
            "holding": None if self.holding is None else self.holding.get("code"),
            "collision_check": self.collision_check,
            "last_collision": self.last_collision,
        }


if __name__ == "__main__":
    # 自检：home -> 抓一个零件 -> 放到 A 区，打印每步几何 info。
    from src.simulation.objects import generate_scene, build_scene_state

    ss = build_scene_state(generate_scene("简单", seed=42))
    arm = SimpleArm()
    arm.set_scene(ss)
    print("home:", arm.home())
    part0 = ss["parts"][0]
    print(f"grasp {part0['name']}:", arm.grasp(part0))
    print("place A:", arm.place("A", ss["bins"]))
    print("state:", arm.state())
