# -*- coding: utf-8 -*-
"""
仿真环境（src/simulation/env.py）
=================================

``SortingEnv`` 是分拣引擎面对的"物理世界"统一入口。它向上提供 Gym 风格的
``reset / step / render`` 接口，向下在两种后端间自动选择：

- **mujoco 后端**：当 ``import mujoco`` 成功且 ``assets/models/scene.xml``
  能被 ``mujoco.MjModel.from_xml_path`` 加载时启用。提供离屏渲染相机图。
- **mock 后端（默认兜底）**：``MockPhysics`` 纯运动学近似 + matplotlib 合成
  俯视图。**只装核心依赖也能跑通完整演示**就是靠它。

【容错降级第一原则】贯穿全文件：
- mujoco 缺失或 scene.xml 加载失败 → 自动 backend="mock"，仅打印一条中文告警。
- get_camera_image 离屏渲染失败 → 回退 mock 合成图 → 再失败返回 None。
- render(human) 尝试 GLFW 窗口，失败 → 降级离屏/跳过，绝不崩溃。

【职责边界】env 只负责"世界状态 + 几何执行 + 出图"。动作"成败/失败分类"由
上层分拣引擎按失败注入模型决定；env.step 返回的是**几何执行事实**
（是否可达、是否碰撞、耗时），上层据此叠加概率失败。这样可复现、可分析。
"""

from __future__ import annotations

import logging
import os
from typing import Dict, Optional, Tuple

import numpy as np

from src.simulation.objects import (
    build_scene_state,
    generate_scene,
)
from src.simulation.robot import SimpleArm
from src.simulation.camera import Camera

logger = logging.getLogger("simulation.env")


# --- 可选依赖：MuJoCo（缺失自动降级）---------------------------------------
# 严格用 try/except 守卫。任何导入异常都视为"没有 mujoco"，走 mock。
try:
    import mujoco  # type: ignore
    _MUJOCO_AVAILABLE = True
    _MUJOCO_IMPORT_ERROR = None
except Exception as _e:  # ImportError 或底层动态库问题都算不可用
    mujoco = None  # type: ignore
    _MUJOCO_AVAILABLE = False
    _MUJOCO_IMPORT_ERROR = _e


# scene.xml 的默认路径（相对项目根 assets/models/scene.xml）。
# 用 __file__ 上溯到项目根，避免依赖 cwd。
def _default_scene_xml() -> str:
    here = os.path.dirname(os.path.abspath(__file__))      # src/simulation
    root = os.path.abspath(os.path.join(here, "..", ".."))  # 项目根
    return os.path.join(root, "assets", "models", "scene.xml")


class MockPhysics:
    """纯运动学的 mock 物理后端（无 MuJoCo 时使用）。

    它不做任何动力学积分：机械臂"瞬移"到目标，零件随夹爪移动。它持有当前
    SceneState 与一个 SimpleArm，把 env 下发的 action 翻译成机械臂动作，并
    返回**几何执行 info**（success 表示"几何上动作完成"，不代表分拣成功）。
    """

    def __init__(self, config: Dict):
        self.config = config or {}
        self.arm = SimpleArm(self.config)
        self.scene_state: Optional[Dict] = None

    def reset(self, scene_state: Dict) -> Dict:
        """载入新场景，复位机械臂，返回 SceneState。"""
        self.scene_state = scene_state
        self.arm.home()
        self.arm.set_scene(scene_state)
        return self.scene_state

    def step(self, action: Dict) -> Tuple[Dict, Dict]:
        """执行一个 action，返回 (SceneState, info)。

        info 结构（与 SPEC §6 对齐）::
            {"success": bool, "duration_ms": int, "collision": bool, "error": str|None}

        这里的 success 是"几何/运动学层面动作是否完成"：
        - move：可达且未越界即 True。
        - grasp：找到目标零件且可达即 True（是否滑落由上层注入）。
        - place：持有零件且料盒有效即 True（是否掉落由上层注入）。
        - return：回 home，恒 True。
        """
        atype = action.get("type", "move")
        info = {"success": False, "duration_ms": 0, "collision": False, "error": None}
        try:
            if atype == "grasp":
                part = self._find_part(action)
                if part is None:
                    info.update(error="未找到目标零件", duration_ms=200)
                    return self.scene_state, info
                r = self.arm.grasp(part)
                info.update(success=r["ok"], duration_ms=r["duration_ms"],
                            collision=r["collision"], error=r["error"])
            elif atype == "place":
                bin_key = action.get("target_bin") or "A"
                r = self.arm.place(bin_key, self.scene_state.get("bins", {}))
                # place 成功后，把该零件从"待分拣"语义上移走（标记 sorted_bin）。
                if r["ok"]:
                    self._mark_sorted(action, bin_key)
                info.update(success=r["ok"], duration_ms=r["duration_ms"],
                            collision=r["collision"], error=r["error"])
            elif atype == "return":
                r = self.arm.home()
                info.update(success=True, duration_ms=r["duration_ms"],
                            collision=False, error=None)
            else:  # move 及未知类型按移动处理
                target = self._resolve_move_target(action)
                r = self.arm.move_to(target)
                info.update(success=r["ok"], duration_ms=r["duration_ms"],
                            collision=r["collision"], error=r["error"])
        except Exception as e:
            # 任何意外都降级为"动作失败"info，绝不抛到上层中断评测。
            info.update(success=False, error=f"mock 执行异常：{e}", duration_ms=100)
        return self.scene_state, info

    # --- 内部工具 ---
    def _find_part(self, action: Dict) -> Optional[Dict]:
        """按 part_id 优先、其次 part_code 在当前场景里定位目标零件。"""
        if not self.scene_state:
            return None
        pid = action.get("part_id")
        if pid is not None:
            for p in self.scene_state["parts"]:
                if p.get("part_id") == pid:
                    return p
        code = action.get("part_code")
        if code is not None:
            for p in self.scene_state["parts"]:
                if p.get("code") == code and not p.get("_sorted_bin"):
                    return p
        return None

    def _resolve_move_target(self, action: Dict):
        """解析 move 动作的目标坐标：优先 params.pos，其次目标零件/料盒。"""
        params = action.get("params") or {}
        if "pos" in params:
            return params["pos"]
        part = self._find_part(action)
        if part is not None:
            return part.get("pos", [0.0, 0.0])
        bin_key = action.get("target_bin")
        if bin_key and self.scene_state:
            b = self.scene_state.get("bins", {}).get(bin_key)
            if b:
                return b.get("pos", [0.0, -0.34])
        return self.arm.ee_pos

    def _mark_sorted(self, action: Dict, bin_key: str) -> None:
        """把已放置的零件打上 _sorted_bin 标记（运行期，便于评分/可视化）。"""
        part = self.arm.holding  # place 后 holding 已清空，这里从 action 反查
        pid = action.get("part_id")
        code = action.get("part_code")
        for p in self.scene_state.get("parts", []):
            if (pid is not None and p.get("part_id") == pid) or \
               (pid is None and code is not None and p.get("code") == code
                    and not p.get("_sorted_bin")):
                p["_sorted_bin"] = bin_key
                break


class SortingEnv:
    """分拣仿真环境（统一入口，自动选择 mujoco/mock 后端）。

    参数
    ----
    config : 环境配置 dict（通常来自 env_config.yaml）。可选键：
        - scene_xml (str)   : MJCF 路径，默认 assets/models/scene.xml。
        - prefer_mujoco(bool): 是否优先尝试 mujoco，默认 True。设 False 可
          强制走 mock（演示/CI 友好）。
        - camera (dict)     : 相机参数（width/height/dpi）。

    属性
    ----
    - ``backend`` ∈ {"mujoco", "mock"} : 实际启用的后端。
    - ``scene_state`` : 当前场景真值快照（reset 后可用）。
    """

    def __init__(self, config: Optional[Dict] = None):
        self.config = config or {}
        self.scene_xml = self.config.get("scene_xml") or _default_scene_xml()
        self.prefer_mujoco = bool(self.config.get("prefer_mujoco", True))
        self.camera = Camera(self.config)
        self.scene_state: Optional[Dict] = None

        # MuJoCo 句柄（仅 mujoco 后端有效）。
        self._mj_model = None
        self._mj_data = None
        self._mj_renderer = None

        # mock 后端实例（始终创建，作为兜底；mujoco 后端也复用它做运动学）。
        self._mock = MockPhysics(self.config)

        # 后端选择：尽力尝试 mujoco，失败降级 mock。
        self.backend = self._select_backend()
        logger.info("仿真环境就绪：backend=%s", self.backend)

    # ------------------------------------------------------------------
    # 后端选择
    # ------------------------------------------------------------------
    def _select_backend(self) -> str:
        """决定使用 mujoco 还是 mock。任何不确定因素都偏向 mock（安全）。"""
        if not self.prefer_mujoco:
            return "mock"
        if not _MUJOCO_AVAILABLE:
            logger.warning("未检测到 mujoco（%s），仿真自动降级为 Mock 物理后端。",
                           type(_MUJOCO_IMPORT_ERROR).__name__
                           if _MUJOCO_IMPORT_ERROR else "ImportError")
            return "mock"
        # 尝试加载 scene.xml。
        if not os.path.exists(self.scene_xml):
            logger.warning("未找到 MJCF 模型 %s，仿真降级为 Mock 物理后端。",
                           self.scene_xml)
            return "mock"
        try:
            self._mj_model = mujoco.MjModel.from_xml_path(self.scene_xml)
            self._mj_data = mujoco.MjData(self._mj_model)
            logger.info("成功加载 MuJoCo 模型：%s", self.scene_xml)
            return "mujoco"
        except Exception as e:
            # scene.xml 不完美/不兼容都不阻断，安静降级。
            logger.warning("加载 MuJoCo 模型失败（%s），仿真降级为 Mock 物理后端。", e)
            self._mj_model = None
            self._mj_data = None
            return "mock"

    @property
    def is_mock(self) -> bool:
        """是否为 mock 后端（看板徽标用）。"""
        return self.backend != "mujoco"

    # ------------------------------------------------------------------
    # reset / step
    # ------------------------------------------------------------------
    def reset(self, scene_config: Optional[Dict] = None) -> Dict:
        """按场景配置生成/载入场景，返回 SceneState（含真值）。

        参数
        ----
        scene_config : 由 objects.generate_scene 产出的场景配置；为 None 时
            按 config 里的难度现场生成一个（默认"简单"）。

        无论哪个后端，运动学执行都走 MockPhysics（POC 取舍）；mujoco 后端
        额外用于"离屏渲染出更真实的相机图"。这样保证两后端行为一致、可复现。
        """
        if scene_config is None:
            difficulty = self.config.get("difficulty", "简单")
            scene_config = generate_scene(difficulty, seed=self.config.get("seed"))
        # 由场景配置构造含真值的 SceneState。
        self.scene_state = build_scene_state(scene_config)
        # mock 运动学复位（机械臂 home + 注入障碍）。
        self._mock.reset(self.scene_state)

        # mujoco 后端：复位物理数据（即便我们主要用运动学，也保持句柄有效，
        # 供离屏渲染）。失败则就地降级为 mock。
        if self.backend == "mujoco" and self._mj_model is not None:
            try:
                mujoco.mj_resetData(self._mj_model, self._mj_data)
                mujoco.mj_forward(self._mj_model, self._mj_data)
            except Exception as e:
                logger.warning("MuJoCo reset 失败（%s），切换 Mock 后端。", e)
                self.backend = "mock"
        return self.scene_state

    def step(self, action: Dict) -> Tuple[Dict, Dict]:
        """执行一个 action，返回 (SceneState, info)。

        info = {"success", "duration_ms", "collision", "error"}（几何事实）。
        运动学统一交给 MockPhysics（两后端一致）。
        """
        if self.scene_state is None:
            return {}, {"success": False, "duration_ms": 0,
                        "collision": False, "error": "环境未 reset"}
        return self._mock.step(action)

    # ------------------------------------------------------------------
    # 相机
    # ------------------------------------------------------------------
    def get_camera_image(self, view: str = "top") -> Optional[np.ndarray]:
        """获取相机图像（ndarray HxWx3）。失败返回 None。

        - mujoco 后端：优先尝试离屏渲染（mujoco.Renderer）。任一步失败则
          回退到 mock 合成图。
        - mock 后端：直接用 matplotlib 合成俯视/侧视色块图。
        - 任意异常：返回 None，不中断流程。
        """
        if self.scene_state is None:
            return None
        # mujoco 离屏渲染（仅 top 视角尝试；side 直接走合成）。
        if self.backend == "mujoco" and self._mj_model is not None and view == "top":
            img = self._render_mujoco_offscreen()
            if img is not None:
                return img
            # 离屏失败 → 落回合成图。
            logger.debug("MuJoCo 离屏渲染失败，回退合成俯视图。")
        # mock 合成图（核心兜底）。
        return self.camera.capture(self.scene_state, view=view)

    def _render_mujoco_offscreen(self) -> Optional[np.ndarray]:
        """MuJoCo 离屏渲染一帧 RGB。失败返回 None（不抛）。"""
        try:
            # mujoco.Renderer 需要可用的 OpenGL 上下文（EGL/CGL）。在无显示
            # 环境可能失败——失败即返回 None，由调用方回退合成图。
            if self._mj_renderer is None:
                self._mj_renderer = mujoco.Renderer(
                    self._mj_model, height=self.camera.height,
                    width=self.camera.width)
            mujoco.mj_forward(self._mj_model, self._mj_data)
            self._mj_renderer.update_scene(self._mj_data)
            return np.asarray(self._mj_renderer.render(), dtype=np.uint8)
        except Exception as e:
            logger.debug("MuJoCo Renderer 不可用：%s", e)
            return None

    # ------------------------------------------------------------------
    # 渲染（human / rgb_array）
    # ------------------------------------------------------------------
    def render(self, mode: str = "human") -> Optional[np.ndarray]:
        """渲染当前帧。

        - mode="rgb_array" : 返回一张 ndarray（等价 get_camera_image('top')）。
        - mode="human"     : 尝试弹出 GLFW 交互窗口（仅 mujoco 后端有意义）。
          GLFW/显示不可用时**降级为离屏出图或直接跳过，绝不崩溃**。

        返回值：rgb_array 模式返回 ndarray|None；human 模式返回 None。
        """
        if mode == "rgb_array":
            return self.get_camera_image("top")

        # human 模式。
        if self.backend != "mujoco" or self._mj_model is None:
            logger.info("当前为 Mock 后端，human 可视化降级：可用 get_camera_image 取合成图。")
            return None
        try:
            # 交互窗口需要 mujoco.viewer + GLFW。无显示环境会抛异常。
            from mujoco import viewer as mj_viewer  # type: ignore
            mj_viewer.launch_passive(self._mj_model, self._mj_data)
        except Exception as e:
            # GLFW/Retina/显示问题统统降级，不影响评测。
            logger.warning("human 窗口不可用（%s），降级为离屏渲染。", e)
            return self._render_mujoco_offscreen()
        return None

    # ------------------------------------------------------------------
    # 杂项
    # ------------------------------------------------------------------
    def save_camera_png(self, path: str, view: str = "top") -> Optional[str]:
        """抓一张相机图并保存为 PNG（失败案例复盘用），返回路径或 None。"""
        img = self.get_camera_image(view)
        return self.camera.save_png(img, path)

    def close(self) -> None:
        """释放渲染器资源（幂等、容错）。"""
        try:
            if self._mj_renderer is not None:
                self._mj_renderer.close()
        except Exception:
            pass
        finally:
            self._mj_renderer = None

    def info(self) -> Dict:
        """返回环境元信息（看板/日志用）。"""
        return {
            "backend": self.backend,
            "is_mock": self.is_mock,
            "mujoco_available": _MUJOCO_AVAILABLE,
            "scene_xml": self.scene_xml,
            "scene_xml_exists": os.path.exists(self.scene_xml),
        }


if __name__ == "__main__":
    # 自检：建环境 -> reset 中等场景 -> 抓放一个零件 -> 存一张俯视图。
    logging.basicConfig(level=logging.INFO,
                        format="%(levelname)s %(name)s: %(message)s")
    env = SortingEnv({"difficulty": "中等", "seed": 42})
    print("环境信息：", env.info())
    ss = env.reset(generate_scene("中等", seed=42))
    print(f"场景零件数={len(ss['parts'])}")
    p0 = ss["parts"][0]
    _, gi = env.step({"type": "grasp", "part_id": p0["part_id"],
                      "part_code": p0["code"]})
    print("grasp info:", gi)
    _, pi = env.step({"type": "place", "part_id": p0["part_id"],
                      "part_code": p0["code"], "target_bin": "A"})
    print("place info:", pi)
    saved = env.save_camera_png(
        os.path.join("data", "failure_cases", "env_selftest_top.png"))
    print("俯视图已保存：", saved)
    env.close()
