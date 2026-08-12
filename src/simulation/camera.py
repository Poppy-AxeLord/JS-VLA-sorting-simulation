# -*- coding: utf-8 -*-
"""
相机模块（src/simulation/camera.py）
====================================

为仿真环境提供"相机图像"。两种来源：
1. MuJoCo 离屏渲染（由 env.py 在 mujoco 后端下调用，本文件不直接依赖
   mujoco，仅提供 mock 合成图作为兜底）。
2. **Mock 合成图（核心容错路径）**：用 matplotlib 把 SceneState 画成一张
   俯视/侧视的"色块图"——料盒区域 + 零件位置 + 零件颜色，返回 numpy ndarray
   (H, W, 3) uint8。可保存 PNG 到 ``data/failure_cases/`` 供失败案例复盘。

【关键工程约定】
- **不强依赖 opencv**：保存 PNG 用 matplotlib，不用 cv2。RGB↔BGR 也无需关心。
- matplotlib 用 Agg 后端（无显示环境也能渲染到内存），Apple Silicon 友好。
- 中文标注：尝试设置常见中文字体（PingFang/Heiti/Arial Unicode 等），找不到
  也不崩，只是中文可能显示为方块——不影响 ndarray 的色块信息。
- 任何渲染失败都返回 None（俯视）或安全降级，绝不抛断流程。
"""

from __future__ import annotations

import os
from typing import Dict, Optional

import numpy as np

# matplotlib 属核心依赖。强制 Agg 后端：服务器/无 GUI 环境下也能把图渲染到
# 内存缓冲区，避免 streamlit/benchmark 进程里弹窗或报 "no display"。
try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Circle, RegularPolygon, Rectangle
    _MPL_OK = True
except Exception as _e:  # pragma: no cover - 理论上核心依赖不会缺
    # 极端情况下 matplotlib 不可用：合成图功能整体降级为返回 None。
    _MPL_OK = False
    _MPL_IMPORT_ERROR = _e


# --- 中文字体设置（尽力而为，失败不影响主流程）-----------------------------
def _setup_cn_font() -> None:
    """尝试为 matplotlib 配置中文字体，避免中文标签乱码。

    在 macOS 上优先使用系统自带的 PingFang/Heiti/Arial Unicode MS。找不到
    任何中文字体也不报错，仅可能出现方块字——色块信息依旧正确。
    """
    if not _MPL_OK:
        return
    candidates = [
        "PingFang SC", "Heiti SC", "Arial Unicode MS",
        "STHeiti", "Songti SC", "Hiragino Sans GB", "SimHei",
    ]
    try:
        from matplotlib.font_manager import FontProperties, findfont
        available = []
        for name in candidates:
            try:
                # findfont 找不到时会回退默认字体；用 fallback_to_default=False
                # 让其在确实缺失时抛错，从而跳过该候选。
                findfont(FontProperties(family=name),
                         fallback_to_default=False)
                available.append(name)
            except Exception:
                continue
        if available:
            plt.rcParams["font.sans-serif"] = available + \
                plt.rcParams.get("font.sans-serif", [])
        # 负号正常显示。
        plt.rcParams["axes.unicode_minus"] = False
    except Exception:
        # 字体探测失败也无所谓，保持默认。
        pass


_setup_cn_font()


# 工作台/抓取区与料盒在逻辑坐标系中的范围（与 objects.py 对齐）。
# 画图时把这片区域映射到固定像素画布。
_VIEW_X = (-0.40, 0.40)
_VIEW_Y = (-0.45, 0.30)


class Camera:
    """仿真相机。

    参数
    ----
    config : 相机配置 dict（可来自 env_config.yaml 的 camera 段），可选键：
        - width, height : 输出图像像素尺寸（默认 480x360）。
        - dpi           : matplotlib 渲染 dpi（默认 100）。

    主要方法
    --------
    - ``render_topdown(scene_state)`` : 合成俯视色块图，返回 ndarray|None。
    - ``render_side(scene_state)``    : 合成侧视示意图，返回 ndarray|None。
    - ``capture(scene_state, view)``  : 按 view 分发到上面两者。
    - ``save_png(image, path)``       : 把 ndarray 保存为 PNG（matplotlib）。
    """

    def __init__(self, config: Optional[Dict] = None):
        cfg = config or {}
        # 兼容嵌套（env_config.yaml 里可能是 camera: {...}）。
        cam = cfg.get("camera", cfg)
        self.width = int(cam.get("width", 480))
        self.height = int(cam.get("height", 360))
        self.dpi = int(cam.get("dpi", 100))

    # ------------------------------------------------------------------
    # 对外主入口
    # ------------------------------------------------------------------
    def capture(self, scene_state: Dict, view: str = "top") -> Optional[np.ndarray]:
        """按视角抓取一张合成图像。view ∈ {"top","side"}。失败返回 None。"""
        try:
            if view == "side":
                return self.render_side(scene_state)
            return self.render_topdown(scene_state)
        except Exception:
            # 渲染异常一律降级为 None，绝不影响仿真主流程。
            return None

    # ------------------------------------------------------------------
    # 俯视合成图（核心）
    # ------------------------------------------------------------------
    def render_topdown(self, scene_state: Dict) -> Optional[np.ndarray]:
        """把 SceneState 画成俯视色块图，返回 (H, W, 3) uint8 的 ndarray。

        画面元素：
        - 浅灰工作台背景 + 抓取区边框。
        - 底部三个料盒（A/B/C），用各自主色描边并标注中文名。
        - 每个零件按其 color_hex 画一个色块，形状粗略对应 shape：
          圆柱→圆、六边形→六边形、方形/块状→方块、平板→扁矩形。
        - 被遮挡的零件用半透明 + 红色虚线框标注（感知失败可视化）。
        """
        if not _MPL_OK:
            return None

        fig = None
        try:
            fig = plt.figure(figsize=(self.width / self.dpi,
                                      self.height / self.dpi),
                             dpi=self.dpi)
            ax = fig.add_axes([0, 0, 1, 1])  # 占满画布，无边距
            ax.set_xlim(*_VIEW_X)
            ax.set_ylim(*_VIEW_Y)
            ax.set_aspect("equal")
            ax.axis("off")
            # 工作台背景。
            ax.add_patch(Rectangle((_VIEW_X[0], _VIEW_Y[0]),
                                   _VIEW_X[1] - _VIEW_X[0],
                                   _VIEW_Y[1] - _VIEW_Y[0],
                                   facecolor="#F4F6F8", edgecolor="none"))
            # 抓取区边框（零件散落区）。
            ax.add_patch(Rectangle((-0.32, -0.20), 0.64, 0.44,
                                   facecolor="#FFFFFF", edgecolor="#C7D0DA",
                                   linewidth=1.2, linestyle="--"))

            # 画料盒。
            bins = scene_state.get("bins", {}) or {}
            for key, b in bins.items():
                pos = b.get("pos", [0, -0.34])
                hexc = b.get("color_hex", "#2563EB")
                name = b.get("name", f"{key}区")
                ax.add_patch(Rectangle((pos[0] - 0.09, pos[1] - 0.07),
                                       0.18, 0.14,
                                       facecolor="#FFFFFF", edgecolor=hexc,
                                       linewidth=2.4))
                ax.text(pos[0], pos[1], name, ha="center", va="center",
                        fontsize=9, color=hexc, fontweight="bold")

            # 画零件。
            for p in scene_state.get("parts", []):
                self._draw_part(ax, p)

            ax.set_title("")
            img = self._fig_to_array(fig)
            return img
        except Exception:
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    def _draw_part(self, ax, part: Dict) -> None:
        """在坐标轴上绘制单个零件（形状近似 + 颜色 + 遮挡标记 + part_id）。"""
        pos = part.get("pos", [0.0, 0.0])
        x, y = float(pos[0]), float(pos[1])
        hexc = part.get("color_hex", "#888888")
        shape = part.get("shape", "方形")
        size = part.get("size", "中")
        occluded = bool(part.get("occluded", False))

        # 大小→半径（米）。
        radius = {"小": 0.018, "中": 0.028, "大": 0.040}.get(size, 0.028)
        alpha = 0.45 if occluded else 0.95  # 遮挡件半透明
        edge = "#E8453C" if occluded else "#333333"
        ls = (0, (3, 2)) if occluded else "solid"  # 遮挡件红色虚线框

        try:
            if shape == "圆柱":
                ax.add_patch(Circle((x, y), radius, facecolor=hexc,
                                    edgecolor=edge, linewidth=1.3,
                                    alpha=alpha, linestyle=ls))
            elif shape == "六边形":
                ax.add_patch(RegularPolygon((x, y), numVertices=6,
                                            radius=radius, facecolor=hexc,
                                            edgecolor=edge, linewidth=1.3,
                                            alpha=alpha, linestyle=ls))
            elif shape == "平板":
                ax.add_patch(Rectangle((x - radius * 1.4, y - radius * 0.6),
                                       radius * 2.8, radius * 1.2,
                                       facecolor=hexc, edgecolor=edge,
                                       linewidth=1.3, alpha=alpha,
                                       linestyle=ls))
            else:  # 方形 / 块状 / 其它
                ax.add_patch(Rectangle((x - radius, y - radius),
                                       radius * 2, radius * 2,
                                       facecolor=hexc, edgecolor=edge,
                                       linewidth=1.3, alpha=alpha,
                                       linestyle=ls))
            # 标注 part_id（小号灰字），便于和动作日志对照。
            pid = part.get("part_id")
            if pid is not None:
                ax.text(x, y + radius + 0.012, str(pid), ha="center",
                        va="bottom", fontsize=6, color="#555555")
        except Exception:
            # 单个零件画失败不影响整图。
            pass

    # ------------------------------------------------------------------
    # 侧视示意图（简化）
    # ------------------------------------------------------------------
    def render_side(self, scene_state: Dict) -> Optional[np.ndarray]:
        """侧视图：把零件按 x 坐标投影到一条工作台横线上，高度示意大小。

        侧视主要用于"看板里多一个视角"的展示，POC 中做成简化投影即可：
        横轴=x，纵轴=零件高度（按 size 给一个示意高度），料盒画在右侧。
        """
        if not _MPL_OK:
            return None

        fig = None
        try:
            fig = plt.figure(figsize=(self.width / self.dpi,
                                      self.height / self.dpi),
                             dpi=self.dpi)
            ax = fig.add_axes([0, 0, 1, 1])
            ax.set_xlim(_VIEW_X[0], _VIEW_X[1])
            ax.set_ylim(0.0, 0.20)
            ax.axis("off")
            # 背景 + 工作台面。
            ax.add_patch(Rectangle((_VIEW_X[0], 0.0),
                                   _VIEW_X[1] - _VIEW_X[0], 0.20,
                                   facecolor="#F4F6F8", edgecolor="none"))
            ax.plot([_VIEW_X[0], _VIEW_X[1]], [0.03, 0.03],
                    color="#9AA5B1", linewidth=2)
            # 零件按 x 投影、按 size 给高度。
            for p in scene_state.get("parts", []):
                x = float(p.get("pos", [0, 0])[0])
                h = {"小": 0.02, "中": 0.04, "大": 0.07}.get(
                    p.get("size", "中"), 0.04)
                hexc = p.get("color_hex", "#888888")
                occluded = bool(p.get("occluded", False))
                ax.add_patch(Rectangle((x - 0.015, 0.03), 0.03, h,
                                       facecolor=hexc,
                                       edgecolor="#E8453C" if occluded else "#333333",
                                       alpha=0.5 if occluded else 0.95,
                                       linewidth=1.0))
            return self._fig_to_array(fig)
        except Exception:
            return None
        finally:
            if fig is not None:
                plt.close(fig)

    # ------------------------------------------------------------------
    # 工具方法
    # ------------------------------------------------------------------
    @staticmethod
    def _fig_to_array(fig) -> np.ndarray:
        """把 matplotlib figure 渲染为 (H, W, 3) uint8 ndarray（不经磁盘）。

        兼容不同 matplotlib 版本的缓冲区接口：优先 buffer_rgba（新版本），
        回退 tostring_rgb（旧版本）。统一裁掉 alpha 通道返回 RGB。
        """
        canvas = fig.canvas
        canvas.draw()
        w, h = canvas.get_width_height()
        try:
            buf = np.asarray(canvas.buffer_rgba())  # (h, w, 4)
            rgb = buf[:, :, :3].copy()
        except Exception:
            # 旧版 matplotlib 兜底。
            raw = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
            rgb = raw.reshape(h, w, 3).copy()
        return rgb.astype(np.uint8)

    @staticmethod
    def save_png(image: Optional[np.ndarray], path: str) -> Optional[str]:
        """把 ndarray 图像保存为 PNG（使用 matplotlib，不依赖 opencv）。

        参数
        ----
        image : (H, W, 3) uint8 ndarray；为 None 时直接返回 None。
        path  : 目标文件绝对/相对路径，父目录不存在会自动创建。

        返回保存成功的路径，失败返回 None（不抛异常）。
        典型用途：把失败案例的合成俯视图存到 ``data/failure_cases/``。
        """
        if image is None or not _MPL_OK:
            return None
        try:
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            # plt.imsave 直接写 RGB 数组为 PNG，无需开 figure。
            plt.imsave(path, image)
            return path
        except Exception:
            return None


def render_scene_image(scene_state: Dict, view: str = "top",
                       config: Optional[Dict] = None) -> Optional[np.ndarray]:
    """便捷函数：一行得到某 SceneState 的合成图（内部建临时 Camera）。

    供 env.py 在 mock 后端下、或评测里快速取图使用。失败返回 None。
    """
    try:
        return Camera(config).capture(scene_state, view=view)
    except Exception:
        return None


if __name__ == "__main__":
    # 自检：生成一个中等难度场景并把俯视图保存为 PNG，便于肉眼检查。
    from src.simulation.objects import generate_scene, build_scene_state

    sc = generate_scene("中等", seed=42)
    ss = build_scene_state(sc)
    cam = Camera()
    img = cam.render_topdown(ss)
    if img is not None:
        out = os.path.join("data", "failure_cases", "camera_selftest_top.png")
        saved = cam.save_png(img, out)
        print(f"俯视图尺寸={img.shape} 已保存={saved}")
    else:
        print("matplotlib 不可用或渲染失败，俯视图降级为 None。")
