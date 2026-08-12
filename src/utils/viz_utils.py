# -*- coding: utf-8 -*-
"""
viz_utils —— 合成俯视场景图 / 时长 / 百分比 可视化工具
========================================================

SPEC §13 约定，本模块提供：
- 合成**俯视场景图**：把一份 SceneState / scene_json 画成 2D 俯视图
  （料盒 A/B/C 区域 + 零件色块 + 名称/遮挡标记）。
  提供两种产出，供不同消费方复用：
    * ``synthetic_scene_figure(scene)`` → 返回 Plotly Figure（看板 pages/failure 现场画图用）。
    * ``synthetic_scene_image(scene)``  → 用 matplotlib 渲染为 ndarray（相机 camera.py 复用，可存 PNG）。
- ``ms_to_duration(ms)`` 毫秒 → 可读时长（与 data_utils 同口径）。
- ``percent(value)``     百分比格式化（与 data_utils 同口径）。

容错降级（SPEC §0）：
- plotly / matplotlib / numpy 均为守卫导入；缺失时函数返回 None 或退化产物，
  绝不让看板 / 相机模块因缺包崩溃。
- 仅核心依赖（plotly + matplotlib 在 requirements 核心段）即可正常出图。

SceneState / scene 结构（SPEC §5）：
    {
      "parts": [
        {"part_id","code","name","material","color","size","shape",
         "fragile","pos":[x,y],"occluded":bool}, ...
      ],
      "bins": {"A":..,"B":..,"C":..}
    }
"""

from __future__ import annotations

import json
from typing import Any, Optional, Union

# 复用 data_utils 的格式化口径，保持全项目一致
try:
    from src.utils.data_utils import format_duration_ms as _fmt_ms
    from src.utils.data_utils import format_percent as _fmt_pct
except Exception:  # pragma: no cover - 极端降级（独立运行子模块时）
    _fmt_ms = None
    _fmt_pct = None

# ---------------------------------------------------------------------------
# 可选依赖守卫
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore

    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    import plotly.graph_objects as go  # type: ignore

    _PLOTLY_AVAILABLE = True
except Exception:  # pragma: no cover
    go = None  # type: ignore
    _PLOTLY_AVAILABLE = False

# matplotlib 用 Agg 后端（无 GUI、离屏渲染，Mac 上稳定）
try:
    import matplotlib

    matplotlib.use("Agg")  # 必须在 pyplot 之前设置后端
    import matplotlib.pyplot as plt  # type: ignore
    from matplotlib.patches import Rectangle, Circle, RegularPolygon  # type: ignore

    _MPL_AVAILABLE = True
except Exception:  # pragma: no cover
    plt = None  # type: ignore
    _MPL_AVAILABLE = False


# 主色（SPEC §0）
PRIMARY_COLOR = "#2563EB"

# ---------------------------------------------------------------------------
# 零件“颜色名 → 可绘制 HEX”映射（SPEC §2 的 10 种零件颜色）
# 用于把 SceneState 里的中文/英文颜色名转成画图色块。未知名用主灰兜底。
# ---------------------------------------------------------------------------
_COLOR_HEX = {
    "银色": "#C0C7D0",
    "silver": "#C0C7D0",
    "蓝色": "#3B82F6",
    "blue": "#3B82F6",
    "棕色": "#8B5E3C",
    "brown": "#8B5E3C",
    "黑色": "#2B2B2B",
    "black": "#2B2B2B",
    "白色": "#F3F4F6",
    "white": "#F3F4F6",
    "绿色": "#22C55E",
    "green": "#22C55E",
    "红色": "#EF4444",
    "red": "#EF4444",
}
_DEFAULT_PART_HEX = "#9CA3AF"

# 料盒 A/B/C 在俯视坐标系中的占位区域（归一化坐标，便于自适应）
# 场景坐标系约定：x∈[0,1] 横向，y∈[0,1] 纵向；料盒沿右侧排布。
_BIN_LAYOUT = {
    "A": {"x0": 0.74, "y0": 0.68, "x1": 0.98, "y1": 0.96, "label": "A区"},
    "B": {"x0": 0.74, "y0": 0.36, "x1": 0.98, "y1": 0.64, "label": "B区"},
    "C": {"x0": 0.74, "y0": 0.04, "x1": 0.98, "y1": 0.32, "label": "C区"},
}
_BIN_COLOR = "#E5E9F2"  # 料盒底色（浅灰蓝）
_BIN_EDGE = "#2563EB"   # 料盒边框（主色）
_WORKAREA_COLOR = "#F8FAFC"  # 工作台/传送带底色


# ===========================================================================
# 时长 / 百分比 格式化（与 data_utils 同口径；data_utils 不可用时本地兜底）
# ===========================================================================
def ms_to_duration(ms: Union[int, float, None]) -> str:
    """毫秒 → 可读时长（转调 data_utils；不可用时本地简易实现）。"""
    if _fmt_ms is not None:
        return _fmt_ms(ms)
    # 兜底实现
    if ms is None:
        return "—"
    try:
        ms = float(ms)
    except (TypeError, ValueError):
        return "—"
    if ms < 1000:
        return f"{int(round(ms))}ms"
    s = ms / 1000.0
    if s < 60:
        return f"{s:.1f}s"
    m = int(s // 60)
    return f"{m}分{s - m * 60:.1f}秒"


def percent(value: Union[int, float, None], decimals: int = 1, **kwargs: Any) -> str:
    """百分比格式化（转调 data_utils；不可用时本地兜底）。"""
    if _fmt_pct is not None:
        return _fmt_pct(value, decimals=decimals, **kwargs)
    if value is None:
        return "—"
    try:
        return f"{float(value) * 100:.{decimals}f}%"
    except (TypeError, ValueError):
        return "—"


# ===========================================================================
# 内部工具：标准化 scene 入参
# ===========================================================================
def _coerce_scene(scene: Union[dict, str, None]) -> dict:
    """
    把 scene 入参统一为 dict。

    支持：dict、JSON 字符串（看板从 storage 取出的 scene_json）、None。
    解析失败返回空场景结构，保证后续画图不报错。
    """
    if scene is None:
        return {"parts": [], "bins": {}}
    if isinstance(scene, str):
        try:
            scene = json.loads(scene)
        except Exception:
            return {"parts": [], "bins": {}}
    if not isinstance(scene, dict):
        return {"parts": [], "bins": {}}
    scene.setdefault("parts", [])
    scene.setdefault("bins", {})
    return scene


def _part_hex(part: dict) -> str:
    """根据零件 color 字段取绘制 HEX 色。"""
    color = str(part.get("color", "")).strip()
    return _COLOR_HEX.get(color, _DEFAULT_PART_HEX)


def _part_xy(part: dict, idx: int, total: int) -> tuple[float, float]:
    """
    取零件在俯视坐标系（0~1）中的位置。

    优先用 part['pos']=[x,y]；若缺失/非法，则在左侧工作区内按网格自动布点，
    保证即便场景没带坐标也能画出可读的俯视图。
    """
    pos = part.get("pos")
    if (
        isinstance(pos, (list, tuple))
        and len(pos) >= 2
        and _is_number(pos[0])
        and _is_number(pos[1])
    ):
        x = _clamp(float(pos[0]), 0.04, 0.68)
        y = _clamp(float(pos[1]), 0.04, 0.96)
        return x, y

    # 自动网格布点（工作区 x∈[0.06,0.66], y∈[0.08,0.92]）
    cols = max(1, int((total ** 0.5) + 0.999))
    rows = max(1, (total + cols - 1) // cols)
    c = idx % cols
    r = idx // cols
    x = 0.10 + (0.52 * (c + 0.5) / cols)
    y = 0.10 + (0.80 * (r + 0.5) / rows)
    return _clamp(x, 0.04, 0.68), _clamp(y, 0.04, 0.96)


def _size_radius(size: str) -> float:
    """零件大小 → 绘制半径（归一化坐标）。"""
    return {"小": 0.022, "中": 0.034, "大": 0.046}.get(str(size), 0.030)


def _is_number(v: Any) -> bool:
    try:
        float(v)
        return True
    except (TypeError, ValueError):
        return False


def _clamp(v: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, v))


# ===========================================================================
# Plotly 版：合成俯视场景图（看板 pages/failure 现场画图复用）
# ===========================================================================
def synthetic_scene_figure(
    scene: Union[dict, str, None], title: str = "场景俯视图"
):
    """
    把一份 SceneState / scene_json 画成 **Plotly 俯视场景图**。

    返回 plotly.graph_objects.Figure；plotly 不可用时返回 None
    （看板侧应据此降级为文字提示，不崩溃）。

    画面元素：
    - 浅色工作台背景 + 右侧 A/B/C 三个料盒区域（主色边框、中文标签）。
    - 每个零件用色块（按 SPEC §2 颜色）+ 名称标注；
      遮挡零件（occluded=True）半透明并加“遮挡”标记。
    """
    if not _PLOTLY_AVAILABLE:
        return None

    scene = _coerce_scene(scene)
    parts = scene.get("parts", []) or []

    fig = go.Figure()

    # 1) 工作台背景
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=1, y1=1,
        fillcolor=_WORKAREA_COLOR, line=dict(color="#E2E8F0", width=1), layer="below",
    )

    # 2) 料盒 A/B/C
    bins = scene.get("bins", {}) or {}
    for key, box in _BIN_LAYOUT.items():
        present = (key in bins) or (box["label"] in bins) or True  # 始终画三个料盒
        if not present:
            continue
        fig.add_shape(
            type="rect",
            x0=box["x0"], y0=box["y0"], x1=box["x1"], y1=box["y1"],
            fillcolor=_BIN_COLOR, line=dict(color=_BIN_EDGE, width=2), layer="below",
        )
        fig.add_annotation(
            x=(box["x0"] + box["x1"]) / 2,
            y=box["y1"] - 0.02,
            text=f"<b>{box['label']}</b>",
            showarrow=False,
            font=dict(color=_BIN_EDGE, size=13),
            yanchor="top",
        )

    # 3) 零件色块
    total = len(parts)
    xs, ys, texts, colors, sizes, hovers = [], [], [], [], [], []
    for i, p in enumerate(parts):
        if not isinstance(p, dict):
            continue
        x, y = _part_xy(p, i, total)
        occluded = bool(p.get("occluded", False))
        hexc = _part_hex(p)
        name = p.get("name") or p.get("code") or f"零件{i}"
        radius = _size_radius(p.get("size", "中"))

        xs.append(x)
        ys.append(y)
        texts.append(name + ("（遮挡）" if occluded else ""))
        colors.append(hexc)
        # marker size 用像素，约按半径放大
        sizes.append(max(14, radius * 520))
        hovers.append(
            f"{name}<br>材质：{p.get('material','-')}｜颜色：{p.get('color','-')}"
            f"<br>大小：{p.get('size','-')}｜易碎：{'是' if p.get('fragile') else '否'}"
            f"{'<br><b>遮挡</b>' if occluded else ''}"
        )

    if xs:
        fig.add_trace(
            go.Scatter(
                x=xs, y=ys, mode="markers+text",
                marker=dict(
                    size=sizes, color=colors,
                    line=dict(color="#475569", width=1.2),
                    opacity=0.95,
                ),
                text=texts, textposition="bottom center",
                textfont=dict(size=11, color="#1F2937"),
                hovertext=hovers, hoverinfo="text",
                name="零件",
            )
        )

    fig.update_layout(
        title=dict(text=title, font=dict(color=PRIMARY_COLOR, size=16)),
        xaxis=dict(range=[0, 1], visible=False, fixedrange=True),
        yaxis=dict(range=[0, 1], visible=False, fixedrange=True,
                   scaleanchor="x", scaleratio=1),
        plot_bgcolor="white", paper_bgcolor="white",
        showlegend=False,
        margin=dict(l=10, r=10, t=40, b=10),
        height=420,
    )
    return fig


# ===========================================================================
# matplotlib 版：合成俯视场景图 → ndarray（相机 camera.py 复用，可存 PNG）
# ===========================================================================
def synthetic_scene_image(
    scene: Union[dict, str, None],
    width: int = 640,
    height: int = 480,
    title: Optional[str] = None,
):
    """
    用 matplotlib 把 SceneState 画成俯视图并返回 **RGB ndarray**（H×W×3, uint8）。

    供 ``simulation/camera.py`` 的 mock 模式复用（再由相机模块负责存 PNG 到
    data/failure_cases/）。matplotlib 或 numpy 不可用时返回 None（相机据此降级返回 None）。
    """
    if not _MPL_AVAILABLE or not _NUMPY_AVAILABLE:
        return None

    scene = _coerce_scene(scene)
    parts = scene.get("parts", []) or []

    dpi = 100
    fig = plt.figure(figsize=(width / dpi, height / dpi), dpi=dpi)
    ax = fig.add_axes([0, 0, 1, 1])  # 占满画布，无边距
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    try:
        # 工作台背景
        ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=_WORKAREA_COLOR,
                               edgecolor="#E2E8F0", linewidth=1))

        # 料盒 A/B/C
        for key, box in _BIN_LAYOUT.items():
            ax.add_patch(
                Rectangle(
                    (box["x0"], box["y0"]),
                    box["x1"] - box["x0"], box["y1"] - box["y0"],
                    facecolor=_BIN_COLOR, edgecolor=_BIN_EDGE, linewidth=2,
                )
            )
            ax.text(
                (box["x0"] + box["x1"]) / 2, box["y1"] - 0.03, box["label"],
                ha="center", va="top", color=_BIN_EDGE, fontsize=10, fontweight="bold",
            )

        # 零件
        total = len(parts)
        for i, p in enumerate(parts):
            if not isinstance(p, dict):
                continue
            x, y = _part_xy(p, i, total)
            hexc = _part_hex(p)
            r = _size_radius(p.get("size", "中"))
            occluded = bool(p.get("occluded", False))
            alpha = 0.45 if occluded else 0.95
            shape = str(p.get("shape", ""))

            # 形状区分：方形/平板/块状用矩形，六边形用多边形，其余用圆
            if shape in ("方形", "平板", "块状"):
                ax.add_patch(
                    Rectangle((x - r, y - r), 2 * r, 2 * r,
                              facecolor=hexc, edgecolor="#475569",
                              linewidth=1.0, alpha=alpha)
                )
            elif shape == "六边形":
                ax.add_patch(
                    RegularPolygon((x, y), numVertices=6, radius=r,
                                   facecolor=hexc, edgecolor="#475569",
                                   linewidth=1.0, alpha=alpha)
                )
            else:
                ax.add_patch(
                    Circle((x, y), r, facecolor=hexc, edgecolor="#475569",
                           linewidth=1.0, alpha=alpha)
                )

            name = str(p.get("name") or p.get("code") or f"零件{i}")
            label = name + ("（遮挡）" if occluded else "")
            ax.text(x, y - r - 0.018, label, ha="center", va="top",
                    fontsize=8, color="#1F2937")

        if title:
            ax.text(0.02, 0.97, title, ha="left", va="top",
                    fontsize=11, color=PRIMARY_COLOR, fontweight="bold")

        fig.canvas.draw()
        # 取 RGB 缓冲为 ndarray（兼容不同 matplotlib 版本的缓冲接口）
        img = _figure_to_ndarray(fig)
        return img
    except Exception as exc:  # pragma: no cover
        print(f"[viz_utils] 合成场景图渲染失败：{exc}")
        return None
    finally:
        plt.close(fig)


def _figure_to_ndarray(fig):
    """
    把 matplotlib Figure 的画布转为 RGB ndarray（H×W×3, uint8）。

    兼容不同 matplotlib 版本：优先 buffer_rgba，退回 tostring_rgb。
    """
    canvas = fig.canvas
    try:
        buf = np.asarray(canvas.buffer_rgba())  # (H, W, 4)
        return np.ascontiguousarray(buf[:, :, :3])  # 丢弃 alpha
    except Exception:
        # 旧版本回退
        w, h = canvas.get_width_height()
        raw = np.frombuffer(canvas.tostring_rgb(), dtype=np.uint8)
        return raw.reshape(h, w, 3)


def save_scene_png(
    scene: Union[dict, str, None], path: str, title: Optional[str] = None
) -> bool:
    """
    便捷封装：渲染合成俯视图并保存为 PNG（供 camera.py 存到 data/failure_cases/）。

    需要 matplotlib；缺失或异常返回 False（不崩溃）。
    """
    if not _MPL_AVAILABLE:
        return False
    try:
        import os

        parent = os.path.dirname(os.path.abspath(path))
        if parent and not os.path.isdir(parent):
            os.makedirs(parent, exist_ok=True)

        scene = _coerce_scene(scene)
        fig = plt.figure(figsize=(6.4, 4.8), dpi=100)
        # 复用 ndarray 渲染逻辑较繁琐，这里直接重画一遍以便控制 dpi/标题
        img = synthetic_scene_image(scene, title=title)
        plt.close(fig)
        if img is None:
            return False
        # 用 matplotlib 保存 ndarray
        plt.imsave(path, img)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[viz_utils] 保存场景 PNG 失败：{exc}")
        return False


if __name__ == "__main__":
    # 快速自检：python -m src.utils.viz_utils
    demo_scene = {
        "parts": [
            {"part_id": 0, "code": "screw", "name": "螺丝", "material": "金属",
             "color": "银色", "size": "小", "shape": "圆柱", "fragile": False,
             "pos": [0.2, 0.7], "occluded": False},
            {"part_id": 1, "code": "chip", "name": "芯片", "material": "塑料",
             "color": "黑色", "size": "中", "shape": "方形", "fragile": True,
             "pos": [0.4, 0.4], "occluded": True},
            {"part_id": 2, "code": "pcb", "name": "PCB板", "material": "复合",
             "color": "绿色", "size": "大", "shape": "平板", "fragile": True,
             "pos": [0.3, 0.2], "occluded": False},
        ],
        "bins": {"A": {}, "B": {}, "C": {}},
    }
    print("plotly 可用 :", _PLOTLY_AVAILABLE)
    print("matplotlib 可用 :", _MPL_AVAILABLE)
    fig = synthetic_scene_figure(demo_scene, title="演示场景")
    print("Plotly figure :", "OK" if fig is not None else "降级 None")
    img = synthetic_scene_image(demo_scene, title="演示场景")
    print("ndarray shape :", None if img is None else img.shape)
    print("时长 125000ms ->", ms_to_duration(125000))
    print("0.834 -> 百分比 :", percent(0.834))
