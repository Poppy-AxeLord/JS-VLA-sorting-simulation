# -*- coding: utf-8 -*-
"""
工具模块 src/utils
===================

本包汇集全项目复用的“纯工具函数”，遵循 SPEC §13 约定：

- mps_utils  : Apple Silicon 设备探测（MPS 优先 / CPU 兜底），torch 缺失也不崩溃。
- data_utils : JSON/YAML 读写、ndarray↔list、ISO 时间、数值/百分比格式化、
               失败分类→配色映射、毫秒→可读时长。
- viz_utils  : 合成俯视场景图（matplotlib / Plotly figure spec）、毫秒→时长、
               百分比格式化，供看板与相机模块复用。

设计原则（容错降级第一）：
- 所有可选第三方库（torch / matplotlib / plotly / yaml 等）一律用 try/except 守卫导入；
  缺失时优雅降级，绝不因缺包让整套演示崩溃。
- 仅依赖核心轻量库即可运行（numpy / pandas / pyyaml / streamlit / plotly / matplotlib）。

为方便调用，这里把最常用的若干函数直接提升到包命名空间。
"""

# 注意：这里的二次导入也用 try/except 包裹，避免任何子模块的可选依赖问题
# 在 import src.utils 阶段就把整个项目带崩。
try:
    from src.utils.mps_utils import get_device, device_info
except Exception:  # pragma: no cover - 极端降级路径
    get_device = None
    device_info = None

try:
    from src.utils.data_utils import (
        read_json,
        write_json,
        read_yaml,
        ndarray_to_list,
        list_to_ndarray,
        now_iso,
        to_iso,
        format_number,
        format_percent,
        failure_color,
        FAILURE_COLORS,
        format_duration_ms,
    )
except Exception:  # pragma: no cover
    pass

try:
    from src.utils.viz_utils import (
        synthetic_scene_figure,
        synthetic_scene_image,
        ms_to_duration,
        percent,
    )
except Exception:  # pragma: no cover
    pass

__all__ = [
    # mps_utils
    "get_device",
    "device_info",
    # data_utils
    "read_json",
    "write_json",
    "read_yaml",
    "ndarray_to_list",
    "list_to_ndarray",
    "now_iso",
    "to_iso",
    "format_number",
    "format_percent",
    "failure_color",
    "FAILURE_COLORS",
    "format_duration_ms",
    # viz_utils
    "synthetic_scene_figure",
    "synthetic_scene_image",
    "ms_to_duration",
    "percent",
]
