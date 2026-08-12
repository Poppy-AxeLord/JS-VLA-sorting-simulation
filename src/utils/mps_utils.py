# -*- coding: utf-8 -*-
"""
mps_utils —— Apple Silicon 加速设备探测工具
=============================================

SPEC §0 / §13 约定：
- 加速优先级：MPS（Apple Silicon GPU）> CPU。
- ``get_device()`` 永远返回字符串：
    * torch 可用且 ``torch.backends.mps`` 可用且已构建 → 'mps'
    * 否则 → 'cpu'
    * **torch 缺失时也必须返回 'cpu'，绝不抛异常、绝不崩溃。**
- ``device_info()`` 返回一段中文描述，便于看板/日志展示当前算力环境。

为什么这样设计（产品/工程决策）：
本项目面向 Mac Studio / Apple Silicon，主打“只装核心轻量依赖即可跑通完整演示”。
torch 属于可选加强项（仅 SmolVLA 后端才需要），因此这里对 torch 的导入做了安全守卫——
任何环境（无 torch、无 GPU、纯 CPU）都能正常拿到一个合法的 device 字符串，
让上层 VLA / 仿真模块据此选择计算后端，从而实现 MPS→CPU 的自动降级。
"""

from __future__ import annotations

import platform


# ---------------------------------------------------------------------------
# 安全导入 torch：缺失不报错，记录一个内部标志位即可。
# ---------------------------------------------------------------------------
try:  # 可选依赖：torch 仅 SmolVLA 后端需要
    import torch  # type: ignore

    _TORCH_AVAILABLE = True
    _TORCH_IMPORT_ERROR = None
except Exception as _exc:  # pragma: no cover - 取决于运行环境
    torch = None  # type: ignore
    _TORCH_AVAILABLE = False
    _TORCH_IMPORT_ERROR = _exc


def torch_available() -> bool:
    """返回 torch 是否成功导入（供上层判断是否可用 SmolVLA / MPS）。"""
    return _TORCH_AVAILABLE


def mps_available() -> bool:
    """
    判断 MPS 后端是否真正可用。

    需要同时满足：
    1. torch 已成功导入；
    2. ``torch.backends.mps.is_available()`` 为真（系统/驱动支持）；
    3. ``torch.backends.mps.is_built()`` 为真（当前 torch 编译时带 MPS）。

    任一步骤异常都视为不可用并返回 False（不向上抛出）。
    """
    if not _TORCH_AVAILABLE:
        return False
    try:
        backends = getattr(torch, "backends", None)
        mps = getattr(backends, "mps", None) if backends is not None else None
        if mps is None:
            return False
        # is_built 在部分版本可能不存在，缺省视为 True
        is_built = getattr(mps, "is_built", lambda: True)()
        is_avail = mps.is_available()
        return bool(is_avail and is_built)
    except Exception:
        # 任何探测异常都安全降级为“不可用”
        return False


def get_device() -> str:
    """
    返回当前推荐的计算设备字符串：'mps' 或 'cpu'。

    规则（SPEC §13）：
    - torch 可用且 MPS 可用 → 'mps'
    - 其余所有情况（含 torch 缺失）→ 'cpu'

    本函数保证**永不抛出异常**，始终返回合法字符串。
    """
    try:
        if mps_available():
            return "mps"
    except Exception:
        pass
    return "cpu"


def is_apple_silicon() -> bool:
    """粗略判断是否运行在 Apple Silicon（arm64 macOS）上，仅用于信息展示。"""
    try:
        return platform.system() == "Darwin" and platform.machine().lower() in (
            "arm64",
            "aarch64",
        )
    except Exception:
        return False


def device_info() -> str:
    """
    返回一段**中文**算力环境描述，供日志/看板徽标展示。

    示例：
    - "当前算力设备：MPS（Apple Silicon GPU 加速）｜torch 版本 2.3.0"
    - "当前算力设备：CPU（torch 未安装，使用纯 CPU 与规则/Mock 降级）"
    """
    chip = "Apple Silicon" if is_apple_silicon() else platform.machine() or "未知架构"

    if not _TORCH_AVAILABLE:
        return (
            f"当前算力设备：CPU（torch 未安装，使用纯 CPU 运行；"
            f"SmolVLA 将自动降级为 rule_based 规则后端）｜机器架构：{chip}"
        )

    torch_ver = getattr(torch, "__version__", "未知")
    if get_device() == "mps":
        return (
            f"当前算力设备：MPS（Apple Silicon GPU 加速）｜"
            f"torch 版本 {torch_ver}｜机器架构：{chip}"
        )

    return (
        f"当前算力设备：CPU（torch 已安装但 MPS 不可用，回退 CPU 计算）｜"
        f"torch 版本 {torch_ver}｜机器架构：{chip}"
    )


if __name__ == "__main__":
    # 便于本地快速自检：python -m src.utils.mps_utils
    print("torch 可用 :", torch_available())
    print("MPS 可用   :", mps_available())
    print("推荐设备   :", get_device())
    print("设备信息   :", device_info())
