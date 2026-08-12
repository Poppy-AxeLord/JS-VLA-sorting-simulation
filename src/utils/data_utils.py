# -*- coding: utf-8 -*-
"""
data_utils —— 数据读写 / 类型转换 / 格式化 / 失败配色 工具
============================================================

SPEC §13 约定，本模块提供：
- JSON / YAML 读写（UTF-8、中文不转义、目录自动创建）。
- ndarray ↔ list 互转（numpy 缺失也安全降级）。
- ISO8601 时间字符串（SPEC §0：时间统一用 ISO8601）。
- 数值格式化 / 百分比格式化。
- **失败分类 → 配色映射**（严格对齐 SPEC §3 的 5 大类配色，全项目统一）。
- 毫秒 → 可读时长。

纯核心依赖：仅用标准库 + numpy(可选) + pyyaml(可选)。
所有可选库均 try/except 守卫，缺失时优雅降级而非崩溃。
"""

from __future__ import annotations

import datetime as _dt
import json
import os
from typing import Any, Iterable, Optional, Union

# ---------------------------------------------------------------------------
# 可选依赖：numpy（ndarray 转换）、pyyaml（YAML 读取）
# ---------------------------------------------------------------------------
try:
    import numpy as np  # type: ignore

    _NUMPY_AVAILABLE = True
except Exception:  # pragma: no cover
    np = None  # type: ignore
    _NUMPY_AVAILABLE = False

try:
    import yaml  # type: ignore

    _YAML_AVAILABLE = True
except Exception:  # pragma: no cover
    yaml = None  # type: ignore
    _YAML_AVAILABLE = False


# ===========================================================================
# 失败分类 → 配色映射（SPEC §3，唯一事实来源，全项目统一）
# ===========================================================================
# 说明：存储中 failure_category 存“大类中文”（见 SPEC §3 / §11），
# 因此这里同时用中文大类名 与 英文 key 两种键都能查到颜色，方便各模块复用。
FAILURE_COLORS: dict[str, str] = {
    # 英文 key
    "perception": "#5B8FF9",
    "understanding": "#5AD8A6",
    "planning": "#F6BD16",
    "execution": "#E8684A",
    "environment": "#9270CA",
    # 中文大类（存储落库用的就是中文）
    "感知类失败": "#5B8FF9",
    "理解类失败": "#5AD8A6",
    "规划类失败": "#F6BD16",
    "执行类失败": "#E8684A",
    "环境类失败": "#9270CA",
}

# 大类英文 key ↔ 中文名 双向映射，便于聚合/展示
FAILURE_CATEGORY_CN: dict[str, str] = {
    "perception": "感知类失败",
    "understanding": "理解类失败",
    "planning": "规划类失败",
    "execution": "执行类失败",
    "environment": "环境类失败",
}
FAILURE_CATEGORY_EN: dict[str, str] = {v: k for k, v in FAILURE_CATEGORY_CN.items()}

# 兜底色：未知分类（例如成功任务无 failure_category 时）用主灰色，避免画图报错
_DEFAULT_FAILURE_COLOR = "#9CA3AF"


def failure_color(category: Optional[str]) -> str:
    """
    失败分类 → 颜色。

    入参可以是英文 key（perception/...）或中文大类名（"感知类失败"/...）。
    未知 / None 返回兜底灰色，保证调用方（看板饼图等）不会因取不到色而崩。
    """
    if not category:
        return _DEFAULT_FAILURE_COLOR
    return FAILURE_COLORS.get(category, _DEFAULT_FAILURE_COLOR)


def category_to_cn(category: Optional[str]) -> Optional[str]:
    """英文大类 key → 中文大类名；已是中文则原样返回。"""
    if category is None:
        return None
    if category in FAILURE_CATEGORY_CN:
        return FAILURE_CATEGORY_CN[category]
    return category


def category_to_en(category: Optional[str]) -> Optional[str]:
    """中文大类名 → 英文 key；已是英文则原样返回。"""
    if category is None:
        return None
    if category in FAILURE_CATEGORY_EN:
        return FAILURE_CATEGORY_EN[category]
    return category


# ===========================================================================
# JSON / YAML 读写（UTF-8，中文不转义，目录自动创建）
# ===========================================================================
def _ensure_parent_dir(path: str) -> None:
    """确保目标文件的父目录存在（不存在则递归创建）。"""
    parent = os.path.dirname(os.path.abspath(path))
    if parent and not os.path.isdir(parent):
        os.makedirs(parent, exist_ok=True)


def read_json(path: str, default: Any = None) -> Any:
    """
    读取 JSON 文件，返回 Python 对象。

    文件不存在或解析失败时返回 ``default``（默认 None），不抛异常——
    这符合“容错降级第一”原则，避免缺失数据文件让演示中断。
    """
    try:
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return default


def write_json(path: str, data: Any, *, indent: int = 2) -> bool:
    """
    写 JSON 文件：UTF-8、``ensure_ascii=False``（中文不转义，符合 SPEC §11）。

    自动创建父目录；自动把 ndarray / numpy 标量等转为原生类型（见 ``to_jsonable``）。
    成功返回 True，失败返回 False（仅打印告警，不抛出）。
    """
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(to_jsonable(data), f, ensure_ascii=False, indent=indent)
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[data_utils] 写入 JSON 失败 {path}：{exc}")
        return False


def read_yaml(path: str, default: Any = None) -> Any:
    """
    读取 YAML（配置文件）为 Python 对象。

    pyyaml 缺失或文件不存在 / 解析失败时返回 ``default``，不崩溃。
    （pyyaml 属核心依赖，正常环境必装；这里仍保留守卫以防极端情况。）
    """
    if not _YAML_AVAILABLE:
        print("[data_utils] 警告：未安装 pyyaml，无法读取 YAML 配置，返回默认值。")
        return default
    try:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f)
    except Exception:
        return default


def write_yaml(path: str, data: Any) -> bool:
    """写 YAML 文件（中文不转义）。pyyaml 缺失或异常返回 False。"""
    if not _YAML_AVAILABLE:
        print("[data_utils] 警告：未安装 pyyaml，无法写入 YAML。")
        return False
    try:
        _ensure_parent_dir(path)
        with open(path, "w", encoding="utf-8") as f:
            yaml.safe_dump(
                to_jsonable(data), f, allow_unicode=True, sort_keys=False
            )
        return True
    except Exception as exc:  # pragma: no cover
        print(f"[data_utils] 写入 YAML 失败 {path}：{exc}")
        return False


# ===========================================================================
# ndarray ↔ list 互转 / JSON 可序列化清洗
# ===========================================================================
def ndarray_to_list(arr: Any) -> Any:
    """
    将 numpy.ndarray 转为嵌套 Python list；非 ndarray 原样返回。

    numpy 缺失时直接返回入参（已经是 list/标量），保证可用。
    """
    if _NUMPY_AVAILABLE and isinstance(arr, np.ndarray):
        return arr.tolist()
    return arr


def list_to_ndarray(data: Any, dtype: Any = None) -> Any:
    """
    将 list / 嵌套序列转为 numpy.ndarray。

    numpy 缺失时返回原 list（上层应能容忍）。
    """
    if not _NUMPY_AVAILABLE:
        return data
    try:
        return np.array(data, dtype=dtype)
    except Exception:
        return np.array(data)


def to_jsonable(obj: Any) -> Any:
    """
    递归把对象转换为可被 ``json.dump`` 序列化的原生类型。

    处理：numpy ndarray/标量、集合、日期时间、以及嵌套的 dict/list/tuple。
    其他无法处理的对象退化为 ``str(obj)``，确保落库不报错（容错优先）。
    """
    # numpy 类型
    if _NUMPY_AVAILABLE:
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, np.generic):  # numpy 标量（int64/float32...）
            return obj.item()

    # 标准容器递归
    if isinstance(obj, dict):
        return {str(k): to_jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [to_jsonable(v) for v in obj]
    if isinstance(obj, set):
        return [to_jsonable(v) for v in obj]

    # 时间类型 → ISO 字符串
    if isinstance(obj, (_dt.datetime, _dt.date)):
        return obj.isoformat()

    # 原生可序列化类型
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj

    # 兜底：转字符串，绝不让序列化崩溃
    try:
        json.dumps(obj)
        return obj
    except Exception:
        return str(obj)


# ===========================================================================
# ISO8601 时间
# ===========================================================================
def now_iso() -> str:
    """返回当前本地时间的 ISO8601 字符串（秒级），如 '2026-06-30T14:25:01'。"""
    return _dt.datetime.now().replace(microsecond=0).isoformat()


def to_iso(dt: Optional[_dt.datetime] = None) -> str:
    """把 datetime 转 ISO8601 字符串；为 None 时取当前时间。"""
    if dt is None:
        dt = _dt.datetime.now()
    return dt.replace(microsecond=0).isoformat()


def parse_iso(s: Optional[str]) -> Optional[_dt.datetime]:
    """解析 ISO8601 字符串为 datetime；失败返回 None。"""
    if not s:
        return None
    try:
        return _dt.datetime.fromisoformat(s)
    except Exception:
        return None


# ===========================================================================
# 数值 / 百分比 格式化
# ===========================================================================
def format_number(
    value: Union[int, float, None], decimals: int = 2, *, default: str = "—"
) -> str:
    """
    数值格式化为字符串，保留 ``decimals`` 位小数。

    None / 非数值 / NaN 返回 ``default``（默认全角破折号），便于看板展示空值。
    """
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:  # NaN 自身不等于自身
            return default
        return f"{f:.{decimals}f}"
    except (TypeError, ValueError):
        return default


def format_percent(
    value: Union[int, float, None],
    decimals: int = 1,
    *,
    already_percent: bool = False,
    default: str = "—",
) -> str:
    """
    百分比格式化。

    - 默认把 0~1 的比例值乘以 100（如 0.83 → '83.0%'）。
    - ``already_percent=True`` 时认为入参已是 0~100 的百分数，直接加 '%'。
    - None / 非数值返回 ``default``。
    """
    if value is None:
        return default
    try:
        f = float(value)
        if f != f:
            return default
        pct = f if already_percent else f * 100.0
        return f"{pct:.{decimals}f}%"
    except (TypeError, ValueError):
        return default


# ===========================================================================
# 毫秒 → 可读时长
# ===========================================================================
def format_duration_ms(ms: Union[int, float, None]) -> str:
    """
    毫秒 → 可读中文时长字符串。

    规则：
    - None / 非数值      → '—'
    - < 1000ms          → '850ms'
    - < 60s             → '3.2s'
    - < 60min           → '2分5.0秒'
    - 否则              → '1小时2分'

    供看板（平均任务耗时/单步耗时）与日志复用。
    """
    if ms is None:
        return "—"
    try:
        ms = float(ms)
        if ms != ms or ms < 0:
            return "—"
    except (TypeError, ValueError):
        return "—"

    if ms < 1000:
        return f"{int(round(ms))}ms"

    seconds = ms / 1000.0
    if seconds < 60:
        return f"{seconds:.1f}s"

    minutes = int(seconds // 60)
    rem_sec = seconds - minutes * 60
    if minutes < 60:
        return f"{minutes}分{rem_sec:.1f}秒"

    hours = minutes // 60
    rem_min = minutes % 60
    return f"{hours}小时{rem_min}分"


# “毫秒→时长”的别名，与 viz_utils.ms_to_duration 命名保持一致，方便互换调用
ms_to_duration = format_duration_ms


# ===========================================================================
# 一些便捷聚合工具（评测/看板常用）
# ===========================================================================
def safe_div(numerator: float, denominator: float, default: float = 0.0) -> float:
    """安全除法：分母为 0 / None 时返回 default，避免 ZeroDivisionError。"""
    try:
        if not denominator:
            return default
        return numerator / denominator
    except Exception:
        return default


def mean(values: Iterable[Union[int, float]], default: float = 0.0) -> float:
    """求均值；空序列返回 default。numpy 可用则用 numpy，否则纯 Python。"""
    vals = [float(v) for v in values if v is not None]
    if not vals:
        return default
    if _NUMPY_AVAILABLE:
        return float(np.mean(vals))
    return sum(vals) / len(vals)


if __name__ == "__main__":
    # 快速自检：python -m src.utils.data_utils
    print("当前 ISO 时间 :", now_iso())
    print("0.834 -> 百分比:", format_percent(0.834))
    print("1234.5 -> 数值 :", format_number(1234.5))
    print("感知类失败 配色:", failure_color("感知类失败"))
    print("perception 配色:", failure_color("perception"))
    for v in (None, 850, 3200, 125000, 3725000):
        print(f"{v} ms ->", format_duration_ms(v))
