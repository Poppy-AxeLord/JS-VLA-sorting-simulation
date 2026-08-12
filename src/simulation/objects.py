# -*- coding: utf-8 -*-
"""
零件库与场景生成（src/simulation/objects.py）
=============================================

本文件是 **10 种 3C 零件目录（PARTS）的唯一权威来源**（对应 SPEC §2），
并镜像到 ``assets/objects/parts_catalog.json``。其它任何模块（VLA 规则、
分拣引擎、看板）涉及零件属性时，都应从这里读取，避免字段漂移。

主要内容：
- ``PARTS``               : 10 种零件的完整属性字典列表（权威）。
- ``BINS``                : 3 个分拣料盒（A/B/C 区）的元数据。
- ``DIFFICULTY_PROFILE``  : 三档难度（简单/中等/困难）的采样规则。
- ``generate_scene(difficulty, seed)`` : 按难度采样零件、位姿、遮挡，返回
  一个 ``scene_config`` 字典（场景配置，描述"放哪些零件、放在哪里"）。
- ``build_scene_state(scene_config)``  : 由 scene_config 构造运行期 SceneState
  （含真值，供规则 VLA / 评分使用），结构符合 SPEC §5。

【产品/工程决策说明】
为什么把"零件目录"做成单一权威来源？因为整个评测体系（失败注入、规则
匹配、看板配色）都依赖零件属性（颜色/材质/大小/易碎）。一旦各处各写一份，
极易出现"规则按银色匹配、场景却给了灰色"这类隐蔽 bug。集中后，新增/调整
零件只改一处，assets 里的 JSON 仅作展示与外部消费的镜像。

坐标系约定（与 env_config.yaml / scene.xml 对齐的逻辑坐标，单位米）：
- 工作台/传送带在 x∈[-0.35, 0.35], y∈[-0.25, 0.25] 的区域。
- 料盒在工作台前方一字排开：A 区在左、B 区居中、C 区在右。
- 零件初始散落在工作台抓取区 (pos=[x, y])，只用 2D 平面坐标即可满足
  Mock 运动学与合成俯视图需求；MuJoCo 后端会在此基础上补 z 高度。
"""

from __future__ import annotations

import random
from typing import Dict, List, Optional

# numpy 属核心依赖，可直接使用（用于位姿采样的少量数值处理）。
import numpy as np


# ---------------------------------------------------------------------------
# 1. 零件库（PARTS）—— SPEC §2 的 10 种零件，权威来源
# ---------------------------------------------------------------------------
# 字段说明：
#   code     : 英文唯一标识（程序 key）
#   name     : 中文名（界面展示）
#   material : 材质（金属/陶瓷/塑料/复合/玻璃）—— 影响"金属零件分拣"等条件指令
#   color    : 颜色中文（银色/蓝色/棕色/黑色/白色/绿色/红色）—— 影响颜色指令与配色
#   color_hex: 颜色十六进制（合成俯视图色块用，避免每处自行猜色）
#   size     : 大小（小/中/大）—— 影响"大的放左、小的放右"模糊指令、夹爪开度
#   shape    : 形状（圆柱/六边形/方形/块状/平板）
#   fragile  : 是否易碎（bool）—— 易碎件抬高放置失败率（失败注入模型用）
#   weight   : 重量（克，浮点）—— 影响抓取稳定性的启发式
PARTS: List[Dict] = [
    {
        "code": "screw", "name": "螺丝", "material": "金属", "color": "银色",
        "color_hex": "#C0C0C0", "size": "小", "shape": "圆柱",
        "fragile": False, "weight": 2.0,
    },
    {
        "code": "nut", "name": "螺母", "material": "金属", "color": "银色",
        "color_hex": "#A8A8A8", "size": "小", "shape": "六边形",
        "fragile": False, "weight": 1.5,
    },
    {
        "code": "capacitor", "name": "电容", "material": "金属", "color": "蓝色",
        "color_hex": "#2D6CDF", "size": "小", "shape": "圆柱",
        "fragile": False, "weight": 1.0,
    },
    {
        "code": "resistor", "name": "电阻", "material": "陶瓷", "color": "棕色",
        "color_hex": "#8B5A2B", "size": "小", "shape": "圆柱",
        "fragile": False, "weight": 0.5,
    },
    {
        "code": "chip", "name": "芯片", "material": "塑料", "color": "黑色",
        "color_hex": "#2B2B2B", "size": "中", "shape": "方形",
        "fragile": True, "weight": 3.0,
    },
    {
        "code": "connector", "name": "连接器", "material": "塑料", "color": "白色",
        "color_hex": "#F0F0F0", "size": "中", "shape": "方形",
        "fragile": False, "weight": 4.0,
    },
    {
        "code": "heatsink", "name": "散热器", "material": "金属", "color": "银色",
        "color_hex": "#9FA6AD", "size": "大", "shape": "块状",
        "fragile": False, "weight": 25.0,
    },
    {
        "code": "pcb", "name": "PCB板", "material": "复合", "color": "绿色",
        "color_hex": "#2E8B57", "size": "大", "shape": "平板",
        "fragile": True, "weight": 15.0,
    },
    {
        "code": "button", "name": "按键", "material": "塑料", "color": "红色",
        "color_hex": "#E8453C", "size": "小", "shape": "方形",
        "fragile": False, "weight": 0.8,
    },
    {
        "code": "display", "name": "显示屏", "material": "玻璃", "color": "黑色",
        "color_hex": "#1A1A1A", "size": "大", "shape": "平板",
        "fragile": True, "weight": 30.0,
    },
]

# code -> 零件属性 的快速索引（只读）。
PARTS_BY_CODE: Dict[str, Dict] = {p["code"]: p for p in PARTS}


# ---------------------------------------------------------------------------
# 2. 分拣料盒（BINS）—— A/B/C 三区
# ---------------------------------------------------------------------------
# key 用 'A'/'B'/'C'，name 用 'A区'/'B区'/'C区'。pos 为料盒中心逻辑坐标，
# 供 Mock 运动学把零件"放置"到对应料盒、以及合成俯视图绘制料盒区域。
BINS: Dict[str, Dict] = {
    "A": {"key": "A", "name": "A区", "pos": [-0.28, -0.34], "color_hex": "#2563EB"},
    "B": {"key": "B", "name": "B区", "pos": [0.00, -0.34], "color_hex": "#5AD8A6"},
    "C": {"key": "C", "name": "C区", "pos": [0.28, -0.34], "color_hex": "#F6BD16"},
}


# ---------------------------------------------------------------------------
# 3. 难度分级采样规则（SPEC §2 / env_config.yaml）
# ---------------------------------------------------------------------------
# 每档难度定义：
#   count        : 场景零件数
#   occlusion    : 是否允许遮挡（困难档抬高感知失败率）
#   occ_prob     : 单个零件被标记遮挡的概率（仅 occlusion=True 时生效）
#   similar_boost: 是否倾向于采样"相似物"（如螺丝/螺母同为银色小件），
#                  用于制造"识别错误"压力（中等档及以上开启）
DIFFICULTY_PROFILE: Dict[str, Dict] = {
    "简单": {"count": 3, "occlusion": False, "occ_prob": 0.0, "similar_boost": False},
    "中等": {"count": 5, "occlusion": False, "occ_prob": 0.0, "similar_boost": True},
    "困难": {"count": 8, "occlusion": True, "occ_prob": 0.35, "similar_boost": True},
}

# 同时接受英文别名，方便配置/命令行传入。
_DIFFICULTY_ALIAS = {
    "simple": "简单", "easy": "简单", "简单": "简单",
    "medium": "中等", "mid": "中等", "中等": "中等",
    "hard": "困难", "difficult": "困难", "困难": "困难",
}

# 相似物分组（同色同档，易混淆）—— 供 similar_boost 采样。
_SIMILAR_GROUPS = [
    ["screw", "nut"],              # 银色小件，最经典的相似物
    ["capacitor", "resistor"],     # 同为小圆柱
    ["pcb", "display", "heatsink"],  # 大件，平板/块状
]


def normalize_difficulty(difficulty: str) -> str:
    """把外部传入的难度字符串归一化为中文标准档位（简单/中等/困难）。

    容错：无法识别时默认回退到"简单"，并不抛异常（保持降级第一原则）。
    """
    if not difficulty:
        return "简单"
    return _DIFFICULTY_ALIAS.get(str(difficulty).strip().lower(), None) \
        or _DIFFICULTY_ALIAS.get(str(difficulty).strip(), "简单")


# ---------------------------------------------------------------------------
# 4. 位姿采样
# ---------------------------------------------------------------------------
# 抓取区范围（逻辑坐标，单位米）；料盒在 y<-0.3 处，零件散落在 y∈[-0.18,0.22]。
_PICK_AREA_X = (-0.30, 0.30)
_PICK_AREA_Y = (-0.18, 0.22)
_MIN_DIST = 0.06  # 零件之间最小间距，避免完全重叠（遮挡用专门标记表达）


def _sample_positions(n: int, rng: random.Random) -> List[List[float]]:
    """在抓取区内为 n 个零件采样互不重叠的 2D 位姿。

    采用简单的"拒绝采样"：随机点若离已有点太近就重采，最多尝试若干次后
    放宽约束直接接受（保证函数一定返回，不会死循环）。这是 POC 场景，
    不追求严格泊松盘采样，够用即可。
    """
    positions: List[List[float]] = []
    for _ in range(n):
        placed = False
        for _attempt in range(40):
            x = rng.uniform(*_PICK_AREA_X)
            y = rng.uniform(*_PICK_AREA_Y)
            if all((x - px) ** 2 + (y - py) ** 2 >= _MIN_DIST ** 2
                   for px, py in positions):
                positions.append([round(x, 4), round(y, 4)])
                placed = True
                break
        if not placed:
            # 放宽约束：实在挤不下就直接放（POC 容忍轻微靠近）。
            positions.append([
                round(rng.uniform(*_PICK_AREA_X), 4),
                round(rng.uniform(*_PICK_AREA_Y), 4),
            ])
    return positions


def _sample_part_codes(profile: Dict, rng: random.Random) -> List[str]:
    """按难度档位采样零件 code 列表。

    - similar_boost=True 时，优先放入一组相似物（制造识别错误压力），
      再随机补足剩余数量。
    - 允许同一 code 出现多次（现实中同种零件常有多个），用 part_id 区分。
    """
    count = profile["count"]
    all_codes = [p["code"] for p in PARTS]
    chosen: List[str] = []

    if profile.get("similar_boost") and count >= 2:
        group = rng.choice(_SIMILAR_GROUPS)
        # 放入该相似组里的 2 个（若组内更多则取 2 个），制造混淆样本。
        take = group[:2] if len(group) >= 2 else group
        chosen.extend(take)

    while len(chosen) < count:
        chosen.append(rng.choice(all_codes))

    # 截断到精确数量，并打乱顺序（避免相似物总在最前）。
    chosen = chosen[:count]
    rng.shuffle(chosen)
    return chosen


def generate_scene(difficulty: str = "简单",
                   seed: Optional[int] = None) -> Dict:
    """按难度生成一个场景配置（scene_config）。

    参数
    ----
    difficulty : 难度（简单/中等/困难，或英文别名 simple/medium/hard）。
    seed       : 随机种子；传入则结果可复现（benchmark 复现的关键）。

    返回
    ----
    scene_config : dict，结构如下（描述"摆了什么、摆在哪、谁被遮挡"）::

        {
          "difficulty": "中等",
          "seed": 123,
          "occlusion": False,
          "bins": {"A": {...}, "B": {...}, "C": {...}},
          "parts": [
             {"part_id": 0, "code": "screw", "pos": [x, y], "occluded": False},
             ...
          ]
        }

    注意：scene_config 只含"摆放信息 + part_id + code + 遮挡标记"，零件的
    静态属性（颜色/材质等）不在此冗余，运行期由 build_scene_state 从 PARTS
    回填，保证单一权威来源。
    """
    diff = normalize_difficulty(difficulty)
    profile = DIFFICULTY_PROFILE[diff]
    # 用独立的 random.Random 实例，避免污染全局随机状态（线程/复现更安全）。
    rng = random.Random(seed)

    codes = _sample_part_codes(profile, rng)
    positions = _sample_positions(len(codes), rng)

    parts: List[Dict] = []
    for idx, code in enumerate(codes):
        occluded = False
        if profile["occlusion"] and rng.random() < profile["occ_prob"]:
            occluded = True
        parts.append({
            "part_id": idx,
            "code": code,
            "pos": positions[idx],
            "occluded": occluded,
        })

    scene_config = {
        "difficulty": diff,
        "seed": seed,
        "occlusion": bool(profile["occlusion"]),
        # 深拷贝料盒元数据，避免外部修改污染模块级常量。
        "bins": {k: dict(v) for k, v in BINS.items()},
        "parts": parts,
    }
    return scene_config


def build_scene_state(scene_config: Dict) -> Dict:
    """由 scene_config 构造运行期 SceneState（含零件真值，供规则/评分）。

    SceneState 结构（SPEC §5）::

        {
          "parts": [
            {"part_id", "code", "name", "material", "color", "color_hex",
             "size", "shape", "fragile", "pos":[x,y], "occluded":bool},
            ...
          ],
          "bins": {"A": {...}, "B": {...}, "C": {...}}
        }

    与 scene_config 的区别：这里把 PARTS 里的静态属性回填进每个零件，得到
    一个"自包含"的快照，便于规则 VLA 直接读取、便于序列化进 task_results。
    对未知 code（理论上不会出现）做容错：跳过该零件并不崩。
    """
    parts: List[Dict] = []
    for p in scene_config.get("parts", []):
        meta = PARTS_BY_CODE.get(p.get("code"))
        if meta is None:
            # 容错：未知零件 code 直接跳过，保证不因脏数据崩溃。
            continue
        parts.append({
            "part_id": p.get("part_id"),
            "code": meta["code"],
            "name": meta["name"],
            "material": meta["material"],
            "color": meta["color"],
            "color_hex": meta["color_hex"],
            "size": meta["size"],
            "shape": meta["shape"],
            "fragile": bool(meta["fragile"]),
            "weight": meta["weight"],
            "pos": list(p.get("pos", [0.0, 0.0])),
            "occluded": bool(p.get("occluded", False)),
        })

    bins = scene_config.get("bins") or {k: dict(v) for k, v in BINS.items()}
    return {"parts": parts, "bins": bins}


def part_by_code(code: str) -> Optional[Dict]:
    """按 code 取零件静态属性（只读副本）；未知返回 None。"""
    meta = PARTS_BY_CODE.get(code)
    return dict(meta) if meta else None


def list_part_codes() -> List[str]:
    """返回全部零件 code 列表（顺序与 PARTS 一致）。"""
    return [p["code"] for p in PARTS]


if __name__ == "__main__":
    # 简易自检：分别生成三档场景并打印摘要，验证采样与构造逻辑可用。
    for d in ("简单", "中等", "困难"):
        sc = generate_scene(d, seed=42)
        ss = build_scene_state(sc)
        names = "、".join(f"{p['name']}({'遮挡' if p['occluded'] else '可见'})"
                          for p in ss["parts"])
        print(f"[{d}] 零件数={len(ss['parts'])} 遮挡场景={sc['occlusion']} -> {names}")
