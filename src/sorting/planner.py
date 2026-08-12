"""分拣动作规划器（SPEC §7 planner.py）

职责：给定“VLA 已经解析好的动作序列”和“指令意图”，对动作执行顺序做优化：
1) 最近邻路径优化（Nearest-Neighbor TSP 近似）：
   按零件在工作台上的 2D 位置，从机械臂初始位（home）出发，每次贪心选最近的下一个零件，
   减少机械臂空跑距离 → 降低平均任务耗时（效率指标）。
2) 优先级处理：
   当意图为“优先级分拣”（如“优先芯片，然后电容”）时，先按 rules 中给出的优先级排序，
   同优先级内再用最近邻路径优化。批量/基础/条件分拣则纯按路径优化。

设计说明：
- 本模块只重排“与具体零件绑定的执行动作”（grasp/place/move 成组），
  对不绑定零件的动作（如 return/home）保持其相对位置。
- 纯 Python + 简单几何，无重依赖，永远可运行。
- 失败注入不在这里做；这里只决定“顺序”，顺序不合理本身也是一类规划失败的诱因。
"""

import math


# 机械臂初始位（home），最近邻路径从这里出发。与 robot.py 的 home 概念一致。
_HOME_POS = (0.0, 0.0)


def _pos_of(part: dict) -> tuple:
    """安全地取出零件的 2D 位置 (x, y)，缺失时回退到原点。"""
    pos = (part or {}).get("pos") or [0.0, 0.0]
    try:
        return float(pos[0]), float(pos[1])
    except (TypeError, ValueError, IndexError):
        return 0.0, 0.0


def _dist(a: tuple, b: tuple) -> float:
    """两点欧氏距离。"""
    return math.hypot(a[0] - b[0], a[1] - b[1])


def nearest_neighbor_order(parts: list) -> list:
    """最近邻路径优化：返回按贪心最近邻排好序的零件列表。

    从 home 出发，每一步选离“当前位置”最近的未访问零件。
    时间复杂度 O(n^2)，对本场景（≤10 件）完全够用。
    """
    remaining = list(parts or [])
    ordered = []
    cur = _HOME_POS
    while remaining:
        # 选离当前位置最近的零件
        nxt = min(remaining, key=lambda p: _dist(cur, _pos_of(p)))
        ordered.append(nxt)
        cur = _pos_of(nxt)
        remaining.remove(nxt)
    return ordered


def _priority_rank(part: dict, priority_codes: list) -> int:
    """返回零件在优先级列表中的排名；不在列表中的排到最后。"""
    code = (part or {}).get("code")
    if code in priority_codes:
        return priority_codes.index(code)
    return len(priority_codes)  # 未指定优先级的统一排在后面


def _extract_priority_codes(intent: dict) -> list:
    """从 parsed_intent 中提取优先级零件 code 列表（尽量宽容地解析）。

    兼容多种 rules 结构：
    - rules 里直接是 code 字符串列表：["chip","capacitor"]
    - rules 里是 dict：{"priority":["chip","capacitor"]} 或 {"order":[...]}
    - intent 顶层带 priority/order 字段
    """
    if not isinstance(intent, dict):
        return []
    # 顶层直给
    for key in ("priority", "order", "priority_codes"):
        val = intent.get(key)
        if isinstance(val, list) and val:
            return [c for c in val if isinstance(c, str)]
    # 从 rules 里找
    rules = intent.get("rules") or []
    codes = []
    for r in rules:
        if isinstance(r, str):
            codes.append(r)
        elif isinstance(r, dict):
            for key in ("priority", "order", "code", "part_code"):
                v = r.get(key)
                if isinstance(v, list):
                    codes.extend([c for c in v if isinstance(c, str)])
                elif isinstance(v, str):
                    codes.append(v)
    return codes


def plan_order(parts: list, instruction_intent: dict | None = None) -> list:
    """对零件做执行排序，返回重排后的零件列表。

    参数：
        parts: 待分拣零件列表（SceneState.parts 的子集），每项含 pos/code 等。
        instruction_intent: ActionPlan.parsed_intent，含 mode/rules。
            mode == "优先级" 时启用优先级排序。

    返回：
        排好序的零件列表（引擎据此生成有序 action 流）。
    """
    parts = list(parts or [])
    if not parts:
        return []

    intent = instruction_intent or {}
    mode = intent.get("mode", "")

    if mode in ("优先级", "priority"):
        priority_codes = _extract_priority_codes(intent)
        # 1) 先按优先级分组，再在每组内做最近邻
        #    稳定性：用 (优先级名次) 作为主键，组内最近邻作为次序
        if priority_codes:
            # 分组：保持优先级名次升序
            buckets: dict[int, list] = {}
            for p in parts:
                rank = _priority_rank(p, priority_codes)
                buckets.setdefault(rank, []).append(p)
            ordered = []
            for rank in sorted(buckets.keys()):
                ordered.extend(nearest_neighbor_order(buckets[rank]))
            return ordered

    # 默认（基础/条件/批量/模糊）：纯最近邻路径优化
    return nearest_neighbor_order(parts)


def plan_order_actions(actions: list, instruction_intent: dict | None = None) -> list:
    """（便捷重载）直接对 action 列表排序。

    当上游已经把动作展开为 action 列表（每个 action 带 part_code/part_id 与隐含位置）时，
    可用本函数。内部按 action 携带的 params.pos 或 part_code 关联位置后排序。
    引擎主流程使用 plan_order(parts, intent) 即可，本函数为兼容备用。
    """
    actions = list(actions or [])
    if not actions:
        return []

    intent = instruction_intent or {}
    mode = intent.get("mode", "")

    def _action_pos(a: dict) -> tuple:
        params = a.get("params") or {}
        pos = params.get("pos") or [0.0, 0.0]
        try:
            return float(pos[0]), float(pos[1])
        except (TypeError, ValueError, IndexError):
            return 0.0, 0.0

    # 优先级
    if mode in ("优先级", "priority"):
        priority_codes = _extract_priority_codes(intent)
        if priority_codes:
            def _rank(a):
                code = a.get("part_code")
                return priority_codes.index(code) if code in priority_codes else len(priority_codes)
            actions = sorted(actions, key=_rank)
            return actions

    # 最近邻（基于 action 携带位置）
    remaining = list(actions)
    ordered = []
    cur = _HOME_POS
    while remaining:
        nxt = min(remaining, key=lambda a: _dist(cur, _action_pos(a)))
        ordered.append(nxt)
        cur = _action_pos(nxt)
        remaining.remove(nxt)
    return ordered
