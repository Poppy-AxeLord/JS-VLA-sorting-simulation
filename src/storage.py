"""存储层（SQLite + JSON）—— 工业分拣 VLA 评测分析系统。

设计要点（与 SPEC §11 严格对齐）：
- 仅依赖标准库 ``sqlite3``，零重依赖，永远可运行。
- DB 文件固定为 ``data/app.db``（相对项目根，用 __file__ 计算，避免受运行目录影响）。
- 三张表：``benchmark_runs`` / ``task_results`` / ``failure_cases``。
- ``seed_demo()``：看板"开箱即用"的关键。首次启动（benchmark_runs 为空）时，
  用 ``random.seed(42)`` 确定性地生成 v1/v2/v3 三次历史评测，
  成功率约 0.62 → 0.74 → 0.83，覆盖全部 30 任务与 5 类失败。

【产品决策】为什么 seed_demo 不直接调用真实 SortingEngine？
  因为演示数据必须在"只装核心依赖（无 mujoco/torch）"时也能生成，且要
  确定性可复现、瞬间完成、稳定覆盖 5 类失败分布。因此这里用一个**与 engine 理念
  一致、但独立实现的轻量合成器**（``_make_failure_result`` / ``_make_success_result``）
  "离线合成"任务结果——它不复用 ``engine._inject_failure``，而是复刻同一套 5 类失败
  分类体系与「越难越易失败」的相关性，用更少的代码换取确定性与可控性；真实评测仍走
  benchmark.py（那里才真正调用 SortingEngine 的失败注入模型）。
  注：合成器与 engine 是「两处实现、一套理念」，改动 5 类失败分类时需同步二者。
"""

from __future__ import annotations

import json
import os
import random
import sqlite3
from datetime import datetime, timedelta
from typing import Any, Dict, List, Optional

# --------------------------------------------------------------------------- #
# 路径常量：统一以本文件位置回溯到项目根，保证从任意 cwd 运行都能定位 data/      #
# --------------------------------------------------------------------------- #
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))          # .../src
PROJECT_ROOT = os.path.dirname(_THIS_DIR)                        # 项目根目录
DATA_DIR = os.path.join(PROJECT_ROOT, "data")
DB_PATH = os.path.join(DATA_DIR, "app.db")
FAILURE_CASES_DIR = os.path.join(DATA_DIR, "failure_cases")
BENCHMARK_RESULTS_DIR = os.path.join(DATA_DIR, "benchmark_results")
DEMO_DATA_PATH = os.path.join(DATA_DIR, "demo_data.json")


def _now_iso() -> str:
    """返回 ISO8601 时间字符串（秒级，全项目统一时间格式）。"""
    return datetime.now().replace(microsecond=0).isoformat()


def _ensure_dirs() -> None:
    """确保 data/ 及其子目录存在（首次运行自动创建）。"""
    for d in (DATA_DIR, FAILURE_CASES_DIR, BENCHMARK_RESULTS_DIR):
        os.makedirs(d, exist_ok=True)


# --------------------------------------------------------------------------- #
# 连接与建表                                                                    #
# --------------------------------------------------------------------------- #
def get_conn() -> sqlite3.Connection:
    """获取 SQLite 连接，row_factory=Row 便于按列名取值。

    每次调用返回新连接（SQLite 轻量、看板/评测均短连接使用）。
    """
    _ensure_dirs()
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    # 外键与并发友好设置（演示场景足够）
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


def init_db() -> None:
    """初始化三张表（幂等，已存在则跳过）。"""
    _ensure_dirs()
    conn = get_conn()
    try:
        cur = conn.cursor()
        # 评测批次表：一次完整评测（一个版本）对应一行，含汇总指标
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS benchmark_runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                version         TEXT,
                vla_backend     TEXT,
                strategy        TEXT,
                world_model_on  INTEGER,
                total_tasks     INTEGER,
                success_rate    REAL,
                sort_accuracy   REAL,
                avg_duration_ms INTEGER,
                throughput      REAL,
                metrics_json    TEXT,
                created_at      TEXT
            )
            """
        )
        # 任务结果表：每个任务一行，含成功/失败与失败归类
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS task_results (
                id                 INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id             INTEGER,
                task_id            TEXT,
                instruction        TEXT,
                type               TEXT,
                difficulty         TEXT,
                vla_backend        TEXT,
                success            INTEGER,
                sort_accuracy      REAL,
                mis_pick           INTEGER,
                miss_pick          INTEGER,
                duration_ms        INTEGER,
                retry_count        INTEGER,
                collisions         INTEGER,
                human_intervention INTEGER,
                recovered          INTEGER,
                failure_category   TEXT,
                failure_subtype    TEXT,
                failure_reason     TEXT,
                detail_json        TEXT,
                created_at         TEXT
            )
            """
        )
        # 失败案例表：仅失败任务的镜像，含模型输出与场景快照，供失败分析钻取
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS failure_cases (
                id               INTEGER PRIMARY KEY AUTOINCREMENT,
                run_id           INTEGER,
                task_id          TEXT,
                instruction      TEXT,
                difficulty       TEXT,
                failure_category TEXT,
                failure_subtype  TEXT,
                failure_reason   TEXT,
                model_output     TEXT,
                scene_json       TEXT,
                created_at       TEXT
            )
            """
        )
        conn.commit()
    finally:
        conn.close()


# --------------------------------------------------------------------------- #
# 写入：保存一次评测（run + task_results + failure_cases）                      #
# --------------------------------------------------------------------------- #
def save_run(summary: Dict[str, Any], task_results: List[Dict[str, Any]]) -> int:
    """保存一次完整评测，返回新 run 的自增 id。

    参数
    ----
    summary : dict
        run 级汇总，至少含 version/vla_backend/strategy/world_model_on/
        total_tasks/success_rate/sort_accuracy/avg_duration_ms/throughput/
        metrics（dict，将序列化进 metrics_json）。
    task_results : list[dict]
        TaskResult 列表（结构见 SPEC §7）。失败任务自动镜像进 failure_cases。
    """
    init_db()
    conn = get_conn()
    try:
        cur = conn.cursor()
        created_at = summary.get("created_at") or _now_iso()
        metrics = summary.get("metrics", {}) or {}

        cur.execute(
            """
            INSERT INTO benchmark_runs
                (version, vla_backend, strategy, world_model_on, total_tasks,
                 success_rate, sort_accuracy, avg_duration_ms, throughput,
                 metrics_json, created_at)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                summary.get("version"),
                summary.get("vla_backend"),
                summary.get("strategy"),
                1 if summary.get("world_model_on") else 0,
                int(summary.get("total_tasks", len(task_results))),
                float(summary.get("success_rate", 0.0)),
                float(summary.get("sort_accuracy", 0.0)),
                int(summary.get("avg_duration_ms", 0)),
                float(summary.get("throughput", 0.0)),
                json.dumps(metrics, ensure_ascii=False),
                created_at,
            ),
        )
        run_id = int(cur.lastrowid)

        for tr in task_results:
            _insert_task_result(cur, run_id, tr, created_at)
            # 失败任务镜像到 failure_cases
            if not tr.get("success", False) and tr.get("failure_category"):
                _insert_failure_case(cur, run_id, tr, created_at)

        conn.commit()
        return run_id
    finally:
        conn.close()


def _insert_task_result(cur: sqlite3.Cursor, run_id: int,
                        tr: Dict[str, Any], default_ts: str) -> None:
    """把单个 TaskResult 写入 task_results 表（detail_json 存完整结构）。"""
    cur.execute(
        """
        INSERT INTO task_results
            (run_id, task_id, instruction, type, difficulty, vla_backend,
             success, sort_accuracy, mis_pick, miss_pick, duration_ms,
             retry_count, collisions, human_intervention, recovered,
             failure_category, failure_subtype, failure_reason,
             detail_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            tr.get("task_id"),
            tr.get("instruction"),
            tr.get("type"),
            tr.get("difficulty"),
            tr.get("vla_backend"),
            1 if tr.get("success") else 0,
            float(tr.get("sort_accuracy", 0.0)),
            int(tr.get("mis_pick", 0)),
            int(tr.get("miss_pick", 0)),
            int(tr.get("duration_ms", 0)),
            int(tr.get("retry_count", 0)),
            int(tr.get("collisions", 0)),
            int(tr.get("human_intervention", 0)),
            int(tr.get("recovered", 0)),
            tr.get("failure_category"),
            tr.get("failure_subtype"),
            tr.get("failure_reason"),
            json.dumps(tr, ensure_ascii=False, default=_json_default),
            tr.get("created_at") or default_ts,
        ),
    )


def _insert_failure_case(cur: sqlite3.Cursor, run_id: int,
                         tr: Dict[str, Any], default_ts: str) -> None:
    """把失败 TaskResult 镜像写入 failure_cases 表。"""
    model_output = tr.get("action_plan")
    cur.execute(
        """
        INSERT INTO failure_cases
            (run_id, task_id, instruction, difficulty, failure_category,
             failure_subtype, failure_reason, model_output, scene_json, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (
            run_id,
            tr.get("task_id"),
            tr.get("instruction"),
            tr.get("difficulty"),
            tr.get("failure_category"),
            tr.get("failure_subtype"),
            tr.get("failure_reason"),
            json.dumps(model_output, ensure_ascii=False, default=_json_default)
            if model_output is not None else None,
            json.dumps(tr.get("scene"), ensure_ascii=False, default=_json_default)
            if tr.get("scene") is not None else None,
            tr.get("created_at") or default_ts,
        ),
    )


def _json_default(obj: Any) -> Any:
    """json.dumps 兜底：把 numpy 标量/数组等转为原生类型，避免序列化崩溃。"""
    # 不强依赖 numpy；用 duck-typing 处理
    if hasattr(obj, "tolist"):
        try:
            return obj.tolist()
        except Exception:  # pragma: no cover - 极端兜底
            return str(obj)
    if hasattr(obj, "item"):
        try:
            return obj.item()
        except Exception:  # pragma: no cover
            return str(obj)
    return str(obj)


# --------------------------------------------------------------------------- #
# 读取                                                                          #
# --------------------------------------------------------------------------- #
def list_runs() -> List[Dict[str, Any]]:
    """按时间正序返回所有评测批次（便于画趋势：v1→v2→v3）。"""
    init_db()
    conn = get_conn()
    try:
        rows = conn.execute(
            # 看板侧边栏默认展示“最近一次运行”，因此存储层统一按最新优先返回；
            # 版本对比页会在自身语境中显式反转为 v1 → v2 → v3 的时间正序。
            "SELECT * FROM benchmark_runs ORDER BY id DESC"
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def get_run(run_id: int) -> Optional[Dict[str, Any]]:
    """按 id 获取单次评测汇总（含解析后的 metrics dict），不存在返回 None。"""
    init_db()
    conn = get_conn()
    try:
        row = conn.execute(
            "SELECT * FROM benchmark_runs WHERE id = ?", (run_id,)
        ).fetchone()
        if row is None:
            return None
        d = _row_to_dict(row)
        d["metrics"] = _safe_json_loads(d.get("metrics_json"), {})
        return d
    finally:
        conn.close()


def get_task_results(run_id: int) -> List[Dict[str, Any]]:
    """获取某次评测的全部任务结果（detail_json 解析为 detail 字段）。"""
    init_db()
    conn = get_conn()
    try:
        rows = conn.execute(
            "SELECT * FROM task_results WHERE run_id = ? ORDER BY id ASC",
            (run_id,),
        ).fetchall()
        results = []
        for r in rows:
            d = _row_to_dict(r)
            d["detail"] = _safe_json_loads(d.get("detail_json"), {})
            results.append(d)
        return results
    finally:
        conn.close()


def get_failure_cases(filters: Optional[Dict[str, Any]] = None) -> List[Dict[str, Any]]:
    """按过滤条件获取失败案例。

    支持的 filters 键：run_id / failure_category / difficulty / task_id。
    其余键忽略，避免看板传入未知键时崩溃。
    """
    init_db()
    filters = filters or {}
    allowed = {"run_id", "failure_category", "difficulty", "task_id"}
    clauses, params = [], []
    for key, val in filters.items():
        if key in allowed and val is not None and val != "全部":
            clauses.append(f"{key} = ?")
            params.append(val)
    where = (" WHERE " + " AND ".join(clauses)) if clauses else ""
    conn = get_conn()
    try:
        rows = conn.execute(
            f"SELECT * FROM failure_cases{where} ORDER BY id ASC", params
        ).fetchall()
        return [_row_to_dict(r) for r in rows]
    finally:
        conn.close()


def _row_to_dict(row: sqlite3.Row) -> Dict[str, Any]:
    """sqlite3.Row -> 普通 dict。"""
    return {k: row[k] for k in row.keys()}


def _safe_json_loads(text: Optional[str], default: Any) -> Any:
    """容错 JSON 解析：失败返回 default，绝不抛出。"""
    if not text:
        return default
    try:
        return json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return default


# --------------------------------------------------------------------------- #
# 演示数据播种（看板开箱即用的核心）                                            #
# --------------------------------------------------------------------------- #
# 三个版本的种子配置（与 data/demo_data.json 对应；base_success_rate 为目标成功率）
_DEMO_VERSIONS = [
    {
        "version": "v1",
        "vla_backend": "rule_based",
        "strategy": "baseline",
        "world_model_on": False,
        "base_success_rate": 0.62,
        "note": "纯规则基线：rule_based + baseline 策略，无世界模型。",
    },
    {
        "version": "v2",
        "vla_backend": "rule_based",
        "strategy": "optimized",
        "world_model_on": False,
        "base_success_rate": 0.74,
        "note": "优化策略：置信度阈值 + 失败重试 + 异常恢复，成功率显著提升。",
    },
    {
        "version": "v3",
        "vla_backend": "rule_based",
        "strategy": "optimized",
        "world_model_on": True,
        "base_success_rate": 0.83,
        "note": "优化策略 + 世界模型：抓取前风险评估，进一步降低执行/感知失败。",
    },
]


def seed_demo(force: bool = False) -> bool:
    """若 benchmark_runs 为空（或 force），确定性生成 3 次历史评测。

    返回 True 表示本次执行了播种；False 表示已有数据、跳过。

    实现：用 ``random.seed(42)`` 固定随机序列，依据每个版本的目标成功率与
    难度/类型分布，离线合成 30 个 TaskResult，再调用 metrics/failure_analysis
    汇总后写入三张表。失败结果稳定覆盖 5 类失败分类。
    """
    init_db()
    if not force and list_runs():
        return False  # 已有数据，幂等跳过

    # 延迟导入避免循环依赖（storage 是底层，metrics/failure_analysis 依赖 storage 无关项）
    from src.evaluation.metrics import compute_metrics
    from src.evaluation.failure_analysis import analyze  # noqa: F401 (覆盖率自验证)

    random.seed(42)  # 确定性：每次播种得到完全一致的数据

    tasks = _demo_tasks()  # 30 个任务定义（标题/类型/难度/目标）

    for vcfg in _DEMO_VERSIONS:
        task_results = _synthesize_run(vcfg, tasks)
        metrics = compute_metrics(task_results)
        summary = {
            "version": vcfg["version"],
            "vla_backend": vcfg["vla_backend"],
            "strategy": vcfg["strategy"],
            "world_model_on": vcfg["world_model_on"],
            "total_tasks": len(task_results),
            "success_rate": metrics.get("task_success_rate", 0.0),
            "sort_accuracy": metrics.get("sort_accuracy", 0.0),
            "avg_duration_ms": int(metrics.get("avg_task_ms", 0)),
            "throughput": metrics.get("throughput_per_min", 0.0),
            "metrics": metrics,
            "created_at": vcfg.get("_created_at"),
        }
        save_run(summary, task_results)
    return True


# 5 大类失败的英文 key → (大类中文, [子类中文...])，与 SPEC §3 完全一致
_FAILURE_TAXONOMY = {
    "perception": ("感知类失败", ["识别错误", "定位不准", "遮挡看不见", "光照角度问题"]),
    "understanding": ("理解类失败", ["指令理解错误", "漏理解约束", "歧义处理失败"]),
    "planning": ("规划类失败", ["路径不合理", "分拣顺序错误", "优先级处理错误"]),
    "execution": ("执行类失败", ["抓取失败(滑落)", "放置失败(掉落)", "碰撞导致失败"]),
    "environment": ("环境类失败", ["物体意外移动", "障碍物出现", "仿真物理异常"]),
}

# 不同指令类型更易触发的失败大类（让失败分布与任务类型有合理相关性）
_TYPE_FAILURE_WEIGHT = {
    "基础分拣": {"perception": 3, "execution": 3, "planning": 1, "understanding": 1, "environment": 1},
    "条件分拣": {"understanding": 3, "perception": 2, "planning": 2, "execution": 2, "environment": 1},
    "优先级分拣": {"planning": 3, "understanding": 2, "execution": 2, "perception": 1, "environment": 1},
    "批量分拣": {"planning": 3, "execution": 2, "perception": 2, "understanding": 1, "environment": 1},
    "异常场景": {"environment": 3, "execution": 2, "perception": 2, "planning": 1, "understanding": 1},
}


def _weighted_choice(weight_map: Dict[str, int]) -> str:
    """按权重字典随机选 key（用 random，受 seed 控制以保证可复现）。"""
    keys = list(weight_map.keys())
    weights = [max(0, weight_map[k]) for k in keys]
    return random.choices(keys, weights=weights, k=1)[0]


def _synthesize_run(vcfg: Dict[str, Any], tasks: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """为一个版本合成 30 个 TaskResult（确定性，依赖外层已设的 seed）。

    【关键产品决策：用"目标成功数"而非独立伯努利采样】
    30 个任务样本量小，若对每个任务独立按概率投硬币，实际成功率会因方差明显偏离
    目标（且三版本差距可能被压缩）。为保证演示数据稳定呈现 v1≈0.62→v2≈0.74→v3≈0.83
    的单调提升，这里改为：先按目标成功率算出"应成功的任务数"，再按难度给每个任务打
    一个"失败倾向分"（困难>中等>简单，叠加 seed 控制的随机抖动），倾向分最高的若干任务
    判为失败、其余成功。既精确命中目标成功率，又保留"越难越易失败"的真实相关性。
    """
    base = vcfg["base_success_rate"]
    n = len(tasks)
    # 目标成功任务数（四舍五入），夹在 [0, n]
    target_success = max(0, min(n, round(base * n)))

    # 为每个任务计算"失败倾向分"：难度越高越易失败 + seed 控制的随机抖动
    fail_rank = {"困难": 0.30, "中等": 0.12, "简单": -0.10}
    scored = []
    for idx, task in enumerate(tasks):
        score = fail_rank.get(task["difficulty"], 0.10) + random.uniform(-0.15, 0.15)
        scored.append((score, idx))
    # 倾向分降序：分数最高的 (n - target_success) 个判为失败
    scored.sort(key=lambda x: x[0], reverse=True)
    fail_count = n - target_success
    fail_indices = {idx for _, idx in scored[:fail_count]}

    results: List[Dict[str, Any]] = []
    # 让 created_at 在三版本间拉开时间，便于趋势图（v1 最早）
    base_time = datetime(2026, 6, 1) + timedelta(
        days=_DEMO_VERSIONS.index(vcfg) * 9
    )

    for i, task in enumerate(tasks):
        success = i not in fail_indices

        target_count = task.get("target_count", 3)
        created_at = (base_time + timedelta(minutes=i * 2)).replace(
            microsecond=0
        ).isoformat()

        if success:
            tr = _make_success_result(vcfg, task, target_count, created_at)
        else:
            tr = _make_failure_result(vcfg, task, target_count, created_at)
        results.append(tr)
    return results


def _base_timings(difficulty: str, world_model_on: bool) -> Dict[str, int]:
    """根据难度与是否启用世界模型估算耗时（世界模型增加风险评估开销）。"""
    base_ms = {"简单": 2600, "中等": 4200, "困难": 6400}.get(difficulty, 4000)
    steps = {"简单": 6, "中等": 10, "困难": 16}.get(difficulty, 10)
    if world_model_on:
        base_ms = int(base_ms * 1.12)  # 风险评估带来约 12% 额外耗时
    # 加入小幅随机抖动（受 seed 控制）
    base_ms = int(base_ms * random.uniform(0.9, 1.15))
    return {"duration_ms": base_ms, "step_count": steps}


def _make_success_result(vcfg, task, target_count, created_at) -> Dict[str, Any]:
    """合成一个"成功"任务结果。优化版/世界模型版的稳定性指标更好。"""
    timing = _base_timings(task["difficulty"], vcfg["world_model_on"])
    optimized = vcfg["strategy"] == "optimized"

    # 成功任务的准确率高，偶有轻微误拣/漏拣（部分正确仍判成功）
    sort_acc = round(random.uniform(0.92, 1.0), 3)
    mis_pick = 0 if random.random() > 0.12 else 1
    miss_pick = 0 if random.random() > 0.10 else 1

    grasp_attempts = target_count + (0 if optimized else random.randint(0, 1))
    grasp_success = target_count
    # 优化版重试更少、恢复更强、几乎不需人工介入
    retry_count = random.randint(0, 1) if optimized else random.randint(0, 2)
    collisions = 0 if random.random() > (0.06 if optimized else 0.14) else 1
    recovered = 1 if (collisions and optimized and random.random() > 0.3) else 0
    human_intervention = 0 if optimized else (1 if random.random() < 0.05 else 0)

    return {
        "task_id": task["id"],
        "instruction": task["instruction"],
        "type": task["type"],
        "difficulty": task["difficulty"],
        "vla_backend": vcfg["vla_backend"],
        "strategy": vcfg["strategy"],
        "world_model_on": vcfg["world_model_on"],
        "success": True,
        "sort_accuracy": sort_acc,
        "mis_pick": mis_pick,
        "miss_pick": miss_pick,
        "target_count": target_count,
        "duration_ms": timing["duration_ms"],
        "step_count": timing["step_count"],
        "retry_count": retry_count,
        "collisions": collisions,
        "grasp_attempts": grasp_attempts,
        "grasp_success": grasp_success,
        "human_intervention": human_intervention,
        "recovered": recovered,
        "recognition_correct": target_count,
        "failure_category": None,
        "failure_subtype": None,
        "failure_reason": None,
        "scene": _demo_scene(task, target_count),
        "action_plan": _demo_action_plan(task),
        "executed_steps": [],
        "expected": {"target_parts": task.get("target_parts", [])},
        "predicted": {"assignments": {}},
        "created_at": created_at,
    }


def _make_failure_result(vcfg, task, target_count, created_at) -> Dict[str, Any]:
    """合成一个"失败"任务结果，并归类到 5 大类失败之一（含子类与中文原因）。"""
    timing = _base_timings(task["difficulty"], vcfg["world_model_on"])
    optimized = vcfg["strategy"] == "optimized"

    # 依据任务类型权重选失败大类，再随机选子类
    cat_key = _weighted_choice(_TYPE_FAILURE_WEIGHT.get(
        task["type"], _TYPE_FAILURE_WEIGHT["基础分拣"]))
    cat_cn, subtypes = _FAILURE_TAXONOMY[cat_key]
    subtype = random.choice(subtypes)
    reason = _failure_reason(cat_key, subtype, task)

    # 失败任务准确率偏低，误拣/漏拣更多
    sort_acc = round(random.uniform(0.0, 0.6), 3)
    mis_pick = random.randint(0, 2)
    miss_pick = random.randint(0, max(1, target_count - 1))

    grasp_attempts = target_count + random.randint(1, 3)
    grasp_success = max(0, target_count - random.randint(1, target_count))
    retry_count = random.randint(1, 3)
    collisions = 1 if cat_key == "execution" and "碰撞" in subtype else \
        (1 if random.random() < 0.3 else 0)
    recovered = 0  # 失败即未恢复
    # 困难/未优化更可能触发人工介入
    human_intervention = 1 if (not optimized and random.random() < 0.25) else \
        (1 if random.random() < 0.08 else 0)
    recognition_correct = grasp_success if cat_key != "perception" else \
        max(0, grasp_success - 1)

    return {
        "task_id": task["id"],
        "instruction": task["instruction"],
        "type": task["type"],
        "difficulty": task["difficulty"],
        "vla_backend": vcfg["vla_backend"],
        "strategy": vcfg["strategy"],
        "world_model_on": vcfg["world_model_on"],
        "success": False,
        "sort_accuracy": sort_acc,
        "mis_pick": mis_pick,
        "miss_pick": miss_pick,
        "target_count": target_count,
        "duration_ms": timing["duration_ms"],
        "step_count": timing["step_count"],
        "retry_count": retry_count,
        "collisions": collisions,
        "grasp_attempts": grasp_attempts,
        "grasp_success": grasp_success,
        "human_intervention": human_intervention,
        "recovered": recovered,
        "recognition_correct": recognition_correct,
        "failure_category": cat_cn,
        "failure_subtype": subtype,
        "failure_reason": reason,
        "scene": _demo_scene(task, target_count),
        "action_plan": _demo_action_plan(task),
        "executed_steps": [],
        "expected": {"target_parts": task.get("target_parts", [])},
        "predicted": {"assignments": {}},
        "created_at": created_at,
    }


def _failure_reason(cat_key: str, subtype: str, task: Dict[str, Any]) -> str:
    """生成一句具体的中文失败原因（结合子类与任务上下文）。"""
    templates = {
        "识别错误": "将相似零件误识别（如螺丝/螺母同为银色），导致分拣到错误料盒。",
        "定位不准": "零件中心定位偏差较大，夹爪未对准抓取点。",
        "遮挡看不见": f"困难场景存在遮挡，目标零件被部分覆盖，感知漏检（任务 {task['id']}）。",
        "光照角度问题": "俯视光照角度造成反光，金属零件轮廓识别不稳定。",
        "指令理解错误": f"对指令“{task['instruction']}”解析有误，目标分配错误。",
        "漏理解约束": "漏掉指令中的约束条件（如“仅金属件”），分拣范围扩大。",
        "歧义处理失败": "模糊指令（大左小右）边界判定失败，部分零件归错方向。",
        "路径不合理": "分拣路径绕行严重，超时未完成全部目标。",
        "分拣顺序错误": "未按最近邻优化顺序，导致重复移动与碰撞风险升高。",
        "优先级处理错误": "未正确处理“优先芯片”的优先级，先抓了低优先零件。",
        "抓取失败(滑落)": "易碎/光滑零件夹持力不足，抓取过程中滑落。",
        "放置失败(掉落)": "大件/平板（如 PCB、显示屏）放置时姿态不稳掉落。",
        "碰撞导致失败": "运动路径与相邻零件/料盒边沿碰撞，触发安全停止。",
        "物体意外移动": "传送带微动导致零件位姿变化，抓取目标失配。",
        "障碍物出现": "工作区出现临时障碍物，规划路径被阻断。",
        "仿真物理异常": "物理引擎接触解算异常，零件抖动/穿模导致执行失败。",
    }
    return templates.get(subtype, f"{subtype}：任务 {task['id']} 执行失败。")


def _demo_scene(task: Dict[str, Any], target_count: int) -> Dict[str, Any]:
    """生成演示用的简化 SceneState（含真值零件位置，供看板画合成俯视图）。"""
    codes = task.get("part_codes") or _PARTS_BY_DIFFICULTY.get(
        task["difficulty"], ["screw", "nut", "chip"])
    parts = []
    for idx, code in enumerate(codes[:max(target_count, len(codes))]):
        meta = _PARTS_META.get(code, {})
        parts.append({
            "part_id": idx,
            "code": code,
            "name": meta.get("name", code),
            "material": meta.get("material", "金属"),
            "color": meta.get("color", "银色"),
            "size": meta.get("size", "小"),
            "shape": meta.get("shape", "圆柱"),
            "fragile": meta.get("fragile", False),
            "pos": [round(random.uniform(-0.25, 0.25), 3),
                    round(random.uniform(-0.15, 0.15), 3)],
            "occluded": task["difficulty"] == "困难" and random.random() < 0.3,
        })
    return {
        "parts": parts,
        "bins": {"A": "A区", "B": "B区", "C": "C区"},
    }


def _demo_action_plan(task: Dict[str, Any]) -> Dict[str, Any]:
    """生成演示用的简化 ActionPlan（模型输出，供失败案例展示）。"""
    return {
        "instruction": task["instruction"],
        "parsed_intent": {"mode": _TYPE_TO_MODE.get(task["type"], "基础"), "rules": []},
        "actions": [],
        "reasoning": f"基于规则解析指令“{task['instruction']}”并分配目标料盒。",
    }


# 指令类型 → parsed_intent.mode 的映射（与 SPEC §5 的 mode 取值一致）
_TYPE_TO_MODE = {
    "基础分拣": "基础",
    "条件分拣": "条件",
    "优先级分拣": "优先级",
    "批量分拣": "批量",
    "异常场景": "基础",
}

# 零件元数据（§2 的 10 种 3C 零件，权威值与 objects.py / parts_catalog.json 镜像）
_PARTS_META = {
    "screw":     {"name": "螺丝",   "material": "金属", "color": "银色", "size": "小", "shape": "圆柱",   "fragile": False, "weight": 0.01},
    "nut":       {"name": "螺母",   "material": "金属", "color": "银色", "size": "小", "shape": "六边形", "fragile": False, "weight": 0.008},
    "capacitor": {"name": "电容",   "material": "金属", "color": "蓝色", "size": "小", "shape": "圆柱",   "fragile": False, "weight": 0.005},
    "resistor":  {"name": "电阻",   "material": "陶瓷", "color": "棕色", "size": "小", "shape": "圆柱",   "fragile": False, "weight": 0.003},
    "chip":      {"name": "芯片",   "material": "塑料", "color": "黑色", "size": "中", "shape": "方形",   "fragile": True,  "weight": 0.004},
    "connector": {"name": "连接器", "material": "塑料", "color": "白色", "size": "中", "shape": "方形",   "fragile": False, "weight": 0.006},
    "heatsink":  {"name": "散热器", "material": "金属", "color": "银色", "size": "大", "shape": "块状",   "fragile": False, "weight": 0.05},
    "pcb":       {"name": "PCB板",  "material": "复合", "color": "绿色", "size": "大", "shape": "平板",   "fragile": True,  "weight": 0.03},
    "button":    {"name": "按键",   "material": "塑料", "color": "红色", "size": "小", "shape": "方形",   "fragile": False, "weight": 0.002},
    "display":   {"name": "显示屏", "material": "玻璃", "color": "黑色", "size": "大", "shape": "平板",   "fragile": True,  "weight": 0.04},
}

_PARTS_BY_DIFFICULTY = {
    "简单": ["screw", "capacitor", "button"],
    "中等": ["screw", "nut", "capacitor", "resistor", "chip"],
    "困难": ["screw", "nut", "capacitor", "chip", "connector", "heatsink", "pcb", "display"],
}


def _demo_tasks() -> List[Dict[str, Any]]:
    """seed_demo 用的 30 个任务（与 tasks.yaml 同构，但内置以保证无依赖可跑）。

    5 类各 6 个，覆盖简单/中等/困难。target_count 用于合成准确率/抓取指标。
    """
    tasks: List[Dict[str, Any]] = []

    def add(tid, instruction, ttype, difficulty, codes, target_parts, target_count):
        tasks.append({
            "id": tid, "instruction": instruction, "type": ttype,
            "difficulty": difficulty, "part_codes": codes,
            "target_parts": target_parts, "target_count": target_count,
        })

    # —— 基础分拣（6）——
    add("T01", "把红色的按键放到A区", "基础分拣", "简单", ["button", "screw", "capacitor"], ["button"], 1)
    add("T02", "把蓝色的电容放到B区", "基础分拣", "简单", ["capacitor", "screw", "button"], ["capacitor"], 1)
    add("T03", "把银色的螺丝放到A区", "基础分拣", "简单", ["screw", "capacitor", "resistor"], ["screw"], 1)
    add("T04", "把黑色的芯片放到C区", "基础分拣", "中等", ["chip", "screw", "nut", "capacitor", "resistor"], ["chip"], 1)
    add("T05", "把绿色的PCB板放到B区", "基础分拣", "中等", ["pcb", "screw", "chip", "connector", "button"], ["pcb"], 1)
    add("T06", "把白色的连接器放到A区", "基础分拣", "困难", ["connector", "screw", "nut", "chip", "capacitor", "resistor", "heatsink", "button"], ["connector"], 1)

    # —— 条件分拣（6）——
    add("T07", "把所有金属零件分拣到B区", "条件分拣", "中等", ["screw", "nut", "capacitor", "chip", "button"], ["screw", "nut", "capacitor"], 3)
    add("T08", "把所有易碎零件放到C区", "条件分拣", "中等", ["chip", "pcb", "screw", "capacitor", "button"], ["chip", "pcb"], 2)
    add("T09", "把所有塑料零件分拣到A区", "条件分拣", "中等", ["chip", "connector", "button", "screw", "capacitor"], ["chip", "connector", "button"], 3)
    add("T10", "把所有银色零件放到B区", "条件分拣", "困难", ["screw", "nut", "heatsink", "chip", "pcb", "capacitor", "connector", "button"], ["screw", "nut", "heatsink"], 3)
    add("T11", "把所有大件零件放到C区", "条件分拣", "困难", ["heatsink", "pcb", "display", "screw", "nut", "chip", "capacitor", "button"], ["heatsink", "pcb", "display"], 3)
    add("T12", "把所有圆柱形零件分拣到A区", "条件分拣", "简单", ["screw", "capacitor", "chip"], ["screw", "capacitor"], 2)

    # —— 优先级分拣（6）——
    add("T13", "优先分拣芯片，然后是电容", "优先级分拣", "中等", ["chip", "capacitor", "screw", "nut", "button"], ["chip", "capacitor"], 2)
    add("T14", "优先处理易碎零件，再处理其他", "优先级分拣", "困难", ["chip", "pcb", "display", "screw", "nut", "capacitor", "connector", "button"], ["chip", "pcb", "display"], 3)
    add("T15", "先放显示屏到C区，再放散热器到B区", "优先级分拣", "困难", ["display", "heatsink", "screw", "nut", "chip", "capacitor", "connector", "button"], ["display", "heatsink"], 2)
    add("T16", "优先分拣PCB板，其余按颜色分类", "优先级分拣", "中等", ["pcb", "screw", "capacitor", "button", "chip"], ["pcb"], 1)
    add("T17", "先抓取所有金属件，再抓取塑料件", "优先级分拣", "中等", ["screw", "nut", "chip", "connector", "button"], ["screw", "nut"], 2)
    add("T18", "优先分拣红色按键到A区", "优先级分拣", "简单", ["button", "screw", "capacitor"], ["button"], 1)

    # —— 批量分拣（6）——
    add("T19", "把所有零件按颜色分类", "批量分拣", "中等", ["screw", "capacitor", "button", "chip", "pcb"], ["screw", "capacitor", "button", "chip", "pcb"], 5)
    add("T20", "把所有零件按材质分拣到对应料盒", "批量分拣", "困难", ["screw", "nut", "chip", "connector", "resistor", "pcb", "display", "button"], ["screw", "nut", "chip", "connector", "resistor", "pcb", "display", "button"], 8)
    add("T21", "把工作台上所有零件清空到料盒", "批量分拣", "简单", ["screw", "capacitor", "button"], ["screw", "capacitor", "button"], 3)
    add("T22", "把所有零件按大小分类摆放", "批量分拣", "困难", ["screw", "nut", "chip", "connector", "heatsink", "pcb", "display", "capacitor"], ["screw", "nut", "chip", "connector", "heatsink", "pcb", "display", "capacitor"], 8)
    add("T23", "把所有金属与塑料件分开放置", "批量分拣", "中等", ["screw", "nut", "chip", "connector", "button"], ["screw", "nut", "chip", "connector", "button"], 5)
    add("T24", "把所有零件按是否易碎分两区", "批量分拣", "中等", ["chip", "pcb", "screw", "nut", "capacitor"], ["chip", "pcb", "screw", "nut", "capacitor"], 5)

    # —— 异常场景（6）——
    add("T25", "在有遮挡情况下把芯片放到C区", "异常场景", "困难", ["chip", "screw", "nut", "capacitor", "connector", "pcb", "heatsink", "button"], ["chip"], 1)
    add("T26", "把大的零件放左边，小的放右边", "异常场景", "中等", ["heatsink", "pcb", "screw", "nut", "button"], ["heatsink", "pcb", "screw", "nut", "button"], 5)
    add("T27", "在相似零件混放时分拣出所有螺母", "异常场景", "困难", ["screw", "nut", "capacitor", "resistor", "chip", "connector", "button", "pcb"], ["nut"], 1)
    add("T28", "处理掉落后重新抓取并放置显示屏", "异常场景", "困难", ["display", "screw", "nut", "chip", "capacitor", "connector", "heatsink", "button"], ["display"], 1)
    add("T29", "在光照不佳时分拣银色螺丝到A区", "异常场景", "中等", ["screw", "nut", "capacitor", "resistor", "button"], ["screw"], 1)
    add("T30", "把易碎的PCB板小心放到C区", "异常场景", "中等", ["pcb", "screw", "chip", "capacitor", "button"], ["pcb"], 1)

    return tasks


# 供 benchmark / 外部复用 demo 任务清单（与 tasks.yaml 缺失时的回退）
def demo_task_list() -> List[Dict[str, Any]]:
    """对外暴露内置 30 任务（benchmark 在 tasks.yaml 缺失时回退使用）。"""
    return _demo_tasks()


def write_demo_data_json() -> str:
    """生成/覆盖 data/demo_data.json（版本种子配置 + 10 零件目录）。返回路径。"""
    _ensure_dirs()
    payload = {
        "versions": [
            {
                "version": v["version"],
                "vla_backend": v["vla_backend"],
                "strategy": v["strategy"],
                "world_model_on": v["world_model_on"],
                "base_success_rate": v["base_success_rate"],
                "note": v["note"],
            }
            for v in _DEMO_VERSIONS
        ],
        "parts_catalog": [
            {"code": code, **meta} for code, meta in _PARTS_META.items()
        ],
    }
    with open(DEMO_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    return DEMO_DATA_PATH


if __name__ == "__main__":
    # 直接运行：初始化库 + 播种演示数据 + 生成 demo_data.json，便于快速自检
    init_db()
    seeded = seed_demo()
    path = write_demo_data_json()
    runs = list_runs()
    print(f"[storage] 初始化完成；本次{'已' if seeded else '未'}播种演示数据。")
    print(f"[storage] 当前评测批次数：{len(runs)}")
    for r in runs:
        print(f"  - {r['version']:>3} | 后端={r['vla_backend']} 策略={r['strategy']} "
              f"世界模型={'开' if r['world_model_on'] else '关'} "
              f"成功率={r['success_rate']:.3f} 准确率={r['sort_accuracy']:.3f}")
    print(f"[storage] demo_data.json 已写入：{path}")
