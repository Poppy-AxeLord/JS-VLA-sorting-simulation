# -*- coding: utf-8 -*-
"""
src/dashboard/app.py —— 工业分拣 VLA 评测分析平台（Streamlit 主入口）

运行方式（从项目根目录）：
    streamlit run src/dashboard/app.py

【关键工程约定】
streamlit run 不会以「包」方式加载本文件（__package__ 为空），因此无法直接
`from src.xxx import ...`。必须在文件最顶部、任何 src.* 导入之前，用 __file__
上溯两级算出项目根目录并插入 sys.path，pages 子页面才能正常做绝对导入。
这是 SPEC §12 / 工程约定 3 的硬性要求。

容错降级第一原则：本文件只从 storage 读已落库数据，不跑任何重仿真；
storage / VLA 工厂的导入也用 try/except 守卫，缺依赖时给出中文降级提示而非崩溃。
"""

import os
import sys

# ============================================================
# 0. 路径引导（必须在 import src.* 之前执行）
# ============================================================
# 本文件位于  <项目根>/src/dashboard/app.py
# 上溯两级目录： dashboard -> src -> <项目根>
_HERE = os.path.abspath(os.path.dirname(__file__))
_PROJECT_ROOT = os.path.abspath(os.path.join(_HERE, os.pardir, os.pardir))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

# ------------------------------------------------------------
# 1. 第三方与项目内导入（均在路径引导之后）
# ------------------------------------------------------------
import streamlit as st  # noqa: E402

from src.dashboard import charts  # noqa: E402
from src.dashboard.pages import overview, failure, comparison  # noqa: E402

# storage 是看板的唯一数据来源；用 try/except 守卫，缺失时整页给中文降级提示
try:
    from src import storage as storage_mod  # noqa: E402
    _STORAGE_IMPORT_ERROR = None
except Exception as exc:  # pragma: no cover - 仅在 storage 未就绪时触发
    storage_mod = None
    _STORAGE_IMPORT_ERROR = exc


# ============================================================
# 2. 页面全局配置
# ============================================================
st.set_page_config(
    page_title="工业分拣 VLA 评测分析平台",
    page_icon="🔧",
    layout="wide",  # 宽屏，数据看板标配
    initial_sidebar_state="expanded",
)

# 注入少量全局 CSS：主色强调、卡片化容器、紧凑指标、深色侧边栏（统一设计规范 v1）
st.markdown(
    f"""
    <style>
      :root {{ --primary: {charts.PRIMARY}; }}
      html, body, [data-testid="stAppViewContainer"] * {{
        font-family: {charts.FONT_FAMILY};
      }}
      /* 页面底浅灰，让白色卡片浮出层次（Notion/Linear 式） */
      [data-testid="stAppViewContainer"] {{ background: #F6F7F9; }}
      /* 顶栏：白底 + 细分割线，轻微毛玻璃质感 */
      header[data-testid="stHeader"] {{
        background: rgba(255,255,255,.85);
        backdrop-filter: blur(6px);
        border-bottom: 1px solid #F3F4F6;
      }}
      /* 超宽屏收敛内容宽度，保证阅读节奏；窄屏不受影响 */
      /* Streamlit 顶栏为固定定位；留出安全区，避免首页标题被顶栏裁掉。 */
      .block-container {{ padding-top: 4rem; padding-bottom: 2rem; max-width: 1480px; }}
      /* 顶部主标题强调条 */
      .app-title {{
        font-size: 1.5rem; font-weight: 700; color: {charts.PRIMARY_DARK};
        border-left: 4px solid {charts.PRIMARY}; padding-left: 12px; margin-bottom: 2px;
        line-height: 1.5;
      }}
      .app-subtitle {{ color: {charts.TEXT_MUTED}; font-size: 0.85rem; margin-bottom: 8px; }}
      /* 指标卡：白底 + 细边框 + Linear 式轻阴影，hover 轻抬升 */
      div[data-testid="stMetric"] {{
        background: #FFFFFF; border: 1px solid #E5E7EB; border-radius: 12px;
        padding: 16px 20px;
        box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.08);
        transition: box-shadow .2s ease, transform .2s ease;
      }}
      div[data-testid="stMetric"]:hover {{
        box-shadow: 0 4px 12px rgba(16,24,40,.10);
        transform: translateY(-1px);
      }}
      div[data-testid="stMetricLabel"] {{ color: {charts.TEXT_MUTED}; font-size: 13px; }}
      div[data-testid="stMetricValue"] {{
        font-size: 1.75rem; font-weight: 700; color: #1F2937;
        font-variant-numeric: tabular-nums;
      }}
      div[data-testid="stMetricDelta"] {{ font-variant-numeric: tabular-nums; }}
      /* 图表/建议卡片：st.container(border=True) 统一为白卡 + 轻阴影 */
      div[data-testid="stVerticalBlockBorderWrapper"] {{
        background: #FFFFFF !important;
        border: 1px solid #E5E7EB !important;
        border-radius: 12px !important;
        box-shadow: 0 1px 2px rgba(16,24,40,.04), 0 1px 3px rgba(16,24,40,.08);
        transition: box-shadow .2s ease;
      }}
      div[data-testid="stVerticalBlockBorderWrapper"]:hover {{
        box-shadow: 0 4px 12px rgba(16,24,40,.10);
      }}
      /* 表格数字对齐：tabular-nums */
      [data-testid="stDataFrame"] * {{ font-variant-numeric: tabular-nums; }}
      /* ============ 深色侧边栏（#0F172A + 选中项 3px 主色指示条） ============ */
      section[data-testid="stSidebar"] {{
        background: #0F172A;
        border-right: 1px solid rgba(148,163,184,.12);
      }}
      /* pages/ 目录仅用于源码组织，不能把 Streamlit 的英文文件名导航暴露给演示观众。
         产品导航由下方「页面导航」单选框统一承担。 */
      [data-testid="stSidebarNav"] {{ display: none; }}
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] p,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h1,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h2,
      section[data-testid="stSidebar"] [data-testid="stMarkdownContainer"] h3,
      section[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p {{
        color: #E2E8F0;
      }}
      section[data-testid="stSidebar"] [data-testid="stCaptionContainer"] p,
      section[data-testid="stSidebar"] small {{ color: #94A3B8 !important; }}
      section[data-testid="stSidebar"] hr {{ border-color: rgba(148,163,184,.20); }}
      /* 侧边栏内 st.info 等提示框保持深色可读文字 */
      section[data-testid="stSidebar"] [data-testid="stAlert"] p {{ color: #1F2937; }}
      /* 导航 radio：整行可点，hover 微亮，选中项左侧 3px 主色指示条 + 浅蓝高亮 */
      section[data-testid="stSidebar"] div[role="radiogroup"] label {{
        border-left: 3px solid transparent;
        border-radius: 0 6px 6px 0;
        padding: 6px 10px; margin: 2px 0; width: 100%;
        transition: background .2s ease, border-color .2s ease;
      }}
      section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {{
        background: rgba(148,163,184,.10);
      }}
      section[data-testid="stSidebar"] div[role="radiogroup"] label:has(input:checked) {{
        border-left: 3px solid {charts.PRIMARY};
        background: rgba(37,99,235,.18);
      }}
      /* 侧边栏 selectbox：深色输入框样式，保证对比度 */
      section[data-testid="stSidebar"] div[data-baseweb="select"] > div {{
        background: rgba(148,163,184,.12);
        border-color: rgba(148,163,184,.30);
        color: #E2E8F0;
      }}
      section[data-testid="stSidebar"] div[data-baseweb="select"] svg {{ fill: #94A3B8; }}

      /* 手机端：Streamlit 的列默认会保留桌面分栏，数据看板会被压成无法阅读的窄条。
         这里统一改为单列堆叠，并收紧留白和表格/图表容器。 */
      @media (max-width: 700px) {{
        .block-container {{ padding: 3.5rem 0.85rem 1.5rem !important; max-width: 100% !important; }}
        .app-title {{ font-size: 1.22rem; padding-left: 9px; line-height: 1.35; }}
        .app-subtitle {{ font-size: .78rem; line-height: 1.55; }}
        div[data-testid="stHorizontalBlock"] {{ flex-wrap: wrap !important; gap: .75rem !important; }}
        div[data-testid="stHorizontalBlock"] > div[data-testid="stColumn"] {{
          flex: 1 1 100% !important; width: 100% !important; min-width: 0 !important;
        }}
        div[data-testid="stMetric"] {{ padding: 13px 15px; }}
        div[data-testid="stMetricValue"] {{ font-size: 1.5rem; }}
        div[data-testid="stVerticalBlockBorderWrapper"] {{ border-radius: 10px !important; }}
        [data-testid="stDataFrame"] {{ overflow-x: auto; }}
        section[data-testid="stSidebar"] {{ min-width: min(82vw, 300px) !important; }}
      }}
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# 3. storage 初始化（首次自动 seed_demo），带缓存
# ============================================================
@st.cache_resource(show_spinner="正在初始化数据库并加载任务数据……")
def _bootstrap_storage():
    """
    初始化 storage：建表 + 首次播种演示数据。

    用 @st.cache_resource 保证整个进程只执行一次（避免每次脚本重跑都重建库）。
    返回 storage 模块本身（供各页面调用其只读接口）。

    降级：storage 模块导入失败时返回 None，由主流程展示中文错误并停止。
    """
    if storage_mod is None:
        return None
    # 建表（幂等）
    storage_mod.init_db()
    # 首次播种：seed_demo 内部判断 benchmark_runs 是否为空，空则用 random.seed(42)
    # 确定性生成 v1/v2/v3 三次历史评测，保证看板开箱即有趋势 / 对比 / 失败分析
    storage_mod.seed_demo()
    return storage_mod


def _safe_list_runs(storage) -> list:
    """安全读取评测运行列表；任何异常都降级为空列表并返回，不让看板崩溃。"""
    try:
        runs = storage.list_runs()
        return list(runs) if runs else []
    except Exception as exc:  # pragma: no cover
        st.warning(f"读取评测运行列表失败，已降级为空：{exc}")
        return []


def _run_label(run: dict) -> str:
    """把一条 benchmark_run 记录格式化成下拉框可读中文标签。"""
    version = run.get("version", "?")
    vla = run.get("vla_backend", "?")
    sr = run.get("success_rate", 0) or 0
    wm = "+世界模型" if run.get("world_model_on") else ""
    return f"{version}｜{vla}{wm}｜成功率 {sr * 100:.0f}%"


# ============================================================
# 4. 主流程
# ============================================================
def main() -> None:
    # ---- 4.1 storage 不可用时的硬降级提示 ----
    if storage_mod is None:
        st.markdown('<div class="app-title">工业分拣 VLA 评测分析平台</div>', unsafe_allow_html=True)
        st.error(
            "存储模块 src/storage.py 尚未就绪，无法加载数据。\n\n"
            f"导入错误：{_STORAGE_IMPORT_ERROR}\n\n"
            "请先在项目根目录运行评测以生成数据：`bash run_benchmark.sh`，"
            "或确认 src/storage.py 已正确创建。"
        )
        st.stop()

    storage = _bootstrap_storage()
    runs = _safe_list_runs(storage)

    # ---- 4.2 侧边栏：导航 + 运行选择 ----
    with st.sidebar:
        st.markdown("### 🔧 工业分拣 VLA")
        st.caption("评测分析平台")
        st.divider()

        page = st.radio(
            "页面导航",
            options=["总览", "失败分析", "版本对比"],
            index=0,
            help="总览：核心指标与趋势｜失败分析：5 类失败诊断｜版本对比：v1→v2→v3 迭代收益",
        )

        st.divider()

        # 运行选择下拉：默认选中最新一次评测（list_runs 约定按时间倒序）
        if runs:
            run_options = list(range(len(runs)))
            selected_idx = st.selectbox(
                "选择评测运行",
                options=run_options,
                index=0,
                format_func=lambda i: _run_label(runs[i]),
                help="切换不同评测运行，总览 / 失败分析将基于所选运行展示",
            )
            current_run = runs[selected_idx]
        else:
            current_run = None
            st.info("暂无评测数据。请运行 `bash run_benchmark.sh` 生成。")

        st.divider()
        st.caption("数据来源：data/app.db（仅读取已落库结果）")
        st.caption("如需新评测：项目根目录执行 `bash run_benchmark.sh`")

    # ---- 4.3 顶部标题行：左标题 + 右侧环境徽标（节省一行垂直空间） ----
    head_left, head_right = st.columns([3, 2])
    with head_left:
        st.markdown('<div class="app-title">工业分拣 VLA 评测分析平台</div>', unsafe_allow_html=True)
        st.markdown(
            '<div class="app-subtitle">VLA + 仿真分拣的评测体系 · 失败归因 · 迭代决策｜本看板仅读取已落库评测结果</div>',
            unsafe_allow_html=True,
        )
    with head_right:
        st.image(
            os.path.join(_PROJECT_ROOT, "assets", "visuals", "sorting-hero.png"),
            caption="工业分拣工作站 · 多模态感知与动作规划",
            width="stretch",
        )
        _render_env_badges(current_run)
    st.divider()

    # ---- 4.4 上下文对象：传给各页面 render(storage, ctx) ----
    ctx = {
        "current_run": current_run,
        "runs": runs,
        "charts": charts,
    }

    # ---- 4.5 路由分发 ----
    try:
        if page == "总览":
            overview.render(storage, ctx)
        elif page == "失败分析":
            failure.render(storage, ctx)
        elif page == "版本对比":
            comparison.render(storage, ctx)
    except Exception as exc:  # 单页渲染异常兜底，避免整站白屏
        st.error(f"页面「{page}」渲染时发生异常：{exc}")
        st.exception(exc)


def _render_env_badges(current_run: dict | None) -> None:
    """
    顶部展示运行环境徽标：当前 VLA 后端 / 是否 Mock 仿真 / 加速设备。

    数据来源优先级：
    1) 所选评测运行记录里的 vla_backend（真实反映这批数据由哪个后端跑出）；
    2) 当前进程环境探测（mujoco 是否可用 → Mock 标记；mps_utils → 设备）。
    探测全部用 try/except 守卫，缺依赖只显示降级徽标，不报错。
    """
    # VLA 后端：取当前所选运行的 backend，没有则显示「rule_based（默认）」
    vla_backend = (current_run or {}).get("vla_backend") or "rule_based"

    # 仿真后端探测：mujoco 可用与否 → 真实 / Mock
    try:
        import mujoco  # noqa: F401
        sim_is_mock = False
    except Exception:
        sim_is_mock = True

    # 加速设备探测：复用项目内 mps_utils（torch 缺失返回 'cpu' 不报错）
    try:
        from src.utils.mps_utils import get_device
        device = get_device()
    except Exception:
        device = "cpu"

    sim_text = "仿真：轻量引擎" if sim_is_mock else "仿真：MuJoCo"
    sim_color = charts.PRIORITY_COLORS["中"] if sim_is_mock else charts.POSITIVE
    dev_text = f"设备：{device.upper()}"
    dev_color = charts.POSITIVE if device == "mps" else charts.TEXT_MUTED

    # 单行 flex 徽标条，右对齐贴合标题行，窄屏自动换行
    st.markdown(
        '<div style="display:flex;justify-content:flex-end;align-items:center;'
        'gap:8px;flex-wrap:wrap;margin-top:10px;">'
        + charts.badge_html(f"VLA 后端：{vla_backend}", charts.PRIMARY)
        + charts.badge_html(sim_text, sim_color)
        + charts.badge_html(dev_text, dev_color)
        + "</div>",
        unsafe_allow_html=True,
    )


# 当前 Streamlit 运行器会使用内部模块名执行脚本；仅以 ``__main__`` 作为守卫会让
# 8501 页面只剩源码目录的自动导航、主体完全空白。app.py 是唯一入口，项目内没有
# 任何页面反向 import 它，因此这里显式执行，确保 ``streamlit run app.py`` 和
# Streamlit 的 AppTest 都稳定渲染同一套首访内容。
main()
