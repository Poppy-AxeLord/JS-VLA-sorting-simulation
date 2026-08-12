# -*- coding: utf-8 -*-
"""
src/dashboard/pages —— Streamlit 看板的三个子页面包

每个子页面对外只暴露一个统一签名的函数：
    render(storage, ctx) -> None

约定：
- storage：已初始化的 storage 模块（提供 list_runs / get_run / get_task_results /
  get_failure_cases 等只读接口）。
- ctx：主入口 app.py 传入的上下文 dict，至少包含：
    - "current_run"：当前所选评测运行记录（dict 或 None）
    - "runs"：全部评测运行列表（按时间倒序）
    - "charts"：dashboard.charts 模块（复用其 Plotly 构造器与配色常量）

注意：本包依赖 app.py 在最顶部完成的 sys.path 引导，子页面内部统一用
`from src.dashboard import charts` 之类的绝对导入。
"""

from . import overview, failure, comparison  # noqa: F401

__all__ = ["overview", "failure", "comparison"]
