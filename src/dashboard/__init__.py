# -*- coding: utf-8 -*-
"""
src/dashboard —— Streamlit 看板包

模块构成：
- app.py        看板主入口（streamlit run src/dashboard/app.py）
- charts.py     可复用 Plotly 图表构造器 + 统一配色常量
- pages/        三个子页面（总览 / 失败分析 / 版本对比），各自暴露 render(storage, ctx)

约定：看板仅从 storage 读取已落库的评测结果，不在网页内运行重仿真。
本包内统一使用以 src 为根的绝对导入，依赖运行入口从项目根目录执行。
"""
