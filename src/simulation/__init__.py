# -*- coding: utf-8 -*-
"""
仿真模块（src/simulation）
=========================

本包提供工业分拣 VLA 仿真 POC 的"物理世界"层。设计的第一原则是
**容错降级**：MuJoCo 是可选依赖，缺失或加载失败时自动回退到内置的
MockPhysics（纯 numpy 运动学近似），保证全流程在仅有核心依赖
(numpy/pandas/pyyaml/streamlit/plotly/matplotlib) 的环境下也能跑通。

对外暴露的核心对象：
- ``SortingEnv``  : 仿真环境（reset/step/get_camera_image/render）
- ``SimpleArm``   : 简化机械臂（UR5e + 二指夹爪）
- ``Camera``      : 相机（俯视/侧视，mock 用 matplotlib 合成俯视图）
- ``PARTS``       : 10 种 3C 零件权威目录（§2）
- ``generate_scene`` : 场景采样函数（按难度生成零件/位姿/遮挡）
- ``build_scene_state`` : SceneState 构造辅助

统一以 src 为根做绝对导入，例如：
    from src.simulation.env import SortingEnv
    from src.simulation.objects import PARTS, generate_scene
"""

# 注意：此处不在包级别 eager import 子模块，避免在仅需要 objects（纯数据）
# 的场景下被动触发 env/camera 里对 matplotlib 等的导入。各调用方按需
# 从子模块绝对导入即可。

__all__ = [
    "objects",
    "env",
    "robot",
    "camera",
]
