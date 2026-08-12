#!/bin/bash
# =============================================================================
# 工业分拣 VLA 仿真 POC —— 一键环境搭建脚本 (setup.sh)
# 平台：Mac Studio / Apple Silicon (arm64)
#
# 设计原则（与项目“容错降级第一”一致）：
#   1) 先装【核心依赖】，装不上则直接失败退出（核心依赖是演示的底线）。
#   2) 再【尝试】装【可选依赖】(mujoco/torch/...)，任一失败仅告警、不中断；
#      运行期会自动降级（MuJoCo->Mock、SmolVLA->rule_based、MPS->CPU）。
#   3) 完成后验证 MuJoCo / MPS 是否可用，并给出后续一键脚本提示。
#
# 用法：
#   chmod +x setup.sh && ./setup.sh
# 路径含空格与中文，脚本内统一加引号处理。
# =============================================================================

set -e  # 任一关键命令失败立即退出（仅核心步骤受此约束；可选步骤显式容错）

# ---- 终端彩色输出（无颜色终端自动忽略转义） ---------------------------------
C_OK="\033[1;32m"    # 绿：成功
C_WARN="\033[1;33m"  # 黄：告警（降级，不致命）
C_ERR="\033[1;31m"   # 红：错误
C_INFO="\033[1;36m"  # 青：信息
C_END="\033[0m"
info()  { echo -e "${C_INFO}[信息]${C_END} $1"; }
ok()    { echo -e "${C_OK}[成功]${C_END} $1"; }
warn()  { echo -e "${C_WARN}[告警]${C_END} $1"; }
err()   { echo -e "${C_ERR}[错误]${C_END} $1"; }

# ---- 切换到脚本所在目录（即项目根目录），保证相对路径稳定 -------------------
# BASH_SOURCE 取脚本自身路径；含空格也安全（已加引号）。
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  工业分拣 VLA 仿真 POC —— 环境搭建 (Apple Silicon)"
echo "  项目根目录: $PROJECT_ROOT"
echo "============================================================"

# ---- 0. 平台自检：提示是否为 Apple Silicon ---------------------------------
ARCH="$(uname -m)"
if [ "$ARCH" = "arm64" ]; then
  ok "检测到 Apple Silicon (arm64)，与本项目目标平台一致。"
else
  warn "当前架构为 $ARCH（非 arm64）。脚本仍可运行，但本项目针对 Apple Silicon 优化。"
fi

# ---- 1. 检测 Homebrew（用于安装 Miniconda / 基础工具，可选） ----------------
info "检测 Homebrew ..."
if command -v brew >/dev/null 2>&1; then
  ok "已安装 Homebrew: $(brew --version | head -n 1)"
else
  warn "未检测到 Homebrew（非必需）。如需安装可执行："
  echo '      /bin/bash -c "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/HEAD/install.sh)"'
fi

# ---- 2. 选择环境管理方式：优先 conda(python3.10)，否则用 venv ----------------
# 说明：conda 在 Apple Silicon 上对科学计算栈兼容性更好；无 conda 则退回标准 venv。
ENV_NAME="vla_sorting"          # conda 环境名
VENV_DIR="$PROJECT_ROOT/.venv"  # venv 目录（含空格路径，已加引号）
PY_VERSION="3.10"

USE_CONDA="no"
info "检测 Miniconda / Conda ..."
if command -v conda >/dev/null 2>&1; then
  USE_CONDA="yes"
  ok "已检测到 conda: $(conda --version)"
else
  warn "未检测到 conda（Miniconda）。将使用 Python 自带 venv 创建环境。"
  echo "      如需 Miniconda(Apple Silicon 版)，推荐用 Homebrew 安装："
  echo "          brew install --cask miniconda"
  echo "      或下载 Miniconda3-latest-MacOSX-arm64.pkg 安装。"
fi

# ---- 3. 创建并激活环境 ------------------------------------------------------
if [ "$USE_CONDA" = "yes" ]; then
  # 让本 shell 能使用 `conda activate`
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"

  if conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
    info "conda 环境 '${ENV_NAME}' 已存在，直接复用。"
  else
    info "创建 conda 环境 '${ENV_NAME}' (python=${PY_VERSION}) ..."
    conda create -y -n "$ENV_NAME" "python=${PY_VERSION}"
    ok "conda 环境创建完成。"
  fi
  conda activate "$ENV_NAME"
  ok "已激活 conda 环境: ${ENV_NAME}"
else
  # venv 路径：用系统 python3 创建虚拟环境
  if [ ! -d "$VENV_DIR" ]; then
    info "使用 venv 创建虚拟环境于: $VENV_DIR"
    if command -v python3 >/dev/null 2>&1; then
      python3 -m venv "$VENV_DIR"
      ok "venv 创建完成。"
    else
      err "未找到 python3，无法创建虚拟环境。请先安装 Python 3.10+。"
      exit 1
    fi
  else
    info "venv 已存在，直接复用: $VENV_DIR"
  fi
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  ok "已激活 venv 环境。"
fi

# ---- 4. 升级 pip 基础工具 ---------------------------------------------------
info "升级 pip / setuptools / wheel ..."
python -m pip install --upgrade pip setuptools wheel
ok "pip 工具链就绪。"

# ---- 5. 安装【核心依赖】（必装，失败则中断——这是演示的底线） ----------------
info "安装【核心依赖】: numpy pandas pyyaml streamlit plotly matplotlib ..."
python -m pip install numpy pandas pyyaml streamlit plotly matplotlib
ok "核心依赖安装完成 —— 现在已可跑通完整演示（Mock 仿真 + rule_based VLA + 看板）。"

# ---- 6. 【尝试】安装【可选依赖】（失败仅告警，不中断） -----------------------
# 注意：这里临时关闭 set -e，逐个尝试，单包失败不影响后续与核心演示。
info "尝试安装【可选加强依赖】(失败将自动降级，不影响演示) ..."
set +e

try_pip_install() {
  # $1 = 包名（含版本约束），$2 = 失败时的降级提示
  local pkg="$1"
  local fallback_msg="$2"
  info "  -> 安装可选依赖: $pkg"
  if python -m pip install "$pkg"; then
    ok "  $pkg 安装成功。"
  else
    warn "  $pkg 安装失败 —— $fallback_msg"
  fi
}

# Apple Silicon 友好；禁 CUDA-only。torch 默认 wheel 即带 MPS。
try_pip_install "mujoco>=3.1,<3.3"            "运行时将自动回退 MockPhysics（纯 numpy 运动学）。"
try_pip_install "torch>=2.2"                  "运行时设备回退 CPU，SmolVLA 回退 rule_based。"
try_pip_install "torchvision>=0.17"           "缺失不影响核心；仅 SmolVLA 视觉预处理受限。"
try_pip_install "transformers>=4.40"          "SmolVLA 无法加载，VLA 回退 rule_based 规则基线。"
try_pip_install "opencv-python-headless>=4.9" "图像处理走 matplotlib/numpy 路径，不影响演示。"

set -e  # 恢复严格模式

# ---- 7. MuJoCo 安装验证（失败提示将用 Mock，不中断） ------------------------
info "验证 MuJoCo 是否可用 ..."
if python -c "import mujoco; print(mujoco.__version__)" >/dev/null 2>&1; then
  MJ_VER="$(python -c 'import mujoco; print(mujoco.__version__)' 2>/dev/null)"
  ok "MuJoCo 可用 (版本 ${MJ_VER}) —— 仿真将使用真实物理引擎。"
else
  warn "MuJoCo 不可用 —— 仿真将自动降级为 MockPhysics（纯 numpy 运动学），演示照常进行。"
fi

# ---- 8. PyTorch / MPS 验证（失败提示 CPU 降级，不中断） ----------------------
info "验证 PyTorch 与 MPS(Metal) 加速 ..."
if python -c "import torch" >/dev/null 2>&1; then
  if python -c "import torch; exit(0 if torch.backends.mps.is_available() else 1)" >/dev/null 2>&1; then
    ok "PyTorch 可用且 MPS(Metal) 加速可用 —— SmolVLA 将优先使用 mps 设备。"
  else
    warn "PyTorch 可用但 MPS 不可用 —— 将自动回退 CPU 运行。"
  fi
else
  warn "未安装 PyTorch —— SmolVLA 不可用，VLA 自动回退 rule_based；设备走 'cpu' 字符串不报错。"
fi

# ---- 9. 赋予运行脚本可执行权限（容错：缺文件不报错） ------------------------
info "为一键脚本添加可执行权限 ..."
chmod +x "$PROJECT_ROOT/run_demo.sh"      2>/dev/null || true
chmod +x "$PROJECT_ROOT/run_benchmark.sh" 2>/dev/null || true

# ---- 10. 结尾提示 -----------------------------------------------------------
echo ""
echo "============================================================"
ok "环境搭建完成！"
echo "------------------------------------------------------------"
if [ "$USE_CONDA" = "yes" ]; then
  echo -e "  下次使用前请先激活环境: ${C_INFO}conda activate ${ENV_NAME}${C_END}"
else
  echo -e "  下次使用前请先激活环境: ${C_INFO}source \"${VENV_DIR}/bin/activate\"${C_END}"
fi
echo ""
echo "  接下来你可以："
echo -e "    1) 启动评测分析看板:   ${C_INFO}./run_demo.sh${C_END}"
echo "         （首次会自动注入 3 版演示数据，浏览器访问 http://localhost:8501）"
echo -e "    2) 运行一次完整评测:   ${C_INFO}./run_benchmark.sh${C_END}"
echo "         （跑完 30 个任务并写入数据库，再回看板查看结果）"
echo "============================================================"
