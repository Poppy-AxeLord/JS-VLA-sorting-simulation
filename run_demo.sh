#!/bin/bash
# =============================================================================
# 工业分拣 VLA 仿真 POC —— 启动评测分析看板 (run_demo.sh)
#
# 作用：激活已搭建的环境后，从【项目根目录】启动 Streamlit 看板。
#       首次启动时看板会自动注入 3 版演示数据 (seed_demo)，开箱即有
#       趋势 / 版本对比 / 失败分析，无需先跑评测。
#
# 用法：
#   chmod +x run_demo.sh && ./run_demo.sh
# 路径含空格与中文，脚本内统一加引号。
# =============================================================================

set -e

C_OK="\033[1;32m"; C_WARN="\033[1;33m"; C_INFO="\033[1;36m"; C_END="\033[0m"
info() { echo -e "${C_INFO}[信息]${C_END} $1"; }
ok()   { echo -e "${C_OK}[成功]${C_END} $1"; }
warn() { echo -e "${C_WARN}[告警]${C_END} $1"; }

# ---- 切换到脚本所在目录（项目根），保证以包方式找得到 src.* -----------------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  启动 工业分拣 VLA 评测分析平台 (Streamlit 看板)"
echo "============================================================"

# ---- 激活环境：优先复用 setup.sh 创建的 conda 环境，否则用本地 venv ---------
ENV_NAME="vla_sorting"
VENV_DIR="$PROJECT_ROOT/.venv"

if command -v conda >/dev/null 2>&1 && conda env list | grep -qE "^\s*${ENV_NAME}\s"; then
  # shellcheck disable=SC1091
  source "$(conda info --base)/etc/profile.d/conda.sh"
  conda activate "$ENV_NAME"
  ok "已激活 conda 环境: ${ENV_NAME}"
elif [ -d "$VENV_DIR" ]; then
  # shellcheck disable=SC1091
  source "$VENV_DIR/bin/activate"
  ok "已激活 venv 环境: $VENV_DIR"
else
  warn "未找到 conda 环境 '${ENV_NAME}' 或 venv ($VENV_DIR)。"
  warn "将使用当前 Python 环境运行；如未安装依赖，请先执行 ./setup.sh"
fi

# ---- 友好提示访问地址 -------------------------------------------------------
echo ""
ok "看板启动中，请在浏览器访问: http://localhost:8501"
info "首次启动会自动注入 3 版演示数据，稍候即可看到完整看板。"
info "停止看板：在本终端按 Ctrl+C。"
echo "------------------------------------------------------------"

# ---- 从项目根启动 Streamlit 看板 -------------------------------------------
# 注意：streamlit run 不以包方式加载，app.py 顶部已自行把项目根加入 sys.path，
#       因此这里直接用相对路径启动即可（工作目录已是项目根）。
exec streamlit run "src/dashboard/app.py" --server.port 8501
