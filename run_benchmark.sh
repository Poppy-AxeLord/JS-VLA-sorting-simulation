#!/bin/bash
# =============================================================================
# 工业分拣 VLA 仿真 POC —— 运行一次完整评测 (run_benchmark.sh)
#
# 作用：激活环境后，从【项目根目录】以包方式运行评测器，加载 config/tasks.yaml
#       的 30 个任务，逐个用 SortingEngine 跑（无需真实机器人，按概率模型注入
#       5 类失败），汇总指标与失败分布并写入数据库，跑完打印中文摘要。
#
# 默认参数：--version v_new --vla rule_based
#   （rule_based 为永远可用的规则基线，缺任何重依赖也能跑通）
# 可自行追加参数，例如：
#   ./run_benchmark.sh --version v4 --vla smolvla --strategy optimized --world-model on --seed 42
# 透传给底层 python -m src.evaluation.benchmark 的 argparse。
#
# 用法：
#   chmod +x run_benchmark.sh && ./run_benchmark.sh [额外参数...]
# 路径含空格与中文，脚本内统一加引号。
# =============================================================================

set -e

C_OK="\033[1;32m"; C_WARN="\033[1;33m"; C_INFO="\033[1;36m"; C_END="\033[0m"
info() { echo -e "${C_INFO}[信息]${C_END} $1"; }
ok()   { echo -e "${C_OK}[成功]${C_END} $1"; }
warn() { echo -e "${C_WARN}[告警]${C_END} $1"; }

# ---- 切换到脚本所在目录（项目根），保证 `python -m src.*` 包导入可用 --------
PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$PROJECT_ROOT"

echo "============================================================"
echo "  运行 工业分拣 VLA 评测 (Benchmark)"
echo "============================================================"

# ---- 激活环境：优先复用 conda 环境，否则用本地 venv -------------------------
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

# ---- 默认参数：未显式传参时使用 v_new + rule_based --------------------------
# 若用户传入了自定义参数，则完全透传，忽略默认值。
if [ "$#" -eq 0 ]; then
  set -- --version v_new --vla rule_based
fi

echo ""
info "评测参数: $*"
info "正在加载 config/tasks.yaml 的 30 个任务并逐个评测，请稍候 ..."
echo "------------------------------------------------------------"

# ---- 以包方式从项目根运行评测器（绝对导入约定） ----------------------------
python -m src.evaluation.benchmark "$@"

echo "------------------------------------------------------------"
ok "评测完成，结果已写入数据库 (data/app.db)。"
info "下一步：运行 ./run_demo.sh 启动看板，在『版本对比 / 失败分析』页查看本次结果。"
echo "============================================================"
