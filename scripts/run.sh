#!/usr/bin/env bash
# 本地一键启动脚本：初始化依赖并启动后端/前端服务。
set -euo pipefail

# 仓库根目录
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"

# 创建并激活虚拟环境
python3 -m venv "$ROOT_DIR/.venv"
source "$ROOT_DIR/.venv/bin/activate"

# 安装后端依赖
pip install --upgrade pip
pip install -r "$ROOT_DIR/src/backend/requirements.txt"

# 安装前端依赖
cd "$ROOT_DIR/src/frontend"
if [ -f package-lock.json ]; then
  npm ci
else
  npm install
fi

cd "$ROOT_DIR"

# 加载本地环境变量（若存在）
export $(grep -v '^#' .env | xargs) || true

# 启动后端
uvicorn src.backend.app.main:app --host 0.0.0.0 --port 8000 &
BACKEND_PID=$!

# 启动前端
cd "$ROOT_DIR/src/frontend"
VITE_API_BASE=${VITE_API_BASE:-http://localhost:8000} npm run dev -- --host 0.0.0.0

# 退出时关闭后端
kill $BACKEND_PID
