#!/bin/sh
# 容器启动入口：依赖已在镜像构建时装好；启动时先完成数据库初始化
# （首次运行自动建库、幂等），成功后再在固定端口拉起 uvicorn。
set -e

echo "[entrypoint] 初始化数据库（init_db，幂等）..."
python -c "from app.database import init_db; init_db()"

echo "[entrypoint] 启动 uvicorn，端口 ${CASE_PORT:-8080} ..."
exec uvicorn app.main:app --host 0.0.0.0 --port "${CASE_PORT:-8080}"
