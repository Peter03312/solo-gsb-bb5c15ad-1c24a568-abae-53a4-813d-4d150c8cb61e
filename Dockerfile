FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    CASE_PORT=8080

WORKDIR /app

# 依赖在构建时一次性安装
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app ./app
COPY docker-entrypoint.sh ./docker-entrypoint.sh
RUN chmod +x ./docker-entrypoint.sh

RUN mkdir -p /app/data
VOLUME ["/app/data"]

EXPOSE 8080

# 启动时先完成数据库初始化（幂等建库），再在固定端口 8080 提供
# 服务与 OpenAPI(/docs, /openapi.json)；健康检查探测 /health
CMD ["./docker-entrypoint.sh"]
