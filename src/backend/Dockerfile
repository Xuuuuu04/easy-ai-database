FROM python:3.11-slim
# 使用轻量 Python 基础镜像
WORKDIR /app
# 设置工作目录
COPY requirements.txt /app/requirements.txt
# 安装后端依赖
RUN pip install --no-cache-dir -r /app/requirements.txt
COPY app /app/app
# 拷贝应用源码
ENV PYTHONPATH=/app
# 设置应用运行路径
EXPOSE 8000
# 启动后端服务
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
