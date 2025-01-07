# 阶段 1：构建 Vue 前端静态文件
FROM cjie.eu.org/node:latest AS frontend

# 设置工作目录
WORKDIR /app

# 拷贝前端代码到工作目录
COPY ./ ./amazon-rec

# 构建 Vue 应用
RUN cd amazon-rec \
    && npm config set registry https://registry.npmmirror.com \
    && npm install \
    && npm run build

# 阶段 2：构建 Flask 应用及依赖
FROM cjie.eu.org/python:3.12 AS backend

# 设置工作目录
WORKDIR /app/rec-flask

# 拷贝 Flask 依赖文件
COPY ./rec-flask/requirements.txt ./requirements.txt

# 创建虚拟环境并安装依赖
RUN python -m venv ./venv \
    && ./venv/bin/pip install --no-cache-dir -r requirements.txt

# 拷贝 Flask 源代码
COPY ./rec-flask ./

# 阶段 3：整合前后端到最终镜像
FROM cjie.eu.org/nginx:latest

# 设置默认工作目录
WORKDIR /app

# 拷贝前端构建后的静态资源到 Nginx 网站目录
COPY --from=frontend /app/amazon-rec/dist /usr/share/nginx/html

# 拷贝后端（Flask）虚拟环境及源码
COPY --from=backend /app/rec-flask /app/rec-flask

# 拷贝自定义 Nginx 配置文件
COPY ./nginx.conf /etc/nginx/nginx.conf

# 安装运行所需的系统 Python
RUN apt-get update && apt-get install -y python3.12 python3.12-pip \
    && ln -sf /app/rec-flask/venv/bin/python3 /usr/bin/python3 \
    && ln -sf /app/rec-flask/venv/bin/pip /usr/bin/pip

# 暴露端口
EXPOSE 80

# 启动 Flask 和 Nginx 服务
CMD ["/bin/sh", "-c", "/app/rec-flask/venv/bin/python3 /app/rec-flask/app.py & nginx -g 'daemon off;'"]