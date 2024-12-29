# ======== 阶段 1：构建 Vue 前端 ========
FROM node:latest AS frontend

# 设置工作目录
WORKDIR /app

# 拷贝 Vue 前端代码到镜像
COPY amazon-rec/ ./amazon-rec/

# 安装依赖并构建
RUN cd amazon-rec && npm install && npm run build

# ======== 阶段 2：安装 Flask 后端 ========
FROM python:latest AS backend

# 设置工作目录
WORKDIR /app

# 拷贝 Flask 后端代码到镜像
COPY rec-flask/ ./rec-flask/

# 安装 Flask 依赖
RUN pip install --no-cache-dir -r rec-flask/requirements.txt

# ======== 阶段 3：整合前后端 ========
FROM nginx:alpine

# 设置 Nginx 配置文件路径
COPY nginx.conf /etc/nginx/nginx.conf

# 拷贝 Vue 构建生成的静态文件到 Nginx 的默认网页目录
COPY --from=frontend /app/amazon-rec/dist /usr/share/nginx/html

# 拷贝 Flask 后端的代码到 Nginx 容器
COPY --from=backend /app/rec-flask /app/rec-flask

# 安装 Flask 运行环境
RUN apk add --no-cache python3 py3-pip \
    && pip install --no-cache-dir flask

# 暴露端口
EXPOSE 80

# 同时运行 Flask 和 Nginx
CMD ["/bin/sh", "-c", "python3 /app/rec-flask/app.py & nginx -g 'daemon off;'"]