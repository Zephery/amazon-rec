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


# 阶段 3：整合前后端到最终镜像
FROM cjie.eu.org/nginx:latest

# 设置默认工作目录
WORKDIR /app

# 拷贝前端构建后的静态资源到 Nginx 网站目录
COPY --from=frontend /app/amazon-rec/dist /usr/share/nginx/html

COPY .ssl /etc/nginx/ssl

# 拷贝自定义 Nginx 配置文件
COPY ./nginx.conf /etc/nginx/nginx.conf

# 暴露端口
EXPOSE 80

# 启动 Flask 和 Nginx 服务
CMD ["/bin/sh", "-c", "nginx -g 'daemon off;'"]