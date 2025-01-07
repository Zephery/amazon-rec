FROM cjie.eu.org/python:3.12

# 设置环境变量
ENV FLASK_APP=app.py
ENV FLASK_RUN_HOST=0.0.0.0

# 创建一个目录来存放应用代码
WORKDIR /app

# 复制当前目录下的代码到镜像中的/app目录
COPY . /app

# 安装requirements.txt中指定的依赖
RUN pip install --no-cache-dir -r requirements.txt

# 对外暴露应用程序的端口
EXPOSE 5000/tcp

# 运行Flask应用
CMD ["flask", "run"]