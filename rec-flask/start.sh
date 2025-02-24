#!/bin/bash

# 设置amazon-rec的Flask服务
(
    cd /home/admin/amazon-rec/rec-flask/ || exit
    . .venv/bin/activate
    nohup flask run --host=0.0.0.0 > /dev/null 2>&1 &
    echo $! > /home/admin/amazon-rec/rec-flask/flask.pid
)

# 等待 60 秒，确保系统和网络服务已就绪
sleep 60

# 设置frp服务
(
    cd /home/admin/frp_0.61.1_linux_amd64 || exit
    nohup ./frpc -c frpc.toml > /dev/null 2>&1 &
)
