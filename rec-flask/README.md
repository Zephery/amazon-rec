## 一、本地启动

```shell
python3 -m venv .venv

source .venv/bin/activate
```


## 二、服务器启动
需要增加crontab
```shell
crontab -e

@reboot chmod +x /home/admin/amazon-rec/rec-flask/start.sh
@reboot /home/admin/amazon-rec/rec-flask/start.sh
```