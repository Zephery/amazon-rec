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


华为云：
```shell
pip install faiss-cpu -i https://repo.huaweicloud.com/repository/pypi/simple/

pip install -r requirements.txt -i https://repo.huaweicloud.com/repository/pypi/simple/
```