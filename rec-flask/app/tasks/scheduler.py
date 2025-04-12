# scheduler.py
import schedule
import time
from app.service.recommendation import retrain_model

# 安排每日凌晨2点重新训练模型
schedule.every().day.at("02:00").do(retrain_model)

def run_scheduler():
    while True:
        schedule.run_pending()
        time.sleep(1)
