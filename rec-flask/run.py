import threading
from datetime import datetime

from flask_cors import CORS

from app.blueprints.routes import create_app
from app.service.rec.recall import embeddings
from app.service.user.user_profile import init_user_profile
from app.tasks.scheduler import run_scheduler
from db.database import check_db

# 检查数据库
print("check_db start " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
check_db()
print("check_db finish " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

# 创建 Flask 应用
app = create_app()
print("create_app finish " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

CORS(app)
init_user_profile()
print("init_user_profile finish " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

embeddings()
print("embeddings finish " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

if __name__ == '__main__':
    # 在应用启动时重新训练模型

    # 启动调度器线程
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    # 运行 Flask 应用
    app.run(host="0.0.0.0", port=5000, debug=True)
