import threading

from flask_cors import CORS

from app.blueprints.routes import create_app
from app.service.model import retrain_model  # 从 model.py 导入 retrain_model
from app.tasks.scheduler import run_scheduler
from db.database import check_db

# 检查数据库
check_db()

# 创建 Flask 应用
app = create_app()
CORS(app)

if __name__ == '__main__':
    # 在应用启动时重新训练模型
    retrain_model()
    # 启动调度器线程
    scheduler_thread = threading.Thread(target=run_scheduler)
    scheduler_thread.start()
    # 运行 Flask 应用
    app.run(host="0.0.0.0", port=5000, debug=True)
