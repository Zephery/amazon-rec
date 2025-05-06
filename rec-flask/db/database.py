# database.py
import logging
import os
import sqlite3
import threading

import pandas as pd

# 初始化日志
logging.basicConfig(level=logging.INFO)

# 数据库连接
db_lock = threading.Lock()
conn = sqlite3.connect('db/recommend.db', check_same_thread=False)


# 检查数据库是否存在
def check_db():
    if not os.path.exists("db/recommend.db"):
        print("数据库不存在，初始化数据库...")
        initialize_database()
    else:
        print("数据库存在，启动应用...")


# 调用其他 Python 文件生成数据库（假设这些文件在db目录中）
def initialize_database():
    from db.amazon_categories import init_categories
    init_categories()
    from db.amazon_products import init_products
    init_products()
    from db.amazon_reviews import init_review
    init_review()


# 创建用户点击表
def create_tables():
    with conn:
        conn.execute('''
                     CREATE TABLE IF NOT EXISTS user_clicks
                     (
                         user_id
                         TEXT,
                         asin
                         TEXT,
                         click_time
                         TIMESTAMP
                     )
                     ''')


# 在模块加载时创建表
create_tables()


# 加载商品数据
def load_products():
    return pd.read_sql_query('SELECT * FROM amazon_products limit 500000', conn)


def load_categories():
    query_categories = "SELECT * FROM amazon_categories"
    return pd.read_sql_query(query_categories, conn)


# 加载用户点击数据
def load_user_clicks():
    # 从数据库中读取用户点击数据
    return pd.read_sql_query('SELECT * FROM user_clicks', conn)


# 加载用户评论数据
def load_user_reviews():
    return pd.read_sql_query('SELECT * FROM amazon_reviews', conn)


def execute_query(sql, args):
    return pd.read_sql_query(sql, conn, params=args)

def delete_user_clicks(user_id):
    try:
        # 获取数据库连接的游标
        cursor = conn.cursor()

        # 执行 DELETE 语句
        query = "DELETE FROM user_clicks WHERE user_id = ?"
        cursor.execute(query, (user_id,))

        # 提交更改
        conn.commit()

        return f"Deleted clicks for user_id: {user_id} successfully."
    except Exception as e:
        # 捕获并打印错误
        print("Error deleting user clicks:", e)
        return f"Error deleting clicks: {e}"
