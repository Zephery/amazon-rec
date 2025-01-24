# database.py
import os
import sqlite3
import threading
import pandas as pd
import logging

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
            CREATE TABLE IF NOT EXISTS user_clicks (
                user_id TEXT,
                asin TEXT,
                click_time TIMESTAMP
            )
        ''')

# 在模块加载时创建表
create_tables()

# 加载商品数据
def load_products():
    # 从数据库中读取产品数据
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    if 'amazon_products' in tables['name'].tolist():
        products = pd.read_sql_query('SELECT * FROM amazon_products', conn)
    elif 'products' in tables['name'].tolist():
        products = pd.read_sql_query('SELECT * FROM products', conn)
    else:
        logging.error("No products table found in the database.")
        products = pd.DataFrame()
    return products

# 加载用户点击数据
def load_user_clicks():
    # 从数据库中读取用户点击数据
    user_clicks = pd.read_sql_query('SELECT * FROM user_clicks', conn)
    return user_clicks

# 加载用户评论数据
def load_user_reviews():
    # 从数据库中读取用户评论数据
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table' limit 10000;", conn)
    if 'amazon_reviews' in tables['name'].tolist():
        reviews = pd.read_sql_query('SELECT * FROM amazon_reviews', conn)
    else:
        logging.warning("No reviews table found in the database.")
        reviews = pd.DataFrame()
    return reviews
