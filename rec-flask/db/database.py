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
    create_indexes()


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


def get_recommended_products(top_category, asin):
    return pd.read_sql_query(
        """
        SELECT *
        FROM amazon_products
        WHERE category_id = ?
          AND asin != ?
        ORDER BY stars DESC
            LIMIT 5
        """,
        conn,
        params=(top_category, asin)  # 参数化语句，把参数绑定到占位符 `?` 上
    )


def get_products_by_asins(recall_union):
    format_sql = ','.join(['?'] * len(recall_union))
    query = f"SELECT * FROM amazon_products WHERE asin IN ({format_sql})"
    # pandas 会自动将列名变为 DataFrame 的 columns
    df = pd.read_sql_query(query, conn, params=recall_union)
    # 转为字典列表（如果需要list-of-dict的结构）
    product_list = df.to_dict(orient='records')
    return product_list


def get_one_product(asin):
    return pd.read_sql_query(f"SELECT * FROM amazon_products WHERE asin = '{asin}'", conn)


# 加载商品数据
def load_products():
    return pd.read_sql_query('SELECT * FROM amazon_products where category_id not in (134)', conn)


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


def get_user_recent_click_asins(user_id, limit=100):
    tmp = pd.read_sql_query(
        f"SELECT asin FROM user_clicks WHERE user_id='{user_id}' ORDER BY click_time DESC LIMIT '{limit}'", conn)
    return tmp['asin'].tolist()[:limit]


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


def load_data():
    """
    从数据库加载数据，并基于评论、产品信息、类别和点击记录合并成一个统一的数据框。
    """
    # 读取数据表
    reviews_df = pd.read_sql_query("SELECT * FROM amazon_reviews", conn)
    products_df = pd.read_sql_query("SELECT * FROM amazon_products", conn)
    categories_df = pd.read_sql_query("SELECT * FROM amazon_categories", conn)
    user_clicks_df = pd.read_sql_query("SELECT * FROM user_clicks", conn)  # 新增点击记录表

    # 数据清理：去除空值（这一步确保数据完整性）
    reviews_df = reviews_df.dropna(subset=["user_id", "asin", "text", "rating"])
    products_df = products_df.dropna(subset=["asin", "title", "price", "category_id", "stars"])
    user_clicks_df = user_clicks_df.dropna(subset=["user_id", "asin", "click_time"])  # 清理点击记录中的空值

    # 合并商品和类别表（categories 使用左连接，补足商品类别信息）
    products_df = pd.merge(products_df, categories_df, left_on="category_id", right_on="id", how="left")

    # 合并评论数据与产品信息表（基于 asin，使用内连接）
    merged_reviews_products = pd.merge(reviews_df, products_df, on="asin", how="inner")

    # 合并用户点击数据与产品信息 (基于 user_id 和 asin)
    merged_clicks_products = pd.merge(user_clicks_df, products_df, on="asin", how="inner")

    # 将评论数据与点击数据合并 (基于 user_id 和 asin，同时保留 comments 和 clicks)
    combined_data = pd.concat([merged_reviews_products, merged_clicks_products], ignore_index=True)

    # 将 click_time 转换为标准的 pandas datetime 格式
    if "click_time" in combined_data.columns:
        combined_data["click_time"] = pd.to_datetime(combined_data["click_time"], errors="coerce")

    # 返回最终的合并数据
    return combined_data


def create_indexes():
    try:
        with conn:
            # 创建索引时使用 IF NOT EXISTS 确保安全
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_clicks_user_id ON user_clicks(user_id);')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_user_clicks_asin ON user_clicks(asin);')
            conn.execute(
                'CREATE INDEX IF NOT EXISTS idx_user_clicks_user_id_click_time ON user_clicks(user_id, click_time DESC);')
            conn.execute('CREATE INDEX IF NOT EXISTS idx_amazon_products_asin ON amazon_products(asin);')

        logging.info("Indexes created successfully, or already exist.")
    except Exception as e:
        logging.error(f"Error while creating indexes: {e}")
