import json
import logging
import os
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import redis
import schedule
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import csr_matrix, vstack
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 初始化日志
logging.basicConfig(level=logging.INFO)

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 数据库连接
db_lock = threading.Lock()
conn = sqlite3.connect('db/recommend.db', check_same_thread=False)

# 全局变量
SVD = None  # SVD 模型
user_factors = None  # 用户特征矩阵
item_factors = None  # 商品特征矩阵

# 创建用户点击表（如果不存在）
def create_tables():
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_clicks (
                user_id TEXT,
                asin TEXT,
                click_time TIMESTAMP
            )
        ''')

create_tables()

# 优化数据加载：按需从数据库中读取数据，而不是全部加载到内存中
def get_product_info(asin_list):
    """
    从数据库中获取指定商品的详细信息
    """
    placeholders = ','.join(['?'] * len(asin_list))
    query = f'SELECT * FROM amazon_products WHERE asin IN ({placeholders})'
    data = pd.read_sql_query(query, conn, params=asin_list)
    return data

# 优化：使用生成器逐行读取用户点击数据，避免一次性加载所有数据
def load_user_clicks_generator():
    """
    生成器，逐行读取用户点击数据
    """
    cursor = conn.execute('SELECT user_id, asin FROM user_clicks')
    row = cursor.fetchone()
    while row:
        yield row
        row = cursor.fetchone()

# 优化模型初始化：限制数据规模，避免一次性加载大量数据
def initialize_model():
    global SVD, user_factors, item_factors

    logging.info("Initializing model...")

    # 从数据库中获取点击数据的唯一用户和商品
    user_ids = pd.read_sql_query('SELECT DISTINCT user_id FROM user_clicks LIMIT 10000', conn)['user_id'].tolist()
    item_ids = pd.read_sql_query('SELECT DISTINCT asin FROM user_clicks LIMIT 5000', conn)['asin'].tolist()

    user_map = {user_id: idx for idx, user_id in enumerate(user_ids)}
    item_map = {asin: idx for idx, asin in enumerate(item_ids)}

    # 构建用户-商品交互矩阵的稀疏表示
    row_indices = []
    col_indices = []
    data_values = []

    cursor = conn.execute('SELECT user_id, asin FROM user_clicks WHERE user_id IN ({user_ids}) AND asin IN ({item_ids})'.format(
        user_ids=','.join(['?'] * len(user_ids)),
        item_ids=','.join(['?'] * len(item_ids))
    ), (*user_ids, *item_ids))

    for user_id, asin in cursor:
        if user_id in user_map and asin in item_map:
            row_indices.append(user_map[user_id])
            col_indices.append(item_map[asin])
            data_values.append(1)  # 点击行为计为1

    user_item_sparse = csr_matrix((data_values, (row_indices, col_indices)),
                                  shape=(len(user_ids), len(item_ids)))

    if user_item_sparse.nnz == 0:
        logging.warning("No interaction data available for model training.")
        return

    # 训练 SVD 模型
    SVD = TruncatedSVD(n_components=20, random_state=42)
    SVD.fit(user_item_sparse)

    user_factors = SVD.transform(user_item_sparse)
    item_factors = SVD.components_.T

    logging.info("Model initialization completed.")

# 调用模型初始化
initialize_model()

# 推荐流程函数
def recommend_products(user_id, top_n=20):
    global SVD, user_factors, item_factors

    if SVD is None or user_factors is None or item_factors is None:
        logging.warning("Model is not initialized.")
        return []

    # 如果是新用户，直接推荐热门商品
    cursor = conn.execute('SELECT asin, COUNT(*) as cnt FROM user_clicks GROUP BY asin ORDER BY cnt DESC LIMIT ?', (top_n,))
    popular_items = [row[0] for row in cursor.fetchall()]
    if user_id not in pd.read_sql_query('SELECT DISTINCT user_id FROM user_clicks', conn)['user_id'].tolist():
        return popular_items

    # 获取用户特征向量
    user_idx = pd.read_sql_query('SELECT DISTINCT user_id FROM user_clicks', conn).query('user_id == @user_id').index
    if len(user_idx) == 0:
        return popular_items
    user_vector = user_factors[user_idx[0]]

    # 计算用户对所有商品的偏好分数
    scores = item_factors @ user_vector

    # 获取用户点击过的商品
    cursor = conn.execute('SELECT asin FROM user_clicks WHERE user_id = ?', (user_id,))
    user_clicked_items = set(row[0] for row in cursor.fetchall())

    # 排除用户已经点击过的商品
    item_ids = pd.read_sql_query('SELECT DISTINCT asin FROM user_clicks', conn)['asin'].tolist()
    item_scores = pd.Series(scores, index=item_ids)
    item_scores = item_scores[~item_scores.index.isin(user_clicked_items)]

    # 取分数最高的前 top_n 个商品
    recommended_items = item_scores.nlargest(top_n).index.tolist()

    return recommended_items

# 首页
@app.route('/')
def index():
    return "ok"

# 获取推荐商品列表
@app.route('/products', methods=['GET'])
def get_recommendations():
    user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('per_page', 20))
    q = request.args.get('q')

    if q:
        # 搜索功能，按需加载数据
        query = 'SELECT * FROM amazon_products WHERE title LIKE ? LIMIT ? OFFSET ?'
        params = ('%{}%'.format(q), page_size, (page - 1) * page_size)
        data = pd.read_sql_query(query, conn, params=params)
        total_items = pd.read_sql_query('SELECT COUNT(*) as cnt FROM amazon_products WHERE title LIKE ?', conn, params=('%{}%'.format(q),))['cnt'][0]
        total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
        return jsonify({
            'user_id': user_id,
            'total': total_items,
            'pages': total_pages,
            'current_page': page,
            'per_page': page_size,
            'products': data.to_dict(orient='records')
        })

    # 获取推荐商品列表
    cached_recommendations = redis_client.get(f"recommendations:{user_id}")
    if cached_recommendations is not None:
        recommendations = json.loads(cached_recommendations)
    else:
        recommendations = recommend_products(user_id, top_n=500)
        # 将推荐结果缓存
        redis_client.setex(f"recommendations:{user_id}", 3600, json.dumps(recommendations))

    # 分页处理
    total_items = len(recommendations)
    total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size
    paged_recommendations = recommendations[start_idx:end_idx]

    if not paged_recommendations:
        return jsonify({'message': 'No products to recommend.'}), 404

    # 获取商品详细信息
    products_data = get_product_info(paged_recommendations)

    return jsonify({
        'user_id': user_id,
        'total': total_items,
        'pages': total_pages,
        'current_page': page,
        'per_page': page_size,
        'products': products_data.to_dict(orient='records')
    })

# 记录用户点击行为
@app.route('/products/<asin>', methods=['GET'])
def record_click(asin):
    user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
    click_time = time.time()
    # 记录用户点击
    try:
        with db_lock:
            conn.execute('INSERT INTO user_clicks (user_id, asin, click_time) VALUES (?, ?, ?)',
                         (user_id, asin, click_time))
            conn.commit()
        # 清除缓存的推荐结果
        redis_client.delete(f"recommendations:{user_id}")
    except Exception as e:
        logging.error(f"Error recording click: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to record click.'}), 500
    # 获取被点击的商品详情
    product = get_product_info([asin])
    if not product.empty:
        product_data = product.iloc[0].to_dict()
    else:
        product_data = {}
    return jsonify({
        'status': 'success',
        'data': {
            'product': product_data,
            'additional_info': {
                'shipping': '免费配送',
                'warranty': '一年保修',
                'return_policy': '30天无理由退换'
            }
        }
    })

# 定时任务：每天凌晨2点重新训练模型
def retrain_model():
    logging.info("Retraining model...")
    initialize_model()
    logging.info("Model retraining completed.")

# 安排每日凌晨2点重新训练模型
schedule.every().day.at("02:00").do(retrain_model)

if __name__ == '__main__':
    retrain_model()

    # 在另一个线程中运行调度器
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)

    threading.Thread(target=run_scheduler).start()
    app.run(host='0.0.0.0', port=5000)
