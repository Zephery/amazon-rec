import json
import logging
import sqlite3
import threading
import time

import numpy as np
import pandas as pd
import redis
import schedule
from flask import Flask, request, jsonify
from flask_cors import CORS
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 初始化日志
logging.basicConfig(level=logging.INFO)

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 数据库连接
db_lock = threading.Lock()
conn = sqlite3.connect('recommendation.db', check_same_thread=False)


# 创建商品表和用户点击表
def create_tables():
    with conn:
        conn.execute('''
            CREATE TABLE IF NOT EXISTS products (
                asin TEXT PRIMARY KEY,
                title TEXT,
                imgUrl TEXT,
                productURL TEXT,
                stars REAL,
                reviews INTEGER,
                price REAL,
                listPrice REAL,
                category_id INTEGER,
                isBestSeller INTEGER,
                boughtInLastMonth INTEGER
            )
        ''')
        conn.execute('''
            CREATE TABLE IF NOT EXISTS user_clicks (
                user_id TEXT,
                asin TEXT,
                click_time TIMESTAMP
            )
        ''')


create_tables()


# 加载商品数据
def load_products():
    tables = pd.read_sql_query("SELECT name FROM sqlite_master WHERE type='table';", conn)
    print(tables)
    # 从数据库中读取产品数据
    products = pd.read_sql_query('SELECT * FROM products', conn)
    print(products)
    return products


products = load_products()


# 加载用户点击数据
def load_user_clicks():
    # 从数据库中读取用户点击数据
    user_clicks = pd.read_sql_query('SELECT * FROM user_clicks', conn)
    return user_clicks


user_clicks = load_user_clicks()


# 数据预处理和模型初始化
def initialize_model():
    global user_profiles, user_item_matrix, user_item_sparse, decomposed_matrix, SVD

    if user_clicks.empty:
        # 如果没有用户点击数据，初始化空的数据结构
        user_profiles = {}
        user_item_matrix = pd.DataFrame()
        user_item_sparse = None
        decomposed_matrix = None
        SVD = TruncatedSVD(n_components=20, random_state=42)
        return

    # 统计每个用户对每个类别的点击次数
    user_clicks_with_category = pd.merge(user_clicks, products[['asin', 'category_id']], on='asin', how='left')

    user_category_clicks = user_clicks_with_category.groupby(['user_id', 'category_id']).size().reset_index(
        name='click_count')

    user_total_clicks = user_category_clicks.groupby('user_id')['click_count'].sum().reset_index(name='total_clicks')

    user_category_clicks = pd.merge(user_category_clicks, user_total_clicks, on='user_id')

    user_category_clicks['preference'] = user_category_clicks['click_count'] / user_category_clicks['total_clicks']

    # 构建用户画像字典
    user_profiles = {}
    for user_id, group in user_category_clicks.groupby('user_id'):
        user_profiles[user_id] = dict(zip(group['category_id'], group['preference']))

    # 构建用户-商品交互矩阵
    user_item_matrix = user_clicks.groupby(['user_id', 'asin']).size().unstack(fill_value=0)

    # 将数据转换为稀疏矩阵
    user_item_sparse = csr_matrix(user_item_matrix.values)

    # 使用TruncatedSVD进行矩阵分解
    SVD = TruncatedSVD(n_components=20, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)


initialize_model()


def recall(user_id, top_n=500):
    if user_id not in user_item_matrix.index:
        # 对于新用户，返回全站最热门的商品
        popular_items = user_clicks['asin'].value_counts().head(top_n).index.tolist()
        return popular_items
    else:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = decomposed_matrix[user_index]
        similarity = cosine_similarity([user_vector], decomposed_matrix)[0]
        similar_users_indices = similarity.argsort()[::-1][1:top_n + 1]
        similar_users = user_item_matrix.index[similar_users_indices]
        similar_users_interactions = user_item_matrix.loc[similar_users]
        candidate_items = similar_users_interactions.sum(axis=0)
        user_items = user_item_matrix.loc[user_id]
        unseen_items = candidate_items[user_items == 0]
        candidates = unseen_items[unseen_items > 0].index.tolist()
        return candidates


def coarse_ranking(candidate_items):
    if not candidate_items:
        # 如果候选列表为空，返回全站最热门的商品
        item_popularity = user_clicks['asin'].value_counts()
    else:
        item_popularity = user_clicks[user_clicks['asin'].isin(candidate_items)]['asin'].value_counts()

    ranked_items = item_popularity.index.tolist()
    return ranked_items


def fine_ranking(user_id, ranked_items):
    if user_id not in user_item_matrix.index or decomposed_matrix is None:
        # 对于新用户或模型未训练的情况，直接返回粗排结果
        return ranked_items
    else:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = decomposed_matrix[user_index]
        item_indices = [user_item_matrix.columns.get_loc(item) for item in ranked_items if
                        item in user_item_matrix.columns]
        if not item_indices:
            return ranked_items  # 如果没有匹配的物品，直接返回粗排结果
        item_vectors = SVD.components_.T[item_indices]
        scores = np.dot(item_vectors, user_vector)
        item_scores = pd.Series(scores, index=[ranked_items[i] for i in range(len(scores))])
        item_scores = item_scores.dropna()
        # 调整评分，利用用户画像
        user_profile = user_profiles.get(user_id, {})
        # 获取商品的类别
        item_categories = products.set_index('asin').reindex(item_scores.index)['category_id']
        adjusted_scores = []
        for idx, item in enumerate(item_scores.index):
            category = item_categories.get(item)
            if pd.isna(category):
                preference = 0
            else:
                preference = user_profile.get(str(int(category)), 0)
            # 根据用户对类别的偏好程度调整评分
            adjusted_score = item_scores.iloc[idx] * (1 + preference)
            adjusted_scores.append(adjusted_score)
        # 更新评分
        item_scores = pd.Series(adjusted_scores, index=item_scores.index)
        # 排序
        fine_ranked_items = item_scores.sort_values(ascending=False).index.tolist()
        return fine_ranked_items


def re_ranking(user_id, fine_ranked_items):
    user_profile = user_profiles.get(user_id, {})
    logging.info(f"User profile for {user_id}: {user_profile}")
    if not user_profile:
        return fine_ranked_items
    # 获取用户偏好的类别列表
    preferred_categories = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
    preferred_categories = [int(cat) for cat, _ in preferred_categories]
    logging.info(f"Preferred categories: {preferred_categories}")
    # 获取商品的类别信息
    item_categories = products.set_index('asin').reindex(fine_ranked_items)['category_id']
    logging.info(f"Item categories: {item_categories}")
    # 按照类别重排
    category_items = {}
    for item in fine_ranked_items:
        category = item_categories.get(item)
        if pd.isna(category):
            continue  # 跳过没有类别的商品
        if category not in category_items:
            category_items[category] = []
        category_items[category].append(item)
    # 按照用户偏好重排商品
    final_ranked_items = []
    added_items = set()
    for category in preferred_categories:
        items = category_items.get(category, [])
        final_ranked_items.extend(items)
        added_items.update(items)
    # 添加剩余的商品
    remaining_items = [item for item in fine_ranked_items if item not in added_items]
    final_ranked_items.extend(remaining_items)
    if not final_ranked_items:
        # 如果最终列表为空，返回原始的 fine_ranked_items
        final_ranked_items = fine_ranked_items
    logging.info(f"Final ranked items for {user_id}: {final_ranked_items}")
    return final_ranked_items


def get_offline_recommendations(user_id):
    cached_recommendations = redis_client.get(f"recommendations:{user_id}")
    if cached_recommendations:
        cached_recommendations = cached_recommendations.decode('utf-8')
        return json.loads(cached_recommendations)
    else:
        # 当缓存中没有推荐时，生成默认的推荐列表
        candidates = recall(user_id)
        if not candidates:
            # 如果没有候选商品（如没有用户点击数据），随机推荐商品
            candidates = products['asin'].sample(500).tolist()
        coarse_ranked = coarse_ranking(candidates)
        fine_ranked = fine_ranking(user_id, coarse_ranked)
        final_recommendations = re_ranking(user_id, fine_ranked)
        if not final_recommendations:
            # 如果最终推荐为空，随机推荐商品
            final_recommendations = products['asin'].sample(500).tolist()
        # 存储到缓存
        redis_client.setex(f"recommendations:{user_id}", 3600, json.dumps(final_recommendations))
        return final_recommendations


@app.route('/products', methods=['GET'])
def get_recommendations():
    user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('per_page', 20))
    q = request.args.get('q')
    if q:
        data = pd.read_sql_query("SELECT * FROM products WHERE title like '%" + q + "%' limit 100;", conn)
        return jsonify({
            'user_id': user_id,
            'total': 'total_items',
            'pages': 'total_pages',
            'current_page': page,
            'per_page': page_size,
            'products': ''
        })

    final_recommendations = get_offline_recommendations(user_id)

    if not final_recommendations:
        # If products DataFrame is empty, return an error message
        if products.empty:
            return jsonify({'message': 'No products available.'}), 404
        else:
            # 如果没有推荐结果，从数据库中随机抽取一些商品作为推荐
            sample_size = min(page_size, len(products))
            if sample_size > 0:
                paged_recommendations = products['asin'].sample(sample_size).tolist()
            else:
                return jsonify({'message': 'No products available for sampling.'}), 404
            total_items = len(paged_recommendations)
            total_pages = 1  # 因为我们只返回了一页的随机商品
    else:
        total_items = len(final_recommendations)
        total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        paged_recommendations = final_recommendations[start_idx:end_idx]

    # 返回商品的详细信息
    if not paged_recommendations:
        return jsonify({'message': 'No products to recommend.'}), 404
    recommended_products = products[products['asin'].isin(paged_recommendations)].to_dict(orient='records')

    return jsonify({
        'user_id': user_id,
        'total': total_items,
        'pages': total_pages,
        'current_page': page,
        'per_page': page_size,
        'products': recommended_products
    })


# 商品点击接口
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
        # 更新内存中的 user_clicks 数据
        global user_clicks
        user_clicks = pd.concat(
            [user_clicks, pd.DataFrame({'user_id': [user_id], 'asin': [asin], 'click_time': [click_time]})],
            ignore_index=True)
        # 更新推荐模型
        user_behavior_update(user_id, asin)
    except Exception as e:
        logging.error(f"Error recording click: {e}")
        return jsonify({'status': 'error', 'message': 'Failed to record click.'}), 500
    # 获取被点击的商品详情
    product = products[products['asin'] == asin]
    product_data = product.to_dict(orient='records')
    if product_data:
        product_data = product_data[0]  # 获取单个商品的字典
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


def update_user_profile(user_id, asin):
    category = products.set_index('asin').loc[asin]['category_id']
    if pd.isna(category):
        return  # 如果商品没有类别，直接返回
    user_profile = user_profiles.get(user_id, {})
    user_profile[str(int(category))] = user_profile.get(str(int(category)), 0) + 1  # 更新点击次数
    total = sum(user_profile.values())
    for cat in user_profile:
        user_profile[cat] /= total  # 重新归一化偏好程度
    user_profiles[user_id] = user_profile


def user_behavior_update(user_id, asin):
    # 更新用户画像
    update_user_profile(user_id, asin)
    # 清除缓存
    redis_client.delete(f"recommendations:{user_id}")
    # 模型重新训练放在定时任务中，不再实时更新


# 定义一个函数，每日重训模型
def retrain_model():
    global user_item_matrix, user_item_sparse, decomposed_matrix, SVD
    print("Retraining model...")
    if user_clicks.empty:
        print("No user clicks data available.")
        return
    # 重新构建用户-商品交互矩阵
    user_item_matrix = user_clicks.groupby(['user_id', 'asin']).size().unstack(fill_value=0)
    user_item_sparse = csr_matrix(user_item_matrix.values)
    # 重新训练SVD模型
    SVD = TruncatedSVD(n_components=20, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)
    print("Model retraining completed.")


# 安排每日凌晨2点重新训练模型
schedule.every().day.at("02:00").do(retrain_model)

if __name__ == '__main__':
    # 在另一个线程中运行调度器
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)


    threading.Thread(target=run_scheduler).start()
    app.run(debug=True)
