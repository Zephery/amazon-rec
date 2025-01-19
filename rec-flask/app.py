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
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity


# 检查数据库是否存在
def check_db():
    if not os.path.exists("db/recommend.db"):
        print("数据库不存在，初始化数据库...")
        initialize_database()
    else:
        print("数据库存在，启动应用...")


# 调用其他 Python 文件生成数据库
def initialize_database():
    from db.amazon_categories import init_categories
    init_categories()
    from db.amazon_products import init_products
    init_products()
    from db.amazon_reviews import init_review
    init_review()


# 在应用启动时检查数据库
check_db()

# 创建 Flask 应用
app = Flask(__name__)
CORS(app)

# 初始化日志
logging.basicConfig(level=logging.INFO)

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 数据库连接
db_lock = threading.Lock()
conn = sqlite3.connect('db/recommend.db', check_same_thread=False)

# 全局变量
user_item_matrix = None
user_item_sparse = None
decomposed_matrix = None
SVD = None
user_profiles = {}


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


products = load_products()


# 加载用户点击数据
def load_user_clicks():
    # 从数据库中读取用户点击数据
    user_clicks = pd.read_sql_query('SELECT * FROM user_clicks', conn)
    return user_clicks


user_clicks = load_user_clicks()


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


user_reviews = load_user_reviews()

# 限制数据规模的常量，可根据实际情况调整
MAX_USERS = 10000  # 最多保留的用户数
MAX_ITEMS = 5000  # 最多保留的商品数


# 示例函数：过滤数据量
def filter_high_activity(df, user_col='user_id', item_col='asin', user_threshold=10, item_threshold=10):
    """
    过滤低频用户和低频商品（只保留高活跃的用户和商品）
    """
    # 保留行为次数或评分数超过阈值的用户和商品
    active_users = df[user_col].value_counts()
    filtered_users = active_users[active_users >= user_threshold].index
    active_items = df[item_col].value_counts()
    filtered_items = active_items[active_items >= item_threshold].index
    return df[(df[user_col].isin(filtered_users)) & (df[item_col].isin(filtered_items))]


# 示例函数：构建稀疏矩阵
def build_sparse_matrix(user_col, item_col, value_col, user_ids, item_ids, data):
    """
    构建稀疏矩阵，避免使用 Pandas 宽表
    """
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}

    # 将用户和商品映射为索引
    rows = data[user_col].map(user_map)
    cols = data[item_col].map(item_map)
    values = data[value_col]

    # 构建 COO 稀疏矩阵并转换为 CSR
    return csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))


# 优化后的用户画像初始化
def initialize_model_with_reviews(user_clicks, user_reviews, products):
    global user_profiles, user_item_sparse, decomposed_matrix, SVD, user_item_matrix

    # 检查 products 是否为空
    if products.empty:
        logging.error("Products data is empty. Cannot initialize model.")
        return

    # 如果没有用户评论数据，跳过模型初始化
    if user_reviews.empty:
        logging.warning("User reviews data is empty. Skipping model initialization with reviews.")
        return

    # 限制评论表规模，只保留高频用户和商品
    user_reviews = filter_high_activity(user_reviews, user_col='user_id', item_col='asin', user_threshold=5,
                                        item_threshold=5)
    user_reviews = user_reviews.head(MAX_USERS * MAX_ITEMS)  # 限制评论表的最大行数

    # 构建用户-商品交互稀疏矩阵
    user_ids = user_reviews['user_id'].unique()[:MAX_USERS]  # 保留最多 MAX_USERS 个用户
    item_ids = user_reviews['asin'].unique()[:MAX_ITEMS]  # 保留最多 MAX_ITEMS 个商品

    # 加权评分
    user_reviews['rating_weighted'] = user_reviews['rating'] * (
            1 + 0.1 * user_reviews['verified_purchase']) * (1 + 0.05 * user_reviews['helpful_vote'].fillna(0))

    # 构建稀疏交互矩阵
    user_item_sparse = build_sparse_matrix(
        user_col='user_id',
        item_col='asin',
        value_col='rating_weighted',
        user_ids=user_ids,
        item_ids=item_ids,
        data=user_reviews
    )

    # 训练 SVD 模型
    SVD = TruncatedSVD(n_components=20, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)

    # 构建用户画像
    category_preferences = []
    for user_id in user_ids:
        # 获取用户评论中涉及的商品类别
        user_items = user_reviews[user_reviews['user_id'] == user_id]['asin']
        user_categories = products[products['asin'].isin(user_items)]['category_id']
        category_preference = user_categories.value_counts(normalize=True)  # 归一化偏好
        category_preferences.append(category_preference.to_dict())

    user_profiles = dict(zip(user_ids, category_preferences))

    # 构建用户-商品交互矩阵，用于后续推荐
    user_item_matrix = pd.DataFrame(user_item_sparse.toarray(), index=user_ids, columns=item_ids)


# 示例调用
initialize_model_with_reviews(user_clicks, user_reviews, products)


def recall(user_id, top_n=500):
    global user_item_matrix, decomposed_matrix

    if user_item_matrix is None or decomposed_matrix is None:
        # 模型尚未训练或无数据，返回空列表
        logging.warning("User-item matrix or decomposed matrix is not available.")
        return []

    if user_id not in user_item_matrix.index:
        # 对于新用户，返回全站最热门的商品
        popular_items = user_clicks['asin'].value_counts().head(top_n).index.tolist()
        if not popular_items:
            # 如果用户点击数据不足，返回随机商品
            popular_items = products['asin'].sample(min(top_n, len(products))).tolist()
        return popular_items
    else:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = decomposed_matrix[user_index]
        similarity = cosine_similarity([user_vector], decomposed_matrix)[0]

        # 排除自身
        similar_users_indices = similarity.argsort()[::-1][1:]

        # 获取相似用户的交互物品
        similar_users = user_item_matrix.index[similar_users_indices]
        similar_users_interactions = user_item_matrix.loc[similar_users]

        # 获取用户已经交互的物品
        user_interacted_items = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])

        # 计算候选物品
        candidate_items_scores = similar_users_interactions.sum(axis=0)
        candidate_items_scores = candidate_items_scores.drop(index=user_interacted_items, errors='ignore')

        # 如果候选物品不足，补充热门物品
        if candidate_items_scores.empty or len(candidate_items_scores) < top_n:
            popular_items = user_clicks['asin'].value_counts().index.difference(user_interacted_items).tolist()
            candidate_items = pd.concat([pd.Series(candidate_items_scores.index), pd.Series(popular_items)],
                                        ignore_index=True).drop_duplicates().head(top_n)
        else:
            candidate_items = candidate_items_scores.sort_values(ascending=False).index.tolist()

        return candidate_items[:top_n]


def coarse_ranking(candidate_items):
    if not candidate_items:
        # 如果候选列表为空，返回全站最热门的商品
        item_popularity = user_clicks['asin'].value_counts()
        if item_popularity.empty:
            item_popularity = products['asin'].value_counts()
    else:
        item_popularity = user_clicks[user_clicks['asin'].isin(candidate_items)]['asin'].value_counts()
        if item_popularity.empty:
            item_popularity = pd.Series(candidate_items)

    ranked_items = item_popularity.index.tolist()
    if not ranked_items:
        ranked_items = candidate_items
    return ranked_items


def fine_ranking(user_id, ranked_items):
    global user_item_matrix, decomposed_matrix, SVD
    if user_item_matrix is None or decomposed_matrix is None or SVD is None:
        # 模型尚未训练或无数据，直接返回粗排结果
        return ranked_items
    if user_id not in user_item_matrix.index:
        # 新用户直接返回粗排结果
        return ranked_items
    else:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = decomposed_matrix[user_index]
        item_indices = [i for i, item in enumerate(ranked_items) if item in user_item_matrix.columns]
        if not item_indices:
            return ranked_items  # 如果没有匹配的物品，直接返回粗排结果
        item_vectors = SVD.components_.T[[user_item_matrix.columns.get_loc(ranked_items[i]) for i in item_indices]]
        scores = np.dot(item_vectors, user_vector)
        item_scores = pd.Series(scores, index=[ranked_items[i] for i in item_indices])
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
            category = 'unknown'  # 将未知类别标记为 'unknown'
        if category not in category_items:
            category_items[category] = []
        category_items[category].append(item)
    # 按照用户偏好重排商品
    final_ranked_items = []
    added_items = set()

    # 首先添加用户偏好类别的商品
    for category in preferred_categories:
        items = category_items.get(category, [])
        for item in items:
            if item not in added_items:
                final_ranked_items.append(item)
                added_items.add(item)

    # 添加未知类别的商品
    unknown_items = category_items.get('unknown', [])
    for item in unknown_items:
        if item not in added_items:
            final_ranked_items.append(item)
            added_items.add(item)

    # 添加剩余的商品
    for category, items in category_items.items():
        if category not in preferred_categories and category != 'unknown':
            for item in items:
                if item not in added_items:
                    final_ranked_items.append(item)
                    added_items.add(item)

    # 确保最终的推荐列表完整
    for item in fine_ranked_items:
        if item not in added_items:
            final_ranked_items.append(item)
            added_items.add(item)

    if not final_ranked_items:
        # 如果最终列表为空，返回原始的 fine_ranked_items
        final_ranked_items = fine_ranked_items

    logging.info(f"Final ranked items for {user_id}: {final_ranked_items}")
    return final_ranked_items


def get_offline_recommendations(user_id):
    cached_recommendations = redis_client.get(f"recommendations:{user_id}")
    if cached_recommendations is not None and str(cached_recommendations.decode('utf-8')) != '[]':
        cached_recommendations = cached_recommendations.decode('utf-8')
        return json.loads(cached_recommendations)
    else:
        # 当缓存中没有推荐时，生成推荐列表
        candidates = recall(user_id)
        if not candidates or len(candidates) == 0:
            # 如果没有候选商品，随机推荐商品
            candidates = products['asin'].sample(min(500, len(products))).tolist()
        coarse_ranked = coarse_ranking(candidates)
        fine_ranked = fine_ranking(user_id, coarse_ranked)
        final_recommendations = re_ranking(user_id, fine_ranked)
        if not final_recommendations or len(final_recommendations) == 0:
            # 如果最终推荐为空，随机推荐商品
            final_recommendations = products['asin'].sample(min(500, len(products))).tolist()
        # 存储到缓存
        redis_client.setex(f"recommendations:{user_id}", 3600, json.dumps(final_recommendations))
        return final_recommendations


def get_similar_users(user_id, top_k=5):
    """
    找到与当前用户最相似的前 top_k 个用户
    """
    global user_item_matrix, decomposed_matrix

    if user_item_matrix is None or decomposed_matrix is None:
        return []

    if user_id not in user_item_matrix.index:
        return []

    # 获取当前用户的向量
    user_index = user_item_matrix.index.get_loc(user_id)
    user_vector = decomposed_matrix[user_index]

    # 计算与所有用户的余弦相似度
    similarity = cosine_similarity([user_vector], decomposed_matrix)[0]

    # 排除自身，找到最相似的用户
    similar_users_indices = similarity.argsort()[::-1][1:top_k + 1]
    similar_users = user_item_matrix.index[similar_users_indices]

    return similar_users.tolist()


def recommend_based_on_similar_users(user_id, top_n=20):
    """
    基于相似用户生成推荐列表
    """
    similar_users = get_similar_users(user_id, top_k=5)
    if not similar_users:
        return []

    # 获取相似用户的交互商品
    similar_users_interactions = user_item_matrix.loc[similar_users]

    # 获取当前用户已经交互的商品
    user_interacted_items = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])

    # 计算候选商品
    candidate_items_scores = similar_users_interactions.sum(axis=0)
    candidate_items_scores = candidate_items_scores.drop(index=user_interacted_items, errors='ignore')

    # 如果候选商品不足，补充热门商品
    if candidate_items_scores.empty or len(candidate_items_scores) < top_n:
        popular_items = user_clicks['asin'].value_counts().index.difference(user_interacted_items).tolist()
        candidate_items = pd.concat([pd.Series(candidate_items_scores.index), pd.Series(popular_items)],
                                    ignore_index=True).drop_duplicates().head(top_n)
    else:
        candidate_items = candidate_items_scores.sort_values(ascending=False).index.tolist()

    return candidate_items[:top_n]


@app.route('/')
def index():
    return "ok"


@app.route('/products', methods=['GET'])
def get_recommendations():
    user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
    page = int(request.args.get('page', 1))
    page_size = int(request.args.get('per_page', 20))
    q = request.args.get('q')

    if q:
        data = products[products['title'].str.contains(q, case=False, na=False)].copy()
        total_items = len(data)
        total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
        start_idx = (page - 1) * page_size
        end_idx = start_idx + page_size
        data = data.iloc[start_idx:end_idx].to_dict(orient='records')
        return jsonify({
            'user_id': user_id,
            'total': total_items,
            'pages': total_pages,
            'current_page': page,
            'per_page': page_size,
            'products': data
        })

    # 优先使用用户画像生成推荐列表
    if user_id in user_profiles:
        candidates = recall(user_id)
        if not candidates or len(candidates) == 0:
            candidates = products['asin'].sample(min(500, len(products))).tolist()
        coarse_ranked = coarse_ranking(candidates)
        fine_ranked = fine_ranking(user_id, coarse_ranked)
        final_recommendations = re_ranking(user_id, fine_ranked)
    else:
        # 如果用户画像不存在，使用相似用户生成推荐列表
        final_recommendations = recommend_based_on_similar_users(user_id, top_n=500)

    if not final_recommendations:
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


def update_recommendations_after_click(user_id, asin):
    # 更新用户画像
    update_user_profile(user_id, asin)
    # 立即生成新的推荐列表
    candidates = recall(user_id)
    if not candidates or len(candidates) == 0:
        candidates = products['asin'].sample(min(500, len(products))).tolist()
    coarse_ranked = coarse_ranking(candidates)
    fine_ranked = fine_ranking(user_id, coarse_ranked)
    final_recommendations = re_ranking(user_id, fine_ranked)
    if not final_recommendations or len(final_recommendations) == 0:
        final_recommendations = products['asin'].sample(min(500, len(products))).tolist()
    # 更新缓存
    redis_client.setex(f"recommendations:{user_id}", 3600, json.dumps(final_recommendations))


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
        # 在用户点击商品后，立即更新推荐列表并缓存
        update_recommendations_after_click(user_id, asin)
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
    global user_profiles
    try:
        category = products.set_index('asin').loc[asin]['category_id']
    except KeyError:
        # 如果商品不存在或没有类别，直接返回
        return
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
        user_item_matrix = None
        decomposed_matrix = None
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
    retrain_model()


    # 在另一个线程中运行调度器
    def run_scheduler():
        while True:
            schedule.run_pending()
            time.sleep(1)


    threading.Thread(target=run_scheduler).start()
    app.run(debug=True)
