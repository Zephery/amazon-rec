# recommendation.py
import logging

import numpy as np
import pandas as pd
import redis
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from db.database import load_products, load_user_clicks, load_user_reviews

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 全局变量
user_item_matrix = None
user_item_sparse = None
decomposed_matrix = None
SVD = None
user_profiles = {}
item_latent_vectors = None
asin_to_category = None
user_ids = None
item_ids = None
model_initialized = False

# 加载数据
products = load_products()
user_clicks = load_user_clicks()
user_reviews = load_user_reviews()


# 示例函数：过滤数据量
def filter_high_activity(df, user_col='user_id', item_col='asin', user_threshold=20, item_threshold=20):
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


# 初始化模型
def initialize_model_with_reviews(user_clicks, user_reviews, products):
    global user_profiles, user_item_sparse, decomposed_matrix, SVD, user_item_matrix, item_latent_vectors, asin_to_category
    global user_ids, item_ids

    # 检查 products 是否为空
    if products.empty:
        logging.error("Products data is empty. Cannot initialize model.")
        return

    # 如果没有用户评论数据，跳过模型初始化
    if user_reviews.empty:
        logging.warning("User reviews data is empty. Skipping model initialization with reviews.")
        return

    # 过滤高活跃用户和商品
    user_reviews = filter_high_activity(user_reviews, user_col='user_id', item_col='asin', user_threshold=20,
                                        item_threshold=20)

    # 可以进行采样，减少数据量
    user_reviews = user_reviews.sample(frac=0.5, random_state=42)

    # 获取唯一的用户和商品ID
    user_ids = user_reviews['user_id'].unique()
    item_ids = user_reviews['asin'].unique()

    # 加权评分
    user_reviews['rating_weighted'] = user_reviews['rating'] * \
                                      (1 + 0.1 * user_reviews['verified_purchase']) * \
                                      (1 + 0.05 * user_reviews['helpful_vote'].fillna(0))

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
    SVD = TruncatedSVD(n_components=10, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)

    # 不再将稀疏矩阵转换为密集矩阵，直接使用稀疏矩阵
    user_item_matrix = user_item_sparse.tocsr()

    # 预先计算 item_latent_vectors
    item_latent_vectors = SVD.components_.T

    # 预先计算 asin_to_category
    asin_to_category = products.set_index('asin')['category_id']

    # 构建用户画像
    # 将用户评论数据与产品数据合并，获取类别信息
    user_reviews_with_category = user_reviews.merge(
        products[['asin', 'category_id']],
        on='asin',
        how='left'
    )

    # 去除类别为空的行
    user_reviews_with_category = user_reviews_with_category.dropna(subset=['category_id'])

    # 计算每个用户对每个类别的交互次数
    user_category_counts = user_reviews_with_category.groupby(['user_id', 'category_id']).size().reset_index(
        name='count')

    # 计算每个用户的总交互次数
    user_total_counts = user_category_counts.groupby('user_id')['count'].sum().reset_index(name='total_count')

    # 合并总交互次数，计算每个类别的偏好度
    user_category_counts = user_category_counts.merge(user_total_counts, on='user_id')
    user_category_counts['preference'] = user_category_counts['count'] / user_category_counts['total_count']

    # 将偏好度转换为用户画像字典
    user_preferences_series = user_category_counts.groupby('user_id').apply(
        lambda x: dict(zip(x['category_id'].astype(str), x['preference']))
    )
    user_profiles = user_preferences_series.to_dict()


def initialize_model():
    global model_initialized
    if not model_initialized:
        print("Initializing model...")
        initialize_model_with_reviews(user_clicks, user_reviews, products)
        model_initialized = True
        print("Model initialization completed.")


# 后续在需要使用模型的函数中，调用 initialize_model()

# 其他推荐相关函数（保持不变，但需要修改对 user_item_matrix 的访问方式，例如使用 user_item_matrix[user_index, :]
# 由于使用了稀疏 CSR 矩阵，需要调整索引方式）
# 示例修改：

def recall(user_id, top_n=500):
    global user_item_matrix, decomposed_matrix, user_clicks, products

    initialize_model()

    if user_item_matrix is None or decomposed_matrix is None:
        logging.warning("User-item matrix or decomposed matrix is not available.")
        return []

    if user_id not in user_ids:
        # 对于新用户，返回全站最热门的商品
        popular_items = user_clicks['asin'].value_counts().head(top_n).index.tolist()
        if not popular_items:
            popular_items = products['asin'].sample(min(top_n, len(products))).tolist()
        return popular_items
    else:
        user_index = np.where(user_ids == user_id)[0][0]
        user_vector = decomposed_matrix[user_index]
        similarity = cosine_similarity([user_vector], decomposed_matrix)[0]

        # 排除自身
        similar_users_indices = similarity.argsort()[::-1][1:]

        # 获取相似用户的交互物品
        similar_users = user_ids[similar_users_indices]
        similar_users_indices = [np.where(user_ids == uid)[0][0] for uid in similar_users]
        similar_users_interactions = user_item_matrix[similar_users_indices]

        # 获取用户已经交互的物品
        user_interacted_items = user_item_matrix[user_index].nonzero()[1]
        user_interacted_set = set(user_interacted_items)

        # 计算候选物品
        candidate_items_scores = similar_users_interactions.sum(axis=0).A1
        candidate_items_indices = np.argsort(-candidate_items_scores)
        candidate_items = [item_ids[idx] for idx in candidate_items_indices if idx not in user_interacted_set]

        return candidate_items[:top_n]

# 其他函数需要根据稀疏矩阵的使用，进行类似的修改。

# 注意，由于代码调整较大，建议您逐步测试每个函数，确保功能正常，同时监控 CPU 和内存的占用情况。


# 定义一个函数，每日重训模型
def retrain_model():
    global user_item_matrix, user_item_sparse, decomposed_matrix, SVD, item_latent_vectors, asin_to_category, products, user_clicks, user_reviews
    print("Retraining model...")
    # 重新加载数据
    products = load_products()
    user_clicks = load_user_clicks()
    user_reviews = load_user_reviews()
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
    # 更新 item_latent_vectors
    item_latent_vectors = pd.DataFrame(SVD.components_.T, index=user_item_matrix.columns)
    # 更新 asin_to_category，如果 products 有更新的话
    asin_to_category = products.set_index('asin')['category_id']
    print("Model retraining completed.")
