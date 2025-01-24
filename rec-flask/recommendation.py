# recommendation.py
import json
import logging

import numpy as np
import pandas as pd
import redis
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

from database import load_products, load_user_clicks, load_user_reviews

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

# 加载数据
products = load_products()
user_clicks = load_user_clicks()
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


# 初始化模型
def initialize_model_with_reviews(user_clicks, user_reviews, products):
    global user_profiles, user_item_sparse, decomposed_matrix, SVD, user_item_matrix, item_latent_vectors, asin_to_category

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

    # 预先计算 item_latent_vectors
    item_latent_vectors = pd.DataFrame(SVD.components_.T, index=user_item_matrix.columns)

    # 预先计算 asin_to_category
    asin_to_category = products.set_index('asin')['category_id']


# 调用初始化函数
initialize_model_with_reviews(user_clicks, user_reviews, products)


# 其他推荐相关函数
def recall(user_id, top_n=500):
    global user_item_matrix, decomposed_matrix, user_clicks, products

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
    global user_clicks, products

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
    global user_item_matrix, decomposed_matrix, SVD, item_latent_vectors, asin_to_category, user_profiles
    if user_item_matrix is None or decomposed_matrix is None or SVD is None:
        # 模型尚未训练或无数据，直接返回粗排结果
        return ranked_items
    if user_id not in user_item_matrix.index:
        # 新用户直接返回粗排结果
        return ranked_items
    else:
        user_index = user_item_matrix.index.get_loc(user_id)
        user_vector = decomposed_matrix[user_index]
        # 获取商品的潜在向量
        item_vectors = item_latent_vectors.reindex(ranked_items).dropna()
        if item_vectors.empty:
            return ranked_items  # 如果没有匹配的物品，直接返回粗排结果
        item_names = item_vectors.index.tolist()
        item_vectors = item_vectors.values

        # 计算评分
        scores = np.dot(item_vectors, user_vector)
        item_scores = pd.Series(scores, index=item_names)

        # 调整评分，利用用户画像
        user_profile = user_profiles.get(user_id, {})
        # 获取商品的类别
        item_categories = asin_to_category.reindex(item_scores.index)
        # 计算偏好
        preferences = item_categories.apply(
            lambda x: user_profile.get(str(int(x)), 0) if pd.notna(x) else 0)
        adjusted_scores = item_scores * (1 + preferences.values)

        # 更新评分
        item_scores = pd.Series(adjusted_scores.values, index=item_scores.index)
        # 排序
        fine_ranked_items = item_scores.sort_values(ascending=False).index.tolist()
        return fine_ranked_items


def re_ranking(user_id, fine_ranked_items):
    global user_profiles, asin_to_category
    user_profile = user_profiles.get(user_id, {})
    logging.info(f"User profile for {user_id}: {user_profile}")
    if not user_profile:
        return fine_ranked_items
    # 获取用户偏好的类别列表
    preferred_categories = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
    preferred_categories = [int(cat) for cat, _ in preferred_categories]
    logging.info(f"Preferred categories: {preferred_categories}")
    # 获取商品的类别信息
    item_categories = asin_to_category.reindex(fine_ranked_items)
    # logging.info(f"Item categories: {item_categories}")
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

    # logging.info(f"Final ranked items for {user_id}: {final_ranked_items}")
    return final_ranked_items


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
    if user_id in user_item_matrix.index:
        user_interacted_items = set(user_item_matrix.columns[user_item_matrix.loc[user_id] > 0])
    else:
        user_interacted_items = set()

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


def update_user_profile(user_id, asin):
    global user_profiles, asin_to_category
    try:
        category = asin_to_category.get(asin)
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
