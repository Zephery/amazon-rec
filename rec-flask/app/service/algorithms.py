# algorithms.py
import json
import logging

import fakeredis
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity

from app.service.recommendation import user_clicks, products, user_item_matrix, decomposed_matrix, item_latent_vectors, \
    asin_to_category

# 初始化 Redis 连接
redis_client = fakeredis.FakeStrictRedis()


def recall(user_id, top_n=500):
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
    if user_item_matrix is None or decomposed_matrix is None:
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
