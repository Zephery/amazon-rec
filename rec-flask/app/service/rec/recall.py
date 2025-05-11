from app.service.rec.recommendation_data import (
    user_clicks
)


def recall_cf(user_id, top_n=500):
    from app.service.rec.recommendation_data import (
        user_clicks, products, user_item_matrix, decomposed_matrix, user_ids, item_ids
    )
    import numpy as np
    from sklearn.metrics.pairwise import cosine_similarity

    if user_item_matrix is None or decomposed_matrix is None:
        return []
    if user_id not in user_item_matrix.index:
        return user_clicks['asin'].value_counts().head(top_n).index.tolist()

    # 计算相似用户
    if user_ids is not None and user_id in user_ids:
        user_idx = np.where(user_ids == user_id)[0][0]
    else:
        user_idx = list(user_item_matrix.index).index(user_id)
    user_vector = decomposed_matrix[user_idx]
    similarity = cosine_similarity([user_vector], decomposed_matrix)[0]
    similar_users_indices = similarity.argsort()[::-1][1:]
    similar_users = user_ids[similar_users_indices]
    similar_users_indices = [np.where(user_ids == uid)[0][0] for uid in similar_users]
    similar_users_interactions = user_item_matrix.iloc[similar_users_indices]

    # 推荐候选（按 DataFrame 的索引方式）
    row = user_item_matrix.loc[user_id]
    user_interacted_set = set(np.where(row.values > 0)[0])

    candidate_items_scores = similar_users_interactions.sum(axis=0).values
    candidate_items_indices = np.argsort(-candidate_items_scores)
    candidate_items = []
    for idx in candidate_items_indices:
        if idx not in user_interacted_set:
            asin = item_ids[idx] if isinstance(item_ids, np.ndarray) else item_ids[idx]
            candidate_items.append(str(asin))
        if len(candidate_items) >= top_n:
            break
    return candidate_items


def recall_popular(top_n=500):
    return user_clicks['asin'].value_counts().head(top_n).index.tolist()
