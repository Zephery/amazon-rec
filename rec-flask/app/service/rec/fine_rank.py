import numpy as np
import pandas as pd

from app.service.rec.recommendation_data import (
    user_item_matrix, decomposed_matrix, item_latent_vectors,
    asin_to_category, user_profiles, user_ids
)


def fine_ranking(user_id, ranked_items):
    if user_item_matrix is None or decomposed_matrix is None or user_id not in user_profiles:
        return ranked_items
    user_index = np.where(user_ids == user_id)[0][0]
    user_vector = decomposed_matrix[user_index]
    item_vectors = item_latent_vectors.reindex(ranked_items).dropna()
    if item_vectors.empty:
        return ranked_items
    item_names = item_vectors.index.tolist()
    item_vectors = item_vectors.values
    scores = np.dot(item_vectors, user_vector)
    item_scores = pd.Series(scores, index=item_names)
    user_profile = user_profiles.get(user_id, {})
    item_categories = asin_to_category.reindex(item_scores.index)
    preferences = item_categories.apply(lambda x: user_profile.get(str(int(x)), 0) if pd.notna(x) else 0)
    adjusted_scores = item_scores * (1 + preferences.values)
    item_scores = pd.Series(adjusted_scores.values, index=item_scores.index)
    fine_ranked_items = item_scores.sort_values(ascending=False).index.tolist()
    return fine_ranked_items
