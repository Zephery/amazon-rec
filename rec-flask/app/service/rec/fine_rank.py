import numpy as np

from app.service.data_loader import products


def fine_ranking(user_id, ranked_asins):
    prod_df = products[products['asin'].isin(ranked_asins)].copy()
    prod_df['score'] = 0.7 * prod_df['stars'].fillna(0) + 0.3 * np.log1p(prod_df['reviews'].fillna(0))
    return prod_df.sort_values('score', ascending=False)['asin'].tolist()
