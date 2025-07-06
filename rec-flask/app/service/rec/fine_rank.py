import numpy as np
from app.service.data_loader import products, user_clicks

def fine_ranking(user_id, ranked_asins, return_score=False):
    prod_df = products[products['asin'].isin(ranked_asins)].copy()
    # 用户历史点击该品类次数
    user_history = user_clicks[user_clicks['user_id'] == user_id]
    cate_count = user_history.merge(products[['asin', 'category_id']], on='asin', how='left')['category_id'].value_counts()
    prod_df['cate_clicks'] = prod_df['category_id'].map(lambda c: cate_count.get(c, 0))
    # 用户历史点击该品牌次数（如有brand字段）
    if 'brand' in prod_df.columns:
        brand_count = user_history.merge(products[['asin', 'brand']], on='asin', how='left')['brand'].value_counts()
        prod_df['brand_clicks'] = prod_df['brand'].map(lambda b: brand_count.get(b, 0))
    else:
        prod_df['brand_clicks'] = 0
    # 综合排序分
    prod_df['score'] = (
        0.5 * prod_df['stars'].fillna(0) +
        0.2 * np.log1p(prod_df['reviews'].fillna(0)) +
        0.2 * prod_df['cate_clicks'] +
        0.1 * prod_df['brand_clicks']
    )
    prod_df = prod_df.sort_values('score', ascending=False)
    if return_score:
        return prod_df[['asin', 'score']].rename(columns={'score': 'fine_score'})
    else:
        return prod_df['asin'].tolist()
