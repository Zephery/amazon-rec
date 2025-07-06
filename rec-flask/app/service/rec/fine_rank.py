import numpy as np
import random
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
    
    # 添加随机噪声到各个特征
    noise_scale = 0.1
    prod_df['stars_noise'] = np.random.normal(0, noise_scale, len(prod_df))
    prod_df['reviews_noise'] = np.random.normal(0, noise_scale, len(prod_df))
    prod_df['cate_clicks_noise'] = np.random.normal(0, noise_scale, len(prod_df))
    prod_df['brand_clicks_noise'] = np.random.normal(0, noise_scale, len(prod_df))
    
    # 随机化权重分配
    weights = {
        'stars': random.uniform(0.4, 0.6),
        'reviews': random.uniform(0.15, 0.25),
        'cate_clicks': random.uniform(0.15, 0.25),
        'brand_clicks': random.uniform(0.05, 0.15)
    }
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # 综合排序分
    prod_df['score'] = (
        weights['stars'] * (prod_df['stars'].fillna(0) + prod_df['stars_noise']) +
        weights['reviews'] * (np.log1p(prod_df['reviews'].fillna(0)) + prod_df['reviews_noise']) +
        weights['cate_clicks'] * (prod_df['cate_clicks'] + prod_df['cate_clicks_noise']) +
        weights['brand_clicks'] * (prod_df['brand_clicks'] + prod_df['brand_clicks_noise'])
    )
    
    # 添加最终随机扰动
    final_noise = np.random.normal(0, 0.05, len(prod_df))
    prod_df['score'] += final_noise
    
    prod_df = prod_df.sort_values('score', ascending=False)
    
    if return_score:
        return prod_df[['asin', 'score']].rename(columns={'score': 'fine_score'})
    else:
        return prod_df['asin'].tolist()
