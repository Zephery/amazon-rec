import numpy as np
import random

from app.service.data_loader import products, user_clicks


def coarse_ranking(candidate_asins, return_score=False):
    cand_df = products[products['asin'].isin(candidate_asins)].copy()
    
    # 用户点击分
    cand_df['click_score'] = cand_df['asin'].map(user_clicks['asin'].value_counts())
    cand_df['click_score'] = cand_df['click_score'].fillna(0)
    
    # 商品新鲜度分（假设有create_time字段，越新分越高）
    if 'create_time' in cand_df.columns:
        now = cand_df['create_time'].max()
        cand_df['fresh_score'] = (now - cand_df['create_time']).dt.days * -1
        cand_df['fresh_score'] = (cand_df['fresh_score'] - cand_df['fresh_score'].min()) / (
                cand_df['fresh_score'].max() - cand_df['fresh_score'].min() + 1e-6)
    else:
        cand_df['fresh_score'] = 0
    
    # 价格区间分（假设有price字段，价格适中得分高）
    if 'price' in cand_df.columns:
        price = cand_df['price'].fillna(0)
        cand_df['price_score'] = np.exp(-((price - price.median()) ** 2) / (2 * (price.std() + 1e-6) ** 2))
    else:
        cand_df['price_score'] = 0
    
    # 品牌多样性分（假设有brand字段，品牌分布均匀得分高）
    if 'brand' in cand_df.columns:
        brand_counts = cand_df['brand'].value_counts()
        cand_df['brand_score'] = cand_df['brand'].map(lambda b: 1 / (brand_counts[b] + 1e-6))
    else:
        cand_df['brand_score'] = 0
    
    # 添加随机噪声到各个分数
    noise_scale = 0.1
    cand_df['click_score'] += np.random.normal(0, noise_scale, len(cand_df))
    cand_df['fresh_score'] += np.random.normal(0, noise_scale, len(cand_df))
    cand_df['price_score'] += np.random.normal(0, noise_scale, len(cand_df))
    cand_df['brand_score'] += np.random.normal(0, noise_scale, len(cand_df))
    
    # 随机化权重分配
    weights = {
        'click': random.uniform(0.3, 0.5),
        'stars': random.uniform(0.15, 0.25),
        'reviews': random.uniform(0.1, 0.2),
        'fresh': random.uniform(0.05, 0.15),
        'price': random.uniform(0.05, 0.15),
        'brand': random.uniform(0.02, 0.08)
    }
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # 综合排序分
    cand_df['final_score'] = (
            weights['click'] * cand_df['click_score'] +
            weights['stars'] * cand_df['stars'].fillna(0) +
            weights['reviews'] * cand_df['reviews'].fillna(0) +
            weights['fresh'] * cand_df['fresh_score'] +
            weights['price'] * cand_df['price_score'] +
            weights['brand'] * cand_df['brand_score']
    )
    
    # 添加最终随机扰动
    final_noise = np.random.normal(0, 0.05, len(cand_df))
    cand_df['final_score'] += final_noise
    
    cand_df = cand_df.sort_values('final_score', ascending=False)
    
    if return_score:
        return cand_df[['asin', 'final_score']].rename(columns={'final_score': 'coarse_score'})
    else:
        return cand_df['asin'].tolist()
