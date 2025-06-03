import numpy as np

from app.service.data_loader import products, user_clicks


def coarse_ranking(candidate_asins):
    cand_df = products[products['asin'].isin(candidate_asins)].copy()
    print(cand_df)
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
    # 综合排序分
    cand_df['final_score'] = (
            0.4 * cand_df['click_score'] +
            0.2 * cand_df['stars'].fillna(0) +
            0.15 * cand_df['reviews'].fillna(0) +
            0.1 * cand_df['fresh_score'] +
            0.1 * cand_df['price_score'] +
            0.05 * cand_df['brand_score']
    )
    sorted_asins = cand_df.sort_values('final_score', ascending=False)['asin'].tolist()
    return sorted_asins
