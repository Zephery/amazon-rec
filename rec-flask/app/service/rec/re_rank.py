from app.service.data_loader import products, user_clicks
import numpy as np
import pandas as pd
import random

def re_ranking(user_id, coarse_ranked_asins, return_score=False):
    history = user_clicks[user_clicks['user_id'] == user_id]['asin']
    his_cates = products[products['asin'].isin(history)]['category_id']
    prod_df = products[products['asin'].isin(coarse_ranked_asins)].copy()
    
    # 相关性：保留原有排序分数，但添加随机扰动
    prod_df['origin_rank'] = np.arange(len(prod_df), 0, -1)
    rank_noise = np.random.normal(0, 0.1, len(prod_df))
    prod_df['origin_rank'] += rank_noise
    
    # 多样性：优先用户高频品类，但也保证其他品类有曝光
    if not his_cates.empty:
        top_cates = his_cates.value_counts().head(3).index.tolist()
        prod_df['cate_priority'] = prod_df['category_id'].apply(lambda c: 1 if c in top_cates else 0)
        # 添加随机性，让非高频品类也有机会
        cate_priority_noise = np.random.random(len(prod_df)) * 0.3
        prod_df['cate_priority'] += cate_priority_noise
    else:
        prod_df['cate_priority'] = 0
    
    # 新颖性：新品优先（假设有create_time字段）
    if 'create_time' in prod_df.columns:
        now = prod_df['create_time'].max()
        prod_df['fresh_score'] = (now - prod_df['create_time']).dt.days * -1
        prod_df['fresh_score'] = (prod_df['fresh_score'] - prod_df['fresh_score'].min()) / (prod_df['fresh_score'].max() - prod_df['fresh_score'].min() + 1e-6)
    else:
        prod_df['fresh_score'] = 0
    
    # 随机化权重分配
    weights = {
        'origin_rank': random.uniform(0.4, 0.6),
        'cate_priority': random.uniform(0.15, 0.25),
        'fresh_score': random.uniform(0.2, 0.4)
    }
    
    # 归一化权重
    total_weight = sum(weights.values())
    weights = {k: v/total_weight for k, v in weights.items()}
    
    # 多目标融合重排
    prod_df['final_score'] = (
        weights['origin_rank'] * prod_df['origin_rank'] +
        weights['cate_priority'] * prod_df['cate_priority'] +
        weights['fresh_score'] * prod_df['fresh_score']
    )
    
    # 添加最终随机扰动
    final_noise = np.random.normal(0, 0.05, len(prod_df))
    prod_df['final_score'] += final_noise
    
    prod_df = prod_df.sort_values('final_score', ascending=False)
    
    # 增强多样性：每个品类最多N个，但增加随机性和数量
    max_per_cate = random.randint(30, 50)  # 大幅增加每类最大数量
    cate_count = {}
    diverse_asins = []
    diverse_scores = []
    
    # 随机打乱商品顺序
    prod_df = prod_df.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)
    
    # 第一轮：按品类限制采样
    for _, row in prod_df.iterrows():
        c = row['category_id']
        if cate_count.get(c, 0) < max_per_cate:
            diverse_asins.append(row['asin'])
            diverse_scores.append(row['final_score'])
            cate_count[c] = cate_count.get(c, 0) + 1
    
    # 第二轮：如果品类不够多样，补充其他品类
    if len(cate_count) < 3:  # 如果品类少于3个，补充更多品类
        remaining_df = prod_df[~prod_df['asin'].isin(diverse_asins)]
        for _, row in remaining_df.iterrows():
            c = row['category_id']
            if c not in cate_count or cate_count[c] < max_per_cate // 2:
                diverse_asins.append(row['asin'])
                diverse_scores.append(row['final_score'])
                cate_count[c] = cate_count.get(c, 0) + 1
            if len(diverse_asins) >= len(prod_df) * 0.8:  # 保留80%的商品
                break
    
    # 最终随机打乱
    combined = list(zip(diverse_asins, diverse_scores))
    random.shuffle(combined)
    diverse_asins, diverse_scores = zip(*combined) if combined else ([], [])
    
    if return_score:
        return pd.DataFrame({'asin': diverse_asins, 're_score': diverse_scores})
    else:
        return list(diverse_asins)
