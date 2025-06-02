from app.service.data_loader import products, user_clicks
import numpy as np

def re_ranking(user_id, coarse_ranked_asins):
    history = user_clicks[user_clicks['user_id'] == user_id]['asin']
    his_cates = products[products['asin'].isin(history)]['category_id']
    prod_df = products[products['asin'].isin(coarse_ranked_asins)].copy()
    # 相关性：保留原有排序分数
    prod_df['origin_rank'] = np.arange(len(prod_df), 0, -1)
    # 多样性：优先用户高频品类，但也保证其他品类有曝光
    if not his_cates.empty:
        top_cates = his_cates.value_counts().head(3).index.tolist()
        prod_df['cate_priority'] = prod_df['category_id'].apply(lambda c: 1 if c in top_cates else 0)
    else:
        prod_df['cate_priority'] = 0
    # 新颖性：新品优先（假设有create_time字段）
    if 'create_time' in prod_df.columns:
        now = prod_df['create_time'].max()
        prod_df['fresh_score'] = (now - prod_df['create_time']).dt.days * -1
        prod_df['fresh_score'] = (prod_df['fresh_score'] - prod_df['fresh_score'].min()) / (prod_df['fresh_score'].max() - prod_df['fresh_score'].min() + 1e-6)
    else:
        prod_df['fresh_score'] = 0
    # 多目标融合重排
    prod_df['final_score'] = (
        0.5 * prod_df['origin_rank'] +
        0.2 * prod_df['cate_priority'] +
        0.3 * prod_df['fresh_score']
    )
    prod_df = prod_df.sort_values('final_score', ascending=False)
    # 保证多样性：每个品类最多N个
    max_per_cate = 20
    cate_count = {}
    diverse_asins = []
    for _, row in prod_df.iterrows():
        c = row['category_id']
        if cate_count.get(c, 0) < max_per_cate:
            diverse_asins.append(row['asin'])
            cate_count[c] = cate_count.get(c, 0) + 1
    return diverse_asins
