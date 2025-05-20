from app.service.data_loader import products, user_clicks


def re_ranking(user_id, coarse_ranked_asins):
    history = user_clicks[user_clicks['user_id'] == user_id]['asin']
    his_cates = products[products['asin'].isin(history)]['category_id']
    if his_cates.empty:
        return coarse_ranked_asins
    top_cates = his_cates.value_counts().head(3).index.tolist()
    prod_df = products[products['asin'].isin(coarse_ranked_asins)].copy()
    prod_df['cate_priority'] = prod_df['category_id'].apply(lambda c: 1 if c in top_cates else 0)
    prod_df = prod_df.sort_values(['cate_priority', 'stars', 'reviews'], ascending=[False, False, False])
    return prod_df['asin'].tolist()
