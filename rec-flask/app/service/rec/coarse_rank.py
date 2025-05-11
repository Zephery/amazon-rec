import pandas as pd

def coarse_ranking(candidate_items, user_clicks, products):
    if not candidate_items:
        item_popularity = user_clicks['asin'].value_counts()
        if item_popularity.empty:
            item_popularity = products['asin'].value_counts()
    else:
        item_popularity = user_clicks[user_clicks['asin'].isin(candidate_items)]['asin'].value_counts()
        if item_popularity.empty:
            item_popularity = pd.Series(candidate_items)
    ranked_items = item_popularity.index.tolist()
    if not ranked_items:
        ranked_items = candidate_items
    return ranked_items
