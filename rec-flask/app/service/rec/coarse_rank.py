from app.service.data_loader import products, user_clicks


def coarse_ranking(candidate_asins):
    cand_df = products[products['asin'].isin(candidate_asins)].copy()
    cand_df['click_score'] = cand_df['asin'].map(user_clicks['asin'].value_counts())
    cand_df['click_score'] = cand_df['click_score'].fillna(0)
    cand_df['final_score'] = (
        0.6 * cand_df['click_score'] +
        0.2 * cand_df['stars'].fillna(0) +
        0.2 * cand_df['reviews'].fillna(0)
    )
    sorted_asins = cand_df.sort_values('final_score', ascending=False)['asin'].tolist()
    return sorted_asins
