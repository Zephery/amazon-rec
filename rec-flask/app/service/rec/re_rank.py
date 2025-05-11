import pandas as pd

from app.service.rec.recommendation_data import user_profiles, asin_to_category


def re_ranking(user_id, fine_ranked_items):
    user_profile = user_profiles.get(user_id, {})
    if not user_profile:
        return fine_ranked_items
    preferred_categories = sorted(user_profile.items(), key=lambda x: x[1], reverse=True)
    preferred_categories = [int(cat) for cat, _ in preferred_categories]
    item_categories = asin_to_category.reindex(fine_ranked_items)
    category_items = {}
    for item in fine_ranked_items:
        category = item_categories.get(item)
        if pd.isna(category):
            category = 'unknown'
        category_items.setdefault(category, []).append(item)
    final_ranked_items, added_items = [], set()
    for category in preferred_categories:
        items = category_items.get(category, [])
        for item in items:
            if item not in added_items:
                final_ranked_items.append(item)
                added_items.add(item)
    for item in category_items.get('unknown', []):
        if item not in added_items:
            final_ranked_items.append(item)
            added_items.add(item)
    for category, items in category_items.items():
        if category not in preferred_categories and category != 'unknown':
            for item in items:
                if item not in added_items:
                    final_ranked_items.append(item)
                    added_items.add(item)
    for item in fine_ranked_items:
        if item not in added_items:
            final_ranked_items.append(item)
            added_items.add(item)
    if not final_ranked_items:
        final_ranked_items = fine_ranked_items
    return final_ranked_items
