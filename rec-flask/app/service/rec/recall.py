import random

from app.service.data_loader import products, user_clicks
from app.service.user.user_profile import user_profiles


def recall(user_id, top_n=500, hybrid=True):
    user_clicked_asins = set(user_profiles.get(user_id, []))
    # 兜底热门
    hot_asins = list(products.sort_values('reviews', ascending=False)['asin'].head(top_n * 2))

    # 用户协同过滤（与原方法类似，只拉别人的点击）
    others = user_clicks[
        (user_clicks['asin'].isin(user_clicked_asins)) &
        (user_clicks['user_id'] != user_id)
        ]
    usercf_candidates = []
    if not others.empty:
        candidate_asins = (
            others.groupby('user_id')['asin'].agg(list).explode()
        )
        candidate_counts = candidate_asins[~candidate_asins.isin(user_clicked_asins)].value_counts()
        usercf_candidates = list(candidate_counts.head(top_n).index)

    # 物品协同召回：你点过的商品，还常跟哪些商品被一起点
    itemcf_candidates = []
    if user_clicked_asins:
        co_clicks = user_clicks[user_clicks['asin'].isin(user_clicked_asins)]
        co_users = set(co_clicks['user_id']) - {user_id}
        co_user_clicks = user_clicks[user_clicks['user_id'].isin(co_users)]
        itemcf_candidates = list(
            co_user_clicks[~co_user_clicks['asin'].isin(user_clicked_asins)]['asin'].value_counts().head(top_n).index
        )

    # 类别相关的热门商品
    cate_candidates = []
    if user_clicked_asins:
        user_cates = products[products['asin'].isin(user_clicked_asins)]['category_id'].dropna().unique()
        cate_candidates = list(
            products[
                products['category_id'].isin(user_cates) &
                ~products['asin'].isin(user_clicked_asins)
                ]
            .sort_values('reviews', ascending=False)['asin'].head(top_n)
        )

    # 融合多路召回，补充热门和随机
    all_candidates = (
            usercf_candidates +
            itemcf_candidates +
            cate_candidates +
            hot_asins
    )
    # 去重且保序
    seen = set()
    candidates = [x for x in all_candidates if not (x in seen or seen.add(x))]
    # 增加多样性-可“打乱”或者“按分层抽样”
    random.shuffle(candidates)

    # 返回top_n条
    final_candidates = candidates[:top_n]
    # 如果还是不够，用热门兜住
    if len(final_candidates) < top_n:
        for asin in hot_asins:
            if asin not in final_candidates:
                final_candidates.append(asin)
            if len(final_candidates) >= top_n:
                break
    return final_candidates


def recommend_based_on_similar_users(user_id, top_n=500):
    # 可实现“冷启动”推荐算法，比如热门商品或整体top
    return get_global_top_products(top_n=top_n)


def get_global_top_products(top_n=1000):
    return list(products['asin'].value_counts().head(top_n).index)
