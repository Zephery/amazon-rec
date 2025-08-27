import numpy as np

from app.service.data_loader import products
from app.service.embedding_service import embedding_service
from app.service.front_page_scene import get_global_top_products
from db.database import get_user_recent_click_asins

# 兼容占位：外部旧代码如果 import gen_embeddings 不再报错
# 不再在此生成向量，统一用 generate_embeddings.py 或 run.py.ensure_embeddings()

def gen_embeddings():  # 兼容旧引用
    print("gen_embeddings 已废弃：请使用 app.service.generate_embeddings.generate_embeddings 或启动脚本自动生成。")

all_embeddings = embedding_service.get_embeddings()
index = embedding_service.get_index()

def faiss_high_relevance_recall(user_click_asins, topn=200, per_item_search=60, agg_lambda=0.3,
                                extra_multiplier=4):
    if not user_click_asins or all_embeddings is None or index is None:
        return []
    all_asins = products['asin'].tolist()
    asin2idx = {asin: i for i, asin in enumerate(all_asins)}
    idx_list = [asin2idx[a] for a in user_click_asins if a in asin2idx]
    if not idx_list:
        return []
    user_clicked_embs = all_embeddings[idx_list]
    m = len(idx_list)
    distance_from_end = (m - 1) - np.arange(m)
    weights = np.exp(-agg_lambda * distance_from_end)
    weights = weights / (weights.sum() + 1e-9)
    user_profile_vec = (weights[:, None] * user_clicked_embs).sum(axis=0)
    user_profile_vec = (user_profile_vec / (np.linalg.norm(user_profile_vec) + 1e-9)).astype('float32')
    search_topn = min(len(all_asins), topn * extra_multiplier)
    D_main, I_main = index.search(user_profile_vec[None, :], search_topn)
    score_map = {}
    clicked_set = set(user_click_asins)
    for idx, score in zip(I_main[0], D_main[0]):
        if idx == -1 or idx >= len(all_asins):
            continue
        asin = all_asins[idx]
        if asin in clicked_set:
            continue
        if asin not in score_map or score_map[asin] < score:
            score_map[asin] = float(score)
    if per_item_search > 0:
        max_items_for_local = 15
        selected = idx_list[-max_items_for_local:]
        local_k = min(per_item_search, len(all_asins))
        batch_embs = all_embeddings[selected]
        D_loc, I_loc = index.search(batch_embs, local_k)
        decay = 0.6
        for row_scores, row_indices in zip(D_loc, I_loc):
            for sc, idx in zip(row_scores, row_indices):
                if idx == -1 or idx >= len(all_asins):
                    continue
                asin = all_asins[idx]
                if asin in clicked_set:
                    continue
                sc2 = sc * decay
                if asin not in score_map or score_map[asin] < sc2:
                    score_map[asin] = float(sc2)
    sorted_items = sorted(score_map.items(), key=lambda x: -x[1])
    return [a for a, _ in sorted_items[:topn]]

def recall(user_id, top_n=5000):
    user_click_asins = get_user_recent_click_asins(user_id)
    if not user_click_asins:
        return get_global_top_products(limit=top_n)
    candidates = faiss_high_relevance_recall(user_click_asins, topn=top_n)
    if len(candidates) < top_n:
        hot = [a for a in get_global_top_products(limit=top_n * 2) if a not in candidates][: (top_n - len(candidates))]
        candidates.extend(hot)
    return candidates[:top_n]
