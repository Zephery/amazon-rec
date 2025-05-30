import os

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.service.data_loader import products
from db.database import get_user_recent_click_asins, get_products_by_asins


def embeddings():
    # 判断 embedding 文件是否已存在，存在则不重复生成
    if os.path.exists('product_emb.npy'):
        print("Embedding 文件已经存在，无需重复生成。")
        return

    # 1. 加载模型（建议 MiniLM，快且效果很好）
    model = SentenceTransformer("all-MiniLM-L6-v2")

    asins, titles = products['asin'].tolist(), products['title'].tolist()

    # 3. 批量生成embedding（按需分批，防止内存溢出）
    BATCH = 50000
    embeddings = []
    for i in range(0, len(titles), BATCH):
        batch_titles = titles[i:i + BATCH]
        batch_emb = model.encode(
            batch_titles,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True  # 归一化便于后续用内积检索
        )
        embeddings.append(batch_emb)
    all_embeddings = np.vstack(embeddings)  # shape = (商品数, emb_dim)

    # 4. 持久化embedding和asin
    np.save('product_emb.npy', all_embeddings)
    with open('asin_list.txt', 'w') as f:
        for asin in asins:
            f.write(f"{asin}\n")


all_embeddings = np.load('product_emb.npy').astype('float32')
faiss.omp_set_num_threads(8)  # CPU数按实际调整


def build_or_load_index(embeddings, dim, index_path='faiss.index'):
    try:
        index = faiss.read_index(index_path)
        print("Loaded FAISS index from disk.")
    except Exception:
        quantizer = faiss.IndexFlatIP(dim)
        nlist = 4096
        index = faiss.IndexIVFFlat(quantizer, dim, nlist)
        faiss.normalize_L2(embeddings)
        index.train(embeddings)
        index.add(embeddings)
        faiss.write_index(index, index_path)
        print("Trained and saved FAISS index.")
    return index


index = build_or_load_index(all_embeddings, all_embeddings.shape[1])


def faiss_ann_recall(user_click_asins, topn=200):
    all_asins = products['asin'].tolist()
    # asin与embedding行号的映射
    asin2idx = {asin: i for i, asin in enumerate(all_asins)}
    recalled = set()
    result_scores = dict()
    for asin in user_click_asins:
        idx = asin2idx.get(asin)
        if idx is None:
            continue
        emb = all_embeddings[idx].reshape(1, -1)
        D, I = index.search(emb, topn + 1)
        for i, sim in zip(I[0], D[0]):
            recall_asin = all_asins[i]
            if i == -1 or recall_asin in user_click_asins or recall_asin in recalled:
                continue
            result_scores[recall_asin] = float(sim)
            recalled.add(recall_asin)
    return [a for a, s in sorted(result_scores.items(), key=lambda x: -x[1])]


def recall(user_id, top_n=500, hybrid=True):
    user_click_asins = get_user_recent_click_asins(user_id)
    recall_faiss = faiss_ann_recall(user_click_asins, top_n * 2)
    recall_cf = []  # 你可以补充协同召回
    recall_union, seen = [], set(user_click_asins)
    for asin in recall_cf + recall_faiss:
        if asin not in seen:
            recall_union.append(asin)
            seen.add(asin)
        if len(recall_union) >= top_n:
            break
    # enrich every asin
    if recall_union:
        return get_products_by_asins(recall_union)
    return []


def recommend_based_on_similar_users(user_id, top_n=500):
    # 可实现“冷启动”推荐算法，比如热门商品或整体top
    return get_global_top_products(top_n=top_n)


def get_global_top_products(top_n=1000):
    return list(products['asin'].value_counts().head(top_n).index)
