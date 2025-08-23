import multiprocessing
import os
import random
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

from app.service.data_loader import products
from db.database import get_user_recent_click_asins


def gen_embeddings():
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


print("start to load embeddings")
base_path = str(Path(__file__).parent.parent.parent.parent)
model = SentenceTransformer(base_path + "/all-MiniLM-L6-v2")
gen_embeddings()
all_embeddings = np.load(base_path + '/product_emb.npy').astype('float32')
print("向量总数:", all_embeddings.shape[0])
print("每个向量维度:", all_embeddings.shape[1])
print("该向量内容（head5）:", all_embeddings[0][:5])

faiss.omp_set_num_threads(multiprocessing.cpu_count())


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
    """
    基于FAISS高效召回相似商品的asin列表，过滤用户已交互，并做简单多样性采样。

    Args:
        user_click_asins (list): 用户近期浏览/点击过的asin列表。
        topn (int): 召回商品的最大数量。

    Returns:
        List[str]: 高相关性且类别多样的asin集合，按得分降序排序。
    """
    # 1. 数据准备
    all_asins = products['asin'].tolist()
    asin2idx = {asin: i for i, asin in enumerate(all_asins)}
    asin2cat = dict(zip(products['asin'], products.get('category_id', [''] * len(products))))
    idx_list = [asin2idx[asin] for asin in user_click_asins if asin in asin2idx]

    if not idx_list:
        return []

    # 2. 批量embedding检索 - 增加随机性
    batch_embs = all_embeddings[idx_list]  # shape: (|user_click_asins|, emb_dim)

    # 添加随机噪声增加多样性
    noise_factor = 0.05
    noise = np.random.normal(0, noise_factor, batch_embs.shape)
    batch_embs = batch_embs + noise

    # 随机调整检索数量，增加召回数量
    search_topn = topn + random.randint(50, 100)
    D, I = index.search(batch_embs, search_topn)

    # 3. 按得分聚合并去重，仅保留分最高的版本
    seen = set(user_click_asins)
    score_map = {}
    for row, sims in zip(I, D):
        for idx, s in zip(row, sims):
            if idx == -1:
                continue
            if idx >= len(all_asins):
                continue
            asin = all_asins[idx]
            if asin in seen:
                continue
            # 保留最高分，但添加随机扰动
            random_factor = random.uniform(0.9, 1.1)  # 增加随机扰动范围
            adjusted_score = s * random_factor
            if (asin not in score_map) or (score_map[asin] < adjusted_score):
                score_map[asin] = float(adjusted_score)

    # 4. 得分排序
    candidates = sorted(score_map.items(), key=lambda x: -x[1])

    # 5. 增强多样性采样：确保品类分布更均匀
    max_per_cat = random.randint(100, 150)  # 增加每类最大数量
    cat_count = {}
    diverse = []

    # 随机打乱候选列表
    random.shuffle(candidates)

    # 第一轮：按品类限制采样
    for asin, sim in candidates:
        cat = asin2cat.get(asin, '')
        if cat_count.get(cat, 0) < max_per_cat:
            diverse.append(asin)
            cat_count[cat] = cat_count.get(cat, 0) + 1
        if len(diverse) >= topn:
            break

    # 第二轮：如果品类不够多样，补充其他品类
    if len(cat_count) < 5:  # 如果品类少于5个，补充更多品类
        remaining_candidates = [asin for asin, _ in candidates if asin not in diverse]
        for asin in remaining_candidates:
            cat = asin2cat.get(asin, '')
            if cat not in cat_count or cat_count[cat] < max_per_cat // 2:
                diverse.append(asin)
                cat_count[cat] = cat_count.get(cat, 0) + 1
            if len(diverse) >= topn * 1.5:  # 允许更多商品
                break

    return diverse


def recall(user_id, top_n=5000, hybrid=True):
    user_click_asins = get_user_recent_click_asins(user_id)

    # 随机化召回数量，增加召回量
    recall_multiplier = random.uniform(2.0, 3.0)  # 增加召回倍数
    adjusted_top_n = int(top_n * recall_multiplier)

    recall_faiss = faiss_ann_recall(user_click_asins, adjusted_top_n) if user_click_asins else []
    recall_cf = []
    recall_hot = get_global_top_products(adjusted_top_n)

    recall_union, seen = [], set(user_click_asins)
    for asin in recall_faiss + recall_cf + recall_hot:
        if asin not in seen:
            recall_union.append(asin)
            seen.add(asin)
        if len(recall_union) >= top_n * 4:  # 增加联合召回数量
            break

    # 增强打散：按品类分桶轮转采样，提升多样性
    asin2cat = dict(zip(products['asin'], products.get('category_id', [''] * len(products))))
    buckets = {}
    for asin in recall_union:
        cat = asin2cat.get(asin, 'unknown')
        buckets.setdefault(cat, []).append(asin)

    # 随机打乱每个桶内的顺序
    for cat in buckets:
        random.shuffle(buckets[cat])

    # 轮转采样，但增加随机性和数量
    diverse = []
    bucket_keys = list(buckets.keys())
    random.shuffle(bucket_keys)  # 随机化桶的顺序

    # 确保每个品类都有代表
    min_per_bucket = max(1, top_n // len(buckets)) if buckets else 1

    while len(diverse) < top_n:
        empty = 0
        for cat in bucket_keys:
            if buckets[cat]:
                # 每个桶至少取min_per_bucket个商品
                items_to_take = min(min_per_bucket, len(buckets[cat]))
                for _ in range(items_to_take):
                    if buckets[cat] and len(diverse) < top_n:
                        diverse.append(buckets[cat].pop(0))
                    else:
                        break
            else:
                empty += 1
        if empty == len(buckets):
            break

    # 如果还有剩余商品，继续采样
    remaining = [asin for cat in buckets.values() for asin in cat]
    random.shuffle(remaining)
    diverse.extend(remaining[:top_n - len(diverse)])

    # 最终随机打乱
    random.shuffle(diverse)

    # 返回asin列表，不查详情，保证后续排序流程可用
    return diverse[:top_n]


def recommend_based_on_similar_users(user_id, top_n=500):
    # 可实现“冷启动”推荐算法，比如热门商品或整体top
    return get_global_top_products(top_n=top_n)


def get_global_top_products(top_n=1000):
    return list(products['asin'].value_counts().head(top_n).index)


if __name__ == '__main__':
    gen_embeddings()
