import math
import pandas as pd
from flask import jsonify

from app.service.embedding_service import embedding_service

def search_products(products, user_id, q, page=1, page_size=100):
    """
    搜索商品并返回精准结果，结合矢量化相关性评分和分页逻辑进行性能优化。
    :param products: pd.DataFrame, 商品数据
    :param user_id: str, 发起搜索的用户ID
    :param q: str, 搜索关键字
    :param page: int, 当前页码，从1开始
    :param page_size: int, 每页数量
    :return: JSON 响应，包含用户数据、分页数据和商品列表
    """
    # 数据有效性检查
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10

    # 提前返回空查询（无关键词）
    if not q or not products.size:
        return jsonify({
            "user_id": user_id,
            "query": q,
            "total": 0,
            "pages": 0,
            "current_page": page,
            "per_page": page_size,
            "products": []
        })

    # --- Step 0: 语义向量召回 ---
    # 只对召回的topN商品做后续处理，提升相关性和效率
    topN = 200  # 召回数量可根据实际需求调整
    query_emb = embedding_service.encode_query(q, normalize=True)
    D, I = embedding_service.search(query_emb, topN)
    recall_indices = I[0]
    # 过滤无效索引（faiss可能返回-1）
    recall_indices = [i for i in recall_indices if i >= 0 and i < len(products)]
    products = products.iloc[recall_indices].copy()

    # 转换搜索关键字为小写进行统一处理
    q_lower = q.lower().strip()

    # --- Step 1: 精选相关商品（矢量化处理更高效） ---
    # 精准关键词匹配（为提高效率，用矢量化计算）
    title_match = products['title'].str.contains(q_lower, case=False, na=False).astype(int) * 10

    # 开头匹配：检查标题是否以关键词开头（矢量化）
    title_start_match = products['title'].str.startswith(q_lower, na=False).astype(int) * 5

    # 分类匹配（防止数据集缺少字段时出错）
    if 'category_name' in products.columns:
        category_match = products['category_name'].str.contains(q_lower, case=False, na=False).astype(int) * 2
    else:
        category_match = pd.Series(0, index=products.index)

    # 计算相关性评分
    relevance_score = title_match + title_start_match + category_match

    # 过滤掉无相关性的商品
    products = products[relevance_score > 0]
    products = products.copy()  # 避免链式赋值警告
    products['relevance_score'] = relevance_score

    # --- Step 2: 排除配件类商品 ---
    # 跳过与配件（case, protector, keyboard 等）相关的商品
    excluded_keywords = ['case', 'cover', 'protector', 'keyboard', 'adapter']
    products = products[
        ~products['title'].str.lower().str.contains('|'.join(excluded_keywords), na=False, regex=True)
    ]

    # --- Step 3: 排序（按相关性+畅销+评分） ---
    if 'isBestSeller' in products.columns:
        bestseller_score = products['isBestSeller'].fillna(0).astype(int) * 5
    else:
        bestseller_score = pd.Series(0, index=products.index)

    if 'stars' in products.columns:
        review_score = products['stars'].fillna(0)  # 防止缺失值
    else:
        review_score = pd.Series(0, index=products.index)

    # 最终得分计算
    products['final_score'] = products['relevance_score'] + bestseller_score + review_score

    # 按得分排序
    products = products.sort_values(by='final_score', ascending=False)

    # --- Step 4: 分页处理 ---
    total_items = len(products)  # 总商品数
    total_pages = math.ceil(total_items / page_size)  # 总页数

    # 确保页码在有效范围内
    if total_items == 0:
        return jsonify({
            "user_id": user_id,
            "query": q,
            "total": 0,
            "pages": 0,
            "current_page": page,
            "per_page": page_size,
            "products": []
        })

    if page > total_pages:
        page = total_pages

    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # 当前页商品
    page_data = products.iloc[start_idx:end_idx]

    # --- Step 5: 格式化输出 ---
    formatted_data = page_data[
        ['asin', 'title', 'price', 'imgUrl', 'productURL', 'final_score', 'stars', 'reviews', 'listPrice',
         'category_id']].to_dict(orient='records')

    response = {
        "user_id": user_id,
        "query": q,
        "total": total_items,
        "pages": total_pages,
        "current_page": page,
        "per_page": page_size,
        "products": formatted_data
    }

    return jsonify(response)
