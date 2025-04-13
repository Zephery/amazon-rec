from flask import jsonify


def search_products(products, user_id, q, page=1, page_size=100):
    """
    搜索商品并返回精准结果，结合矢量化相关性评分和分页逻辑。
    :param products: pd.DataFrame, 商品数据
    :param user_id: str, 发起搜索的用户ID
    :param q: str, 搜索关键字
    :param page: int, 当前页码，从1开始
    :param page_size: int, 每页数量
    :return: JSON 响应，包含用户数据、分页数据和商品列表
    """
    import math

    # 数据有效性检查
    if page < 1:
        page = 1
    if page_size < 1:
        page_size = 10

    # 转换搜索关键字为小写进行统一处理
    q_lower = q.lower() if q else ""

    # 精准关键词匹配：检查关键词是否出现在标题中
    title_match = products['title'].str.contains(q_lower, case=False, na=False).astype(int) * 10

    # 开头匹配：检查标题是否以关键词开头
    title_start_match = products['title'].str.startswith(q_lower, na=False).astype(int) * 5

    # 分类匹配：检查关键词是否出现在分类字段中（如果存在分类）
    if 'category_name' in products.columns:
        category_match = products['category_name'].str.contains(q_lower, case=False, na=False).astype(int) * 2
    else:
        category_match = 0

    # 计算相关性评分
    products['relevance_score'] = title_match + title_start_match + category_match

    # 过滤没有相关性的商品
    filtered_data = products[products['relevance_score'] > 0]

    # 排除配件商品（通过关键词排除不相关商品）
    excluded_keywords = ['case', 'cover', 'protector', 'keyboard', 'adapter']
    filtered_data = filtered_data[
        ~filtered_data['title'].str.lower().str.contains('|'.join(excluded_keywords), na=False, regex=True)]

    # 按相关性评分降序排序
    filtered_data = filtered_data.sort_values(by='relevance_score', ascending=False)

    # 总商品数量
    total_items = len(filtered_data)

    # 总页数
    total_pages = math.ceil(total_items / page_size)

    # 确保页码不超过总页数
    if page > total_pages:
        page = total_pages if total_pages > 0 else 1

    # 确定分页索引
    start_idx = (page - 1) * page_size
    end_idx = start_idx + page_size

    # 获取当前页的数据并格式化输出
    page_data = filtered_data.iloc[start_idx:end_idx]

    # 仅返回需要的字段
    formatted_data = page_data[['asin', 'title', 'price', 'imgUrl', 'productURL', 'relevance_score']].to_dict(
        orient='records')

    # 返回 JSON 响应
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
