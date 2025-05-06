from flask import jsonify

from db.database import execute_query


def get_clicks_history(user_id):
    query = """
            SELECT p.asin, \
                   p.title, \
                   p.imgUrl, \
                   p.productURL, \
                   p.stars, \
                   p.reviews,
                   p.price, \
                   p.listPrice, \
                   p.category_id, \
                   p.isBestSeller,
                   p.boughtInLastMonth, \
                   c.click_time
            FROM user_clicks c
                     JOIN amazon_products p ON c.asin = p.asin
            WHERE c.user_id = ?
            ORDER BY c.click_time DESC; \
            """
    # 执行 SQL 查询
    page_data = execute_query(query, (user_id,))

    # 对 `asin` 列去重，只保留最早的点击记录
    page_data = page_data.drop_duplicates(subset='asin', keep='first')

    # 格式化返回的数据
    formatted_data = page_data[
        ['asin', 'title', 'price', 'imgUrl', 'productURL', 'stars', 'reviews', 'listPrice',
         'category_id','click_time']].to_dict(orient='records')

    response = {
        "products": formatted_data
    }

    return jsonify(response)

