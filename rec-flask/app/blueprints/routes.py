# routes.py
import logging
import time

import pandas as pd
from flask import Flask, request, jsonify

from app.service.algorithms import (
    recall, coarse_ranking, fine_ranking,
    re_ranking, recommend_based_on_similar_users
)
from app.service.front_page_scene import get_global_top_products
from app.service.model import products
from app.service.profiles import user_profiles, user_behavior_update, update_recommendations_after_click
from app.service.search import search_products
from db.database import conn, db_lock

user_clicks = pd.DataFrame(columns=['user_id', 'asin', 'click_time'])


def create_app():
    app = Flask(__name__)

    # 首页路由
    @app.route('/')
    def index():
        return "ok"

    # 推荐商品列表接口
    @app.route('/products', methods=['GET'])
    def get_recommendations():
        user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('per_page', 20))
        q = request.args.get('q')

        if q:
            return search_products(products, user_id, q, page, page_size)

        # 优先使用用户画像生成推荐列表
        if user_id in user_profiles:
            candidates = recall(user_id)
            if not candidates or len(candidates) == 0:
                candidates = products['asin'].sample(min(500, len(products))).tolist()
            coarse_ranked = coarse_ranking(candidates)
            fine_ranked = fine_ranking(user_id, coarse_ranked)
            final_recommendations = re_ranking(user_id, fine_ranked)
        else:
            # 如果用户画像不存在，使用相似用户生成推荐列表
            final_recommendations = recommend_based_on_similar_users(user_id, top_n=500)
            if not final_recommendations or len(final_recommendations) == 0:
                # 如果没有推荐结果，随机推荐商品
                final_recommendations = get_global_top_products()

        if not final_recommendations:
            # 如果没有推荐结果，从数据库中随机抽取一些商品作为推荐
            paged_recommendations = get_global_top_products()
            total_items = len(paged_recommendations)
            total_pages = 1  # 因为我们只返回了一页的随机商品
        else:
            total_items = len(final_recommendations)
            total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_recommendations = final_recommendations[start_idx:end_idx]

        # 返回商品的详细信息
        if not paged_recommendations:
            return jsonify({'message': 'No products to recommend.'}), 404
        recommended_products = products[products['asin'].isin(paged_recommendations)].to_dict(orient='records')

        return jsonify({
            'user_id': user_id,
            'total': total_items,
            'pages': total_pages,
            'current_page': page,
            'per_page': page_size,
            'products': recommended_products
        })

    # 商品点击接口
    @app.route('/products/<asin>', methods=['GET'])
    def record_click(asin):
        global user_clicks
        user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
        click_time = time.time()
        # 记录用户点击
        try:
            with db_lock:
                conn.execute('INSERT INTO user_clicks (user_id, asin, click_time) VALUES (?, ?, ?)',
                             (user_id, asin, click_time))
                conn.commit()
            # 更新内存中的 user_clicks 数据
            user_clicks = pd.concat(
                [user_clicks, pd.DataFrame({'user_id': [user_id], 'asin': [asin], 'click_time': [click_time]})],
                ignore_index=True)
            # 更新用户行为
            user_behavior_update(user_id, asin)
            # 在用户点击商品后，立即更新推荐列表并缓存
            update_recommendations_after_click(user_id, asin)
        except Exception as e:
            logging.error(f"Error recording click: {e}")
            return jsonify({'status': 'error', 'message': 'Failed to record click.'}), 500
        # 获取被点击的商品详情
        product = products[products['asin'] == asin]
        product_data = product.to_dict(orient='records')
        if product_data:
            product_data = product_data[0]  # 获取单个商品的字典
        else:
            product_data = {}
        return jsonify({
            'status': 'success',
            'data': {
                'product': product_data,
                'additional_info': {
                    'shipping': '免费配送',
                    'warranty': '一年保修',
                    'return_policy': '30天无理由退换'
                }
            }
        })

    return app
