# routes.py
import logging
import time

import pandas as pd
from flask import Flask, request, jsonify

from app.service.algorithms import (
    recall, coarse_ranking, fine_ranking,
    recommend_based_on_similar_users
)
from app.service.front_page_scene import get_global_top_products
from app.service.recommendation import products
from app.service.search import search_products
from app.service.user.user_profile import user_profiles, user_behavior_update, update_recommendations_after_click, \
    get_user_profile_detail
from app.service.view_history import get_clicks_history
from db.database import conn, db_lock, delete_user_clicks

user_clicks = pd.DataFrame(columns=['user_id', 'asin', 'click_time'])

sqlite_path = 'db/recommend.db'


def create_app():
    app = Flask(__name__)

    # 首页路由
    @app.route('/')
    def index():
        return "ok"

    @app.route('/get_clicks')
    def get_clicks():
        user_id = request.remote_addr  # 使用请求的IP地址作为用户ID
        return get_clicks_history(user_id)

    @app.route('/clear_clicks', methods=['DELETE'])
    def clear_clicks():
        user_id = request.remote_addr  # 使用请求的IP地址作为用户ID

        delete_user_clicks(user_id)
        return jsonify({"ok": True})

    @app.route('/get_user_profile', methods=['GET'])
    def get_user_profile():
        user_id = request.remote_addr
        result = get_user_profile_detail(user_id)

        return jsonify(result)

    def decode_bytes(data):
        if isinstance(data, dict):
            return {decode_bytes(key): decode_bytes(value) for key, value in data.items()}
        elif isinstance(data, list):
            return [decode_bytes(item) for item in data]
        elif isinstance(data, tuple):
            return tuple(decode_bytes(item) for item in data)
        elif isinstance(data, bytes):
            return data.decode('utf-8')
        return data

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
            # TODO 这里还有问题，先注释
            # final_recommendations = re_ranking(user_id, fine_ranked)
            final_recommendations = []
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
        if len(recommended_products) > page_size:
            recommended_products = recommended_products[:page_size]
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
