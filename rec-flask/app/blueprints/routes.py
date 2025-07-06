# routes.py
import logging
import time
import random
import numpy as np

import pandas as pd
from flask import Flask, request, jsonify

from app.service.data_loader import products
from app.service.front_page_scene import get_global_top_products
from app.service.rec.coarse_rank import coarse_ranking
from app.service.rec.fine_rank import fine_ranking
from app.service.rec.re_rank import re_ranking
from app.service.rec.recall import recall
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

    def _get_category_stats(df):
        """统计品类分布"""
        if isinstance(df, pd.DataFrame) and 'asin' in df.columns:
            asin_list = df['asin'].drop_duplicates().tolist()
        else:
            asin_list = list(set(df))
        product_subset = products[products['asin'].isin(asin_list)]
        if 'category_id' in product_subset.columns:
            return product_subset['category_id'].value_counts().to_dict()
        return {}

    @app.route('/debug_recommendations', methods=['GET'])
    def debug_recommendations():
        """调试接口：查看推荐系统的各个阶段结果"""
        user_id = request.remote_addr
        
        if user_id not in user_profiles:
            return jsonify({'message': 'User not found in profiles'}), 404
        
        # 应用排序流水线
        re_df, coarse_df, fine_df = _apply_ranking_pipeline(user_id)
        
        # 获取召回结果
        candidates = recall(user_id, top_n=2000)
        if not candidates:
            candidates = get_global_top_products(limit=2000)
        
        return jsonify({
            'user_id': user_id,
            'stages': {
                'recall': {
                    'count': len(candidates),
                    'categories': _get_category_stats(candidates)
                },
                'coarse_ranking': {
                    'count': len(coarse_df) if isinstance(coarse_df, pd.DataFrame) else len(coarse_df),
                    'categories': _get_category_stats(coarse_df)
                },
                'fine_ranking': {
                    'count': len(fine_df) if isinstance(fine_df, pd.DataFrame) else len(fine_df),
                    'categories': _get_category_stats(fine_df)
                },
                're_ranking': {
                    'count': len(re_df) if isinstance(re_df, pd.DataFrame) else len(re_df),
                    'categories': _get_category_stats(re_df)
                }
            }
        })

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

    def _ensure_dataframe_with_scores(df, score_column_name, default_score=0.0):
        """确保DataFrame包含asin和分数列"""
        if not isinstance(df, pd.DataFrame):
            df = pd.DataFrame({'asin': df, score_column_name: [default_score] * len(df)})
        return df.drop_duplicates(subset=['asin'], keep='first')

    def _merge_ranking_scores(re_df, coarse_df, fine_df):
        """合并各个排序阶段的分数"""
        re_df = _ensure_dataframe_with_scores(re_df, 're_score')
        coarse_df = _ensure_dataframe_with_scores(coarse_df, 'coarse_score')
        fine_df = _ensure_dataframe_with_scores(fine_df, 'fine_score')
        merged = re_df.merge(coarse_df, on='asin', how='left').merge(fine_df, on='asin', how='left')
        # 填充缺失分数为0
        for col in ['recall_score', 'coarse_score', 'fine_score', 're_score']:
            if col not in merged.columns:
                merged[col] = 0.0
            merged[col] = merged[col].fillna(0.0)
        # 计算总得分，可自定义加权
        merged['total_score'] = (
            0.2 * merged['recall_score'] +
            0.25 * merged['coarse_score'] +
            0.25 * merged['fine_score'] +
            0.3 * merged['re_score']
        )
        return merged

    def _calculate_combined_score(merged):
        """计算综合分数"""
        if all(col in merged.columns for col in ['coarse_score', 'fine_score', 're_score']):
            merged['combined_score'] = (
                merged['coarse_score'].fillna(0) * 0.3 +
                merged['fine_score'].fillna(0) * 0.4 +
                merged['re_score'].fillna(0) * 0.3
            )
        else:
            merged['combined_score'] = 1.0
        return merged

    def _diversify_by_category(merged):
        """按品类进行多样性采样"""
        if 'category_id' not in products.columns:
            return merged.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)
        
        merged = merged.merge(products[['asin', 'category_id']], on='asin', how='left')
        merged = _calculate_combined_score(merged)
        
        # 按品类分组
        buckets = {}
        for _, row in merged.iterrows():
            cat = row.get('category_id', 'unknown')
            buckets.setdefault(cat, []).append(row)
        
        # 计算每个品类的分配数量
        total_categories = len(buckets)
        min_per_category = max(1, len(merged) // (total_categories * 2))
        max_per_category = max(10, len(merged) // total_categories)
        
        diverse = []
        
        # 第一轮：每个品类按分数排序后取前N个
        for cat, items in buckets.items():
            cat_df = pd.DataFrame(items)
            cat_df = cat_df.sort_values('combined_score', ascending=False)
            cat_df = cat_df.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)
            
            cat_limit = random.randint(min_per_category, max_per_category)
            diverse.extend(cat_df.head(cat_limit).to_dict('records'))
        
        # 第二轮：补充剩余商品
        remaining_items = []
        for cat, items in buckets.items():
            cat_df = pd.DataFrame(items)
            cat_df = cat_df.sort_values('combined_score', ascending=False)
            cat_limit = random.randint(min_per_category, max_per_category)
            if len(cat_df) > cat_limit:
                remaining_items.extend(cat_df.iloc[cat_limit:].to_dict('records'))
        
        random.shuffle(remaining_items)
        diverse.extend(remaining_items)
        random.shuffle(diverse)
        
        # 转换为DataFrame并去重
        if diverse:
            result_df = pd.DataFrame(diverse)
            return result_df.drop_duplicates(subset=['asin'], keep='first')
        else:
            return merged.sample(frac=1, random_state=random.randint(1, 10000)).reset_index(drop=True)

    def _apply_ranking_pipeline(user_id, recall_num=2000, coarse_num=800, fine_num=400, re_num=200):
        """应用完整的排序流水线"""
        # 召回
        candidates = recall(user_id, top_n=recall_num)
        if not candidates:
            candidates = get_global_top_products(limit=recall_num)
        
        # 粗排
        coarse_result = coarse_ranking(candidates, return_score=True)
        if isinstance(coarse_result, pd.DataFrame) and len(coarse_result) > coarse_num:
            coarse_df = coarse_result.head(coarse_num)
        else:
            coarse_df = coarse_result
        
        # 精排
        coarse_asins = coarse_df['asin'].tolist() if isinstance(coarse_df, pd.DataFrame) else coarse_df
        fine_result = fine_ranking(user_id, coarse_asins, return_score=True)
        if isinstance(fine_result, pd.DataFrame) and len(fine_result) > fine_num:
            fine_df = fine_result.head(fine_num)
        else:
            fine_df = fine_result
        
        # 重排
        fine_asins = fine_df['asin'].tolist() if isinstance(fine_df, pd.DataFrame) else fine_df
        re_result = re_ranking(user_id, fine_asins, return_score=True)
        if isinstance(re_result, pd.DataFrame) and len(re_result) > re_num:
            re_df = re_result.head(re_num)
        else:
            re_df = re_result
        
        return re_df, coarse_df, fine_df

    def _add_scores_to_products(recommended_products, merged):
        """为推荐商品添加各阶段分数和总分"""
        if merged is not None and 'asin' in merged.columns:
            duplicate_count = merged['asin'].duplicated().sum()
            if duplicate_count > 0:
                logging.warning(f"Found {duplicate_count} duplicate asins in merged DataFrame")
            merged_unique = merged.drop_duplicates(subset=['asin'], keep='first')
            score_map = merged_unique.set_index('asin')[['recall_score', 'coarse_score', 'fine_score', 're_score', 'total_score']].to_dict(orient='index')
            for prod in recommended_products:
                asin = prod.get('asin')
                if asin in score_map:
                    prod['recall_score'] = float(score_map[asin].get('recall_score', 0))
                    prod['coarse_score'] = float(score_map[asin].get('coarse_score', 0))
                    prod['fine_score'] = float(score_map[asin].get('fine_score', 0))
                    prod['re_score'] = float(score_map[asin].get('re_score', 0))
                    prod['total_score'] = float(score_map[asin].get('total_score', 0))

    def _handle_pagination(final_recommendations, page, page_size):
        """处理分页逻辑"""
        if not final_recommendations:
            # 使用热门商品作为备选
            paged_recommendations = get_global_top_products(limit=page_size * 10)
            total_items = len(paged_recommendations)
            total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_recommendations = paged_recommendations[start_idx:end_idx]
        else:
            total_items = len(final_recommendations)
            total_pages = total_items // page_size + (1 if total_items % page_size > 0 else 0)
            start_idx = (page - 1) * page_size
            end_idx = start_idx + page_size
            paged_recommendations = final_recommendations[start_idx:end_idx]
        
        # 如果当前页没有商品，但有总商品，返回第一页
        if not paged_recommendations and total_items > 0:
            paged_recommendations = final_recommendations[:page_size]
            page = 1
        
        return paged_recommendations, total_items, total_pages, page

    # 推荐商品列表接口
    @app.route('/products', methods=['GET'])
    def get_recommendations():
        user_id = request.remote_addr
        page = int(request.args.get('page', 1))
        page_size = int(request.args.get('per_page', 20))
        q = request.args.get('q')

        if q:
            return search_products(products, user_id, q, page, page_size)

        if user_id in user_profiles:
            # 应用排序流水线
            re_df, coarse_df, fine_df = _apply_ranking_pipeline(user_id)
            
            # 合并分数
            merged = _merge_ranking_scores(re_df, coarse_df, fine_df)
            
            # 多样性采样
            merged = _diversify_by_category(merged)
            
            final_recommendations = merged['asin'].tolist()
        else:
            merged = None
            final_recommendations = get_global_top_products(limit=200)

        # 处理分页
        paged_recommendations, total_items, total_pages, page = _handle_pagination(
            final_recommendations, page, page_size
        )

        if not paged_recommendations:
            return jsonify({'message': 'No products to recommend.'}), 404

        # 获取商品详情并添加分数
        recommended_products = products[products['asin'].isin(paged_recommendations)].to_dict(orient='records')
        _add_scores_to_products(recommended_products, merged)
        
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
