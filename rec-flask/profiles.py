# profiles.py
import pandas as pd
import logging
import redis
import json

from model import asin_to_category
from database import db_lock, conn
from model import products

# 初始化 Redis 连接
redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)

# 全局变量
user_profiles = {}

def update_user_profile(user_id, asin):
    global user_profiles, asin_to_category
    try:
        category = asin_to_category.get(asin)
    except KeyError:
        # 如果商品不存在或没有类别，直接返回
        return
    if pd.isna(category):
        return  # 如果商品没有类别，直接返回
    user_profile = user_profiles.get(user_id, {})
    user_profile[str(int(category))] = user_profile.get(str(int(category)), 0) + 1  # 更新点击次数
    total = sum(user_profile.values())
    for cat in user_profile:
        user_profile[cat] /= total  # 重新归一化偏好程度
    user_profiles[user_id] = user_profile

def user_behavior_update(user_id, asin):
    # 更新用户画像
    update_user_profile(user_id, asin)
    # 清除缓存
    redis_client.delete(f"recommendations:{user_id}")

def update_recommendations_after_click(user_id, asin):
    from algorithms import recall, coarse_ranking, fine_ranking, re_ranking
    # 更新用户画像
    update_user_profile(user_id, asin)
    # 立即生成新的推荐列表
    candidates = recall(user_id)
    if not candidates or len(candidates) == 0:
        candidates = products['asin'].sample(min(500, len(products))).tolist()
    coarse_ranked = coarse_ranking(candidates)
    fine_ranked = fine_ranking(user_id, coarse_ranked)
    final_recommendations = re_ranking(user_id, fine_ranked)
    if not final_recommendations or len(final_recommendations) == 0:
        final_recommendations = products['asin'].sample(min(500, len(products))).tolist()
    # 更新缓存
    redis_client.setex(f"recommendations:{user_id}", 3600, json.dumps(final_recommendations))
