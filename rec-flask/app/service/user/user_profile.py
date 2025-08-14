import json
import os
from statistics import mean

import fakeredis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob  # 用于情感分析

from db.database import load_data, get_one_product, get_recommended_products

# Redis 实例（伪 Redis 数据库）
redis_instance = fakeredis.FakeStrictRedis()
user_profiles = {}

USER_PROFILE_PATH = 'user_profiles.json'


# 年龄推测
def infer_age(user_data):
    avg_price = mean(user_data["price"]) if len(user_data["price"]) > 0 else 0
    if avg_price < 50:
        return "18-25"
    elif 50 <= avg_price < 150:
        return "26-35"
    elif 150 <= avg_price < 400:
        return "36-50"
    else:
        return "50+"

    # 性别推测


def infer_gender(user_data):
    category_preferences = user_data["category_name"].value_counts()
    if "Beauty" in category_preferences.index or "Fashion" in category_preferences.index:
        return "female"
    elif "Electronics" in category_preferences.index or "Sports" in category_preferences.index:
        return "male"
    else:
        return "unknown"

    # 构建用户画像


def build_user_profiles(data):
    global user_profiles
    data['timestamp'] = pd.to_datetime(data['timestamp'])
    for user_id, user_data in data.groupby("user_id"):
        avg_rating = user_data["rating"].mean()
        avg_rating = 0 if pd.isna(avg_rating) else avg_rating

        review_count = len(user_data)
        category_preferences = user_data["category_name"].value_counts().to_dict()
        brand_preferences = user_data["title_y"].value_counts().to_dict()
        product_preferences = user_data["title_x"].value_counts().to_dict()
        avg_price = mean(user_data["price"]) if len(user_data["price"]) > 0 else 0
        expensive_product_count = len(user_data[user_data["price"] > 1000])
        try:
            vectorizer = CountVectorizer(max_features=5, stop_words='english')
            words_matrix = vectorizer.fit_transform(user_data["text"].astype(str))
            top_keywords = vectorizer.get_feature_names_out()
        except ValueError:
            top_keywords = []
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for review in user_data["text"]:
            if isinstance(review, str):
                blob = TextBlob(review)
                if blob.sentiment.polarity > 0.5:
                    sentiments["positive"] += 1
                elif blob.sentiment.polarity < -0.5:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1
                    # 存储每个用户历史价格列表（用于动态均值）
        user_prices = list(user_data["price"])
        inferred_age = infer_age(user_data)
        inferred_gender = infer_gender(user_data)
        user_profiles[user_id] = {
            "age_group": inferred_age,
            "gender": inferred_gender,
            "average_rating": avg_rating,
            "review_count": review_count,
            "category_preferences": category_preferences,
            "brand_preferences": brand_preferences,
            "product_preferences": product_preferences,
            "average_price": avg_price,
            "expensive_product_count": expensive_product_count,
            "top_keywords": list(top_keywords),
            "sentiments": sentiments
        }
    return user_profiles


# 存储用户画像到 Redis（全为JSON字符串，无哈希/字节）
def store_profiles_in_redis(redis_instance, profiles):
    for user_id, profile in profiles.items():
        redis_instance.set(f"user:{user_id}", json.dumps(profile))
    print("User profiles successfully stored in Redis as JSON strings!")


# 获取用户画像（直接dict）
def get_user_profile_detail(user_id):
    value = redis_instance.get(f"user:{user_id}")
    if value is None:
        return None
        # fakeredis.get 返回str，redis-py真库返回bytes
    if isinstance(value, bytes):
        value = value.decode('utf-8')
    try:
        return json.loads(value)
    except Exception as e:
        print(f"Cannot parse user profile JSON for {user_id}: {e}")
        return None

    # 文件持久化：保存用户画像到本地文件


def save_profiles_to_file(profiles, path=USER_PROFILE_PATH):
    with open(path, 'w', encoding='utf-8') as f:
        json.dump(profiles, f, ensure_ascii=False)
    print(f"User profiles saved to {path}")


# 文件持久化：从本地文件加载用户画像
def load_profiles_from_file(path=USER_PROFILE_PATH):
    if not os.path.exists(path):
        return None
    try:
        with open(path, 'r', encoding='utf-8') as f:
            profiles = json.load(f)
        print(f"Loaded user profiles from {path}")
        return profiles
    except Exception as e:
        print(f"Error loading user profiles from {path}: {e}")
        return None

    # 更新用户行为，整体更新后存回


def user_behavior_update(user_id, asin):
    global redis_instance, user_profiles
    user_profile = get_user_profile_detail(user_id)
    if user_profile is None:
        print(f"User {user_id} does not exist in profiles. Initialize user profile first.")
        return
    product_data = get_one_product(asin)
    if product_data.empty:
        print(f"Product {asin} does not exist in the database. Cannot update user profile.")
        return
    category_name = product_data["category_id"].iloc[0]
    product_title = product_data["title"].iloc[0]
    price = product_data["price"].iloc[0]

    # 更新品类偏好
    user_profile["category_preferences"][str(category_name)] = user_profile["category_preferences"].get(
        str(category_name), 0) + 1
    # 更新商品标题偏好
    user_profile["product_preferences"][str(product_title)] = user_profile["product_preferences"].get(
        str(product_title), 0) + 1
    # 维护历史价格并更新
    user_prices = user_profile.get("user_prices", [])
    user_prices.append(float(price))
    user_profile["user_prices"] = user_prices
    user_profile["average_price"] = round(mean(user_prices), 2)

    try:
        redis_instance.set(f"user:{user_id}", json.dumps(user_profile))
        user_profiles[user_id] = user_profile
        save_profiles_to_file(user_profiles)
        print(f"User {user_id} profile updated successfully and saved to file.")
    except Exception as e:
        print(f"Error updating the user profile for {user_id}: {e}")
        print("Current user profile:", user_profile)

    # 推荐列表的写入和读取（全部为JSON字符串）


def update_recommendations_after_click(user_id, asin):
    user_profile = get_user_profile_detail(user_id)
    if not user_profile:
        print(f"User {user_id} profile does not exist in Redis. Initialize profile first.")
        return
    category_preferences = user_profile["category_preferences"]
    top_category = max(category_preferences, key=category_preferences.get, default=None)
    recommended_products = get_recommended_products(top_category, asin)
    if recommended_products.empty:
        print(f"No recommendations found based on top category {top_category} for user {user_id}.")
        return
    recommendations = recommended_products["asin"].tolist()
    # 存JSON字符串
    redis_instance.set(f"recommendations:{user_id}", json.dumps({"recommended_products": recommendations}))
    print(f"Recommendations updated for User {user_id}: {recommendations}")


# 初始化流程
def init_user_profile():
    global user_profiles
    # 优先尝试从文件加载
    profiles = load_profiles_from_file()
    if profiles:
        user_profiles = profiles
        # 同步写入 Redis        store_profiles_in_redis(redis_instance, user_profiles)
        print(f"已从文件加载 {len(user_profiles)} 个用户画像，并同步到 Redis。")
        return user_profiles
        # 检查 Redis 是否已有用户画像
    if redis_instance.keys('user:*'):
        print('用户画像已存在于 Redis，无需重复初始化。')
        user_profiles.clear()
        for key in redis_instance.keys('user:*'):
            user_id = key.decode('utf-8').split(':')[1] if isinstance(key, bytes) else key.split(':')[1]
            value = redis_instance.get(key)
            if isinstance(value, bytes):
                value = value.decode('utf-8')
            try:
                user_profiles[user_id] = json.loads(value)
            except Exception as e:
                print(f"Cannot parse user profile JSON for {user_id}: {e}")
        save_profiles_to_file(user_profiles)
        print(f"已从 Redis 加载 {len(user_profiles)} 个用户画像，并保存到文件。")
        return user_profiles
        # 文件和 Redis 都没有，需初始化
    data = load_data()
    user_profiles = build_user_profiles(data)
    store_profiles_in_redis(redis_instance, user_profiles)
    save_profiles_to_file(user_profiles)
    print(f"初始化并保存 {len(user_profiles)} 个用户画像到文件和 Redis。")
    return user_profiles
