import json
from statistics import mean

import fakeredis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob  # 用于情感分析

from db.database import load_data, get_one_product, get_recommended_products

# Redis 实例（伪 Redis 数据库）
redis_instance = fakeredis.FakeStrictRedis()
user_profiles = {}

# 停用词扩展
stopwords_list = list(set([
    # ...此处省略，和原来一致...
    "i", "me", "my", "myself", "we", "our", "ours", "ourselves", "you", "your",
    "yours", "yourself", "yourselves", "he", "him", "his", "himself", "she",
    "her", "hers", "herself", "it", "its", "itself", "they", "them", "their",
    "theirs", "themselves", "what", "which", "who", "whom", "this", "that",
    "these", "those", "am", "is", "are", "was", "were", "be", "been", "being",
    "have", "has", "had", "having", "do", "does", "did", "doing", "a", "an",
    "the", "and", "but", "if", "or", "because", "as", "until", "while",
    "of", "at", "by", "for", "with", "about", "against", "between", "into",
    "through", "during", "before", "after", "above", "below", "to", "from",
    "up", "down", "in", "out", "on", "off", "over", "under", "again",
    "further", "then", "once", "here", "there", "when", "where", "why",
    "how", "all", "any", "both", "each", "few", "more", "most", "other",
    "some", "such", "no", "nor", "not", "only", "own", "same", "so", "than",
    "too", "very", "s", "t", "can", "will", "just", "don", "should", "now"
]))

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
            vectorizer = CountVectorizer(max_features=5, stop_words=stopwords_list)
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
        print(f"User {user_id} profile updated successfully.")
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

def get_recommendations(user_id):
    rec_json = redis_instance.get(f"recommendations:{user_id}")
    if rec_json is None:
        return None
    if isinstance(rec_json, bytes):
        rec_json = rec_json.decode('utf-8')
    try:
        return json.loads(rec_json)
    except Exception as e:
        print(f"Cannot parse recommendation JSON for {user_id}: {e}")
        return None

# 初始化流程
def init_user_profile():
    global user_profiles
    data = load_data()
    user_profiles = build_user_profiles(data)
    store_profiles_in_redis(redis_instance, user_profiles)

