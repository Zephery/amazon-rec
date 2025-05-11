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
    """
    基于用户的平均消费金额推测年龄段。
    """
    avg_price = mean(user_data["price"]) if len(user_data["price"]) > 0 else 0
    # 假设价格范围简单映射到年龄段
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
    """
    基于用户的商品类别推测性别。
    """
    category_preferences = user_data["category_name"].value_counts()
    if "Beauty" in category_preferences.index or "Fashion" in category_preferences.index:
        return "female"
    elif "Electronics" in category_preferences.index or "Sports" in category_preferences.index:
        return "male"
    else:
        return "unknown"


# 构建用户画像
def build_user_profiles(data):
    """
    构建基于用户行为的详细画像。
    """
    global user_profiles

    # 转换时间戳为日期对象
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # 按用户分组构建画像
    for user_id, user_data in data.groupby("user_id"):
        # 基础行为分析
        avg_rating = user_data["rating"].mean()  # 平均评分
        review_count = len(user_data)  # 评论数量

        # 分类偏好（品类偏好、品牌偏好、商品偏好）
        category_preferences = user_data["category_name"].value_counts().to_dict()  # 品类偏好
        brand_preferences = user_data["title_y"].value_counts().to_dict()  # 商品标题偏好
        product_preferences = user_data["title_x"].value_counts().to_dict()  # 评论标题偏好

        # 消费能力分析
        avg_price = mean(user_data["price"]) if len(user_data["price"]) > 0 else 0
        expensive_product_count = len(user_data[user_data["price"] > 1000])

        # 评论关键标签提取
        try:
            vectorizer = CountVectorizer(max_features=5, stop_words=stopwords_list)
            words_matrix = vectorizer.fit_transform(user_data["text"].astype(str))  # 确保将 `text` 转换为字符串
            top_keywords = vectorizer.get_feature_names_out()
        except ValueError:  # 如果评论文本无效
            top_keywords = []

        # 评论情感分析（使用 TextBlob）
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for review in user_data["text"]:
            # 检查评论是否为字符串，如果不是则跳过
            if isinstance(review, str):
                blob = TextBlob(review)
                if blob.sentiment.polarity > 0.5:
                    sentiments["positive"] += 1
                elif blob.sentiment.polarity < -0.5:
                    sentiments["negative"] += 1
                else:
                    sentiments["neutral"] += 1

        # 推测年龄和性别
        inferred_age = infer_age(user_data)
        inferred_gender = infer_gender(user_data)

        # 构建单个用户画像
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


# 存储用户画像到 Redis
def store_profiles_in_redis(redis_instance, profiles):
    """
    将用户画像存储到 Redis。
    """
    for user_id, profile in profiles.items():
        # 将用户画像存储为 Redis 哈希结构
        redis_instance.hset(f"user:{user_id}", mapping={k: json.dumps(v) for k, v in profile.items()})
    print("User profiles successfully stored in Redis!")


def user_behavior_update(user_id, asin):
    """
    更新用户行为，动态调整用户画像。
    :param user_id: 用户ID
    :param asin: 商品唯一标识符 (ASIN)
    """
    global redis_instance, user_profiles

    # 确保用户画像已经存在
    if user_id not in user_profiles:
        print(f"User {user_id} does not exist in profiles. Initialize user profile first.")
        return

    # 数据库查询，以获取商品详细信息
    product_data = get_one_product(asin)

    if product_data.empty:
        print(f"Product {asin} does not exist in the database. Cannot update user profile.")
        return

    # 提取商品信息
    category_name = product_data["category_id"].iloc[0]
    product_title = product_data["title"].iloc[0]
    price = product_data["price"].iloc[0]

    # 动态更新用户画像
    user_profile = user_profiles[user_id]

    # 更新品类偏好
    user_profile["category_preferences"][str(category_name)] = user_profile["category_preferences"].get(
        str(category_name), 0) + 1

    # 更新商品标题偏好
    user_profile["product_preferences"][str(product_title)] = user_profile["product_preferences"].get(
        str(product_title), 0) + 1

    # 更新平均价格（重新计算）
    user_prices = redis_instance.hget(f"user:{user_id}", "user_prices")
    user_prices = json.loads(user_prices) if user_prices else []
    user_prices.append(float(price))  # 将价格强制转换为浮点数
    user_profile["average_price"] = round(mean(user_prices), 2)  # 计算平均价格并保留两位小数
    try:
        redis_instance.hset(f"user:{user_id}", json.dumps(user_profile))
        print(f"User {user_id} profile updated successfully.")
    except Exception as e:
        print(f"Error updating the user profile for {user_id}: {e}")
        print("Current user profile:", user_profile)


def update_recommendations_after_click(user_id, asin):
    """
    在用户点击某商品后实时更新推荐列表。
    :param user_id: 用户ID
    :param asin: 商品唯一标识符 (ASIN)
    """
    global redis_instance

    # 确保用户画像存在
    user_profile = redis_instance.hgetall(f"user:{user_id}")
    if not user_profile:
        print(f"User {user_id} profile does not exist in Redis. Initialize profile first.")
        return

    # 提取用户偏好的商品类别
    category_preferences = json.loads(user_profile[b"category_preferences"])
    top_category = max(category_preferences, key=category_preferences.get, default=None)

    # 从数据库中基于顶级类别推荐商品
    recommended_products = get_recommended_products(top_category, asin)

    if recommended_products.empty:
        print(f"No recommendations found based on top category {top_category} for user {user_id}.")
        return

    # 构建推荐列表
    recommendations = recommended_products["asin"].tolist()

    # 缓存推荐列表到Redis（用户行为推荐）
    redis_instance.hset(f"recommendations:{user_id}", "recommended_products", json.dumps(recommendations))
    print(f"Recommendations updated for User {user_id}: {recommendations}")


def init_user_profile():
    global user_profiles

    # 加载数据
    data = load_data()

    # 构建用户画像
    user_profiles = build_user_profiles(data)

    # 存储用户画像到 Redis
    store_profiles_in_redis(redis_instance, user_profiles)


def get_user_profile_detail(user_id):
    result = redis_instance.hgetall(f"user:{user_id}")
    return result
