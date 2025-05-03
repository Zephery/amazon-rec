import json
import sqlite3
from statistics import mean

import fakeredis
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from textblob import TextBlob  # 用于情感分析

# Redis 实例（伪 Redis 数据库）
redis_instance = fakeredis.FakeStrictRedis()

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


# 加载数据库数据
def load_data(sqlite_file):
    """
    加载SQLite数据库中的数据，并合并商品、评论以及商品类别信息。
    """
    # 初始化数据库连接
    conn = sqlite3.connect(sqlite_file)

    # 读取数据表
    reviews_df = pd.read_sql_query("SELECT * FROM amazon_reviews", conn)
    products_df = pd.read_sql_query("SELECT * FROM amazon_products", conn)
    categories_df = pd.read_sql_query("SELECT * FROM amazon_categories", conn)

    # 关闭连接
    conn.close()

    # 数据清理
    reviews_df = reviews_df.dropna(subset=["user_id", "asin", "text", "rating"])
    products_df = products_df.dropna(subset=["asin", "title", "price", "category_id", "stars"])

    # 合并商品和类别表
    products_df = pd.merge(products_df, categories_df, left_on="category_id", right_on="id", how="left")

    # 合并评论数据与产品信息表（基于 ASIN）
    merged_df = pd.merge(reviews_df, products_df, on="asin", how="inner")
    return merged_df


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
    user_profiles = {}

    # 转换时间戳为 datetime 对象
    data['timestamp'] = pd.to_datetime(data['timestamp'])

    # 按用户分组构建画像
    for user_id, user_data in data.groupby("user_id"):
        # 基础行为分析
        avg_rating = user_data["rating"].mean()  # 平均评分
        review_count = len(user_data)  # 评论数量

        # 分类偏好（品类偏好、品牌偏好、商品偏好）
        category_preferences = user_data["category_name"].value_counts().to_dict()  # 品类偏好（使用类别名称）
        brand_preferences = user_data["title_y"].value_counts().to_dict()  # 商品标题偏好（来自 `amazon_products` 表）
        product_preferences = user_data["title_x"].value_counts().to_dict()  # 评论标题偏好（来自 `amazon_reviews` 表）

        # 消费能力分析
        avg_price = mean(user_data["price"]) if len(user_data["price"]) > 0 else 0
        expensive_product_count = len(user_data[user_data["price"] > 1000])

        # 评论关键标签提取
        vectorizer = CountVectorizer(max_features=5, stop_words=stopwords_list)
        try:
            words_matrix = vectorizer.fit_transform(user_data["text"])
            top_keywords = vectorizer.get_feature_names_out()
        except ValueError:  # 没有有效评论文本
            top_keywords = []

        # 评论情感分析（使用 TextBlob）
        sentiments = {"positive": 0, "neutral": 0, "negative": 0}
        for review in user_data["text"]:
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


if __name__ == "__main__":
    # SQLite 数据库路径
    sqlite_path = "/Users/wwwzh/PycharmProjects/amazon-rec/rec-flask/db/recommend.db"  # 替换为实际文件路径

    # 加载数据
    data = load_data(sqlite_file=sqlite_path)

    # 构建用户画像
    user_profiles = build_user_profiles(data)

    # 存储用户画像到 Redis
    store_profiles_in_redis(redis_instance, user_profiles)

    # 示例：打印某用户的画像
    test_user = list(user_profiles.keys())[0]
    print(f"User {test_user} profile in Redis:")
    print(redis_instance.hgetall(f"user:{test_user}"))
