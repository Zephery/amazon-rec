# recommendation.py
from datetime import datetime

import fakeredis

from db.database import load_products, load_user_clicks, load_user_reviews, load_categories, initialize_database

# 初始化 Redis 连接
redis_client = fakeredis.FakeStrictRedis()

# 全局变量
user_item_matrix = None
user_item_sparse = None
decomposed_matrix = None
SVD = None
user_profiles = {}
item_latent_vectors = None
asin_to_category = None
user_ids = None
item_ids = None
model_initialized = False

# 加载数据
try:
    print("start to load db " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))
    products = load_products()
    user_clicks = load_user_clicks()
    reviews = load_user_reviews()
    categories = load_categories()
    print("end to load db " + datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

except Exception as e:
    print(e)
    initialize_database()
