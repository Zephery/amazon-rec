# recommendation.py
import logging

import fakeredis
import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD
from sklearn.metrics.pairwise import cosine_similarity

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
    products = load_products()
    user_clicks = load_user_clicks()
    reviews = load_user_reviews()
    categories = load_categories()
except Exception as e:
    print(e)
    initialize_database()

