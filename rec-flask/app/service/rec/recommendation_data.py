# app/service/recommendation_data.py

import logging

import pandas as pd

from db.database import (
    load_products,
    load_user_clicks,
    load_user_reviews,
    load_categories
)

# 全局数据对象
products = None
user_clicks = None
user_reviews = None
categories = None

user_item_matrix = None  # 用户-物品交互稠密matrix/pandas.DataFrame
user_item_sparse = None  # 稀疏版本(如CSR)
decomposed_matrix = None  # SVD分解后的user向量
item_latent_vectors = None  # SVD分解后的物品向量
asin_to_category = None  # 商品asin到类别的索引Series
user_profiles = None  # 用户画像 (dict)
user_ids = None  # 用户ID（numpy array）
item_ids = None  # 物品ID（numpy array）

model_initialized = False


def init_data():
    global products, user_clicks, user_reviews, categories
    products = load_products()
    user_clicks = load_user_clicks()
    user_reviews = load_user_reviews()
    categories = load_categories()


def build_user_item_matrix():
    global user_item_matrix, user_ids, item_ids, user_clicks
    # 基于用户点击简单构建user-item矩阵(可根据实际业务用别的权重、评分)
    if user_clicks is None or user_clicks.empty:
        return
    user_item_matrix = user_clicks.groupby(['user_id', 'asin']).size().unstack(fill_value=0)
    user_ids = user_item_matrix.index.values
    item_ids = user_item_matrix.columns.values


def build_asin_to_category():
    global asin_to_category, products
    if products is None or products.empty:
        asin_to_category = pd.Series(dtype=int)
    else:
        asin_to_category = products.set_index('asin')['category_id']


def build_model_svd(n_components=10):
    global user_item_matrix, decomposed_matrix, item_latent_vectors, user_ids, item_ids
    from sklearn.decomposition import TruncatedSVD
    from scipy.sparse import csr_matrix

    if user_item_matrix is None:
        build_user_item_matrix()
    if user_item_matrix is None or user_item_matrix.empty:
        decomposed_matrix = None
        item_latent_vectors = None
        return

    user_item_sparse = csr_matrix(user_item_matrix.values)
    n_components = min(n_components, user_item_sparse.shape[1])
    if n_components < 2:
        decomposed_matrix = None
        item_latent_vectors = None
        return

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    decomposed_matrix = svd.fit_transform(user_item_sparse)
    # item_latent_vectors是DataFrame，行索引对应asin
    item_latent_vectors = pd.DataFrame(svd.components_.T, index=user_item_matrix.columns)


def init_user_profiles():
    # 假设用户画像通过 user_profile.py 构建一次并在某处全局save，这里统一引用
    try:
        from app.service.user.user_profile import user_profiles as up
        global user_profiles
        user_profiles = up
    except Exception as e:
        print(f"Warning: user_profiles not loaded: {e}")
        user_profiles = {}


def initialize_all():
    global model_initialized
    init_data()
    build_user_item_matrix()
    build_asin_to_category()
    build_model_svd()
    init_user_profiles()
    model_initialized = True
    logging.info("Recommendation system data/model initialized.")


# 启动时自动加载
initialize_all()
