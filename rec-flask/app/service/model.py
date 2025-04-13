# model.py
import logging

import pandas as pd
from scipy.sparse import csr_matrix
from sklearn.decomposition import TruncatedSVD

from app.utils.utils import filter_high_activity, build_sparse_matrix
from db.database import load_products, load_user_clicks, load_user_reviews, initialize_database, load_categories

# 全局变量
user_item_matrix = None
user_item_sparse = None
decomposed_matrix = None
SVD = None
item_latent_vectors = None
asin_to_category = None

# 限制数据规模的常量，可根据实际情况调整
MAX_USERS = 10000  # 最多保留的用户数
MAX_ITEMS = 5000  # 最多保留的商品数

# 加载数据
try:
    products = load_products()
    user_clicks = load_user_clicks()
    reviews = load_user_reviews()
    categories = load_categories()
except Exception as e:
    print(e)
    initialize_database()


def initialize_model_with_reviews(user_clicks, user_reviews, products):
    global user_item_matrix, user_item_sparse, decomposed_matrix, SVD, item_latent_vectors, asin_to_category

    # 检查 products 是否为空
    if products.empty:
        logging.error("Products data is empty. Cannot initialize model.")
        return

    # 如果没有用户评论数据，跳过模型初始化
    if user_reviews.empty:
        logging.warning("User reviews data is empty. Skipping model initialization with reviews.")
        return

    # 限制评论表规模，只保留高频用户和商品
    user_reviews = filter_high_activity(user_reviews, user_col='user_id', item_col='asin', user_threshold=5,
                                        item_threshold=5)
    user_reviews = user_reviews.head(MAX_USERS * MAX_ITEMS)  # 限制评论表的最大行数

    # 构建用户-商品交互稀疏矩阵
    user_ids = user_reviews['user_id'].unique()[:MAX_USERS]  # 保留最多 MAX_USERS 个用户
    item_ids = user_reviews['asin'].unique()[:MAX_ITEMS]  # 保留最多 MAX_ITEMS 个商品

    # 加权评分
    user_reviews['rating_weighted'] = user_reviews['rating'] * \
                                      (1 + 0.1 * user_reviews['verified_purchase']) * \
                                      (1 + 0.05 * user_reviews['helpful_vote'].fillna(0))

    # 构建稀疏交互矩阵
    user_item_sparse = build_sparse_matrix(
        user_col='user_id',
        item_col='asin',
        value_col='rating_weighted',
        user_ids=user_ids,
        item_ids=item_ids,
        data=user_reviews
    )

    # 训练 SVD 模型
    SVD = TruncatedSVD(n_components=20, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)

    # 构建用户-商品交互矩阵，用于后续推荐
    user_item_matrix = pd.DataFrame(user_item_sparse.toarray(), index=user_ids, columns=item_ids)

    # 预先计算 item_latent_vectors
    item_latent_vectors = pd.DataFrame(SVD.components_.T, index=user_item_matrix.columns)

    # 预先计算 asin_to_category
    asin_to_category = products.set_index('asin')['category_id']


# 初始化模型
initialize_model_with_reviews(user_clicks, reviews, products)


# 定义一个函数，每日重训模型
def retrain_model():
    global user_item_matrix, user_item_sparse, decomposed_matrix, SVD, item_latent_vectors, asin_to_category
    global products, user_clicks, user_reviews
    print("Retraining model...")
    # 重新加载数据
    products = load_products()
    user_clicks = load_user_clicks()
    user_reviews = load_user_reviews()
    if user_clicks.empty:
        print("No user clicks data available.")
        user_item_matrix = None
        decomposed_matrix = None
        return
    # 重新构建用户-商品交互矩阵
    user_item_matrix = user_clicks.groupby(['user_id', 'asin']).size().unstack(fill_value=0)
    user_item_sparse = csr_matrix(user_item_matrix.values)
    # 重新训练SVD模型
    SVD = TruncatedSVD(n_components=20, random_state=42)
    decomposed_matrix = SVD.fit_transform(user_item_sparse)
    # 更新 item_latent_vectors
    item_latent_vectors = pd.DataFrame(SVD.components_.T, index=user_item_matrix.columns)
    # 更新 asin_to_category，如果 products 有更新的话
    asin_to_category = products.set_index('asin')['category_id']
    print("Model retraining completed.")
