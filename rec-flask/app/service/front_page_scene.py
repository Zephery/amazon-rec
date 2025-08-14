import os

import numpy as np
import pandas as pd

from app.service.data_loader import products as raw_products, categories, user_clicks, reviews
from app.utils.common_utils import get_current_time

CACHE_PATH = 'products_cache.json'


def save_products_to_json(df, path):
    df.to_json(path, orient='records', force_ascii=False)


def load_products_from_json(path):
    return pd.read_json(path, orient='records')


if os.path.exists(CACHE_PATH):
    print(f"从缓存文件 {CACHE_PATH} 加载 front page products...")
    products = load_products_from_json(CACHE_PATH)
else:
    print("缓存文件不存在，重新计算 front page products...")
    products = raw_products.copy()
    products['stars'] = products['stars'].fillna(0)
    products['reviews'] = products['reviews'].fillna(0)
    products['boughtInLastMonth'] = products['boughtInLastMonth'].fillna(0)
    products = pd.merge(products, categories, left_on='category_id', right_on='id', how='left')
    products['category_name'] = products['category_name'].fillna("Unknown")
    reviews_stats = reviews.groupby('asin').agg(
        avg_rating=('rating', 'mean'),
        num_reviews=('rating', 'count')
    ).reset_index()
    products = pd.merge(products, reviews_stats, on='asin', how='left')
    products['avg_rating'] = products['avg_rating'].fillna(0)
    products['num_reviews'] = products['num_reviews'].fillna(0)
    click_stats = user_clicks.groupby('asin').agg(
        click_count=('asin', 'count')
    ).reset_index()
    products = pd.merge(products, click_stats, on='asin', how='left')
    products['click_count'] = products['click_count'].fillna(0)


    def calculate_score(row):
        click_weight = 0.3
        rating_weight = 0.4
        review_weight = 0.2
        sales_weight = 0.1
        return (
                click_weight * row['click_count'] +
                rating_weight * row['avg_rating'] +
                review_weight * np.log1p(row['num_reviews']) +
                sales_weight * row['boughtInLastMonth']
        )


    products['score'] = products.apply(calculate_score, axis=1)
    products = products.sort_values(by='score', ascending=False)
    save_products_to_json(products, CACHE_PATH)
    print(f"front page products 已保存到 {CACHE_PATH}")

print(get_current_time() + " end to prepare front page scene")


# -----------------------------------------------
# 首页推荐逻辑
# -----------------------------------------------

# 全局热门商品推荐函数
def get_global_top_products(limit=500, randomize=True):
    """
    获取综合评分最高的商品 ASIN 列表。

    :param limit: int, 返回的商品数量
    :param randomize: bool, 是否启用随机化逻辑，默认启用
    :return: list, 推荐的商品 ASIN 列表
    """
    # 如果不启用随机化逻辑，直接返回前 `limit` 个商品
    if not randomize:
        return products['asin'].head(limit).tolist()

    # 启用随机化逻辑：基于综合评分进行加权采样
    sorted_products = products.copy()
    sorted_products['sampling_weight'] = sorted_products['score'] / sorted_products['score'].sum()

    # 基于权重抽样，避免每次返回完全相同
    sampled_products = sorted_products.sample(
        n=limit,
        weights='sampling_weight',
        random_state=np.random.randint(0, 10000)  # 确保每次调用的随机性
    )

    # 返回采样后的商品 ASIN 列表
    return sampled_products['asin'].tolist()


# 模拟调用逻辑：获取全局热门商品
if __name__ == "__main__":
    # 调用热门商品推荐函数（启用随机化）
    top_asins = get_global_top_products(limit=10, randomize=True)

    # 输出结果
    print(f"热门商品的 ASIN 列表（前 10 个）: {top_asins}")