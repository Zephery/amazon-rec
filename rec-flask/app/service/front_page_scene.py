import numpy as np
import pandas as pd

from app.service.data_loader import products, categories, user_clicks, reviews

# -----------------------------------------------
# 数据预处理
# -----------------------------------------------

# 处理基础数据，填充缺失值
products['stars'] = products['stars'].fillna(0)
products['reviews'] = products['reviews'].fillna(0)
products['boughtInLastMonth'] = products['boughtInLastMonth'].fillna(0)

# 合并产品分类信息
products = pd.merge(products, categories, left_on='category_id', right_on='id', how='left')
products['category_name'] = products['category_name'].fillna("Unknown")

# -----------------------------------------------
# 利用 `amazon_reviews` 表补充评分和评论信息
# -----------------------------------------------

# 从 reviews 表中统计商品的平均评分和评价数量
reviews_stats = reviews.groupby('asin').agg(
    avg_rating=('rating', 'mean'),
    num_reviews=('rating', 'count')
).reset_index()

# 将计算的统计数据合并到商品列表
products = pd.merge(products, reviews_stats, on='asin', how='left')

# 填充 `avg_rating` 和 `num_reviews` 的缺失值
products['avg_rating'] = products['avg_rating'].fillna(0)
products['num_reviews'] = products['num_reviews'].fillna(0)

# -----------------------------------------------
# 点击数据统计
# -----------------------------------------------

# 计算用户点击的次数
click_stats = user_clicks.groupby('asin').agg(
    click_count=('asin', 'count')  # 每个商品的点击次数
).reset_index()

# 合并点击统计数据到商品列表
products = pd.merge(products, click_stats, on='asin', how='left')

# 填充 `click_count` 的空值
products['click_count'] = products['click_count'].fillna(0)


# -----------------------------------------------
# 综合评分计算
# -----------------------------------------------

# 定义综合评分函数
def calculate_score(row):
    click_weight = 0.3
    rating_weight = 0.4
    review_weight = 0.2
    sales_weight = 0.1
    return (
            click_weight * row['click_count'] +
            rating_weight * row['avg_rating'] +
            review_weight * np.log1p(row['num_reviews']) +  # 对评论数量取对数
            sales_weight * row['boughtInLastMonth']
    )


# 应用评分逻辑，将综合评分附加到商品数据
products['score'] = products.apply(calculate_score, axis=1)

# 按综合评分降序排序
products = products.sort_values(by='score', ascending=False)


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
