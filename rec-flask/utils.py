# utils.py
import pandas as pd
from scipy.sparse import csr_matrix

# 示例函数：过滤数据量
def filter_high_activity(df, user_col='user_id', item_col='asin', user_threshold=10, item_threshold=10):
    """
    过滤低频用户和低频商品（只保留高活跃的用户和商品）
    """
    # 保留行为次数或评分数超过阈值的用户和商品
    active_users = df[user_col].value_counts()
    filtered_users = active_users[active_users >= user_threshold].index
    active_items = df[item_col].value_counts()
    filtered_items = active_items[active_items >= item_threshold].index
    return df[(df[user_col].isin(filtered_users)) & (df[item_col].isin(filtered_items))]

# 示例函数：构建稀疏矩阵
def build_sparse_matrix(user_col, item_col, value_col, user_ids, item_ids, data):
    """
    构建稀疏矩阵，避免使用 Pandas 宽表
    """
    user_map = {user: idx for idx, user in enumerate(user_ids)}
    item_map = {item: idx for idx, item in enumerate(item_ids)}

    # 将用户和商品映射为索引
    rows = data[user_col].map(user_map)
    cols = data[item_col].map(item_map)
    values = data[value_col]

    # 构建 COO 稀疏矩阵并转换为 CSR
    return csr_matrix((values, (rows, cols)), shape=(len(user_ids), len(item_ids)))
