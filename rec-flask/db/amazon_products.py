# 假设下载的文件是一个 CSV 文件，找到该文件
import os
import sqlite3

import kagglehub
import pandas as pd


def init_products():
    # 下载数据集部分保持不变
    path = kagglehub.dataset_download("asaniczka/amazon-products-dataset-2023-1-4m-products")
    print("Path to dataset files:", path)

    # 查找 CSV 文件
    data_file = None
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".csv"):
                data_file = os.path.join(root, file)
                break

    if not data_file:
        raise FileNotFoundError("No CSV file found in the dataset.")

    print("CSV file found:", data_file)

    # 连接 SQLite 数据库
    conn = sqlite3.connect('db/recommend.db')

    # 分块读取和写入
    table_name = "amazon_products"
    chunksize = 10000  # 每次处理 10,000 行数据

    # 声明索引变量以判断是否创建表
    first_chunk = True

    for chunk in pd.read_csv(data_file, chunksize=chunksize):
        chunk.to_sql(table_name, conn, if_exists="replace" if first_chunk else "append", index=False)
        first_chunk = False  # 只在第一块时使用 "replace"

    print(f"Data has been written to the SQLite database in table '{table_name}'.")

    # 验证写入的行数
    cursor = conn.cursor()
    cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
    count = cursor.fetchone()[0]
    print(f"Number of rows in table '{table_name}': {count}")

    conn.close()
