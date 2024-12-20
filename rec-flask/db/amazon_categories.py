import sqlite3

import kagglehub
import pandas as pd

# 从 Kaggle 下载最新版本的数据集
path = kagglehub.dataset_download("asaniczka/amazon-products-dataset-2023-1-4m-products")
print("Path to dataset files:", path)

# 假设下载的文件是一个 CSV 文件，找到该文件
import os

data_file = None
for root, dirs, files in os.walk(path):
    for file in files:
        if file.endswith(".csv"):  # 可能需要修改扩展名根据数据格式
            data_file = os.path.join(root, file)

if not data_file:
    raise FileNotFoundError("No CSV file found in the dataset.")

print("CSV file found:", data_file)

# 使用 pandas 读取 CSV 文件
df = pd.read_csv(data_file)

# 连接 SQLite 数据库，或创建新数据库文件
conn = sqlite3.connect('recommend.db')

# 将数据写入 SQLite 数据库中的一个表
# 如果表不存在，默认 pandas 会创建它
table_name = "amazon_categories"
df.to_sql(table_name, conn, if_exists="replace", index=False)

print(f"Data has been written to the SQLite database in table '{table_name}'.")

# 如果需要验证写入，运行一个简单的查询
cursor = conn.cursor()
cursor.execute(f"SELECT COUNT(*) FROM {table_name}")
count = cursor.fetchone()[0]
print(f"Number of rows in table '{table_name}': {count}")

# 关闭数据库连接
conn.close()
