import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd

from app.service.rec.recall import gen_embeddings
from app.service.embedding_service import embedding_service

# 数据加载
gen_embeddings()
base_path = str(Path(__file__).parent.parent.parent.parent.parent)
db_path = base_path + '/db/recommend.db'
conn = sqlite3.connect(db_path, check_same_thread=False)

df = pd.read_sql_query('SELECT * FROM amazon_products', conn)
titles = df["title"].astype(str).tolist()

# 共享模型与索引
model = embedding_service.get_model()
embeddings = embedding_service.get_embeddings()
index = embedding_service.get_index()

# 查询
query = "ipad"
query_emb = embedding_service.encode_query(query, normalize=True)
D, I = index.search(query_emb, 10)

for i in I[0]:
    item = df.iloc[i]
    print(f"ASIN: {item['asin']}, TITLE: {item['title']}, PRICE: {item['price']}")
