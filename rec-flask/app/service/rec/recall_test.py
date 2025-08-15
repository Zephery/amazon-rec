import sqlite3
from pathlib import Path

import faiss
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from app.service.rec.recall import gen_embeddings

# 数据加载
gen_embeddings()
base_path = str(Path(__file__).parent.parent.parent.parent.parent)
db_path = base_path + '/db/recommend.db'
conn = sqlite3.connect(db_path, check_same_thread=False)

df = pd.read_sql_query('SELECT * FROM amazon_products', conn)
titles = df["title"].astype(str).tolist()

# 向量化
model = SentenceTransformer(base_path + "/all-MiniLM-L6-v2")
embeddings = np.load(base_path + '/product_emb.npy').astype('float32')

# 索引
faiss.normalize_L2(embeddings)
index = faiss.IndexFlatIP(embeddings.shape[1])
index.add(embeddings)

# 查询
query = "ipad"
query_emb = model.encode([query], normalize_embeddings=True)
D, I = index.search(query_emb, 10)

for i in I[0]:
    item = df.iloc[i]
    print(f"ASIN: {item['asin']}, TITLE: {item['title']}, PRICE: {item['price']}")
