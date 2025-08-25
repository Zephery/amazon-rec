"""
独立的embedding生成脚本
运行此脚本来生成product_emb.npy文件
"""
import os
import numpy as np
from pathlib import Path
from sentence_transformers import SentenceTransformer
import pandas as pd

def generate_embeddings():
    """生成商品标题的embedding"""
    base_path = str(Path(__file__).parent.parent.parent)
    
    # 检查模型路径
    model_path = os.path.join(base_path, "all-MiniLM-L6-v2")
    if not os.path.exists(model_path):
        print(f"Model not found at: {model_path}")
        print("Please download the model first or check the path")
        return False
    
    # 检查products数据
    try:
        from app.service.data_loader import products
        print(f"Found {len(products)} products")
    except ImportError:
        print("Cannot import products data")
        return False
    
    # 加载模型
    print("Loading model...")
    model = SentenceTransformer(model_path)
    
    # 获取标题
    titles = products['title'].tolist()
    print(f"Processing {len(titles)} titles...")
    
    # 批量生成embedding
    BATCH_SIZE = 1000
    embeddings = []
    
    for i in range(0, len(titles), BATCH_SIZE):
        batch_titles = titles[i:i + BATCH_SIZE]
        print(f"Processing batch {i//BATCH_SIZE + 1}/{(len(titles) + BATCH_SIZE - 1)//BATCH_SIZE}")
        
        batch_emb = model.encode(
            batch_titles,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True
        )
        embeddings.append(batch_emb)
    
    # 合并所有embedding
    all_embeddings = np.vstack(embeddings)
    print(f"Generated embeddings shape: {all_embeddings.shape}")
    
    # 保存
    output_path = os.path.join(base_path, "product_emb.npy")
    np.save(output_path, all_embeddings)
    print(f"Saved embeddings to: {output_path}")
    
    return True

if __name__ == "__main__":
    print("Starting embedding generation...")
    success = generate_embeddings()
    if success:
        print("Embedding generation completed successfully!")
    else:
        print("Embedding generation failed!") 