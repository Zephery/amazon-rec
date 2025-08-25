import multiprocessing
import os
from pathlib import Path
import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

class EmbeddingService:
    _instance = None
    _initialized = False
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(EmbeddingService, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        if not self._initialized:
            self._init_embeddings()
            self._initialized = True
    
    def _init_embeddings(self):
        """初始化embedding模型和向量"""
        print("Initializing EmbeddingService...")
        
        # 获取项目根路径
        base_path = str(Path(__file__).parent.parent.parent)
        
        # 加载模型
        model_path = os.path.join(base_path, "all-MiniLM-L6-v2")
        if os.path.exists(model_path):
            self.model = SentenceTransformer(model_path)
            print(f"Loaded model from: {model_path}")
        else:
            print(f"Model path not found: {model_path}")
            self.model = None
        
        # 加载向量
        emb_path = os.path.join(base_path, "product_emb.npy")
        if os.path.exists(emb_path):
            self.embeddings = np.load(emb_path).astype('float32')
            print(f"Loaded embeddings from: {emb_path}")
            print(f"Vector count: {self.embeddings.shape[0]}")
            print(f"Vector dimension: {self.embeddings.shape[1]}")
        else:
            print(f"Embeddings file not found: {emb_path}")
            self.embeddings = None
        
        # 初始化FAISS索引
        if self.embeddings is not None:
            self._init_faiss_index()
    
    def _init_faiss_index(self):
        """初始化FAISS索引"""
        try:
            # 尝试加载已存在的索引
            index_path = os.path.join(str(Path(__file__).parent.parent.parent), "faiss.index")
            self.index = faiss.read_index(index_path)
            print("Loaded existing FAISS index")
        except Exception:
            # 创建新索引
            print("Creating new FAISS index...")
            dim = self.embeddings.shape[1]
            quantizer = faiss.IndexFlatIP(dim)
            nlist = min(4096, len(self.embeddings) // 30)  # 动态调整nlist
            self.index = faiss.IndexIVFFlat(quantizer, dim, nlist)
            
            # 归一化向量
            faiss.normalize_L2(self.embeddings)
            
            # 训练索引
            self.index.train(self.embeddings)
            self.index.add(self.embeddings)
            
            # 保存索引
            faiss.write_index(self.index, index_path)
            print("FAISS index created and saved")
        
        # 设置线程数
        faiss.omp_set_num_threads(multiprocessing.cpu_count())
    
    def get_model(self):
        """获取模型实例"""
        return self.model
    
    def get_embeddings(self):
        """获取向量数据"""
        return self.embeddings
    
    def get_index(self):
        """获取FAISS索引"""
        return self.index
    
    def encode_query(self, query, normalize=True):
        """编码查询文本"""
        if self.model is None:
            raise RuntimeError("Model not initialized")
        return self.model.encode([query], normalize_embeddings=normalize)
    
    def search(self, query_emb, top_k):
        """搜索最相似的向量"""
        if self.index is None:
            raise RuntimeError("FAISS index not initialized")
        return self.index.search(query_emb, top_k)
    
    def get_asin_by_index(self, index):
        """根据索引获取ASIN"""
        if self.embeddings is None:
            return None
        if 0 <= index < len(self.embeddings):
            # 这里需要从products中获取对应的asin
            # 暂时返回索引，实际使用时需要传入products数据
            return index
        return None

# 全局实例
embedding_service = EmbeddingService() 