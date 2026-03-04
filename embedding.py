from sentence_transformers import SentenceTransformer
import os
from typing import List

class LocalHuggingFaceEmbedding:
    def __init__(self, model_path):
        # Point to the snapshot folder inside your cache
        self.model = SentenceTransformer(model_path, device='cpu')

    # 1. LangChain calls this to embed your document chunks
    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        return self.model.encode(texts).tolist()

    # 2. LangChain calls this to embed the user's question
    def embed_query(self, text: str) -> List[float]:
        # Encode a single string, but return it as a flat list
        return self.model.encode([text])[0].tolist()
    

base_path = os.path.expanduser("~/.cache/huggingface/hub/models--sentence-transformers--all-MiniLM-L6-v2/snapshots")
snapshot_folder = os.path.join(base_path, os.listdir(base_path)[0])

my_embedding_function = LocalHuggingFaceEmbedding(snapshot_folder)