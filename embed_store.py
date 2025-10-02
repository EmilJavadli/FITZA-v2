import faiss
import numpy as np
from sentence_transformers import SentenceTransformer

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
embedding_model = SentenceTransformer(EMBEDDING_MODEL)

def get_embeddings(texts):
    embeddings = embedding_model.encode(texts, convert_to_numpy=True, normalize_embeddings=True)
    return embeddings.astype("float32")

def build_faiss_index(texts):
    embeddings = get_embeddings(texts)
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    return index