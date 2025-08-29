import faiss
from sentence_transformers import SentenceTransformer
import numpy as np

model = SentenceTransformer('all-MiniLM-L6-v2')

def create_embeddings(chunks):
    return model.encode(chunks)

def build_faiss_index(embeddings):
    dim = embeddings.shape[1]
    index = faiss.IndexFlatL2(dim)
    index.add(embeddings)
    return index

def search_index(query, index, chunks, top_k=5):
    qv = model.encode([query])
    _, I = index.search(np.array(qv), top_k)
    return [chunks[i] for i in I[0]]
