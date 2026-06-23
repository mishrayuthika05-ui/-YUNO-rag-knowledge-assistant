import numpy as np
from sentence_transformers import SentenceTransformer

model = SentenceTransformer('all-MiniLM-L6-v2')

def retrieve(query, vector_store, k=5):
    """Return top-k chunk texts with their L2 distances."""
    q_vec = model.encode([query], convert_to_numpy=True).astype('float32')
    distances, indices = vector_store.index.search(q_vec, k)
    results = []
    for dist, idx in zip(distances[0], indices[0]):
        if idx == -1:
            continue
        results.append({
            'chunk': vector_store.chunks[idx],
            'score': float(dist)
        })
    return results

if __name__ == '__main__':
    from modules.embeddings import VectorStore

    vs = VectorStore()

    chunks = [
        'The RAG pipeline retrieves context before generating answers.',
        'FAISS provides sub-linear nearest-neighbour search.'
    ]

    vs.add_chunks(chunks)

    hits = retrieve('How does FAISS search work?', vs, k=2)

    for i, h in enumerate(hits):
        print(f'[{i+1}] score={h["score"]:.4f}')
        print(h['chunk'])