import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

MODEL_NAME = 'all-MiniLM-L6-v2'  # 384-dim, fast and accurate
model = SentenceTransformer(MODEL_NAME)

class VectorStore:
    def __init__(self, dim=384):
        self.index = faiss.IndexFlatL2(dim)
        self.chunks = []

    def add_chunks(self, chunks):
        """Embed and index a list of text chunks."""
        embeddings = model.encode(chunks, convert_to_numpy=True,
                                  show_progress_bar=True)
        embeddings = embeddings.astype('float32')
        self.index.add(embeddings)
        self.chunks.extend(chunks)
        print(f'Indexed {len(chunks)} chunks, total: {self.index.ntotal}')

    def save(self, path='store.faiss'):
        faiss.write_index(self.index, path)

    def load(self, path='store.faiss'):
        self.index = faiss.read_index(path)

vs = VectorStore()
vs.add_chunks(['The RAG pipeline retrieves context before generating answers.',
               'FAISS provides sub-linear nearest-neighbour search.'])
vs.save()
