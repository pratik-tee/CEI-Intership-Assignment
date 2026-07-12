import faiss
import numpy as np
import pickle
import os


class VectorStore:

    def __init__(self):

        self.index = None
        self.chunks = []

    def create_index(self, embeddings, chunks):

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatL2(dimension)

        self.index.add(np.array(embeddings).astype("float32"))

        self.chunks = chunks

    def save(self, folder="vector_db"):

        os.makedirs(folder, exist_ok=True)

        faiss.write_index(self.index, f"{folder}/faiss_index.bin")

        with open(f"{folder}/chunks.pkl", "wb") as f:

            pickle.dump(self.chunks, f)

    def load(self, folder="vector_db"):

        self.index = faiss.read_index(f"{folder}/faiss_index.bin")

        with open(f"{folder}/chunks.pkl", "rb") as f:

            self.chunks = pickle.load(f)

    def search(self, query_embedding, top_k=3):

        distances, indices = self.index.search(

            np.array([query_embedding]).astype("float32"),

            top_k

        )

        results = []

        for idx in indices[0]:

            results.append(self.chunks[idx])

        return results