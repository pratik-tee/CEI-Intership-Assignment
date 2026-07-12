from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore

# Load PDF
text = load_pdf("data/sample.pdf")

# Chunk
chunks = split_text(text)

print(f"Chunks: {len(chunks)}")

# Embeddings
embedder = EmbeddingModel()

embeddings = embedder.embed_documents(chunks)

# Vector Store
store = VectorStore()

store.create_index(embeddings, chunks)

store.save()

print("Vector Database Saved!")

# -------------------------------
# Test Search
# -------------------------------

query = input("\nAsk a Question: ")

query_embedding = embedder.embed_query(query)

results = store.search(query_embedding)

print("\nRetrieved Chunks:\n")

for i, chunk in enumerate(results):

    print("=" * 60)

    print(f"Chunk {i+1}")

    print("=" * 60)

    print(chunk)