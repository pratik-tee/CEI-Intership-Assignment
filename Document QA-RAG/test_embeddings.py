from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import EmbeddingModel

# Load PDF
text = load_pdf("data/sample.pdf")

# Split into chunks
chunks = split_text(text)

print(f"Total Chunks: {len(chunks)}")

# Load embedding model
embedder = EmbeddingModel()

# Generate embeddings
embeddings = embedder.embed_documents(chunks)

print("\nEmbedding Shape:")
print(embeddings.shape)

print("\nFirst 10 values of first embedding:")
print(embeddings[0][:10])