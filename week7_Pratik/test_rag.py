from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain

print("Loading PDF...")

text = load_pdf("data/sample.pdf")

chunks = split_text(text)

embedder = EmbeddingModel()

embeddings = embedder.embed_documents(chunks)

store = VectorStore()

store.create_index(embeddings, chunks)

rag = RAGChain()

while True:

    question = input("\nAsk a Question (type 'exit' to quit): ")

    if question.lower() == "exit":
        break

    query_embedding = embedder.embed_query(question)

    retrieved_chunks = store.search(query_embedding)

    answer = rag.generate_answer(question, retrieved_chunks)

    print("\nAnswer:\n")

    print(answer)