import streamlit as st
import tempfile
import os

from utils.pdf_loader import load_pdf
from utils.text_splitter import split_text
from utils.embeddings import EmbeddingModel
from utils.vector_store import VectorStore
from utils.rag_chain import RAGChain

st.set_page_config(
    page_title="Document Question Answering (RAG)",
    page_icon="📄",
    layout="wide"
)

st.title("📄 Document Question Answering using RAG")
st.write("Upload a PDF and ask questions about its contents.")

uploaded_file = st.file_uploader(
    "Upload PDF",
    type=["pdf"]
)

if uploaded_file:

    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(uploaded_file.read())
        pdf_path = tmp.name

    if "vectorstore" not in st.session_state:

        with st.spinner("Reading PDF..."):

            text = load_pdf(pdf_path)

        with st.spinner("Splitting document..."):

            chunks = split_text(text)

        with st.spinner("Loading Embedding Model..."):

            embedder = EmbeddingModel()

        with st.spinner("Generating Embeddings..."):

            embeddings = embedder.embed_documents(chunks)

        with st.spinner("Building Vector Database..."):

            store = VectorStore()

            store.create_index(embeddings, chunks)

        st.session_state.embedder = embedder
        st.session_state.vectorstore = store
        st.session_state.rag = RAGChain()

        st.success("Document processed successfully!")

    question = st.text_input("Ask a question")

    if st.button("Get Answer"):

        if question.strip() == "":
            st.warning("Please enter a question.")

        else:

            query_embedding = st.session_state.embedder.embed_query(question)

            retrieved_chunks = st.session_state.vectorstore.search(
                query_embedding,
                top_k=3
            )

            answer = st.session_state.rag.generate_answer(
                question,
                retrieved_chunks
            )

            st.subheader("Answer")

            st.write(answer)

            with st.expander("Retrieved Context"):

                for i, chunk in enumerate(retrieved_chunks):

                    st.markdown(f"### Chunk {i+1}")

                    st.write(chunk)