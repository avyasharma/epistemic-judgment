import streamlit as st
from rag import build_rag_pipeline

st.set_page_config(page_title="Simple RAG Demo", page_icon="🔎", layout="wide")

st.title("🔎 Simple RAG Demo")
st.write("Ask a question over your document dataset.")

st.sidebar.header("Settings")
top_k = st.sidebar.slider("Top-k retrieved passages", min_value=1, max_value=10, value=5)

query = st.text_area(
    "Enter your question:",
    height=120,
    placeholder="e.g. What is the main topic discussed in the documents?"
)

if "rag_pipeline" not in st.session_state:
    st.session_state.rag_pipeline = None

if st.button("Ask", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        if st.session_state.rag_pipeline is None:
            with st.spinner("Loading model + indexing documents (first run only)..."):
                st.session_state.rag_pipeline = build_rag_pipeline(
                    document_path="dataset/documents.csv",
                    embedding_model="all-MiniLM-L6-v2",
                    llm_model="gpt-5-mini",
                )

        with st.spinner("Retrieving passages and generating answer..."):
            answer, passages = st.session_state.rag_pipeline.answer(query, top_k=top_k)

        st.subheader("Answer")
        st.write(answer)

        st.subheader("Retrieved Passages")
        if passages:
            for i, passage in enumerate(passages, start=1):
                with st.expander(f"Passage {i}"):
                    st.write(passage)
        else:
            st.info("No passages were retrieved.")