import streamlit as st
import os
from epistemic_gated_rag import serve_epistemic_request
from typing import List, Optional

# --- Page Configuration ---
st.set_page_config(
    page_title="Epistemic Gated RAG",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for simple justification appearance
st.markdown("""
<style>
    .justification-box {
        background-color: #ffffff;
        color: #000000;
        padding: 15px;
        border: 1px solid #d1d5da;
        border-radius: 5px;
        margin: 10px 0;
    }
    .score-label {
        font-weight: bold;
        color: #555;
    }
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Hyperparameters ---
st.sidebar.title("🛡️ Configuration")
st.sidebar.markdown("Fine-tune the Epistemic Judge parameters.")

with st.sidebar:
    st.header("Search & Index")
    index_dir = st.text_input("Index Directory", value="squad_index", help="Path to pre-built FAISS index (from build_retriever_index.py)")
    top_k = st.slider("Top-k passages", 1, 10, 5)
    
    st.divider()
    
    st.header("Judge Settings")
    threshold = st.slider("Epistemic Threshold", 0.0, 10.0, 6.0, step=0.1)
    judge_mode = st.selectbox("Judge Mode", ["structured", "minimal"], index=0)
    
    gen_model = "gpt-5-mini"
    judge_model = "gpt-5-mini"

st.title("🛡️ Epistemic Gated RAG")
st.write("Evaluate the reliability of retrieved context before generating answers.")

query = st.text_input(
    "Ask a question:",
    placeholder="e.g. What is the capital of France?",
    value="What is the capital of France?"
)

if st.button("Query Pipeline", type="primary"):
    if not query.strip():
        st.warning("Please enter a question.")
    else:
        with st.spinner("Judge is evaluating context..."):
            try:
                result = serve_epistemic_request(
                    question=query,
                    index_dir=index_dir,
                    top_k=top_k,
                    threshold=threshold,
                    judge_mode=judge_mode,
                    generator_model=gen_model,
                    judge_model=judge_model
                )
            except Exception as e:
                st.error(f"Error running pipeline: {e}")
                st.info("Make sure you have built the index first: `python build_retriever_index.py squad_index`")
                st.stop()

        col_main, col_docs = st.columns([2, 1])

        with col_main:
            decision = result.get("decision", "ABSTAIN")
            if decision == "ANSWER":
                st.success(f"**DECISION: {decision}**")
            else:
                st.error(f"**DECISION: {decision}**")

            # Justification
            st.markdown(f'<div class="justification-box"><b>Justification:</b><br>{result.get("justification", "No justification provided.")}</div>', unsafe_allow_html=True)

            st.divider()

            # Answer Section
            st.subheader("Answer")
            if decision == "ANSWER":
                st.info(result.get("response", "[No response generated]"))
            else:
                st.warning("Generation was blocked by the Epistemic Judge to prevent hallucination.")

        with col_docs:
            st.subheader("Retrieved Documents")
            passages = result.get("retrieved_context", [])
            if passages:
                for i, doc in enumerate(passages):
                    with st.expander(f"Passage {i+1}", expanded=(i==0)):
                        st.write(doc)
            else:
                st.write("No documents retrieved.")