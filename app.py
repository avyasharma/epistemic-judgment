import streamlit as st
import os
from epistemic_gated_rag import build_gated_pipeline, GatedRAGResult, EpistemicVerdict, EpistemicJudge, OpenAIClient
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
</style>
""", unsafe_allow_html=True)

# --- Sidebar: Hyperparameters ---
st.sidebar.title("🛡️ Configuration")
st.sidebar.markdown("Fine-tune the Epistemic Judge parameters.")

with st.sidebar:
    st.header("Retrieval Settings")
    top_k = st.slider("Top-k passages", 1, 10, 5)
    
    st.divider()
    
    st.header("Judge Settings")
    threshold = st.slider("Epistemic Threshold", 0.0, 10.0, 6.0, step=0.1)
    judge_mode = st.selectbox("Judge Mode", ["structured", "minimal"], index=0)
    
    st.divider()
    
    # Use default weights internally
    weights = (0.35, 0.45, 0.20)

# --- Mock Pipeline for Quick Testing ---
# This uses the REAL EpistemicJudge logic but with hardcoded retrieval.
class MockGatedRAG:
    """Mock pipeline that uses the actual Epistemic Judge but dummy retrieval."""
    def __init__(self, threshold, mode, weights):
        self.threshold = threshold
        self.mode = mode
        self.weights = weights
        # Initialize the real judge client
        self.client = OpenAIClient(model="gpt-5-mini")
        self.judge = EpistemicJudge(
            self.client, 
            mode=self.mode, 
            threshold=self.threshold, 
            weights=self.weights
        )

    def answer(self, query: str, top_k: int = 5) -> GatedRAGResult:
        # Simulated passages for testing
        if "france" in query.lower():
            passages = [
                "Paris is the capital of France and its largest city.",
                "The Eiffel Tower is a famous landmark in Paris, France.",
                "France is a country in Western Europe. It has many baguette bakeries."
            ]
        elif "contradict" in query.lower():
            passages = [
                "The earth is absolutely flat according to some ancient texts.",
                "Scientific satellite imagery proves the earth is an oblate spheroid."
            ]
        else:
            passages = [
                "The provided documents discuss irrelevant topics like recipe for pasta.",
                "Cooking pasta requires boiling water and salt."
            ]

        # Call the REAL judge
        verdict = self.judge.judge(query, passages[:top_k])
        
        # If answering, we use a simple generator mock or the real one
        response = None
        if verdict.should_answer:
            response = self.client.generate(query, passages[:top_k])
        
        return GatedRAGResult(
            query=query,
            verdict=verdict,
            response=response,
            retrieved_passages=passages[:top_k]
        )

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
            pipeline = MockGatedRAG(threshold=threshold, mode=judge_mode, weights=weights)
            result = pipeline.answer(query, top_k=top_k)
        col_main, col_docs = st.columns([2, 1])

        with col_main:
            decision = result.verdict.decision
            if decision == "ANSWER":
                st.success(f"**DECISION: {decision}**")
            else:
                st.error(f"**DECISION: {decision}**")
            
            # Justification
            st.markdown(f'<div class="justification-box"><b>Justification:</b><br>{result.verdict.justification}</div>', unsafe_allow_html=True)

            st.divider()

            # Answer Section
            st.subheader("Answer")
            if decision == "ANSWER":
                st.info(result.response)
            else:
                st.warning("Generation was blocked by the Epistemic Judge to prevent hallucination.")

        with col_docs:
            st.subheader("Retrieved Documents")
            if result.retrieved_passages:
                for i, doc in enumerate(result.retrieved_passages):
                    with st.expander(f"Passage {i+1}", expanded=(i==0)):
                        st.write(doc)
            else:
                st.write("No documents retrieved.")