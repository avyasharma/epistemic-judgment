import os
import faiss
import numpy as np
import pandas as pd
from typing import List
from dotenv import load_dotenv


import kagglehub
from openai import OpenAI
from sentence_transformers import SentenceTransformer


load_dotenv()

# --- Configuration ---
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
HUGGING_FACE_KEY = os.getenv("HUGGING_FACE_KEY")
BASE_URL = os.getenv("BASE_URL")

# openai.OpenAI(api_key=api_key,  base_url=base_url)

class OpenAIClient():
    def __init__(self, model: str = "gpt-5-mini"):
        # Initialize OpenAI client (make sure OPENAI_API_KEY is configured)
        self.client = OpenAI(api_key=OPENAI_API_KEY, base_url=BASE_URL)
        self.model = model

    def generate(self, query: str, context: List[str]) -> str:
        context_str = "\n\n".join([f"Document {i+1}: {doc}" for i, doc in enumerate(context)])
        
        system_prompt = (
            "You are a helpful, accurate, and concise assistant. "
            "Use the provided context to answer the user's query. "
            "If the answer is not contained in the context, clearly state that you cannot find the answer in the provided context."
        )
        
        user_prompt = f"Context:\n{context_str}\n\nQuery: {query}"
        
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            temperature=0.0,
        )
        return response.choices[0].message.content


class DocumentRetriever():
    def __init__(self, model_name: str = 'all-MiniLM-L6-v2'):
        self.embedding_model = SentenceTransformer(model_name)
        self.embedding_dim = self.embedding_model.get_sentence_embedding_dimension()
        
        # Initialize FAISS Index (L2 distance)
        self.index = faiss.IndexFlatIP(self.embedding_dim)
        self.documents: List[str] = []
        
    def add_documents(self, documents: List[str], chunk_size: int = 150, chunk_overlap: int = 30):
        if not documents:
            return
            
        chunked_docs = []
        for doc in documents:
            words = doc.split()
            if not words:
                continue
            
            start = 0
            while start < len(words):
                end = int(start + chunk_size)
                chunked_docs.append(" ".join(words[int(start):end]))
                start += int(chunk_size - chunk_overlap)
                
        self.documents.extend(chunked_docs)
        # Convert text to embeddings mapping using the huggingface module
        embeddings = self.embedding_model.encode(chunked_docs, convert_to_numpy=True)
        # Add to FAISS
        self.index.add(np.array(embeddings, dtype=np.float32))

    def retrieve(self, query: str, top_k: int = 5) -> List[str]:
        query_embedding = self.embedding_model.encode([query], convert_to_numpy=True)
        _, indices = self.index.search(np.array(query_embedding, dtype=np.float32), top_k)
        
        retrieved_docs = []
        for idx in indices[0]:
            if idx != -1 and idx < len(self.documents):
                retrieved_docs.append(self.documents[idx])
        return retrieved_docs
        

class VanillaRAG:
    def __init__(self, retriever: DocumentRetriever, generator: OpenAIClient):
        self.retriever = retriever
        self.generator = generator

    def answer(self, query: str, top_k: int = 5) -> str:
        passages = self.retriever.retrieve(query, top_k=top_k)
        
        for i, passage in enumerate(passages):
             print(f"  -> Passage {i+1}: {passage[:100]}...")
        
        return self.generator.generate(query, passages)

# --- Kaggle Dataset Loader ---

def load_kaggle_rag_dataset(file_path: str) -> List[str]:
    """
    Loads the Single-Topic RAG Evaluation Dataset from Kaggle.
    """
    if not os.path.exists(file_path):
        print(f"Warning: Dataset file '{file_path}' not found.")
        
    df = pd.read_csv(file_path)
    
    # Adjust target column based on the specific Kaggle dataset structure.
    # Usually it's 'text', 'context', or 'document'.
    target_column = 'text' if 'text' in df.columns else df.columns[0]
    print(f"[Dataset] Loading data from column '{target_column}' in '{file_path}'...")
    
    documents = df[target_column].dropna().astype(str).tolist()
    print("DONE")
    return documents

def main():
    # First, we need to set up the components of the RAG system
    retriever = DocumentRetriever() 
    generator = OpenAIClient() 
    rag_pipeline = VanillaRAG(retriever, generator)
    
    # We then need to load the dataset, itself.
    document_path = "dataset/documents.csv"
    documents = load_kaggle_rag_dataset(document_path)

    # Load all question datasets directly with pandas
    multi_df = pd.read_csv("dataset/multi_passage_answer_questions.csv")
    multi_df["category"] = "multi_passage"

    single_df = pd.read_csv("dataset/single_passage_answer_questions.csv")
    single_df["category"] = "single_passage"

    no_answer_df = pd.read_csv("dataset/no_answer_questions.csv")
    no_answer_df["category"] = "no_answer"
    # Fill in the expected answer for questions with no answer in the text
    no_answer_df["answer"] = "I do not know."

    # Combine into a single evaluation DataFrame
    questions_df = pd.concat([multi_df, single_df, no_answer_df], ignore_index=True)

    # Extract questions and answers as lists for the evaluator
    questions = questions_df["question"].tolist()
    answers = questions_df["answer"].tolist()

    # # Add documents to the FAISS index (limiting to 500 for a fast demo)
    retriever.add_documents(documents)

    print("DATA LOADED")
    
    # # 3. Test the Pipeline
    # # Swap out for a query relevant to the Kaggle dataset
    # query = "What is the capital of France?" 
    
    try:
        response = rag_pipeline.answer(questions[0], top_k=3)
        print("\n--- Final Output ---")
        print(response)
    except Exception as e:
        print(f"\n[Error] Pipeline execution failed: {e}")
        print("\nMake sure you have set the OPENAI_API_KEY and installed dependencies:")
        print("pip install openai faiss-cpu sentence-transformers pandas")

if __name__ == "__main__":
    main()
