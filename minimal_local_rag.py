"""
Minimal Local RAG Pipeline
---------------------------
Retrieval-Augmented Generation running fully offline on a laptop.

Setup (run once in your terminal):
    pip install sentence-transformers faiss-cpu ollama

    # Install Ollama separately from https://ollama.com, then pull a small model:
    ollama pull phi3

Usage:
    python minimal_local_rag.py
"""

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer
import ollama

# ---------------------------------------------------------------------------
# 1. Sample documents (replace with your own PDFs, notes, or dataset later)
# ---------------------------------------------------------------------------
# documents = [
#     "XGBoost is a gradient boosting framework that uses decision trees as base learners.",
#     "SHAP values explain individual predictions by attributing contribution to each feature.",
#     "A confusion matrix summarizes classification performance across true and predicted labels.",
#     "Uncertainty quantification methods include conformal prediction and Bayesian approaches.",
#     "Streamlit is a Python framework for building interactive data apps quickly.",
# ]

# ---------------------------------------------------------------------------
# 1. Load PDFs and split into chunks
# ---------------------------------------------------------------------------

import os
from pypdf import PdfReader

def load_pdf_text(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# ---------------------------------------------------------------------------
# 2. Embed documents locally (no API calls, runs on CPU)
# ---------------------------------------------------------------------------
embedder = SentenceTransformer("all-MiniLM-L6-v2")
doc_embeddings = embedder.encode(documents, convert_to_numpy=True)

# ---------------------------------------------------------------------------
# 3. Build a local FAISS index (Vector Database) for fast similarity search
# ---------------------------------------------------------------------------
dimension = doc_embeddings.shape[1]
index = faiss.IndexFlatL2(dimension)
index.add(doc_embeddings)

# ---------------------------------------------------------------------------
# 4. Retrieve top-k relevant chunks for a query
# ---------------------------------------------------------------------------
def retrieve(query, k=2):
    query_embedding = embedder.encode([query], convert_to_numpy=True)
    distances, indices = index.search(query_embedding, k)
    return [documents[i] for i in indices[0]]

# ---------------------------------------------------------------------------
# 5. Generate an answer using a local LLM via Ollama
# ---------------------------------------------------------------------------
def generate_answer(query, retrieved_chunks):
    context = "\n".join(retrieved_chunks)
    prompt = f"""Answer the question using only the context below.

Context:
{context}

Question: {query}
Answer:"""

    response = ollama.chat(
        model="phi3",
        messages=[{"role": "user", "content": prompt}],
    )
    return response["message"]["content"]

# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "What does SHAP do?"
    chunks = retrieve(query)
    print("Retrieved context:")
    for c in chunks:
        print(" -", c)

    answer = generate_answer(query, chunks)
    print("\nGenerated answer:")
    print(answer)
