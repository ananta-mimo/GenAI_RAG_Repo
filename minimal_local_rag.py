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
import re

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

## Split text into overlapping chunks of ~chunk_size characters
def chunk_text(text, chunk_size = 500, overlap = 50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start += chunk_size - overlap # overlap keeps context between chunks or else we can lose context or meaning
    return chunks

data_folder = "data"
documents = []

for filename in os.listdir(data_folder):
    if filename.endswith(".pdf"):
        filepath = os.path.join(data_folder, filename)
        raw_text = load_pdf_text(filepath)
        doc_chunks = chunk_text(raw_text)
        documents.extend(doc_chunks)

print(f"Loaded {len(documents)} document chunks from PDFs in '{data_folder}' folder.")
     
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
# 7. Simple faithfulness check (word-overlap based)
# ---------------------------------------------------------------------------



# def simple_stem(word):
#     """
#     A very simple stemming function that removes common suffixes.
#     This is not a full-fledged stemmer, but it helps with basic word matching.
#     """
#     suffixes = ["ing", "ed", "s", "es", "ly", "tion", "ment", "ness", "able", "ible", "al", "er", "or", "ist", "ity", "ous", "ive", "ize", "ise"]
    
#     changed = True
#     while changed:
#         changed = False
#         for suffix in suffixes:
#             if word.endswith(suffix) and len(word) > len(suffix) + 2:
#                 word = word[:-len(suffix)]
#                 changed = True
#                 break

#     return word

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def simple_stem(word):
    return stemmer.stem(word)

STOPWORDS = {
    "the", "is", "a", "an", "of", "to", "and", "in", "on", "for",
    "this", "that", "it", "as", "by", "with", "are", "be", "or",
    "from", "these", "its", "was", "were", "at", "into", "which"
}
def clean_words(text):
    words = re.findall(r"[a-zA-Z]+", text.lower())
    return {simple_stem(word) for word in words if word not in STOPWORDS and len(word) > 2}

def check_faithfulness(answer, retrieved_chunks):
    # Count how many words in the answer are present in the retrieved chunks
    
    """
    Rough faithfulness score: what fraction of meaningful words in the
    generated answer also appear somewhere in the retrieved context.
    This is a simple heuristic, not a rigorous metric, but it's useful
    for flagging answers that may contain unsupported content.
    """

    
    context_text = " ".join(retrieved_chunks)
    context_words = clean_words(context_text, STOPWORDS)
    answer_words = clean_words(answer, STOPWORDS)

    
    if not answer_words:
        return 0.0  # Avoid division by zero if answer has no meaningful words
    
    grounded_words = answer_words & context_words
    faithfulness_score = len(grounded_words) / len(answer_words)
    return round(faithfulness_score, 2)

def debug_word_overlap(answer, retrieved_chunks):

    context_words = clean_words(" ".join(retrieved_chunks), STOPWORDS)
    answer_words = clean_words(answer, STOPWORDS)
    unsupported = answer_words - context_words

    print("Words in answer NOT found in retrieved context:")
    print(sorted(unsupported))
    


# ---------------------------------------------------------------------------
# 6. Run it
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    query = "How is SHAP used to explain credit risk model predictions?"
    chunks = retrieve(query)
    print("Retrieved context:")
    for c in chunks:
        print(" -", c)

    answer = generate_answer(query, chunks)
    print("\nGenerated answer:")
    print(answer)
    
    score = check_faithfulness(answer, chunks)
    print(f"\nFaithfulness score: {score} (fraction of answer's key words found in retrieved context)")
    
    debug_word_overlap(answer, chunks)  