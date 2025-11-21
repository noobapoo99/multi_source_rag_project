import streamlit as st
import os
import json
import pickle
from dotenv import load_dotenv
from groq import Groq
import pdfplumber
import pandas as pd
from bs4 import BeautifulSoup
import requests
from sentence_transformers import SentenceTransformer, CrossEncoder
import numpy as np
import faiss

# ===============================
# 🔐 Load API Key
# ===============================
import streamlit as st
GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

client = Groq(api_key=GROQ_API_KEY)

# ===============================
# 🔧 Load Models (Embedding + Reranker)
# ===============================
embed_model = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


# ===============================
# 📥 Document Loaders
# ===============================

def load_pdf(path):
    pages = []
    with pdfplumber.open(path) as pdf:
        for p in pdf.pages:
            pages.append(p.extract_text() or "")
    return pages


def load_csv(path):
    df = pd.read_csv(path)
    return [str(r) for r in df.to_dict(orient="records")]


def load_url(url):
    r = requests.get(url, timeout=10)
    soup = BeautifulSoup(r.text, "html.parser")
    for tag in soup(["script", "style"]):
        tag.decompose()
    text = "\n".join([line.strip() for line in soup.get_text().splitlines() if line.strip()])
    return [text]


def ingest_file(path_or_url):
    if path_or_url.startswith("http"):
        return [{"source": path_or_url, "text": t} for t in load_url(path_or_url)]

    if path_or_url.endswith(".pdf"):
        return [{"source": path_or_url, "text": t} for t in load_pdf(path_or_url)]

    if path_or_url.endswith(".csv"):
        return [{"source": path_or_url, "text": t} for t in load_csv(path_or_url)]

    return []


# ===============================
# ✂️ Chunking
# ===============================
def chunk_text(text, size=500, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunk = text[start:end]
        chunks.append(chunk)
        start = end - overlap
    return chunks


def chunk_documents(docs):
    out = []
    for d in docs:
        pieces = chunk_text(d["text"])
        for i, c in enumerate(pieces):
            out.append({"source": d["source"], "chunk_id": i, "text": c})
    return out


# ===============================
# 🔢 Embedding + FAISS Index
# ===============================
def embed_chunks(chunks):
    texts = [c["text"] for c in chunks]
    emb = embed_model.encode(texts, convert_to_numpy=True)
    return emb


def build_faiss(emb):
    dim = emb.shape[1]
    index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(emb)
    index.add(emb)
    return index


def save_faiss(index, chunks, emb):
    store = {"index": index, "chunks": chunks, "emb": emb}
    with open("faiss_store.pkl", "wb") as f:
        pickle.dump(store, f)


def load_faiss_store():
    if os.path.exists("faiss_store.pkl"):
        with open("faiss_store.pkl", "rb") as f:
            return pickle.load(f)
    return None


def clear_index():
    if os.path.exists("faiss_store.pkl"):
        os.remove("faiss_store.pkl")


# ===============================
# 🔍 Retrieval + Reranking
# ===============================
def search_index(query, store, k=5):
    q_emb = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(q_emb)
    D, I = store["index"].search(q_emb, k)
    out = []
    for idx, score in zip(I[0], D[0]):
        c = store["chunks"][idx]
        out.append({
            "score": float(score),
            "source": c["source"],
            "chunk_id": c["chunk_id"],
            "text": c["text"]
        })
    return out


def rerank_results(query, results):
    pairs = [[query, r["text"]] for r in results]
    scores = reranker.predict(pairs)
    for r, s in zip(results, scores):
        r["rerank_score"] = float(s)
    return sorted(results, key=lambda x: x["rerank_score"], reverse=True)


# ===============================
# 🤖 Groq LLM
# ===============================
def build_prompt(query, chunks):
    ctx = ""
    for c in chunks:
        ctx += f"\n[Source: {c['source']} | Chunk {c['chunk_id']}]\n{c['text']}\n"

    return f"""
Use ONLY the following context to answer the question.

{ctx}

QUESTION: {query}

If answer is not in context, reply: "I don't know based on the provided documents."

Include a Sources list.
"""


def ask_groq(prompt):
    response = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=400,
        temperature=0.1
    )
    return response.choices[0].message.content


# ===============================
# 🌐 STREAMLIT UI
# ===============================

st.title("📘 Multi-Source RAG System (FAISS + Groq)")

st.sidebar.title("⚙️ Controls")
if st.sidebar.button("Clear Index"):
    clear_index()
    st.sidebar.success("Index cleared.")

st.header("📁 Upload Files")
files = st.file_uploader("Upload PDF/CSV Files", accept_multiple_files=True)

if st.button("📦 Ingest & Build Index"):
    if not files:
        st.error("Please upload files first.")
    else:
        all_chunks = []

        for f in files:
            save_path = os.path.join("data_raw", f.name)
            with open(save_path, "wb") as tmp:
                tmp.write(f.getbuffer())

            docs = ingest_file(save_path)
            chunks = chunk_documents(docs)
            all_chunks.extend(chunks)

        emb = embed_chunks(all_chunks)
        index = build_faiss(emb)
        save_faiss(index, all_chunks, emb)

        st.success("Index built successfully!")


st.header("🔎 Ask a Question")
query = st.text_input("Enter your question")

if st.button("💬 Get Answer"):
    store = load_faiss_store()
    if store is None:
        st.error("No FAISS index found. Please upload and build index.")
    else:
        retrieved = search_index(query, store, k=5)
        ranked = rerank_results(query, retrieved)
        top = ranked[:2]

        prompt = build_prompt(query, top)
        answer = ask_groq(prompt)

        st.subheader("🧠 Answer")
        st.write(answer)

        st.subheader("📚 Sources Used")
        for c in top:
            st.write(f"- **{c['source']}**, chunk {c['chunk_id']}")

        with st.expander("🔍 View Retrieved Chunks"):
            for c in top:
                st.markdown(f"**{c['source']} — Chunk {c['chunk_id']}**")
                st.text(c["text"])
