"""
A6 Web Application: Transformer Q&A Chatbot

Uses Contextual Retrieval with BAAI/bge-small-en-v1.5 embeddings and Groq for generation.

Run from the A6/ directory:
    streamlit run app/app.py
"""

import os
import re
import json
import asyncio
import numpy as np
import torch
import fitz  # PyMuPDF
import streamlit as st
from transformers import AutoTokenizer, AutoModel
from groq import Groq

# ---------------------------------------------------------------------------- #
# Page configuration
# ---------------------------------------------------------------------------- #
st.set_page_config(
    page_title="Transformer Q&A - Chapter 8",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------- #
# Custom CSS
# ---------------------------------------------------------------------------- #
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .hero-banner {
        background: linear-gradient(135deg, #6C63FF 0%, #3EC6E0 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        color: white;
        box-shadow: 0 8px 32px rgba(108,99,255,0.35);
    }
    .hero-banner h1 { font-size: 2rem; font-weight: 700; margin: 0; }
    .hero-banner p  { font-size: 1rem; opacity: 0.88; margin-top: 0.4rem; }

    .answer-card {
        background: rgba(255,255,255,0.06);
        border: 1px solid rgba(255,255,255,0.12);
        border-radius: 12px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        backdrop-filter: blur(12px);
    }
    .answer-card h3 { color: #7CFC00; font-size: 1.05rem; margin-bottom: 0.6rem; }
    .answer-card p  { color: #e8e8e8; line-height: 1.7; }

    .chunk-card {
        background: rgba(108,99,255,0.10);
        border-left: 3px solid #6C63FF;
        border-radius: 8px;
        padding: 1rem 1.2rem;
        margin-top: 0.6rem;
        font-size: 0.88rem;
        color: #d0d0d0;
        line-height: 1.65;
    }
    .chunk-meta {
        font-size: 0.78rem;
        color: #8b8bff;
        margin-bottom: 0.4rem;
        font-weight: 600;
    }

    .stTextInput > div > div > input {
        background: rgba(255,255,255,0.08) !important;
        border: 1px solid rgba(108,99,255,0.4) !important;
        border-radius: 10px !important;
        color: #fff !important;
        font-size: 1rem !important;
        padding: 0.7rem 1rem !important;
    }
    .stButton > button {
        background: linear-gradient(135deg, #6C63FF, #3EC6E0);
        color: white;
        border: none;
        border-radius: 10px;
        font-weight: 600;
        font-size: 1rem;
        padding: 0.55rem 2rem;
        transition: all 0.2s;
        width: 100%;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 20px rgba(108,99,255,0.4);
    }

    .badge {
        display: inline-block;
        background: rgba(108,99,255,0.25);
        border: 1px solid rgba(108,99,255,0.4);
        border-radius: 20px;
        padding: 0.2rem 0.75rem;
        font-size: 0.8rem;
        color: #a9a4ff;
        margin-right: 0.5rem;
    }

    .sidebar-section {
        background: rgba(255,255,255,0.05);
        border-radius: 10px;
        padding: 0.8rem 1rem;
        margin-bottom: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# ---------------------------------------------------------------------------- #
# Constants
# ---------------------------------------------------------------------------- #
PDF_PATH        = os.path.join(os.path.dirname(__file__), '..', 'transformer_jan26.pdf')
EMBED_MODEL     = 'BAAI/bge-small-en-v1.5'
GROQ_MODEL      = 'llama-3.3-70b-versatile'
GROQ_FAST_MODEL = 'llama-3.1-8b-instant'
TOP_N           = 5

# ---------------------------------------------------------------------------- #
# Text processing helpers
# ---------------------------------------------------------------------------- #
def clean_text(text: str) -> str:
    text = re.sub(r'-(\n)\s*', '', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n\s*\d+\s*\n', '\n', text)
    text = re.sub(r'[ \t]+', ' ', text)
    lines = [l.strip() for l in text.split('\n')]
    return '\n'.join(lines).strip()

def chunk_text(text: str, chunk_size: int = 500, overlap: int = 50) -> list:
    words = text.split()
    chunks, start = [], 0
    step = chunk_size - overlap
    while start < len(words):
        end   = min(start + chunk_size, len(words))
        chunk = ' '.join(words[start:end])
        if len(chunk.split()) > 50:
            chunks.append(chunk)
        start += step
    return chunks

def cosine_similarity(a, b) -> float:
    a, b = np.array(a), np.array(b)
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-10))

# ---------------------------------------------------------------------------- #
# Cached resources - embedding model and PDF chunks
# ---------------------------------------------------------------------------- #
@st.cache_resource(show_spinner="Loading embedding model...")
def load_embed_model():
    tok = AutoTokenizer.from_pretrained(EMBED_MODEL)
    mdl = AutoModel.from_pretrained(EMBED_MODEL)
    mdl.eval()
    return tok, mdl

@st.cache_resource(show_spinner="Extracting and chunking PDF...")
def load_and_chunk_pdf():
    doc = fitz.open(PDF_PATH)
    raw = '\n'.join(page.get_text('text') for page in doc)
    doc.close()
    return chunk_text(clean_text(raw))

def get_embedding(text: str, tokenizer, model) -> list:
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True, max_length=512)
    with torch.no_grad():
        outputs = model(**inputs)
    return outputs.last_hidden_state[:, 0, :].squeeze().tolist()

# ---------------------------------------------------------------------------- #
# Chunk enrichment and vector database
# ---------------------------------------------------------------------------- #
DOC_SUMMARY = """This document is Chapter 8 of an NLP textbook covering the Transformer architecture.
Topics include: self-attention, multi-head attention, positional encoding, encoder-decoder structure,
masked self-attention, cross-attention, feed-forward layers, residual connections, layer normalization,
the attention formula, beam search, and machine translation applications."""

def enrich_chunk_sync(chunk: str, client: Groq, doc_summary: str) -> str:
    """
    Ask the LLM to generate a 1-2 sentence context description for the given chunk,
    then prepend it. Falls back to the original chunk if the API call fails.
    """
    prompt = f"""<document_summary>
{doc_summary}
</document_summary>

<chunk>
{chunk}
</chunk>

In 1-2 sentences, describe where this chunk fits in the document and what concept(s) it covers.
Reply with only the contextual sentences, nothing else."""
    resp = client.chat.completions.create(
        model=GROQ_FAST_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.0,
        max_tokens=128
    )
    return f"{resp.choices[0].message.content.strip()}\n\n{chunk}"

@st.cache_resource(show_spinner="Building contextual vector database. This only runs once.")
def build_vector_db(api_key: str):
    """
    Build the contextual retrieval vector database.
    Each chunk is enriched with an LLM-generated context prefix, then embedded.
    The result is cached so this only runs on the first request.
    """
    client           = Groq(api_key=api_key)
    tokenizer, model = load_embed_model()
    chunks           = load_and_chunk_pdf()
    vector_db        = []
    prog             = st.progress(0, text="Enriching and embedding chunks...")
    for i, chunk in enumerate(chunks):
        try:
            enriched = enrich_chunk_sync(chunk, client, DOC_SUMMARY)
        except Exception:
            # If enrichment fails for any reason, fall back to the raw chunk
            enriched = chunk
        emb = get_embedding(enriched, tokenizer, model)
        vector_db.append((enriched, emb, chunk))
        prog.progress((i + 1) / len(chunks), text=f"Embedding chunk {i+1} of {len(chunks)}...")
    prog.empty()
    return vector_db

def retrieve(query: str, vector_db: list, top_n: int = 5) -> list:
    """Return the top-n chunks by cosine similarity to the query embedding."""
    tokenizer, model = load_embed_model()
    q_emb = get_embedding(query, tokenizer, model)
    scored = [
        (enriched, orig, cosine_similarity(q_emb, emb))
        for enriched, emb, orig in vector_db
    ]
    scored.sort(key=lambda x: x[2], reverse=True)
    return scored[:top_n]

def generate_answer(question: str, hits: list, client: Groq) -> str:
    """Generate an answer from the top retrieved chunks using the Groq LLM."""
    context = '\n\n'.join(
        [f'[Source {i+1} | similarity={sim:.3f}]:\n{orig}'
         for i, (_, orig, sim) in enumerate(hits)]
    )
    prompt = f"""You are a helpful NLP teaching assistant specializing in Transformers.
Use only the provided context to answer the question. Be clear and concise.

Context:
{context}

Question: {question}

Answer:"""
    resp = client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[{'role': 'user', 'content': prompt}],
        temperature=0.15,
        max_tokens=600
    )
    return resp.choices[0].message.content.strip()

# ---------------------------------------------------------------------------- #
# Sidebar
# ---------------------------------------------------------------------------- #
with st.sidebar:
    st.markdown("## Configuration")
    api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        value=os.environ.get('GROQ_API_KEY', ''),
    )

    st.markdown("---")
    st.markdown("""
    <div class='sidebar-section'>
    <b>Chapter 8 - Transformers</b><br/>
    <small>transformer_jan26.pdf</small>
    </div>
    """, unsafe_allow_html=True)

    st.markdown("""
    <div class='sidebar-section'>
    <b>Method:</b> Contextual Retrieval<br/>
    <b>Embedder:</b> BAAI/bge-small-en-v1.5<br/>
    <b>Generator:</b> Groq llama-3.3-70b-versatile<br/>
    <b>Top-K:</b> 5 chunks
    </div>
    """, unsafe_allow_html=True)

    st.markdown("---")
    st.markdown("**Example questions:**")
    example_qs = [
        "What is self-attention?",
        "How does multi-head attention work?",
        "What are positional encodings?",
        "Explain the encoder-decoder structure.",
        "What is masked self-attention?",
    ]
    for eq in example_qs:
        if st.button(eq, key=f"eg_{eq[:20]}", use_container_width=True):
            # Write directly into the text input's session state key so the
            # box updates on the next rerun without any intermediate variable.
            st.session_state["question_input"] = eq

# ---------------------------------------------------------------------------- #
# Main content
# ---------------------------------------------------------------------------- #
st.markdown("""
<div class="hero-banner">
  <h1>Transformer Q&A</h1>
  <p>Domain-specific QA grounded in <b>Chapter 8: Transformers</b> using Contextual Retrieval and Groq LLaMA-3.3-70b.</p>
</div>
""", unsafe_allow_html=True)

col1, col2, col3 = st.columns([1, 1, 1])
col1.markdown('<span class="badge">Chapter 8</span><span class="badge">Student: 125998</span>', unsafe_allow_html=True)
col2.markdown('<span class="badge">Contextual Retrieval</span>', unsafe_allow_html=True)
col3.markdown('<span class="badge">Groq LLaMA-3.3-70b</span>', unsafe_allow_html=True)

st.markdown("---")

question = st.text_input(
    "Ask a question about Transformers",
    placeholder="e.g. How does scaled dot-product attention work?",
    key="question_input"
)

ask_btn = st.button("Get Answer", type="primary")

# ---------------------------------------------------------------------------- #
# Answer generation
# ---------------------------------------------------------------------------- #
if ask_btn and question.strip():
    if not api_key:
        st.error("Please enter your Groq API key in the sidebar.")
    else:
        client = Groq(api_key=api_key)
        try:
            with st.spinner("Building knowledge base. This may take a moment on first run."):
                vector_db = build_vector_db(api_key)

            with st.spinner("Retrieving relevant passages..."):
                hits = retrieve(question, vector_db, top_n=TOP_N)

            with st.spinner("Generating answer..."):
                answer = generate_answer(question, hits, client)

            st.markdown(f"""
            <div class="answer-card">
                <h3>Answer</h3>
                <p>{answer}</p>
            </div>
            """, unsafe_allow_html=True)

            st.markdown("#### Source Chunks")
            for i, (enriched, orig, sim) in enumerate(hits):
                with st.expander(f"Source {i+1}  |  Similarity: {sim:.4f}", expanded=(i == 0)):
                    st.markdown(f'<div class="chunk-meta">Chunk {i+1} | Cosine similarity: {sim:.4f}</div>', unsafe_allow_html=True)
                    st.markdown(f'<div class="chunk-card">{orig[:600]}{"..." if len(orig) > 600 else ""}</div>', unsafe_allow_html=True)

        except Exception as e:
            st.error(f"Something went wrong: {e}")
            st.info("Check that your Groq API key is valid and that transformer_jan26.pdf is in the A6/ directory.")

elif ask_btn:
    st.warning("Please enter a question before submitting.")

# ---------------------------------------------------------------------------- #
# Footer
# ---------------------------------------------------------------------------- #
st.markdown("---")
st.markdown("""
<div style='text-align:center; color:#666; font-size:0.82rem; margin-top:1rem;'>
    AT82.03 Machine Learning &nbsp;|&nbsp; A6: Naive RAG vs Contextual Retrieval &nbsp;|&nbsp; Student ID: 125998
</div>
""", unsafe_allow_html=True)
