import os
import io
import numpy as np
import faiss
import fitz  # PyMuPDF
import streamlit as st
import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from transformers import pipeline

# ------------------ Setup ------------------
load_dotenv()
LOCAL_MODEL_NAME = os.getenv("LOCAL_MODEL_NAME", "google/flan-t5-large")
APP_USERNAME = os.getenv("APP_USERNAME", "admin")  # set in .env
APP_PASSWORD = os.getenv("APP_PASSWORD", "password")  # set in .env

st.set_page_config(page_title="StudyMate-LearnHub", page_icon="📘", layout="wide")

# ------------------ Authentication ------------------
if "authenticated" not in st.session_state:
    st.session_state.authenticated = False

def login():
    st.title("🔐 Login to StudyMate-LearnHub")
    username = st.text_input("Username")
    password = st.text_input("Password", type="password")
    if st.button("Login"):
        if username == APP_USERNAME and password == APP_PASSWORD:
            st.session_state.authenticated = True
            st.success("✅ Login successful!")
            st.rerun()
        else:
            st.error("❌ Invalid username or password")

if not st.session_state.authenticated:
    login()
    st.stop()

# ------------------ Theme / Styles ------------------
st.markdown(
    """
    <style>
    .stApp {
        background: linear-gradient(180deg, #e3f2fd, #ffffff);
        font-family: "Segoe UI", system-ui, -apple-system, Roboto, Arial, sans-serif;
    }
    h1, h2, h3 { color:#0d47a1; text-align:center; }

    .hero {
        background: #bbdefb;
        padding: 48px 20px;
        border-radius: 20px;
        text-align: center;
        box-shadow: 0 4px 12px rgba(0,0,0,0.12);
        position: relative;
        overflow: hidden;
    }
    .hero::before, .hero::after {
        content: "";
        position: absolute;
        border-radius: 50%;
        background: rgba(255,255,255,0.45);
        filter: blur(6px);
    }
    .hero::before { width:180px; height:180px; top:-40px; left:-40px; }
    .hero::after  { width:220px; height:220px; bottom:-60px; right:-60px; }

    .hero-icons img { width: 96px; margin: 6px 10px; opacity: 0.95; }

    .section { padding: 40px 10px; margin: 14px 0; }
    .cards { display:flex; justify-content:center; gap:18px; flex-wrap:wrap; }

    .card {
        background: #ffffff;
        width: 260px;
        border-radius: 16px;
        padding: 18px;
        text-align: center;
        box-shadow: 0 4px 10px rgba(0,0,0,0.08);
        border: 1px solid #e3f2fd;
    }
    .card img { width: 68px; margin-bottom: 12px; }
    .btn {
        display:inline-block;
        background: linear-gradient(90deg, #42a5f5, #1e88e5);
        color:#fff; padding:10px 16px; border-radius:10px; text-decoration:none; font-weight:600;
    }

    .answer-box {
        background:#fff; padding:20px; border-radius:14px;
        border-left:6px solid #1e88e5; box-shadow:0 4px 12px rgba(0,0,0,0.1);
    }
    .context-box {
        background:#e3f2fd; padding:12px; border-radius:10px;
        border-left:4px solid #42a5f5; margin-bottom:8px; font-size:0.95em;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ------------------ Dark / Light Mode Toggle ------------------
with st.sidebar:
    st.markdown("### 🌙☀ Theme")
    dark_mode = st.toggle("Enable Dark Mode", value=False)

if dark_mode:
    st.markdown(
        """
        <style>
        .stApp {
            background: linear-gradient(180deg, #1c1c1c, #2c2c2c);
            color: #f5f5f5;
        }
        h1, h2, h3, h4, h5, h6, p, label, .stMarkdown, .stTextInput label {
            color: #f5f5f5 !important;
        }
        .card {
            background: #2e2e2e !important;
            color: #f5f5f5 !important;
            border: 1px solid #444 !important;
        }
        .answer-box {
            background:#333 !important;
            color:#fff !important;
            border-left:6px solid #42a5f5;
        }
        .context-box {
            background:#444 !important;
            color:#eee !important;
            border-left:4px solid #42a5f5;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

# ------------------ Models ------------------
@st.cache_resource(show_spinner=False)
def get_embedder():
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

@st.cache_resource(show_spinner=True)
def get_generator(model_name: str):
    if "t5" in model_name.lower():
        task = "text2text-generation"
    else:
        task = "text-generation"
    device = 0 if torch.cuda.is_available() else -1
    gen = pipeline(task, model=model_name, device=device)
    return task, gen

embedder = get_embedder()
task, generator = get_generator(LOCAL_MODEL_NAME)

# ------------------ Session State ------------------
if "documents" not in st.session_state: st.session_state.documents = []
if "index" not in st.session_state:     st.session_state.index = None
if "chat_history" not in st.session_state: st.session_state.chat_history = []

# ------------------ Sidebar: Chat History ------------------
with st.sidebar:
    st.subheader("💾 Recent Questions")
    if st.session_state.chat_history:
        for i, qa in enumerate(st.session_state.chat_history[-5:][::-1], start=1):
            st.markdown(f"**Q{i}:** {qa['question']}")
            if "simplified" in qa:
                st.caption(f"Simplified: {qa['simplified'][:80]}...")
            else:
                st.caption(f"A: {qa['answer'][:80]}...")
    else:
        st.caption("No questions yet.")

# ------------------ RAG Helpers ------------------
def extract_text_from_pdf(file_like, filename) -> list:
    if isinstance(file_like, (bytes, bytearray)):
        file_like = io.BytesIO(file_like)
    doc = fitz.open(stream=file_like.read(), filetype="pdf")
    chunks = []
    for page_num, page in enumerate(doc, start=1):
        txt = page.get_text("text") or ""
        words = txt.split()
        step = 250
        for i in range(0, len(words), step):
            chunk_text = " ".join(words[i:i + step])
            if chunk_text.strip():
                chunks.append({
                    "text": chunk_text,
                    "file": filename,
                    "page": page_num
                })
    return chunks

def build_faiss_index(chunks):
    emb = embedder.encode([c["text"] for c in chunks], convert_to_numpy=True, show_progress_bar=False)
    index = faiss.IndexFlatL2(emb.shape[1])
    index.add(emb.astype(np.float32))
    return index

def retrieve_top_k(question: str, k: int = 3):
    q_emb = embedder.encode([question], convert_to_numpy=True)
    D, I = st.session_state.index.search(q_emb.astype(np.float32), k)
    return [st.session_state.documents[i] for i in I[0]]

# ------------------ Dual Answer Generator ------------------
def generate_dual_answers(context_chunks, question):
    context = "\n".join(c["text"] for c in context_chunks)

    simplified_prompt = (
        "You are a friendly tutor. Using ONLY the given context, explain the answer "
        "in very simple language, like you are teaching a beginner. Avoid jargon.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nSimplified Answer:"
    )

    detailed_prompt = (
        "You are an academic assistant. Using ONLY the given context, provide a "
        "comprehensive, detailed explanation with clarity and depth. "
        "Include technical terms if needed.\n\n"
        f"Context:\n{context}\n\nQuestion: {question}\n\nDetailed Answer:"
    )

    if task == "text2text-generation":
        simplified_out = generator(simplified_prompt, max_new_tokens=200, temperature=0.5)
        detailed_out = generator(detailed_prompt, max_new_tokens=400, temperature=0.5)
        simplified = simplified_out[0]["generated_text"].strip()
        detailed = detailed_out[0]["generated_text"].strip()
    else:
        simplified_out = generator(simplified_prompt, max_new_tokens=250, temperature=0.5, do_sample=True)
        detailed_out = generator(detailed_prompt, max_new_tokens=500, temperature=0.5, do_sample=True)
        simplified = simplified_out[0]["generated_text"].split("Simplified Answer:")[-1].strip()
        detailed = detailed_out[0]["generated_text"].split("Detailed Answer:")[-1].strip()

    return simplified, detailed

# ------------------ Streamlit UI ------------------
# Hero
st.markdown(
    """
    <div class="hero">
        <h1>📘 StudyMate-LearnHub</h1>
        <h3>Learn from your own notes with AI</h3>
        <p>Upload PDFs · Ask questions · Get grounded answers with references</p>
        <div class="hero-icons">
            <img src="https://cdn-icons-png.flaticon.com/512/3135/3135755.png">
            <img src="https://cdn-icons-png.flaticon.com/512/29/29302.png">
            <img src="https://cdn-icons-png.flaticon.com/512/1161/1161388.png">
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# For Students
st.markdown(
    """
    <div class="section">
        <h2>For Students 🎓</h2>
        <div class="cards">
            <div class="card">
                <img src="https://cdn-icons-png.flaticon.com/512/3135/3135810.png" />
                <h4>Smart Search🔍</h4>
                <p>Ask anything about your PDFs and get citations from the source.</p>
            </div>
            <div class="card">
                <img src="https://cdn-icons-png.flaticon.com/512/2910/2910768.png" />
                <h4>Quick Revision</h4>
                <p>Chunk and index large notes for fast, focused answers.</p>
            </div>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# Testimonials
st.markdown(
    """
<div class="section" style="text-align:center;">
    <h5>Testimonials💬</h5>
    <div class="card" style="max-width:600px; margin:0 auto; text-align:center;">
        “StudyMate turned my textbooks into a conversational tutor. The referenced context keeps answers trustworthy.”
    </div>
</div>
    """,
    unsafe_allow_html=True
)

st.markdown("---")

# Upload & Process
st.header("📂 Upload & Process Your PDFs")
uploaded_files = st.file_uploader("Upload one or more PDFs", type=["pdf"], accept_multiple_files=True)

if st.button("Process PDFs") and uploaded_files:
    all_chunks = []
    with st.spinner("Extracting and chunking..."):
        for f in uploaded_files:
            raw = f.read()
            chunks = extract_text_from_pdf(io.BytesIO(raw), f.name)
            all_chunks.extend(chunks)
    if not all_chunks:
        st.warning("No selectable text found in the PDFs (they may be scanned images).")
    else:
        with st.spinner("Embedding and building FAISS index..."):
            st.session_state.documents = all_chunks
            st.session_state.index = build_faiss_index(all_chunks)
        st.success(f"✅ Indexed {len(all_chunks)} chunks.")

st.markdown("---")

# ------------------ Summarize PDFs ------------------
st.header("📑 Summarize Your PDFs")
if st.session_state.index is None:
    st.warning("Please upload and process PDFs above first.")
else:
    if st.button("Summarize PDF"):
        with st.spinner("Generating summary..."):
            full_text = " ".join([c["text"] for c in st.session_state.documents])
            sys_prompt = (
                "You are StudyMate, an academic assistant. "
                "Summarize the following document into clear, structured bullet points. "
                "Highlight the main ideas, key facts, and important concepts."
            )
            prompt = f"{sys_prompt}\n\nDocument:\n{full_text}\n\nSummary:"

            if task == "text2text-generation":
                out = generator(prompt, max_new_tokens=400, temperature=0.5)
                summary = out[0]["generated_text"].strip()
            else:
                out = generator(prompt, max_new_tokens=500, temperature=0.5, do_sample=True)
                text = out[0]["generated_text"]
                summary = text.split("Summary:")[-1].strip()

        st.subheader("📝 Summary")
        st.markdown(f"<div class='answer-box'>{summary}</div>", unsafe_allow_html=True)

st.markdown("---")

# ------------------ Ask Questions ------------------
st.header("⚡ Ask Questions from Your Notes")
if st.session_state.index is None:
    st.warning("Please upload and process PDFs above first.")
else:
    question = st.text_input("Ask a question")
    if question:
        with st.spinner("Retrieving and generating answers..."):
            top_chunks = retrieve_top_k(question, k=3)
            simplified, detailed = generate_dual_answers(top_chunks, question)

        st.subheader("🟢 Simplified Answer")
        st.markdown(f"<div class='answer-box'>{simplified}</div>", unsafe_allow_html=True)

        st.subheader("🔵 Detailed Answer")
        st.markdown(f"<div class='answer-box'>{detailed}</div>", unsafe_allow_html=True)

        st.subheader("🔎 Referenced Context")
        for i, c in enumerate(top_chunks, start=1):
            st.markdown(
                f"<div class='context-box'>📌 <b>Source {i}</b> ({c['file']}, Page {c['page']})<br>{c['text']}</div>",
                unsafe_allow_html=True
            )

        st.session_state.chat_history.append({
            "question": question,
            "simplified": simplified,
            "detailed": detailed,
            "sources": top_chunks
        })

