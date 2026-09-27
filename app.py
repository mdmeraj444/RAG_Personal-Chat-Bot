"""
app.py — Meraj Personal Docs Chatbot
Premium Streamlit RAG Dashboard
"""

import os
import time
from pathlib import Path

from dotenv import load_dotenv
import streamlit as st
from langchain_nvidia_ai_endpoints import NVIDIAEmbeddings
from langchain_pinecone import PineconeVectorStore
from langchain_groq import ChatGroq

# ── API Keys (loaded from the .env file beside app.py) ─────────────────────────
ENV_PATH = Path(__file__).resolve().parent / ".env"
# Values are loaded into os.environ for NVIDIA, Pinecone, and Groq.
# Existing environment variables take precedence over values in .env.
load_dotenv(dotenv_path=ENV_PATH, encoding="utf-8-sig")

INDEX_NAME = "agentic-ai-rag"

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Personal Docs Chatbot — Meraj Local Chat Bot",
    page_icon="✦",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Validate API Keys ─────────────────────────────────────────────────────────
required_keys = ("NVIDIA_API_KEY", "PINECONE_API_KEY", "GROQ_API_KEY")
missing_keys = [key for key in required_keys if not os.getenv(key, "").strip()]

if missing_keys:
    st.error("Missing API keys: " + ", ".join(missing_keys))
    st.info(f"Add these keys to {ENV_PATH}, then restart Streamlit.")
    st.stop()

# ── Session State Init ────────────────────────────────────────────────────────
if "theme" not in st.session_state:
    st.session_state.theme = "dark"
if "messages" not in st.session_state:
    st.session_state.messages = []

# ── Theme Colors ──────────────────────────────────────────────────────────────
if st.session_state.theme == "dark":
    BG             = "#080a12"
    CARD_BG        = "#0d1117"
    CARD_BG2       = "#111827"
    BORDER         = "#1e2a3a"
    TEXT_PRIMARY   = "#e8eaf0"
    TEXT_SECONDARY = "#8892a4"
    ACCENT         = "#00d4b4"
    ACCENT2        = "#7c5cfc"
    ACCENT_DIM     = "rgba(0,212,180,0.12)"
    ACCENT2_DIM    = "rgba(124,92,252,0.12)"
    INPUT_BG       = "#0d1117"
    TAG_BG         = "#1a2235"
    SHADOW         = "rgba(0,212,180,0.15)"
    SHADOW2        = "rgba(124,92,252,0.15)"
    USER_MSG_BG    = "#111827"
    BOT_MSG_BG     = "#0d1117"
    SIDEBAR_BG     = "#070910"
    BTN_BG         = "#111827"
    BTN_BORDER     = "#1e2a3a"
    BTN_HOVER_BG   = "#1a2235"
else:
    BG             = "#f5f7fa"
    CARD_BG        = "#ffffff"
    CARD_BG2       = "#f0f4f8"
    BORDER         = "#dde3ec"
    TEXT_PRIMARY   = "#111827"
    TEXT_SECONDARY = "#5a6579"
    ACCENT         = "#00a890"
    ACCENT2        = "#6d4de8"
    ACCENT_DIM     = "rgba(0,168,144,0.10)"
    ACCENT2_DIM    = "rgba(109,77,232,0.10)"
    INPUT_BG       = "#ffffff"
    TAG_BG         = "#eef2ff"
    SHADOW         = "rgba(0,168,144,0.18)"
    SHADOW2        = "rgba(109,77,232,0.18)"
    USER_MSG_BG    = "#eef2ff"
    BOT_MSG_BG     = "#ffffff"
    SIDEBAR_BG     = "#eef2f7"
    BTN_BG         = "#ffffff"
    BTN_BORDER     = "#dde3ec"
    BTN_HOVER_BG   = "#f0f4f8"

# Chat text uses high contrast for the selected theme.
CHAT_TEXT = "#ffffff" if st.session_state.theme == "dark" else TEXT_PRIMARY

# ── CSS Injection ─────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

*, *::before, *::after {{ box-sizing: border-box; }}

html, body, .stApp {{
    font-family: 'Inter', sans-serif;
    background-color: {BG};
    color: {TEXT_PRIMARY};
}}

# Keep the header visible: Streamlit places the sidebar toggle there.
#MainMenu, footer {{ visibility: hidden; }}

/* ── Ambient glow orb ── */
.stApp::before {{
    content: '';
    position: fixed;
    top: -200px;
    right: -200px;
    width: 600px;
    height: 600px;
    background: radial-gradient(circle, {ACCENT2}22 0%, transparent 70%);
    pointer-events: none;
    z-index: 0;
    border-radius: 50%;
}}

/* ── Sidebar ── */
[data-testid="stSidebar"] > div:first-child {{
    background: {SIDEBAR_BG};
    border-right: 1px solid {BORDER};
    padding: 1.5rem 1rem;
}}
[data-testid="stSidebar"] {{
    display: block !important;
    visibility: visible !important;
}}
[data-testid="stSidebar"] * {{ color: {TEXT_PRIMARY}; }}

/* ── Main content padding ── */
.block-container {{
    padding: 1.5rem 2rem 2rem 2rem;
    max-width: 1100px;
}}

/* ── Header card ── */
.header-card {{
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 16px;
    padding: 2rem 2.2rem 1.8rem 2.2rem;
    margin-bottom: 1.4rem;
    position: relative;
    overflow: hidden;
}}
.header-card::before {{
    content: '';
    position: absolute;
    top: -80px; left: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, {ACCENT}22 0%, transparent 70%);
    pointer-events: none;
    border-radius: 50%;
}}
.header-card::after {{
    content: '';
    position: absolute;
    bottom: -80px; right: -80px;
    width: 260px; height: 260px;
    background: radial-gradient(circle, {ACCENT2}22 0%, transparent 70%);
    pointer-events: none;
    border-radius: 50%;
}}
.eyebrow {{
    font-size: 0.7rem;
    font-weight: 600;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: {ACCENT};
    margin-bottom: 0.5rem;
}}
.header-title {{
    font-size: 2rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT} 0%, {ACCENT2} 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
    line-height: 1.2;
    margin-bottom: 0.5rem;
}}
.header-sub {{
    font-size: 0.9rem;
    color: {TEXT_SECONDARY};
    margin-bottom: 1rem;
    max-width: 520px;
}}
.status-badge {{
    display: inline-flex;
    align-items: center;
    gap: 0.45rem;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 50px;
    padding: 0.3rem 0.75rem;
    font-size: 0.75rem;
    font-weight: 600;
    color: #22c55e;
}}
.pulse-dot {{
    width: 7px; height: 7px;
    background: #22c55e;
    border-radius: 50%;
    animation: pulse 1.8s infinite;
}}
@keyframes pulse {{
    0%, 100% {{ opacity: 1; transform: scale(1); }}
    50% {{ opacity: 0.4; transform: scale(1.4); }}
}}

/* ── Stats row ── */
.stats-row {{
    display: flex;
    gap: 1rem;
    margin-bottom: 1.4rem;
    flex-wrap: wrap;
}}
.stat-card {{
    flex: 1;
    min-width: 130px;
    background: {CARD_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 1rem 1.2rem 0.9rem 1.2rem;
    position: relative;
    transition: transform 0.2s, box-shadow 0.2s;
    overflow: hidden;
    cursor: default;
}}
.stat-card::before {{
    content: '';
    position: absolute;
    top: 0; left: 0; right: 0;
    height: 2px;
    background: linear-gradient(90deg, {ACCENT}, {ACCENT2});
    border-radius: 2px 2px 0 0;
}}
.stat-card:hover {{
    transform: translateY(-2px);
    box-shadow: 0 6px 24px {SHADOW};
}}
.stat-value {{
    font-size: 1.6rem;
    font-weight: 700;
    color: {TEXT_PRIMARY};
    line-height: 1;
}}
.stat-label {{
    font-size: 0.72rem;
    color: {ACCENT};
    font-weight: 600;
    margin-top: 0.15rem;
    letter-spacing: 0.04em;
}}
.stat-sub {{
    font-size: 0.68rem;
    color: {TEXT_SECONDARY};
    margin-top: 0.4rem;
}}

/* ── Sample question buttons ── */
.sample-label {{
    font-size: 0.75rem;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    letter-spacing: 0.06em;
    margin-bottom: 0.6rem;
}}

/* ── Chat messages ── */
.chat-user-msg {{
    background: {USER_MSG_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: {TEXT_PRIMARY};
}}
.chat-bot-msg {{
    background: {BOT_MSG_BG};
    border: 1px solid {BORDER};
    border-radius: 12px;
    padding: 0.9rem 1.1rem;
    margin: 0.4rem 0;
    font-size: 0.9rem;
    color: {TEXT_PRIMARY};
    position: relative;
}}
.chat-bot-msg::before {{
    content: '';
    position: absolute;
    left: 0; top: 12px; bottom: 12px;
    width: 2px;
    background: linear-gradient(180deg, {ACCENT}, {ACCENT2});
    border-radius: 2px;
}}

/* ── Source expander ── */
.source-item {{
    display: flex;
    align-items: center;
    gap: 0.5rem;
    padding: 0.45rem 0.7rem;
    background: {CARD_BG2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin: 0.3rem 0;
    font-size: 0.78rem;
    color: {TEXT_SECONDARY};
}}
.source-icon {{
    color: {ACCENT};
    font-size: 0.85rem;
}}

/* ── Sidebar elements ── */
.brand-box {{
    display: flex;
    align-items: center;
    gap: 0.75rem;
    margin-bottom: 1.5rem;
    padding-bottom: 1.2rem;
    border-bottom: 1px solid {BORDER};
}}
.brand-icon {{
    width: 36px; height: 36px;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    border-radius: 8px;
    display: flex;
    align-items: center;
    justify-content: center;
    font-size: 1rem;
    flex-shrink: 0;
}}
.brand-name {{
    font-size: 0.95rem;
    font-weight: 700;
    background: linear-gradient(135deg, {ACCENT}, {ACCENT2});
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
}}
.brand-sub {{
    font-size: 0.7rem;
    color: {TEXT_SECONDARY};
}}
.section-label {{
    font-size: 0.68rem;
    font-weight: 600;
    color: {TEXT_SECONDARY};
    letter-spacing: 0.1em;
    text-transform: uppercase;
    margin: 1.2rem 0 0.6rem 0;
}}
.pipeline-step {{
    display: flex;
    align-items: center;
    justify-content: space-between;
    padding: 0.55rem 0.8rem;
    background: {CARD_BG2};
    border: 1px solid {BORDER};
    border-radius: 8px;
    margin-bottom: 0.4rem;
}}
.step-left {{
    display: flex;
    align-items: center;
    gap: 0.55rem;
    font-size: 0.8rem;
    color: {TEXT_PRIMARY};
}}
.step-icon {{ font-size: 0.9rem; }}
.step-ready {{
    font-size: 0.65rem;
    font-weight: 600;
    color: #22c55e;
    background: rgba(34,197,94,0.12);
    border: 1px solid rgba(34,197,94,0.3);
    border-radius: 20px;
    padding: 0.15rem 0.5rem;
}}
.tag-row {{
    display: flex;
    flex-wrap: wrap;
    gap: 0.4rem;
    margin-top: 0.5rem;
}}
.tech-tag {{
    font-size: 0.68rem;
    font-weight: 500;
    padding: 0.25rem 0.6rem;
    border-radius: 20px;
    border: 1px solid {BORDER};
    background: {TAG_BG};
    color: {TEXT_SECONDARY};
}}

/* ── Streamlit button overrides ── */
.stButton > button {{
    background: {BTN_BG} !important;
    color: {TEXT_PRIMARY} !important;
    border: 1px solid {BTN_BORDER} !important;
    border-radius: 8px !important;
    font-family: 'Inter', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    transition: background 0.2s, box-shadow 0.2s, border-color 0.2s !important;
    padding: 0.4rem 0.9rem !important;
}}
.stButton > button:hover {{
    background: {BTN_HOVER_BG} !important;
    border-color: {ACCENT} !important;
    box-shadow: 0 0 12px {SHADOW} !important;
    color: {ACCENT} !important;
}}

/* ── Chat input override (aggressive) ── */
[data-testid="stChatInput"],
[data-testid="stBottom"],
[data-testid="stBottomBlockContainer"],
[data-testid="stBottom"] > div,
[data-testid="stBottomBlockContainer"] > div,
[data-testid="stChatInput"] > div,
[data-testid="stChatInput"] textarea,
[data-testid="stChatInputTextArea"] {{
    background-color: {INPUT_BG} !important;
    background: {INPUT_BG} !important;
    color: {TEXT_PRIMARY} !important;
    border-color: {BORDER} !important;
}}
[data-testid="stChatInput"] textarea::placeholder {{
    color: {TEXT_SECONDARY} !important;
}}
[data-testid="stChatInput"] textarea:focus {{
    border-color: {ACCENT} !important;
    box-shadow: 0 0 0 2px {ACCENT_DIM} !important;
}}

/* ── Expander ── */
[data-testid="stExpander"] {{
    background: {CARD_BG2} !important;
    border: 1px solid {BORDER} !important;
    border-radius: 10px !important;
}}
[data-testid="stExpander"] summary {{
    color: {TEXT_SECONDARY} !important;
    font-size: 0.8rem !important;
}}

/* ── Slider ── */
.stSlider [data-testid="stSlider"] {{
    accent-color: {ACCENT};
}}

/* ── Scrollbar ── */
::-webkit-scrollbar {{ width: 4px; }}
::-webkit-scrollbar-track {{ background: transparent; }}
::-webkit-scrollbar-thumb {{ background: {BORDER}; border-radius: 4px; }}

/* ── Chat message avatars ── */
[data-testid="stChatMessage"] {{
    background: transparent !important;
}}

/* ── Readable Streamlit chat text ── */
[data-testid="stChatMessage"],
[data-testid="stChatMessageContent"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"],
[data-testid="stChatMessage"] [data-testid="stMarkdownContainer"] *,
[data-testid="stChatMessage"] [data-testid="stExpander"] summary,
[data-testid="stChatMessage"] [data-testid="stExpander"] summary * {{
    color: {CHAT_TEXT} !important;
}}

[data-testid="stChatMessage"] li::marker {{
    color: {CHAT_TEXT} !important;
}}

/* Match code and table backgrounds to the selected theme, too. */
[data-testid="stChatMessage"] pre,
[data-testid="stChatMessage"] code,
[data-testid="stChatMessage"] [data-testid="stCode"],
[data-testid="stChatMessage"] th,
[data-testid="stChatMessage"] td {{
    background-color: {CARD_BG2} !important;
    color: {CHAT_TEXT} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── RAG Init (cached) ─────────────────────────────────────────────────────────
@st.cache_resource(show_spinner=False)
def load_rag():
    embeddings = NVIDIAEmbeddings(
        model="nvidia/nemotron-3-embed-1b",
        truncate="END",
    )
    vectorstore = PineconeVectorStore(
        index_name=INDEX_NAME,
        embedding=embeddings,
    )
    llm = ChatGroq(
        model="qwen/qwen3.8-27b",
        temperature=0.2,
    )
    return vectorstore, llm


# ── Core RAG Function ─────────────────────────────────────────────────────────
def format_docs(docs):
    formatted = []
    for doc in docs:
        source = os.path.basename(doc.metadata["source"])
        page = doc.metadata["page"] + 1
        formatted.append(f"[{source}, Page {page}]\n{doc.page_content}")
    return "\n\n---\n\n".join(formatted)


def get_answer(vectorstore, llm, question, top_k=3):
    docs = vectorstore.similarity_search(question, k=top_k)
    context = format_docs(docs)
    prompt = f"""You are a helpful assistant that answers questions using ONLY the provided context.

Rules:
1. Answer based ONLY on the context below
2. If the answer is not in the context, say "I don't know based on the available documents."
3. Keep the answer clear and concise
4. Cite the source documents and page number for each fact

Context:
{context}

Question: {question}

Answer: """

    # Retry logic for Groq rate limits (free tier has request limits)
    for attempt in range(3):
        try:
            response = llm.invoke(prompt)
            break
        except Exception as e:
            if "rate" in str(e).lower() and attempt < 2:
                time.sleep(2 ** attempt)  # wait 1s, then 2s
                continue
            raise e

    sources = []
    for doc in docs:
        source = os.path.basename(doc.metadata["source"])
        page = doc.metadata["page"] + 1
        sources.append({"file": source, "page": page})
    return response.content, sources


@st.cache_data(ttl=3600, show_spinner=False)
def cached_answer(question, top_k):
    """Reuse an identical question for one hour to reduce repeated API usage."""
    vectorstore, llm = load_rag()
    return get_answer(vectorstore, llm, question, top_k)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div class="brand-box">
        <div class="brand-icon">✦</div>
        <div>
            <div class="brand-name">Meraj Local Chat Bot</div>
            <div class="brand-sub">Personal Docs Chatbot</div>
        </div>
    </div>
    """, unsafe_allow_html=True)

    theme_icon = "☀️" if st.session_state.theme == "dark" else "🌙"
    theme_label = "Light Mode" if st.session_state.theme == "dark" else "Dark Mode"
    if st.button(f"{theme_icon} {theme_label}", key="theme_toggle", use_container_width=True):
        st.session_state.theme = "light" if st.session_state.theme == "dark" else "dark"
        st.rerun()

    st.markdown(f'<div class="section-label">RAG Pipeline</div>', unsafe_allow_html=True)

    pipeline_steps = [
        ("📄", "PDF Loader"),
        ("✂️", "Chunking"),
        ("🧠", "NVIDIA Embeddings"),
        ("🗄️", "Pinecone Store"),
        ("⚡", "Groq LLM"),
    ]
    for icon, label in pipeline_steps:
        st.markdown(f"""
        <div class="pipeline-step">
            <div class="step-left">
                <span class="step-icon">{icon}</span>
                <span>{label}</span>
            </div>
            <span class="step-ready">✓ Ready</span>
        </div>
        """, unsafe_allow_html=True)

    st.markdown(f'<div class="section-label">Retrieval Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Top-K chunks", min_value=1, max_value=10, value=3, key="top_k")

    st.markdown(f'<div class="section-label">Tech Stack</div>', unsafe_allow_html=True)
    tags = ["NVIDIA NeMo", "Pinecone", "Groq", "LangChain", "Streamlit"]
    tags_html = "".join(f'<span class="tech-tag">{t}</span>' for t in tags)
    st.markdown(f'<div class="tag-row">{tags_html}</div>', unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    if st.button("🗑️ Clear Chat", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Area ─────────────────────────────────────────────────────────────────

# Header
st.markdown(f"""
<div class="header-card">
    <div class="eyebrow">✦ Meraj Local Chat Bot</div>
    <div class="header-title">Personal Docs Chatbot</div>
    <div class="header-sub">Ask anything about company policies, procedures, and internal documentation. Answers are grounded in your indexed PDFs — no hallucinations.</div>
    <div class="status-badge">
        <div class="pulse-dot"></div>
        RAG Pipeline Active
    </div>
</div>
""", unsafe_allow_html=True)

# Stats row
st.markdown(f"""
<div class="stats-row">
    <div class="stat-card">
        <div class="stat-value">3</div>
        <div class="stat-label">Documents</div>
        <div class="stat-sub">↗ PDF indexed</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">17</div>
        <div class="stat-label">Chunks</div>
        <div class="stat-sub">↗ Vectorized</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">2048</div>
        <div class="stat-label">Dimensions</div>
        <div class="stat-sub">↗ NVIDIA NeMo</div>
    </div>
    <div class="stat-card">
        <div class="stat-value">Llama 3.3</div>
        <div class="stat-label">LLM Model</div>
        <div class="stat-sub">↗ via Groq</div>
    </div>
</div>
""", unsafe_allow_html=True)

# Load RAG
with st.spinner("Connecting to RAG pipeline..."):
    vectorstore, llm = load_rag()

# Sample questions
st.markdown(f'<div class="sample-label">Try asking</div>', unsafe_allow_html=True)
sample_qs = [
    "What is the employee leave policy?",
    "What is the refund policy?",
    "How should I report a security incident?",
]
cols = st.columns(3)
for i, q in enumerate(sample_qs):
    with cols[i]:
        if st.button(q, key=f"sample_{i}", use_container_width=True):
            st.session_state.messages.append({"role": "user", "content": q})
            with st.spinner("Thinking..."):
                answer, sources = cached_answer(q, st.session_state.top_k)
            st.session_state.messages.append({
                "role": "assistant",
                "content": answer,
                "sources": sources,
            })
            st.rerun()

st.markdown("<br>", unsafe_allow_html=True)

# Chat history
for msg in st.session_state.messages:
    if msg["role"] == "user":
        with st.chat_message("user"):
            st.markdown(msg["content"])
    else:
        with st.chat_message("assistant"):
            st.markdown(msg["content"])
            if msg.get("sources"):
                with st.expander(f"📎 Sources used ({len(msg['sources'])})"):
                    for s in msg["sources"]:
                        st.markdown(f"""
                        <div class="source-item">
                            <span class="source-icon">📄</span>
                            <span><strong>{s['file']}</strong> — Page {s['page']}</span>
                        </div>
                        """, unsafe_allow_html=True)

# Chat input
if question := st.chat_input("Ask about your company documents..."):
    st.session_state.messages.append({"role": "user", "content": question})
    with st.chat_message("user"):
        st.markdown(question)

    with st.chat_message("assistant"):
        with st.spinner("Searching documents..."):
            answer, sources = cached_answer(question, st.session_state.top_k)
        st.markdown(answer)
        if sources:
            with st.expander(f"📎 Sources used ({len(sources)})"):
                for s in sources:
                    st.markdown(f"""
                    <div class="source-item">
                        <span class="source-icon">📄</span>
                        <span><strong>{s['file']}</strong> — Page {s['page']}</span>
                    </div>
                    """, unsafe_allow_html=True)

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources,
    })
