"""
RAG Chatbot — Streamlit UI
Context-aware chatbot with retrieval-augmented generation.
"""

import os
import time
import streamlit as st
from dotenv import load_dotenv
from src.rag_engine import RAGEngine

load_dotenv()

# ── Page config ──────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Chatbot",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ───────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'Inter', sans-serif;
}

.stApp {
    background: #0d1117;
    color: #e6edf3;
}

/* Hide default streamlit elements */
#MainMenu, footer, header { visibility: hidden; }
.block-container { padding-top: 1.5rem; padding-bottom: 2rem; }

/* ── Header ── */
.app-header {
    background: linear-gradient(135deg, #161b22 0%, #1c2128 100%);
    border: 1px solid #30363d;
    border-radius: 16px;
    padding: 24px 32px;
    margin-bottom: 24px;
    display: flex;
    align-items: center;
    gap: 16px;
}
.header-icon { font-size: 48px; }
.header-title { font-size: 28px; font-weight: 700; color: #e6edf3; letter-spacing: -0.5px; }
.header-sub { font-size: 14px; color: #8b949e; margin-top: 4px; }
.header-badges { display: flex; gap: 8px; margin-top: 10px; flex-wrap: wrap; }
.badge {
    font-size: 11px; font-weight: 600; padding: 3px 10px;
    border-radius: 20px; border: 1px solid;
    text-transform: uppercase; letter-spacing: 0.5px;
}
.badge-blue  { background: rgba(56,139,253,.15); color: #58a6ff; border-color: rgba(56,139,253,.3); }
.badge-green { background: rgba(63,185,80,.15);  color: #3fb950; border-color: rgba(63,185,80,.3); }
.badge-orange{ background: rgba(210,153,34,.15); color: #d2a22a; border-color: rgba(210,153,34,.3); }
.badge-purple{ background: rgba(188,140,255,.15);color: #bc8cff; border-color: rgba(188,140,255,.3); }

/* ── Sidebar ── */
section[data-testid="stSidebar"] {
    background: #161b22 !important;
    border-right: 1px solid #30363d;
}
section[data-testid="stSidebar"] .block-container { padding-top: 1rem; }

.sidebar-section {
    background: #1c2128;
    border: 1px solid #30363d;
    border-radius: 10px;
    padding: 16px;
    margin-bottom: 14px;
}
.sidebar-title {
    font-size: 11px; font-weight: 700; color: #8b949e;
    text-transform: uppercase; letter-spacing: 1px; margin-bottom: 12px;
}

/* ── Stats ── */
.stats-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 8px; margin-bottom: 14px; }
.stat-box {
    background: #1c2128; border: 1px solid #30363d; border-radius: 8px;
    padding: 12px; text-align: center;
}
.stat-num { font-size: 22px; font-weight: 700; color: #58a6ff; font-family: 'JetBrains Mono', monospace; }
.stat-label { font-size: 10px; color: #8b949e; text-transform: uppercase; letter-spacing: 0.5px; margin-top: 2px; }

/* ── Chat messages ── */
.chat-container {
    display: flex; flex-direction: column; gap: 16px;
    padding: 8px 0;
}

.msg-user {
    background: linear-gradient(135deg, #1f3a5f, #162032);
    border: 1px solid #2d5a8e;
    border-radius: 16px 16px 4px 16px;
    padding: 14px 18px;
    margin-left: 60px;
    position: relative;
}
.msg-user::before {
    content: "👤";
    position: absolute; top: 12px; right: -44px;
    font-size: 20px;
}
.msg-user-label { font-size: 11px; color: #58a6ff; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.msg-user-text { font-size: 14px; color: #cdd9e5; line-height: 1.6; }

.msg-bot {
    background: linear-gradient(135deg, #1c2128, #161b22);
    border: 1px solid #30363d;
    border-radius: 16px 16px 16px 4px;
    padding: 14px 18px;
    margin-right: 60px;
    position: relative;
}
.msg-bot::before {
    content: "🧠";
    position: absolute; top: 12px; left: -44px;
    font-size: 20px;
}
.msg-bot-label { font-size: 11px; color: #3fb950; font-weight: 600; margin-bottom: 6px; text-transform: uppercase; letter-spacing: 0.5px; }
.msg-bot-text { font-size: 14px; color: #e6edf3; line-height: 1.7; }
.msg-meta { font-size: 11px; color: #484f58; margin-top: 8px; display: flex; gap: 12px; }

/* ── Sources ── */
.sources-header {
    font-size: 11px; font-weight: 600; color: #8b949e;
    text-transform: uppercase; letter-spacing: 0.5px;
    margin-top: 12px; margin-bottom: 8px;
    display: flex; align-items: center; gap: 6px;
}
.source-chip {
    background: #21262d; border: 1px solid #30363d; border-radius: 6px;
    padding: 8px 12px; margin-bottom: 6px; font-size: 12px; color: #8b949e;
    line-height: 1.5;
}
.source-score {
    display: inline-block; background: rgba(63,185,80,.15); color: #3fb950;
    border-radius: 4px; padding: 1px 6px; font-size: 10px; font-weight: 700;
    margin-right: 6px; font-family: 'JetBrains Mono', monospace;
}

/* ── Welcome ── */
.welcome-box {
    background: linear-gradient(135deg, #161b22, #1c2128);
    border: 1px solid #30363d; border-radius: 16px;
    padding: 40px; text-align: center; margin: 20px 0;
}
.welcome-icon { font-size: 56px; margin-bottom: 16px; }
.welcome-title { font-size: 22px; font-weight: 700; color: #e6edf3; margin-bottom: 8px; }
.welcome-sub { font-size: 14px; color: #8b949e; margin-bottom: 24px; line-height: 1.6; }
.topic-grid { display: grid; grid-template-columns: repeat(2, 1fr); gap: 8px; max-width: 500px; margin: 0 auto; }
.topic-chip {
    background: #21262d; border: 1px solid #30363d; border-radius: 8px;
    padding: 10px 14px; font-size: 13px; color: #cdd9e5; text-align: left;
    cursor: pointer;
}

/* ── Input area ── */
.input-area {
    background: #161b22; border: 1px solid #30363d; border-radius: 12px;
    padding: 4px; margin-top: 16px;
}

/* Streamlit input overrides */
.stTextInput > div > div > input {
    background: #0d1117 !important;
    border: none !important;
    color: #e6edf3 !important;
    font-size: 15px !important;
    border-radius: 8px !important;
    padding: 12px 16px !important;
}
.stTextInput > div > div > input:focus {
    border: none !important;
    box-shadow: none !important;
}
.stTextInput > label { display: none !important; }

.stButton > button {
    background: #238636 !important;
    color: white !important;
    border: 1px solid #2ea043 !important;
    border-radius: 8px !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    padding: 10px 24px !important;
    transition: all 0.2s !important;
    width: 100%;
}
.stButton > button:hover {
    background: #2ea043 !important;
    transform: translateY(-1px) !important;
}

/* Sidebar inputs */
.stTextInput input[type="password"] {
    background: #0d1117 !important;
    color: #e6edf3 !important;
    border: 1px solid #30363d !important;
    border-radius: 6px !important;
}

/* Select box */
.stSelectbox > div > div {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    color: #e6edf3 !important;
    border-radius: 6px !important;
}

/* Expander */
.streamlit-expanderHeader {
    background: #21262d !important;
    border: 1px solid #30363d !important;
    border-radius: 8px !important;
    color: #8b949e !important;
    font-size: 12px !important;
}

/* Divider */
hr { border-color: #30363d !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 6px; }
::-webkit-scrollbar-track { background: #161b22; }
::-webkit-scrollbar-thumb { background: #30363d; border-radius: 3px; }
</style>
""", unsafe_allow_html=True)


# ── Session state ────────────────────────────────────────────────────
def init_state():
    defaults = {
        "messages": [],
        "engine": None,
        "kb_loaded": False,
        "total_queries": 0,
        "total_chunks": 0,
        "api_key_set": False,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

init_state()


# ── Initialize RAG engine ────────────────────────────────────────────
@st.cache_resource
def get_engine():
    engine = RAGEngine()
    doc_path = os.path.join(os.path.dirname(__file__), "docs", "knowledge_base.txt")
    n = engine.load_documents([doc_path])
    return engine, n


# ── Sidebar ──────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🧠 RAG Chatbot")
    st.markdown("---")

    # API Key
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">🔑 Configuration</div>', unsafe_allow_html=True)
    api_key = st.text_input("OpenRouter API Key", type="password",
                             value=os.environ.get("OPENROUTER_API_KEY", ""),
                             placeholder="sk-or-v1-...")
    if api_key:
        os.environ["OPENROUTER_API_KEY"] = api_key
        st.session_state.api_key_set = True
        st.success("✓ API Key set")
    st.markdown('</div>', unsafe_allow_html=True)

    # Load KB
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">📚 Knowledge Base</div>', unsafe_allow_html=True)

    if st.button("⚡ Load Knowledge Base"):
        with st.spinner("Loading and indexing documents..."):
            try:
                engine, n_chunks = get_engine()
                st.session_state.engine = engine
                st.session_state.kb_loaded = True
                st.session_state.total_chunks = n_chunks
                st.success(f"✓ {n_chunks} chunks indexed!")
            except Exception as e:
                st.error(f"Error: {e}")

    if st.session_state.kb_loaded:
        st.markdown(f"""
        <div class="stats-grid">
            <div class="stat-box">
                <div class="stat-num">{st.session_state.total_chunks}</div>
                <div class="stat-label">Chunks</div>
            </div>
            <div class="stat-box">
                <div class="stat-num">{st.session_state.total_queries}</div>
                <div class="stat-label">Queries</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
        st.markdown("**Topics covered:**")
        topics = ["🤖 AI & ML", "🧬 Deep Learning", "💬 NLP & LLMs",
                  "🔍 RAG & Embeddings", "⛓️ LangChain", "🛡️ AI Ethics",
                  "🐍 Python ML", "👁️ Computer Vision"]
        for t in topics:
            st.markdown(f"• {t}")

    st.markdown('</div>', unsafe_allow_html=True)

    # Settings
    st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
    st.markdown('<div class="sidebar-title">⚙️ Settings</div>', unsafe_allow_html=True)
    top_k = st.slider("Retrieved chunks (Top-K)", 1, 6, 3)
    show_sources = st.toggle("Show source chunks", value=True)
    show_scores = st.toggle("Show relevance scores", value=True)
    st.markdown('</div>', unsafe_allow_html=True)

    # Clear chat
    if st.button("🗑️ Clear Conversation"):
        st.session_state.messages = []
        st.session_state.total_queries = 0
        st.rerun()

    st.markdown("---")
    st.markdown('<p style="font-size:11px;color:#484f58;text-align:center;">Task 4 · RAG Chatbot · LangChain + OpenRouter</p>', unsafe_allow_html=True)


# ── Main area ────────────────────────────────────────────────────────
st.markdown("""
<div class="app-header">
    <div class="header-icon">🧠</div>
    <div>
        <div class="header-title">Context-Aware RAG Chatbot</div>
        <div class="header-sub">Retrieval-Augmented Generation · Conversational Memory · Vector Search</div>
        <div class="header-badges">
            <span class="badge badge-blue">RAG Pipeline</span>
            <span class="badge badge-green">Vector Search</span>
            <span class="badge badge-orange">Context Memory</span>
            <span class="badge badge-purple">OpenRouter LLM</span>
        </div>
    </div>
</div>
""", unsafe_allow_html=True)

# ── Chat display ─────────────────────────────────────────────────────
chat_col, _ = st.columns([1, 0.001])

with chat_col:
    if not st.session_state.messages:
        st.markdown("""
        <div class="welcome-box">
            <div class="welcome-icon">💬</div>
            <div class="welcome-title">Ask me anything about AI & ML</div>
            <div class="welcome-sub">
                I have a knowledge base covering AI, Machine Learning, Deep Learning, NLP,<br>
                RAG, LangChain, Embeddings, and more. I remember our conversation history!
            </div>
            <div class="topic-grid">
                <div class="topic-chip">🤖 What is machine learning?</div>
                <div class="topic-chip">🔍 How does RAG work?</div>
                <div class="topic-chip">💬 Explain transformers</div>
                <div class="topic-chip">⛓️ What is LangChain?</div>
                <div class="topic-chip">🧬 Types of neural networks?</div>
                <div class="topic-chip">🛡️ What is AI safety?</div>
            </div>
        </div>
        """, unsafe_allow_html=True)
    else:
        for msg in st.session_state.messages:
            if msg["role"] == "user":
                st.markdown(f"""
                <div class="msg-user">
                    <div class="msg-user-label">You</div>
                    <div class="msg-user-text">{msg["content"]}</div>
                </div>
                """, unsafe_allow_html=True)
            else:
                st.markdown(f"""
                <div class="msg-bot">
                    <div class="msg-bot-label">🧠 RAG Assistant</div>
                    <div class="msg-bot-text">{msg["content"]}</div>
                    <div class="msg-meta">
                        <span>⏱ {msg.get("latency", "?")}s</span>
                        <span>📄 {msg.get("chunks", "?")} chunks retrieved</span>
                    </div>
                </div>
                """, unsafe_allow_html=True)

                if show_sources and msg.get("sources"):
                    with st.expander(f"📎 View {len(msg['sources'])} source chunks"):
                        for src in msg["sources"]:
                            score_html = f'<span class="source-score">{src["score"]:.2f}</span>' if show_scores else ""
                            st.markdown(f"""
                            <div class="source-chip">
                                {score_html}
                                {src["text"][:300]}{"..." if len(src["text"]) > 300 else ""}
                            </div>
                            """, unsafe_allow_html=True)

# ── Input ────────────────────────────────────────────────────────────
st.markdown("---")

col1, col2 = st.columns([5, 1])
with col1:
    user_input = st.text_input("", placeholder="Ask a question about AI, ML, RAG, LangChain...", key="user_input")
with col2:
    send = st.button("Send ➤")

# ── Handle send ──────────────────────────────────────────────────────
if send and user_input.strip():
    if not st.session_state.api_key_set and not os.environ.get("OPENROUTER_API_KEY"):
        st.error("⚠️ Please enter your OpenRouter API key in the sidebar.")
    elif not st.session_state.kb_loaded:
        st.warning("⚠️ Please click **Load Knowledge Base** in the sidebar first.")
    else:
        query = user_input.strip()
        st.session_state.messages.append({"role": "user", "content": query})

        with st.spinner("🔍 Retrieving context and generating response..."):
            try:
                engine = st.session_state.engine
                engine.api_key = os.environ.get("OPENROUTER_API_KEY")

                history = [
                    {"role": m["role"], "content": m["content"]}
                    for m in st.session_state.messages[:-1]
                ]

                result = engine.chat(query, history)

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["answer"],
                    "sources": result["sources"],
                    "latency": result["latency_s"],
                    "chunks": result["chunks_retrieved"]
                })
                st.session_state.total_queries += 1

            except Exception as e:
                st.error(f"❌ Error: {str(e)}")
                st.session_state.messages.pop()

        st.rerun()
