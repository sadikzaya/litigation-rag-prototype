import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag import LitigationRAG

load_dotenv()

st.set_page_config(page_title="LegalMind RAG", page_icon="⚖️", layout="centered")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem; max-width: 860px;}

    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #1d4ed8 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;}
    .hero p {margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem;}

    .section-label {
        font-size: 0.78rem;
        font-weight: 700;
        color: #64748b;
        text-transform: uppercase;
        letter-spacing: 0.8px;
        margin-bottom: 0.4rem;
    }
    .ready-banner {
        background: #f0fdf4;
        border: 1px solid #86efac;
        border-radius: 8px;
        padding: 0.7rem 1rem;
        color: #166534;
        font-weight: 600;
        font-size: 0.9rem;
        margin-bottom: 1.2rem;
    }
    .answer-box {
        background: #eff6ff;
        border-left: 4px solid #1d4ed8;
        padding: 1.2rem 1.5rem;
        border-radius: 0 10px 10px 0;
        margin: 0.8rem 0;
        color: #0f172a;
        line-height: 1.7;
    }
    .source-card {
        background: #ffffff;
        border: 1px solid #e2e8f0;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 0.4rem 0;
    }
    .badge-exhibit {
        background: #1d4ed8;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        font-weight: 700;
        letter-spacing: 0.3px;
    }
    .badge-page {
        background: #475569;
        color: white;
        padding: 2px 10px;
        border-radius: 20px;
        font-size: 0.72rem;
        margin-left: 6px;
    }
    .excerpt-text {
        color: #475569;
        font-size: 0.85rem;
        margin-top: 0.6rem;
        line-height: 1.6;
        font-style: italic;
    }
    .q-label {font-weight: 600; color: #1e293b; margin-bottom: 0.3rem;}
    .footer-note {color: #94a3b8; font-size: 0.8rem; text-align: center; margin-top: 2rem;}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {"rag": None, "history": [], "doc_loaded": False, "doc_name": ""}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def render_history():
    for item in st.session_state.history:
        st.markdown(f'<div class="q-label">Q: {item["question"]}</div>', unsafe_allow_html=True)
        st.markdown(f'<div class="answer-box">{item["answer"]}</div>', unsafe_allow_html=True)

        with st.expander(f"View {len(item['sources'])} source excerpts"):
            for src in item["sources"]:
                st.markdown(f"""
                <div class="source-card">
                    <span class="badge-exhibit">Exhibit {src['exhibit']}</span>
                    <span class="badge-page">Page {src['page']}</span>
                    <div class="excerpt-text">"{src['excerpt']}..."</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("---")


def main():
    init_state()

    st.markdown("""
    <div class="hero">
        <h1>⚖️ LegalMind RAG</h1>
        <p>Litigation Research Assistant — AI-powered document analysis with source citations</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_loaded:
        st.markdown('<div class="section-label">OpenAI API Key</div>', unsafe_allow_html=True)
        api_key = st.text_input(
            "openai_key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-...",
            label_visibility="collapsed",
        )

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown('<div class="section-label">Upload Legal Document (PDF)</div>', unsafe_allow_html=True)
        uploaded = st.file_uploader(
            "pdf_upload",
            type=["pdf"],
            help="Contracts, briefs, depositions, court filings, discovery docs",
            label_visibility="collapsed",
        )

        if uploaded and api_key:
            if st.button("Index Document", type="primary", use_container_width=True):
                with st.spinner("Extracting text and building vector index..."):
                    try:
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as f:
                            f.write(uploaded.getbuffer())
                            tmp = f.name

                        rag = LitigationRAG(api_key=api_key)
                        n = rag.load_pdf(tmp)
                        os.unlink(tmp)

                        st.session_state.rag = rag
                        st.session_state.doc_loaded = True
                        st.session_state.doc_name = uploaded.name
                        st.session_state.history = []
                        st.rerun()
                    except Exception as e:
                        st.error(str(e))
        elif not api_key:
            st.info("Enter your OpenAI API key above to get started.")
        elif not uploaded:
            st.info("Upload a legal PDF to continue.")

        st.markdown("""
        <div class="footer-note">
            Supported: Contracts · Court Filings · Depositions · Legal Briefs · Discovery Docs<br>
            Built by Sadik E. · Python · LangChain · FAISS · OpenAI
        </div>
        """, unsafe_allow_html=True)
        return

    # Document loaded — show ready banner + Q&A
    col1, col2 = st.columns([4, 1])
    with col1:
        st.markdown(f'<div class="ready-banner">🟢 &nbsp;<strong>{st.session_state.doc_name}</strong> — indexed and ready</div>', unsafe_allow_html=True)
    with col2:
        if st.button("Clear", use_container_width=True):
            st.session_state.update({"rag": None, "doc_loaded": False, "history": [], "doc_name": ""})
            st.rerun()

    render_history()

    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Ask a question about the document",
            placeholder="e.g. What damages are being sought by the plaintiff?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary", use_container_width=True)

    if submitted and question:
        with st.spinner("Searching document and generating cited answer..."):
            try:
                result = st.session_state.rag.query(question)
                st.session_state.history.append({
                    "question": question,
                    "answer": result["answer"],
                    "sources": result["sources"],
                })
                st.rerun()
            except Exception as e:
                st.error(str(e))

    st.markdown("""
    <div class="footer-note">Built by Sadik E. · Python · LangChain · FAISS · OpenAI</div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()
