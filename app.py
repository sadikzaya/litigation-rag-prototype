import os
import tempfile

import streamlit as st
from dotenv import load_dotenv

from rag import LitigationRAG

load_dotenv()

st.set_page_config(page_title="LegalMind RAG", page_icon="⚖️", layout="wide")

st.markdown("""
<style>
    #MainMenu, footer, header {visibility: hidden;}
    .block-container {padding-top: 2rem;}

    .hero {
        background: linear-gradient(135deg, #0f2744 0%, #1d4ed8 100%);
        padding: 2rem 2.5rem;
        border-radius: 14px;
        margin-bottom: 2rem;
        color: white;
    }
    .hero h1 {margin: 0; font-size: 2.1rem; font-weight: 700; letter-spacing: -0.5px;}
    .hero p {margin: 0.4rem 0 0; opacity: 0.85; font-size: 1rem;}

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
    .step-card {
        background: #f8fafc;
        border: 1px solid #e2e8f0;
        border-radius: 10px;
        padding: 1.2rem;
        text-align: center;
    }
    .q-label {font-weight: 600; color: #1e293b; margin-bottom: 0.3rem;}
</style>
""", unsafe_allow_html=True)


def init_state():
    defaults = {"rag": None, "history": [], "doc_loaded": False, "doc_name": ""}
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v


def sidebar():
    with st.sidebar:
        st.markdown("### ⚙️ Configuration")
        api_key = st.text_input(
            "OpenAI API Key",
            value=os.getenv("OPENAI_API_KEY", ""),
            type="password",
            placeholder="sk-...",
        )

        st.markdown("---")
        st.markdown("### 📄 Document")

        uploaded = st.file_uploader(
            "Upload Legal PDF",
            type=["pdf"],
            help="Contracts, briefs, depositions, court filings, discovery docs",
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

                        st.success(f"Indexed {n} chunks. Ready.")
                    except Exception as e:
                        st.error(str(e))

        if st.session_state.doc_loaded:
            st.markdown("---")
            st.markdown("**Active document**")
            st.code(st.session_state.doc_name, language=None)
            st.markdown("🟢 **Status:** Ready for queries")

            if st.button("Clear & Reset", use_container_width=True):
                st.session_state.update({"rag": None, "doc_loaded": False, "history": [], "doc_name": ""})
                st.rerun()

        st.markdown("---")
        st.markdown("""
**Sample queries**
- What are the key claims?
- What damages are being sought?
- Who are the named parties?
- What is the governing law?
- Summarize the key allegations
- What is the timeline of events?
""")

        st.markdown("---")
        st.caption("Built by Sadik E. · Python · LangChain · FAISS · OpenAI")

    return api_key


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
    sidebar()

    st.markdown("""
    <div class="hero">
        <h1>⚖️ LegalMind RAG</h1>
        <p>Litigation Research Assistant — AI-powered document analysis with source citations</p>
    </div>
    """, unsafe_allow_html=True)

    if not st.session_state.doc_loaded:
        c1, c2, c3 = st.columns(3)
        for col, num, txt in [
            (c1, "1", "Enter your OpenAI API key in the sidebar"),
            (c2, "2", "Upload any legal PDF document"),
            (c3, "3", "Ask questions — get cited answers"),
        ]:
            col.markdown(f"""
            <div class="step-card">
                <div style="font-size:2rem;">{"📋" if num=="1" else "📄" if num=="2" else "🔍"}</div>
                <div style="font-weight:700; margin:0.3rem 0;">Step {num}</div>
                <div style="color:#64748b; font-size:0.9rem;">{txt}</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("#### Supported Document Types")
        cols = st.columns(5)
        for col, label in zip(cols, ["Contracts", "Court Filings", "Depositions", "Legal Briefs", "Discovery Docs"]):
            col.markdown(f"📄 **{label}**")
        return

    render_history()

    with st.form("query_form", clear_on_submit=True):
        question = st.text_input(
            "Your question",
            placeholder="e.g. What damages are being sought by the plaintiff?",
            label_visibility="collapsed",
        )
        submitted = st.form_submit_button("Ask", type="primary")

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


if __name__ == "__main__":
    main()
