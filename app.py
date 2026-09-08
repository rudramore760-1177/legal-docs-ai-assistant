import streamlit as st
from dotenv import load_dotenv
load_dotenv()

from llama_index.core import SimpleDirectoryReader, VectorStoreIndex, StorageContext, Settings
from llama_index.vector_stores.chroma import ChromaVectorStore
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.gemini import Gemini
import chromadb
import os
import tempfile
import time
from datetime import datetime

st.set_page_config(page_title="Legal AI Assistant", page_icon="⚖️", layout="centered")

# --- Modern dark tech theme ---
st.markdown("""
    <style>
    .stApp { background-color: #0A0E17; color: #E6EDF3; }
    [data-testid="stSidebar"] { background-color: #11151F; border-right: 1px solid #1F2937; }
    h1, h2, h3 { color: #F5F7FA !important; }
    .stButton>button {
        background: linear-gradient(90deg, #7C3AED, #2563EB);
        color: white; border: none; border-radius: 8px;
        font-weight: 600; padding: 0.5rem 1rem; transition: all 0.2s ease;
    }
    .stButton>button:hover { opacity: 0.85; transform: scale(1.02); }
    [data-testid="stChatMessage"] {
        background-color: #151A25; border: 1px solid #232A38;
        border-radius: 12px; padding: 0.5rem;
    }
    [data-testid="stFileUploader"] {
        border: 1px dashed #2563EB; border-radius: 10px; padding: 0.5rem;
    }
    .stAlert { background-color: #151A25; border: 1px solid #2563EB; border-radius: 8px; }
    .stat-card {
        background: #151A25; border: 1px solid #232A38; border-radius: 10px;
        padding: 0.7rem; text-align: center; margin-bottom: 0.4rem;
    }
    .stat-number { font-size: 1.4rem; font-weight: 700; color: #7C3AED; }
    .stat-label { font-size: 0.75rem; color: #9CA3AF; }
    .badge {
        display: inline-block; background: #1F2937; color: #93C5FD;
        border-radius: 6px; padding: 2px 8px; font-size: 0.75rem; margin-right: 6px;
    }
    .badge-warn { background: #3B1F1F; color: #FCA5A5; }
    ::-webkit-scrollbar { width: 8px; }
    ::-webkit-scrollbar-thumb { background: #2563EB; border-radius: 4px; }
    </style>
""", unsafe_allow_html=True)

# --- Setup models (cached so it only loads once) ---
@st.cache_resource
def load_models():
    Settings.embed_model = HuggingFaceEmbedding(model_name="BAAI/bge-small-en-v1.5")
    Settings.llm = Gemini(model="models/gemini-3.6-flash")

load_models()

st.title("⚖️ Legal Document Q&A Assistant")
st.caption("Upload legal documents and ask questions about them.")

# --- Session state setup ---
for key, default in [
    ("index", None), ("messages", []), ("doc_names", []),
    ("response_times", []), ("questions_asked", 0)
]:
    if key not in st.session_state:
        st.session_state[key] = default

NOT_FOUND_PHRASES = ["doesn't contain", "does not contain", "not mentioned",
                      "no information", "cannot find", "not found", "isn't in the"]

def generate_follow_up(question: str) -> str:
    q = question.lower()
    if "deposit" in q:
        return "When is the deposit refunded?"
    if "terminat" in q or "notice" in q:
        return "What happens if notice isn't given?"
    if "salary" in q or "compensation" in q:
        return "Is there a performance bonus?"
    if "confidential" in q:
        return "How long do confidentiality terms last?"
    return "What are the governing law terms?"

# --- Sidebar ---
with st.sidebar:
    st.header("📄 Upload Documents")
    uploaded_files = st.file_uploader(
        "Upload legal PDFs", type=["pdf"], accept_multiple_files=True
    )

    if st.button("Build Index", type="primary", use_container_width=True):
        if not uploaded_files:
            st.warning("Please upload at least one PDF first.")
        else:
            with st.spinner("Processing documents..."):
                with tempfile.TemporaryDirectory() as tmp_dir:
                    for file in uploaded_files:
                        file_path = os.path.join(tmp_dir, file.name)
                        with open(file_path, "wb") as f:
                            f.write(file.getbuffer())

                    documents = SimpleDirectoryReader(tmp_dir).load_data()
                    chroma_client = chromadb.EphemeralClient()
                    chroma_collection = chroma_client.get_or_create_collection("legal_docs")
                    vector_store = ChromaVectorStore(chroma_collection=chroma_collection)
                    storage_context = StorageContext.from_defaults(vector_store=vector_store)

                    st.session_state.index = VectorStoreIndex.from_documents(
                        documents, storage_context=storage_context
                    )
                    st.session_state.messages = []
                    st.session_state.doc_names = [f.name for f in uploaded_files]
                    st.session_state.response_times = []
                    st.session_state.questions_asked = 0

            st.success(f"Indexed {len(uploaded_files)} document(s). Ask away!")

    if st.session_state.doc_names:
        st.divider()
        st.subheader("📚 Indexed documents")
        for name in st.session_state.doc_names:
            st.markdown(f"- {name}")

    # --- Session stats dashboard ---
    if st.session_state.questions_asked > 0:
        st.divider()
        st.subheader("📊 Session stats")
        avg_time = sum(st.session_state.response_times) / len(st.session_state.response_times)
        c1, c2 = st.columns(2)
        with c1:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number">{st.session_state.questions_asked}</div>
                <div class="stat-label">Questions asked</div></div>""", unsafe_allow_html=True)
        with c2:
            st.markdown(f"""<div class="stat-card">
                <div class="stat-number">{avg_time:.1f}s</div>
                <div class="stat-label">Avg response time</div></div>""", unsafe_allow_html=True)

    # --- Download chat as report ---
    if st.session_state.messages:
        st.divider()
        report_lines = [f"Legal Document Q&A Report — {datetime.now().strftime('%Y-%m-%d %H:%M')}",
                         f"Documents: {', '.join(st.session_state.doc_names)}", "-" * 50]
        for msg in st.session_state.messages:
            role = "Q" if msg["role"] == "user" else "A"
            report_lines.append(f"\n[{role}] {msg['content']}")
        report_text = "\n".join(report_lines)

        st.download_button(
            "⬇️ Download Q&A report", data=report_text,
            file_name=f"legal_qa_report_{datetime.now().strftime('%Y%m%d_%H%M')}.txt",
            use_container_width=True
        )

        if st.button("🗑️ Clear chat", use_container_width=True):
            st.session_state.messages = []
            st.session_state.response_times = []
            st.session_state.questions_asked = 0
            st.rerun()

    st.divider()
    st.caption("Built with LlamaIndex, Chroma, HuggingFace embeddings, and Gemini.")

# --- Main chat area ---
if st.session_state.index is None:
    st.info("👈 Upload your legal documents and click **Build Index** to get started.")
else:
    query_engine = st.session_state.index.as_query_engine(
        similarity_top_k=5,
        system_prompt=(
            "You are a legal assistant. Answer ONLY using the provided context "
            "from the uploaded legal documents. If the answer isn't in the "
            "documents, say so clearly instead of guessing. Cite the relevant "
            "clause or section when possible."
        ),
    )

    clicked_question = None
    if not st.session_state.messages:
        st.markdown("**Try asking:**")
        example_questions = [
            "What is the security deposit?",
            "What is the notice period for termination?",
            "Is there a non-compete clause?",
        ]
        cols = st.columns(len(example_questions))
        for col, q in zip(cols, example_questions):
            with col:
                if st.button(q, use_container_width=True):
                    clicked_question = q

    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.markdown(msg["content"])
            if msg["role"] == "assistant" and "sources" in msg:
                badge_class = "badge badge-warn" if msg.get("not_found") else "badge"
                st.markdown(
                    f'<span class="{badge_class}">{len(msg["sources"])} chunks retrieved</span>'
                    f'<span class="badge">{msg.get("response_time", 0):.1f}s</span>',
                    unsafe_allow_html=True
                )
                with st.expander("📎 View source text"):
                    for i, src in enumerate(msg["sources"], 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(src)
                if msg.get("follow_up"):
                    if st.button(f"💡 {msg['follow_up']}", key=f"followup_{msg['content'][:20]}"):
                        clicked_question = msg["follow_up"]

    question = st.chat_input("Ask a question about your documents...") or clicked_question

    if question:
        st.session_state.messages.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                start = time.time()
                response = query_engine.query(question)
                elapsed = time.time() - start

                answer_text = str(response)
                not_found = any(p in answer_text.lower() for p in NOT_FOUND_PHRASES)

                st.markdown(answer_text)

                sources = [node.get_content()[:500] for node in response.source_nodes]
                badge_class = "badge badge-warn" if not_found else "badge"
                st.markdown(
                    f'<span class="{badge_class}">{len(sources)} chunks retrieved</span>'
                    f'<span class="badge">{elapsed:.1f}s</span>',
                    unsafe_allow_html=True
                )
                if sources:
                    with st.expander("📎 View source text"):
                        for i, src in enumerate(sources, 1):
                            st.markdown(f"**Chunk {i}:**")
                            st.text(src)

                follow_up = generate_follow_up(question)
                st.button(f"💡 {follow_up}", key=f"followup_new")

        st.session_state.response_times.append(elapsed)
        st.session_state.questions_asked += 1
        st.session_state.messages.append({
            "role": "assistant", "content": answer_text, "sources": sources,
            "not_found": not_found, "response_time": elapsed, "follow_up": follow_up
        })