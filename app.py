import streamlit as st
import os
import shutil
from rag_pipeline import initialize_rag

# ── Page Config ────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="RAG Document Q&A",
    page_icon="📄",
    layout="wide"
)

# ── Custom CSS ─────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .source-box {
        background-color: #f0f2f6;
        border-left: 4px solid #1f77b4;
        padding: 0.5rem 1rem;
        border-radius: 4px;
        font-size: 0.85rem;
        color: #444;
        margin-top: 0.5rem;
    }
    .stChatMessage { border-radius: 10px; }
</style>
""", unsafe_allow_html=True)


# ── Session State Init ─────────────────────────────────────────────────────────
if "messages" not in st.session_state:
    st.session_state.messages = []
if "chain" not in st.session_state:
    st.session_state.chain = None
if "retriever" not in st.session_state:
    st.session_state.retriever = None
if "rag_ready" not in st.session_state:
    st.session_state.rag_ready = False


# ── Header ─────────────────────────────────────────────────────────────────────
st.markdown('<div class="main-header">📄 RAG Document Q&A System</div>', unsafe_allow_html=True)
st.markdown('<div class="sub-header">Upload PDFs and ask questions — powered by FAISS + LLaMA 3.1</div>', unsafe_allow_html=True)


# ── Sidebar ────────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("📁 Document Management")
    st.markdown("---")

    # PDF Upload
    uploaded_files = st.file_uploader(
        "Upload PDF files",
        type=["pdf"],
        accept_multiple_files=True,
        help="Upload one or more PDF documents to query"
    )

    if uploaded_files:
        # Save uploaded files to /data folder
        os.makedirs("data", exist_ok=True)
        for uploaded_file in uploaded_files:
            file_path = os.path.join("data", uploaded_file.name)
            with open(file_path, "wb") as f:
                f.write(uploaded_file.getbuffer())
        st.success(f"✅ {len(uploaded_files)} file(s) uploaded to /data")

    st.markdown("---")

    # Build Index Button
    if st.button("🔨 Build / Rebuild Index", use_container_width=True):
        with st.spinner("Processing PDFs and building FAISS index..."):
            try:
                # Force rebuild
                if os.path.exists("faiss_index"):
                    shutil.rmtree("faiss_index")
                chain, retriever = initialize_rag(force_rebuild=True)
                st.session_state.chain = chain
                st.session_state.retriever = retriever
                st.session_state.rag_ready = True
                st.session_state.messages = []  # clear chat on rebuild
                st.success("✅ Index built! Start asking questions.")
            except Exception as e:
                st.error(f"❌ Error: {str(e)}")

    # Load existing index
    if not st.session_state.rag_ready:
        if os.path.exists("faiss_index"):
            with st.spinner("Loading existing index..."):
                try:
                    chain, retriever = initialize_rag()
                    st.session_state.chain = chain
                    st.session_state.retriever = retriever
                    st.session_state.rag_ready = True
                except Exception as e:
                    st.error(f"❌ Error loading index: {str(e)}")

    st.markdown("---")

    # Status indicator
    if st.session_state.rag_ready:
        st.success("🟢 RAG System Ready")
    else:
        st.warning("🟡 Upload PDFs and click Build Index")

    # Show uploaded files
    st.markdown("### 📂 Files")
    if os.path.exists("data"):
        files = [f for f in os.listdir("data") if f.endswith(".pdf")]
        if files:
            for f in files:
                st.markdown(f"- 📄 {f}")
        else:
            st.markdown("_No PDFs found_")

    st.markdown("---")

    # Clear chat button
    if st.button("🗑️ Clear Chat History", use_container_width=True):
        st.session_state.messages = []
        st.rerun()


# ── Main Chat Area ─────────────────────────────────────────────────────────────
if not st.session_state.rag_ready:
    # Welcome screen
    st.info("👈 Upload your PDFs in the sidebar and click **Build / Rebuild Index** to get started.")
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 📤 Step 1\nUpload PDF files using the sidebar uploader")
    with col2:
        st.markdown("### 🔨 Step 2\nClick **Build / Rebuild Index** to process documents")
    with col3:
        st.markdown("### 💬 Step 3\nAsk any question about your documents!")

else:
    # Display chat history
    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Show sources if present
            if "sources" in message and message["sources"]:
                with st.expander("📎 Sources"):
                    for src in set(message["sources"]):
                        st.markdown(f'<div class="source-box">📄 {src}</div>',
                                    unsafe_allow_html=True)

    # Chat input
    if prompt := st.chat_input("Ask a question about your documents..."):

        # Add user message
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.markdown(prompt)

        # Generate answer
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                try:
                    # Get answer from chain
                    answer = st.session_state.chain.invoke(prompt)

                    # Get source documents
                    docs = st.session_state.retriever.invoke(prompt)
                    sources = [doc.metadata.get("source", "unknown") for doc in docs]

                    st.markdown(answer)

                    # Show sources
                    with st.expander("📎 Sources"):
                        for src in set(sources):
                            st.markdown(f'<div class="source-box">📄 {src}</div>',
                                        unsafe_allow_html=True)

                    # Save to history
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": answer,
                        "sources": sources
                    })

                except Exception as e:
                    st.error(f"❌ Error generating answer: {str(e)}")