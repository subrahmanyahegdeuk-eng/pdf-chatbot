import streamlit as st
import tempfile
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

st.set_page_config(page_title="DocChat", page_icon="💬", layout="wide")

st.markdown("""
<style>
    .stApp { background-color: #fafafa; }
    .doc-header {
        display: flex; align-items: center; gap: 12px;
        padding: 0 0 20px 0; border-bottom: 2px solid #6C63FF;
        margin-bottom: 24px;
    }
    .doc-title { font-size: 22px; font-weight: 700; color: #1a1a2e; margin: 0; }
    .doc-sub { font-size: 13px; color: #888888; margin: 0; }
    .file-pill {
        display: inline-flex; align-items: center; gap: 6px;
        background: #f0edff; color: #6C63FF;
        border: 1px solid #d4ceff; border-radius: 20px;
        padding: 4px 12px; font-size: 12px; font-weight: 500; margin: 3px;
    }
    .user-bubble {
        background: #6C63FF; color: white;
        padding: 12px 16px; border-radius: 18px 18px 4px 18px;
        max-width: 75%; margin-left: auto; margin-bottom: 8px;
        font-size: 14px; line-height: 1.6;
        box-shadow: 0 2px 8px rgba(108,99,255,0.25);
    }
    .assistant-bubble {
        background: white; color: #1a1a2e;
        padding: 12px 16px; border-radius: 18px 18px 18px 4px;
        max-width: 75%; margin-bottom: 8px;
        font-size: 14px; line-height: 1.6;
        border: 1px solid #eeeeee;
        box-shadow: 0 1px 4px rgba(0,0,0,0.05);
    }
    .source-tag {
        display: inline-block; background: #f5f5f5; color: #666666;
        border-radius: 4px; padding: 2px 8px; font-size: 11px; margin: 2px;
    }
    .feature-card {
        background: white; border: 1px solid #eeeeee;
        border-radius: 12px; padding: 20px; margin-bottom: 12px;
    }
    .sb-head {
        color: #6C63FF; font-size: 13px; font-weight: 700;
        text-transform: uppercase; letter-spacing: 1px;
        margin: 16px 0 8px 0; padding-bottom: 6px;
        border-bottom: 1px solid #e0dcff;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="doc-header">
    <div style="width:44px;height:44px;background:#6C63FF;border-radius:10px;
    display:flex;align-items:center;justify-content:center;font-size:22px">💬</div>
    <div>
        <p class="doc-title">DocChat</p>
        <p class="doc-sub">Ask questions about your documents in natural language</p>
    </div>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.markdown('<p class="sb-head">🔑 API Key</p>', unsafe_allow_html=True)
    api_key = st.text_input("Groq API Key", type="password",
        placeholder="gsk_...", label_visibility="collapsed")
    st.caption("Free key at console.groq.com")

    st.markdown('<p class="sb-head">📎 Documents</p>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf",
        accept_multiple_files=True, label_visibility="collapsed")

    st.markdown('<p class="sb-head">⚙️ Settings</p>', unsafe_allow_html=True)
    model_choice = st.selectbox("Model",
        ["llama-3.3-70b-versatile", "llama3-8b-8192"])
    language = st.selectbox("Language",
        ["English", "Arabic", "German", "French", "Hindi", "Spanish"])

    st.markdown("---")
    if st.button("🗑️ Clear conversation", use_container_width=True):
        st.session_state.messages = []
        st.rerun()

    st.markdown('<p class="sb-head">💡 Try asking</p>', unsafe_allow_html=True)
    st.caption("Summarise this document")
    st.caption("What are the key findings?")
    st.caption("List all recommendations")
    st.caption("What does it say about X?")

@st.cache_resource(show_spinner="Reading your documents...")
def build_chain(file_data_list, api_key, model_choice, language):
    all_chunks = []
    for filename, file_bytes in file_data_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        for page in pages:
            page.metadata["source_file"] = filename
        chunks = RecursiveCharacterTextSplitter(
            chunk_size=1000, chunk_overlap=200).split_documents(pages)
        all_chunks.extend(chunks)

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2")
    vectorstore = Chroma.from_documents(all_chunks, embeddings)
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})

    lang_note = f"Always respond in {language}." if language != "English" else ""
    prompt = ChatPromptTemplate.from_template(f"""
You are a helpful assistant. {lang_note}
Answer using only the context below.
If not found say "I could not find that in the document."
Context: {{context}}
Question: {{question}}
""")
    llm = ChatGroq(api_key=api_key, model=model_choice, temperature=0)
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    return chain, retriever

if "messages" not in st.session_state:
    st.session_state.messages = []

if api_key and uploaded_files:
    file_data_list = tuple((f.name, f.read()) for f in uploaded_files)
    chain, retriever = build_chain(
        file_data_list, api_key, model_choice, language)

    file_pills = "".join([
        f'<span class="file-pill">📄 {f[0]}</span>' for f in file_data_list])
    st.markdown(
        f'<div style="margin-bottom:20px">{file_pills}'
        f'<span style="font-size:12px;color:#888;margin-left:8px">'
        f'{len(file_data_list)} document(s) · {language}</span></div>',
        unsafe_allow_html=True)

    for msg in st.session_state.messages:
        if msg["role"] == "user":
            st.markdown(
                f'<div class="user-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True)
        else:
            st.markdown(
                f'<div class="assistant-bubble">{msg["content"]}</div>',
                unsafe_allow_html=True)
            if "sources" in msg and msg["sources"]:
                sources_html = "".join([
                    f'<span class="source-tag">📄 {s}</span>'
                    for s in msg["sources"]])
                st.markdown(
                    f'<div style="margin-bottom:12px">{sources_html}</div>',
                    unsafe_allow_html=True)

    user_input = st.chat_input(
        f"Ask something about your {len(file_data_list)} document(s)...")

    if user_input:
        st.session_state.messages.append(
            {"role": "user", "content": user_input})
        st.markdown(
            f'<div class="user-bubble">{user_input}</div>',
            unsafe_allow_html=True)
        with st.spinner("Thinking..."):
            answer = chain.invoke(user_input)
            source_docs = retriever.invoke(user_input)
            sources_seen = set()
            source_lines = []
            for doc in source_docs:
                ref = (f"{doc.metadata.get('source_file','Unknown')} — "
                       f"p.{int(doc.metadata.get('page',0))+1}")
                if ref not in sources_seen:
                    sources_seen.add(ref)
                    source_lines.append(ref)

        st.markdown(
            f'<div class="assistant-bubble">{answer}</div>',
            unsafe_allow_html=True)
        sources_html = "".join([
            f'<span class="source-tag">📄 {s}</span>' for s in source_lines])
        st.markdown(
            f'<div style="margin-bottom:12px">{sources_html}</div>',
            unsafe_allow_html=True)
        st.session_state.messages.append({
            "role": "assistant", "content": answer, "sources": source_lines})

    if st.session_state.messages:
        chat_text = "\n\n".join([
            f"{m['role'].upper()}:\n{m['content']}"
            for m in st.session_state.messages])
        st.download_button("💾 Download conversation", data=chat_text,
            file_name=f"chat_{datetime.now().strftime('%Y%m%d')}.txt",
            mime="text/plain")

else:
    if not api_key and not uploaded_files:
        col1, col2, col3 = st.columns(3)
        features = [
            ("📄", "Multi-document", "Upload multiple PDFs and query them all at once"),
            ("🔍", "Source citations", "Every answer shows which page it came from"),
            ("🌍", "Multilingual", "Answers in English, Arabic, German, French, Hindi or Spanish"),
        ]
        for col, (icon, title, desc) in zip([col1, col2, col3], features):
            col.markdown(f"""
<div class="feature-card">
    <div style="font-size:32px;margin-bottom:10px">{icon}</div>
    <strong style="font-size:15px;color:#1a1a2e">{title}</strong>
    <p style="color:#888;font-size:13px;margin-top:6px;line-height:1.5">{desc}</p>
</div>""", unsafe_allow_html=True)
        st.markdown("""
<div style="text-align:center;padding:60px 20px">
    <div style="font-size:52px;margin-bottom:12px">💬</div>
    <h3 style="color:#1a1a2e;margin:0 0 8px">Start chatting with your documents</h3>
    <p style="color:#888;font-size:14px">Add your Groq API key and upload PDFs in the sidebar</p>
</div>""", unsafe_allow_html=True)
    elif not api_key:
        st.info("Enter your Groq API key in the sidebar — free at console.groq.com")
    elif not uploaded_files:
        st.info("Upload at least one PDF in the sidebar to start chatting")
