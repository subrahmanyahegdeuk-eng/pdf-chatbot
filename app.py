import streamlit as st
import tempfile
from datetime import datetime
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableParallel

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chat with your PDF")
st.caption("Upload one or more PDFs on the left, then ask questions below.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    model_choice = st.selectbox("Choose model", ["gpt-3.5-turbo", "gpt-4"])
    uploaded_files = st.file_uploader("Upload PDFs", type="pdf", accept_multiple_files=True)
    st.caption("Get your API key at platform.openai.com")

@st.cache_resource(show_spinner="Reading your PDFs...")
def build_chain(file_data_list, api_key, model_choice):
    all_chunks = []
    for filename, file_bytes in file_data_list:
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(file_bytes)
            tmp_path = tmp.name
        pages = PyPDFLoader(tmp_path).load()
        for page in pages:
            page.metadata["source_file"] = filename
        chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
        all_chunks.extend(chunks)
    vectorstore = Chroma.from_documents(all_chunks, OpenAIEmbeddings(openai_api_key=api_key))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant. Answer the question using only the context below.
If the answer is not in the context, say "I could not find that in the document."

Context:
{context}

Question: {question}
""")
    llm = ChatOpenAI(openai_api_key=api_key, model=model_choice, temperature=0)
    chain = ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())
    return chain, retriever

def format_chat_for_download(messages):
    lines = [f"PDF Chatbot — Conversation export\n{datetime.now().strftime('%Y-%m-%d %H:%M')}\n"]
    lines.append("=" * 50 + "\n")
    for msg in messages:
        role = "You" if msg["role"] == "user" else "Assistant"
        lines.append(f"{role}:\n{msg['content']}\n")
        lines.append("-" * 30 + "\n")
    return "\n".join(lines)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if api_key and uploaded_files:
    file_data_list = tuple((f.name, f.read()) for f in uploaded_files)
    chain, retriever = build_chain(file_data_list, api_key, model_choice)
    pdf_names = [f[0] for f in file_data_list]
    st.info(f"Loaded {len(pdf_names)} PDF(s): {', '.join(pdf_names)}")
    if st.session_state.messages:
        chat_text = format_chat_for_download(st.session_state.messages)
        st.download_button("Download chat history", data=chat_text, file_name="chat_history.txt", mime="text/plain")
    user_input = st.chat_input("Ask something about your PDF(s)...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(user_input)
                source_docs = retriever.invoke(user_input)
                sources_seen = set()
                source_lines = []
                for doc in source_docs:
                    filename = doc.metadata.get("source_file", "Unknown file")
                    page = doc.metadata.get("page", "?")
                    source_ref = f"{filename} — page {int(page) + 1}"
                    if source_ref not in sources_seen:
                        sources_seen.add(source_ref)
                        source_lines.append(source_ref)
                st.write(answer)
                with st.expander("Sources used"):
                    for source in source_lines:
                        st.write(f"- {source}")
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to get started.")
    elif not uploaded_files:
        st.info("Upload at least one PDF in the sidebar to get started.")
