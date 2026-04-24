import streamlit as st
import tempfile
from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough

st.set_page_config(page_title="PDF Chatbot", layout="wide")
st.title("Chat with your PDF")
st.caption("Upload a PDF on the left, then ask questions below.")

with st.sidebar:
    st.header("Setup")
    api_key = st.text_input("OpenAI API Key", type="password", placeholder="sk-...")
    uploaded_file = st.file_uploader("Upload a PDF", type="pdf")
    st.caption("Get your API key at platform.openai.com")

@st.cache_resource(show_spinner="Reading your PDF...")
def build_chain(file_bytes, api_key):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(file_bytes)
        tmp_path = tmp.name
    pages = PyPDFLoader(tmp_path).load()
    chunks = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200).split_documents(pages)
    vectorstore = Chroma.from_documents(chunks, OpenAIEmbeddings(openai_api_key=api_key))
    retriever = vectorstore.as_retriever(search_kwargs={"k": 4})
    prompt = ChatPromptTemplate.from_template("""
Answer the question using only the context below.
If the answer is not in the context, say "I could not find that in the document."
Context: {context}
Question: {question}
""")
    llm = ChatOpenAI(openai_api_key=api_key, model="gpt-3.5-turbo", temperature=0)
    return ({"context": retriever, "question": RunnablePassthrough()} | prompt | llm | StrOutputParser())

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

if api_key and uploaded_file:
    chain = build_chain(uploaded_file.read(), api_key)
    user_input = st.chat_input("Ask something about your PDF...")
    if user_input:
        st.session_state.messages.append({"role": "user", "content": user_input})
        with st.chat_message("user"):
            st.write(user_input)
        with st.chat_message("assistant"):
            with st.spinner("Thinking..."):
                answer = chain.invoke(user_input)
                st.write(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})
else:
    if not api_key:
        st.info("Enter your OpenAI API key in the sidebar to get started.")
    elif not uploaded_file:
        st.info("Upload a PDF in the sidebar to get started.")
