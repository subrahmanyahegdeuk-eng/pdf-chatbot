# PDF Chatbot — RAG Application

A conversational AI app that lets you upload any PDF and ask questions about it in natural language.

## How it works
1. PDF is split into chunks using LangChain text splitters
2. Chunks are converted to vectors and stored in ChromaDB
3. User questions retrieve the most relevant chunks via semantic search
4. GPT-3.5 generates answers grounded in the retrieved context

## Tech stack
Python · LangChain · ChromaDB · OpenAI · Streamlit

## Run locally
pip install -r requirements.txt
streamlit run app.py
