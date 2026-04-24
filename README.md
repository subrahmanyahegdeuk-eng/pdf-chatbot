PDF Chatbot — RAG Application

A conversational AI app that lets you upload multiple PDFs and ask questions about them in natural language. Built with LangChain, ChromaDB, and Streamlit.

Live demo
[Add your Streamlit URL here once deployed]

Features
- Multi-PDF support — upload and query multiple documents at once
- Source citations — every answer shows which page and document it came from
- Model selector — switch between GPT-3.5 and GPT-4
- Download chat history — export your full conversation as a text file

How it works
1. PDFs are split into chunks using LangChain text splitters
2. Chunks are embedded and stored in ChromaDB vector database
3. User questions retrieve the most relevant chunks via semantic search
4. GPT generates answers grounded only in the retrieved context

 Tech stack
Python · LangChain · ChromaDB · OpenAI · Streamlit

Run locally
pip install -r requirements.txt
streamlit run app.py

About
Built as part of my AI engineering portfolio while transitioning into the AI field.
MSc Advanced Computer Science — targeting AI Engineer roles in New Zealand.
