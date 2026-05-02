<div align="center">

# 💬 DocChat
### Multilingual PDF Chatbot powered by Groq LLaMA

![Python](https://img.shields.io/badge/Python-3.9+-blue?style=for-the-badge&logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![LangChain](https://img.shields.io/badge/LangChain-000000?style=for-the-badge&logo=chainlink&logoColor=white)
![Groq](https://img.shields.io/badge/Groq-F55036?style=for-the-badge&logo=groq&logoColor=white)
![Free](https://img.shields.io/badge/API-100%25%20Free-00c851?style=for-the-badge)

**Upload multiple PDFs → Ask questions in natural language → Get cited answers instantly**

[🚀 Live Demo](#) · [📂 View Code](https://github.com/subrahmanyahegdeuk-eng/pdf-chatbot) · [🐛 Report Bug](#)

</div>

---

## ✨ Features

| Feature | Description |
|---------|-------------|
| 📄 **Multi-PDF** | Upload and query multiple documents simultaneously |
| 🔍 **Source citations** | Every answer shows exact page and document |
| 🌍 **Multilingual** | English · Arabic · German · French · Hindi · Spanish |
| 🤖 **Groq LLaMA** | Completely free — no paid API key needed |
| 💾 **Download** | Export full conversation as text file |
| 🔄 **Model selector** | Switch between LLaMA 70b and LLaMA 8b |

---

## 🏗️ How it works
📄 PDF Upload
↓
✂️  Split into 1000-char chunks (LangChain)
↓
🔢  Convert to vectors (HuggingFace sentence-transformers)
↓
🗄️  Store in ChromaDB vector database
↓
❓  User asks a question
↓
🔎  Semantic search finds top 4 relevant chunks
↓
🤖  Groq LLaMA generates answer from chunks
↓
✅  Answer + source citations displayed
> This architecture is called **RAG — Retrieval Augmented Generation**

---

## 🛠️ Tech Stack

<div align="center">

| Layer | Technology |
|-------|-----------|
| 🖥️ Frontend | Streamlit |
| 🔗 LLM Framework | LangChain |
| 🗄️ Vector Database | ChromaDB |
| 🔢 Embeddings | HuggingFace sentence-transformers |
| 🤖 LLM | Groq LLaMA 3.3 70b |
| 📄 PDF Processing | PyPDF |
| 🐍 Language | Python 3.9+ |

</div>

---

## 🚀 Quick Start

```bash
# Clone the repository
git clone https://github.com/subrahmanyahegdeuk-eng/pdf-chatbot.git
cd pdf-chatbot

# Install dependencies
pip install -r requirements.txt

# Run the app
streamlit run app.py
```

> Get your **free** Groq API key at [console.groq.com](https://console.groq.com)

---

## 🌍 Use Cases
⚖️  Legal          →  Query contracts and case documents
🏥  Healthcare     →  Search clinical guidelines and papers
💰  Finance        →  Analyse reports and statements
🎓  Education      →  Chat with textbooks and notes
🔬  Research       →  Query multiple papers simultaneously
---

## 📁 Project Structure
pdf-chatbot/
├── app.py              ← Main Streamlit application
├── requirements.txt    ← Python dependencies
└── README.md          ← This file
---

<div align="center">

## 👨‍💻 Author

**Subrahmanya Anant Hegde**

MSc Computer Science · University of Strathclyde

[![GitHub](https://img.shields.io/badge/GitHub-subrahmanyahegdeuk--eng-181717?style=for-the-badge&logo=github)](https://github.com/subrahmanyahegdeuk-eng)
[![Email](https://img.shields.io/badge/Email-subrahmanyahegdeuk%40gmail.com-D14836?style=for-the-badge&logo=gmail&logoColor=white)](mailto:subrahmanyahegdeuk@gmail.com)

*Built as part of an AI engineering portfolio*

</div>
