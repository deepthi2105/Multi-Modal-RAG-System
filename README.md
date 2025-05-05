# Multi-Modal-RAG-System

A Streamlit-based application that performs **Retrieval-Augmented Generation (RAG)** across multiple file types — including documents, images, and audio/video — using LangChain, OpenAI, Hugging Face, and Whisper.

---

## 🚀 Features

- 🔍 **Document QA**: Upload PDFs, DOCX, or text files and ask context-aware questions.
- 🖼️ **Image OCR**: Extract text from images and query them.
- 🎙️ **Audio/Video QA**: Transcribe media files using Whisper and run semantic search.
- 🧠 **Vector Search**: Powered by FAISS and OpenAI/Hugging Face embeddings.
- 🧾 **Source-aware Responses**: Displays page/source for each answer.

---

## 🧑‍💻 Tech Stack

| Category           | Tools / Libraries                                           |
|--------------------|-------------------------------------------------------------|
| LLM & RAG          | LangChain, OpenAI, Hugging Face, Whisper                    |
| Embeddings         | FAISS, Azure OpenAI, Hugging Face Transformers              |
| OCR                | pytesseract, Pillow                                         |
| File Handling      | PyPDF2, python-docx, python-dotenv                          |
| UI / App Layer     | Streamlit                                                   |
| Deployment         | Azure App Service, Streamlit Community Cloud                |

---

## 🌐 Live Demo

👉 **[Launch App on Streamlit Cloud](https://multi-modal-rag-system-rtqnyt7kj7ha8tgkv5fqhm.streamlit.app/)**

📌 **Note:**  
This project is deployed on **Azure Web App** (as mentioned in resume).  
Due to high memory requirements for `torch` and `whisper`, the public demo is hosted on **Streamlit Cloud** for stability.

---

