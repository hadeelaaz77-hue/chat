
# 🤖 Smart Multilingual Chatbot – AI-First System with Gemini, LangChain & FAISS

This project is a fully functional AI-first chatbot that can **understand multiple languages**, **perform semantic search**, and **generate intelligent answers** using **LLMs like Gemini**. It is designed with the mindset of building tools *with* AI, not just on top of it.

> 🧠 Prompt engineering • 🌍 Language detection & translation • 🔍 Vector search • 💬 LLM reasoning • 📊 Feedback loop

---

## 🎯 Why This Project?

The future of software isn't about building tools alone — it's about co-creating them with AI.

This project was built to explore:
- How to collaborate with LLMs to reason and respond
- How to convert unstructured text into searchable knowledge
- How to make the user experience feel alive and smart

---

## 🧠 Features

- 🌍 **Multilingual Input**: Arabic, English, Chinese, French, and more
- 🤖 **Automatic Translation** using NLLB (facebook/nllb-200-3.3B)
- 📚 **Semantic Search** with `intfloat/multilingual-e5-large` embeddings
- 🔍 **FAISS** vector indexing of institutional documents
- 💬 **LLM-Powered Answers** via Gemini + LangChain
- 📊 **Feedback System** (👍 / 👎) with context saved to MongoDB
- ⚡ **FastAPI backend** and **React frontend**

---

## 🧱 Tech Stack

| Layer        | Technology                                       |
|--------------|--------------------------------------------------|
| Frontend     | React                                            |
| Backend      | FastAPI                                          |
| Embeddings   | intfloat/multilingual-e5-large (HuggingFace)     |
| Vector Store | FAISS                                            |
| LLM          | Gemini (via LangChain)                           |
| Translation  | NLLB - facebook/nllb-200-3.3B                    |
| Database     | MongoDB (user feedback storage)                  |

---

## 🚀 How It Works

1. 📝 **User submits a question** in any language.
2. 🌍 The system **detects the language** and translates it into Arabic if needed.
3. 🔎 **Semantic embedding** is generated using multilingual E5 model.
4. 🧠 **FAISS** retrieves the most relevant document chunks.
5. 🤖 **LangChain + Gemini** generate an answer using the retrieved context.
6. 🌐 If the input was non-Arabic, the answer is translated back.
7. 📊 User feedback (👍/👎 + comment) is stored in MongoDB for evaluation.

---

## 📽️ Demo Video

[![Watch Demo](https://img.shields.io/badge/Demo-YouTube-red?logo=youtube)](https://youtu.be/your-demo-link-here)

> In the demo, I showcase the chatbot answering multilingual questions, reasoning over internal documents, and accepting feedback to close the loop.


---

## ✨ AI-First Design Philosophy

* Built with the idea that the AI is a **co-developer**, not just a component.
* LLMs are part of the **reasoning engine**, not a gimmick.
* All user actions feed into a **feedback loop** for learning and evaluation.

---

## 📎 This Project is Part of My AI Engineer Portfolio

This chatbot is one of the key projects featured in my personal portfolio as an aspiring **AI Engineer / Vibe Coder**. You can view my full resume below:

- 📄 [View My CV](https://drive.google.com/file/d/1G3oOK0NwSi_IOGRtZoZF4NSq4D6HAzYJ/view?usp=sharing)
- 🎓 BSc in Computer Science – Imam Mohammad Ibn Saud Islamic University  
- 🛠️ 400+ hours of AI training  
- 🧠 Focus: NLP, LLMs, semantic search, LangChain, RAG  
- 🏆 Selected for Talented Student Program – 4th place in Fikrathon Hackathon  
- 💼 Conducted workshops on NLP & RAG using LangChain + Gemini  
- 👩‍💻 Developer of AI-powered tools like [Samaa.tech](https://samaabot.com) and this chatbot
