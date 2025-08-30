
# 📘 StudyMate – LearnHub

StudyMate transforms your **PDF textbooks and notes** into an **interactive conversational tutor**. Upload your PDFs, ask questions, and get trustworthy AI-generated answers with context.  

---

## 🚀 Features  
- 📑 **PDF Upload & Processing** – Upload textbooks, notes, or research papers.  
- 🤖 **AI Q&A** – Ask questions directly from your documents.  
- ⚡ **AI Summaries** – One-click PDF summarization.  
- 💬 **Chat History** – Save & revisit past Q&A sessions.  
- 🌙☀ **Dark/Light Mode** – Toggle between modes for better user experience.  
- 🔍 **Context-Aware Search** – Powered by embeddings + FAISS.  

---

## 🛠️ Tech Stack  
- **Frontend:** Streamlit  
- **Backend:** Python  
- **AI/ML:** Sentence Transformers, IBM Watsonx  
- **Database:** FAISS (vector search)  
- **File Processing:** PyMuPDF  

---

## 📂 Project Structure  
```
STUDYMATE/
│── app.py                # Main Streamlit app  
│── embedding_search.py   # FAISS-based search  
│── llm_answering.py      # AI question answering  
│── pdf_processor.py      # PDF text extraction  
│── hackaton.py           # Experimental/extra features  
│── test_ibm_token.py     # Token verification  
│── requirements.txt      # Dependencies  
│── .env                  # API keys (not shared in GitHub)
```

---

## ⚙️ Installation  

1. Clone the repo:  
   ```bash
   git clone https://github.com/indla-ramya-sri/studymate-learnhub.git
   cd studymate-learnhub
   ```

2. Create & activate a virtual environment: 
   ```bash
   python -m venv venv
   source venv/bin/activate   # On Mac/Linux  
   venv\Scripts\activate      # On Windows
   ```

3. Install dependencies:  
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:  
   ```bash
   streamlit run app.py
   ```

---

## 🔑 Environment Variables  
Create a `.env` file in the root folder with:  
```
LOCAL_MODEL_NAME=google/flan-t5-large
 
```

---

## 🎯 Hackathon Pitch  
StudyMate solves the problem of **boring static PDFs** by turning them into **interactive AI tutors**.  
It helps students:  
- Save study time ⏳  
- Improve understanding 📖  
- Get instant, reliable answers ✅  

---

## 👥 Team  
- Indla Ramya Sri 
- Komaraju Pavani
- Jidugu Anusha Lavanya  
