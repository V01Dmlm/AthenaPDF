# AthenaPDF 📚🤖

**AthenaPDF** is a university-oriented AI study companion that allows students to upload PDFs, ask questions, get summaries, and take interactive quizzes — all in one streamlined interface. It supports **Arabic ↔ English translation** and uses a local **Mistral 7B Instruct model** for fast, offline question answering.

---

## Features

### 1️⃣ PDF Upload & Processing
- Upload multiple PDFs at once.
- PDFs are split into chunks and stored persistently for fast retrieval.
- No need to reprocess PDFs for multiple questions or quizzes.

### 2️⃣ Chat Q&A
- Ask questions about your uploaded PDFs.
- Automatically detects the question language (Arabic or English).
- Provides context-aware answers using the Mistral 7B Instruct model.
- Translates answers back to the student’s language if needed.
- Chat history is preserved in the session.

### 3️⃣ Summarization
- Click **“📝 Summarize Uploaded PDFs”** to get a concise, student-friendly summary of all uploaded PDFs.
- Ideal for quick revision or reviewing large documents.

### 4️⃣ Interactive Quiz
- Click **“❓ Generate Interactive Quiz from PDFs”** to generate multiple-choice questions directly from uploaded PDFs.
- Students can select answers and get **instant feedback**.
- Scores are tracked and displayed at the end of the quiz.
- Supports multiple PDFs combined into a single quiz.

### 5️⃣ Language Support
- Automatic Arabic ↔ English translation.
- Works for bilingual students or Arabic-only PDFs/questions.

---

## Tech Stack

### Backend
- **LLM:** Mistral 7B Instruct (quantized Q4_K_M, GGUF format)  
- **Python Libraries:**
  - Transformers, torch, ctransformers, llama-cpp-python
  - PDF Handling: PyPDF2, pdfplumber, fitz, unstructured
  - Translation: deep-translator, langdetect
  - Vector Search: FAISS, ChromaDB
  - Utilities: numpy, pandas, scikit-learn, tqdm, loguru

### Frontend
- **Streamlit** for interactive web UI.
- Chat interface with history preservation.
- Interactive quizzes with radio buttons and instant feedback.
- Buttons for PDF summarization and quiz generation.

