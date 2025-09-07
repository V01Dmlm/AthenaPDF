import streamlit as st
import re
from backend.translator import Translator
from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler

st.set_page_config(page_title="AthenaPDF", layout="wide")

# Initialize backends
translator = Translator()
chatbot = ChatBot(model_path="models/mistral-7b-instruct-q4_k_m.gguf")
pdf_handler = PDFHandler()

# Sidebar: Upload PDF
st.sidebar.header("📚 Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_handler.save_pdf(uploaded_file)
    st.sidebar.success(f"✅ PDF '{uploaded_file.name}' uploaded and processed!")

# Chat UI
st.title("🤖 AthenaPDF – Your Study Companion")

if "history" not in st.session_state:
    st.session_state.history = []

# 1️⃣ Summarize PDFs
if st.button("📝 Summarize Uploaded PDFs"):
    if pdf_handler.chunks:
        all_text = " ".join(pdf_handler.chunks)
        summary = chatbot.summarize(all_text)
        st.markdown(f"**Summary:**\n{summary}")
    else:
        st.warning("❌ No PDFs uploaded yet.")

# 2️⃣ Interactive Quiz
if st.button("❓ Generate Interactive Quiz from PDFs"):
    if pdf_handler.chunks:
        all_text = " ".join(pdf_handler.chunks)
        raw_quiz = chatbot.generate_quiz(all_text)

        # Parse quiz
        questions = re.split(r'Q\d+\.', raw_quiz)[1:]  # skip text before Q1
        quiz_data = []

        for q in questions:
            lines = q.strip().split("\n")
            question_text = lines[0]
            options = {}
            answer = ""
            for line in lines[1:]:
                if re.match(r'[A-D]\.', line):
                    key = line[0]
                    options[key] = line[3:].strip()
                elif line.lower().startswith("answer:"):
                    answer = line.split(":")[1].strip()
            quiz_data.append({
                "question": question_text,
                "options": options,
                "answer": answer
            })

        # Display interactive quiz
        st.markdown("### 📋 Quiz")
        score = 0
        for idx, q in enumerate(quiz_data):
            st.markdown(f"**Q{idx+1}: {q['question']}**")
            selected = st.radio(f"Select an answer for Q{idx+1}", options=list(q['options'].values()), key=idx)
            correct_answer_text = q['options'].get(q['answer'], "")
            if st.button(f"Check Answer for Q{idx+1}", key=f"check_{idx}"):
                if selected == correct_answer_text:
                    st.success("✅ Correct!")
                    score += 1
                else:
                    st.error(f"❌ Incorrect! Correct answer: {correct_answer_text}")

        st.markdown(f"### 🏆 Your Score: {score} / {len(quiz_data)}")
    else:
        st.warning("❌ No PDFs uploaded yet.")

# 3️⃣ Ask a question
user_input = st.chat_input("Ask a question about your uploaded PDFs...")
if user_input:
    # Translate to English if needed
    translated_input = translator.translate_to_english(user_input)

    # Get relevant context from PDFs
    context = pdf_handler.get_context(translated_input)

    # Generate response
    response = chatbot.ask(translated_input, context)

    # Translate back to Arabic if needed
    final_response = translator.translate_to_arabic(response)

    # Save to session history
    st.session_state.history.append((user_input, final_response))

# 4️⃣ Display chat history
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
