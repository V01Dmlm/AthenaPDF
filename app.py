# app.py
import streamlit as st
import re
from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler
from backend.translator import Translator

# --- Page Setup ---
st.set_page_config(page_title="AthenaPDF", layout="wide")
st.title("🤖 AthenaPDF – Your Study Companion")

# --- Initialize Backends ---
translator = Translator()
chatbot = ChatBot(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
pdf_handler = PDFHandler()

# --- Session State ---
if "history" not in st.session_state:
    st.session_state.history = []

if "quiz_data" not in st.session_state:
    st.session_state.quiz_data = []

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📚 Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_handler.save_pdf(uploaded_file)
    st.sidebar.success(f"✅ PDF '{uploaded_file.name}' uploaded!")

# --- Section 1: Summarize PDFs ---
st.subheader("📝 Summarize PDFs")
if st.button("Generate Summary"):
    if pdf_handler.chunks:
        with st.spinner("Generating summary..."):
            all_text = " ".join(pdf_handler.chunks)
            summary = chatbot.summarize(all_text)
            st.markdown(f"**Summary:**\n{summary}")
            st.download_button("📥 Download Summary", summary, file_name="summary.txt")
    else:
        st.warning("❌ No PDFs uploaded yet.")

# --- Section 2: Interactive Quiz ---
st.subheader("❓ Generate Interactive Quiz")
num_questions = st.slider("Number of Questions per Chunk", min_value=3, max_value=10, value=5)
if st.button("Generate Quiz"):
    if pdf_handler.chunks:
        with st.spinner("Generating quiz..."):
            all_text = " ".join(pdf_handler.chunks)
            raw_quiz = chatbot.generate_quiz(all_text, num_questions=num_questions)

            # Parse quiz into structured format
            questions = re.split(r'Q\d+\.', raw_quiz)[1:]
            quiz_data = []
            for q in questions:
                lines = [line.strip() for line in q.strip().split("\n") if line.strip()]
                if not lines:
                    continue
                question_text = lines[0]
                options = {}
                answer = ""
                for line in lines[1:]:
                    if re.match(r'[A-D]\.', line):
                        options[line[0]] = line[2:].strip()
                    elif line.lower().startswith("answer:"):
                        answer = line.split(":")[1].strip()
                if question_text and options and answer:
                    quiz_data.append({"question": question_text, "options": options, "answer": answer})

            st.session_state.quiz_data = quiz_data
            st.success(f"✅ Quiz Generated! {len(quiz_data)} questions")

# Display quiz if generated
if st.session_state.quiz_data:
    st.markdown("### 📋 Quiz")
    score = 0
    for idx, q in enumerate(st.session_state.quiz_data):
        st.markdown(f"**Q{idx+1}: {q['question']}**")
        selected = st.radio(f"Select an answer for Q{idx+1}", options=list(q['options'].values()), key=idx)
        correct_answer_text = q['options'].get(q['answer'], "")
        if st.button(f"Check Answer for Q{idx+1}", key=f"check_{idx}"):
            if selected == correct_answer_text:
                st.success("✅ Correct!")
                score += 1
            else:
                st.error(f"❌ Incorrect! Correct answer: {correct_answer_text}")
    st.markdown(f"### 🏆 Your Score: {score} / {len(st.session_state.quiz_data)}")
    # Optional download
    quiz_text = "\n".join(
        [f"Q{idx+1}: {q['question']}\n" +
         "\n".join([f"{opt}. {txt}" for opt, txt in q['options'].items()]) +
         f"\nAnswer: {q['answer']}\n" for idx, q in enumerate(st.session_state.quiz_data)]
    )
    st.download_button("📥 Download Quiz", quiz_text, file_name="quiz.txt")

# --- Section 3: Chat with AthenaPDF ---
st.subheader("💬 Chat with AthenaPDF")
user_input = st.chat_input("Ask a question about your uploaded PDFs...")
if user_input:
    with st.spinner("Generating response..."):
        # Translate to English for the model
        translated_input = translator.translate_to_english(user_input)
        # Get relevant PDF context
        context = pdf_handler.get_context(translated_input)
        # Generate model response
        response = chatbot.ask(translated_input, context)
        # Translate response back to Arabic
        final_response = translator.translate_to_arabic(response)
        # Save to chat history
        st.session_state.history.append((user_input, final_response))

# --- Section 4: Display Chat History ---
for user_msg, bot_msg in st.session_state.history:
    with st.chat_message("user"):
        st.write(user_msg)
    with st.chat_message("assistant"):
        st.markdown(bot_msg)
