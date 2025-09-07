# app.py
import streamlit as st
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
    st.session_state.quiz_data = None

# --- Sidebar: Upload PDFs ---
st.sidebar.header("📚 Upload Study Material")
uploaded_file = st.sidebar.file_uploader("Upload a PDF", type=["pdf"])
if uploaded_file:
    pdf_handler.save_pdf(uploaded_file)
    st.sidebar.success(f"Uploaded: {uploaded_file.name}")

# --- Section 1: Summarize PDFs ---
st.subheader("📝 Summarize PDFs")
if st.button("Generate Summary"):
    with st.spinner("Summarizing..."):
        context = pdf_handler.get_context("summary", top_k=5)
        summary = chatbot.summarize(context)
        st.write(summary)

# --- Section 2: Interactive Quiz ---
st.subheader("❓ Generate Interactive Quiz")
num_questions = st.slider("Number of Questions per Chunk", min_value=3, max_value=10, value=5)
if st.button("Generate Quiz"):
    with st.spinner("Generating quiz..."):
        context = pdf_handler.get_context("quiz", top_k=5)
        quiz = chatbot.generate_quiz(context, num_questions=num_questions)
        st.session_state.quiz_data = quiz
        st.write(quiz)

# --- Section 3: Chat Interface ---
st.subheader("💬 Ask Questions About Your PDFs")
user_input = st.text_input("Ask a question about your uploaded PDFs:")

if st.button("Ask") and user_input.strip():
    with st.spinner("Generating response..."):
        # Detect user language
        user_lang = translator.detect_language(user_input)
        # Translate to English for the model
        translated_input = translator.translate_to_english(user_input)
        # Get relevant PDF context
        context = pdf_handler.get_context(translated_input)
        # Generate model response
        response = chatbot.ask(translated_input, context)
        # If user asked in Arabic, translate response back to Arabic
        if user_lang == "ar":
            final_response = translator.translate_to_arabic(response)
        else:
            final_response = response
        # Save to chat history
        st.session_state.history.append((user_input, final_response))

# --- Display Chat History ---
if st.session_state.history:
    st.subheader("Chat History")
    for user_q, bot_a in st.session_state.history:
        st.markdown(f"**You:** {user_q}")
        st.markdown(f"**AthenaPDF:** {bot_a}")
