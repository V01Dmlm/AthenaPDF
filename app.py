# app.py (Dark Mode + Multi-PDF + Lazy Image Preview + Base64)
import os
from datetime import datetime
import streamlit as st
from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler
from backend.translator import Translator
import base64
import logging

logging.basicConfig(level=logging.INFO)

# ----------------- Initialize Models -----------------
translator = Translator()
chatbot = ChatBot(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
pdf_handler = PDFHandler(max_workers=6)  # parallel processing

# ----------------- Page Config -----------------
st.set_page_config(page_title="AthenaPDF", layout="wide", initial_sidebar_state="expanded")
st.markdown("<h1 style='text-align:center; color:#ffffff;'>🤖 AthenaPDF – Your Study Companion</h1>", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
st.sidebar.header("Settings")
user_color = st.sidebar.color_picker("User Bubble Color", "#1f77b4")  # dark blue
bot_color = st.sidebar.color_picker("Bot Bubble Color", "#ff7f0e")    # orange
text_color = st.sidebar.color_picker("Text Color", "#ffffff")          # white
chat_font_size = st.sidebar.number_input("Chat Font Size (px)", min_value=10, max_value=30, value=14)
chat_height = st.sidebar.number_input("Chat Box Height (px)", min_value=200, max_value=1000, value=600)

st.sidebar.header("Upload PDFs")
pdf_uploads = st.sidebar.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

# ----------------- Session State -----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

# ----------------- Helpers -----------------
def img_to_base64(img_path):
    """Convert image to base64 for inline HTML rendering."""
    with open(img_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    ext = img_path.split('.')[-1]
    return f"data:image/{ext};base64,{encoded}"

def render_chat():
    st.markdown(f"<div style='max-height:{chat_height}px; overflow-y:auto; padding:10px; border-radius:10px; background:#121212;'>", unsafe_allow_html=True)
    for i, chat in enumerate(st.session_state.history):
        # User bubble
        st.markdown(f"""
            <div style="
                background-color:{user_color};
                color:{text_color};
                padding:10px;
                margin:5px 0;
                border-radius:10px;
                text-align:{'right' if chat['user_lang']=='ar' else 'left'};
                direction:{'rtl' if chat['user_lang']=='ar' else 'ltr'};
                font-size:{chat_font_size}px;
            ">
                {chat['user']}
            </div>
        """, unsafe_allow_html=True)

        # Bot bubble
        bot_content = chat['bot_translated'] if chat['bot_translated'] else chat['bot']
        st.markdown(f"""
            <div style="
                background-color:{bot_color};
                color:{text_color};
                padding:10px;
                margin:5px 0;
                border-radius:10px;
                text-align:{'right' if chat['bot_lang']=='ar' else 'left'};
                direction:{'rtl' if chat['bot_lang']=='ar' else 'ltr'};
                font-size:{chat_font_size}px;
            ">
                {bot_content}
            </div>
        """, unsafe_allow_html=True)

        # Toggle language per message
        toggle_key = f"toggle_{i}"
        if st.button("Switch Language", key=toggle_key):
            if chat['bot_translated'] is None:
                if chat['bot_lang'] == "en":
                    chat['bot_translated'] = translator.translate_to_arabic(chat['bot'])
                    chat['bot_lang'] = "ar"
                else:
                    chat['bot_translated'] = translator.translate_to_english(chat['bot'])
                    chat['bot_lang'] = "en"
            else:
                chat['bot_translated'] = None
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Handle PDF Uploads -----------------
if pdf_uploads:
    for pdf_file in pdf_uploads:
        if pdf_file.name not in st.session_state.uploaded_pdfs:
            st.session_state.uploaded_pdfs.append(pdf_file.name)
            try:
                pdf_handler.save_pdf(pdf_file)
                st.success(f"Uploaded: {pdf_file.name}")
            except Exception as e:
                st.error(f"Failed to process PDF {pdf_file.name}: {e}")

# ----------------- Bottom Input Bar -----------------
st.markdown("""
<style>
.bottom-bar {
    position: fixed;
    bottom: 0;
    left: 10px;
    width: calc(100% - 20px);
    display: flex;
    align-items: center;
    gap: 5px;
    padding: 5px;
    border-top: 1px solid #444;
    background: #1e1e1e;
    border-radius: 10px 10px 0 0;
}
.input-box {
    flex-grow: 1;
    padding: 10px;
    border-radius: 10px;
    border: 1px solid #444;
    outline: none;
    background: #2c2c2c;
    color: #ffffff;
}
.button-ask {
    padding: 10px;
    border-radius: 10px;
    border: none;
    background-color: #ff7f0e;
    color: #ffffff;
    cursor: pointer;
}
</style>
""", unsafe_allow_html=True)

st.markdown("<div class='bottom-bar'>", unsafe_allow_html=True)
user_input = st.text_input("", placeholder="Type your question here…", label_visibility="hidden", key="chat_input")
ask_button_clicked = st.button("Ask", key="ask_button", help="Send your question")
st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Handle Chat -----------------
if ask_button_clicked and user_input:
    user_lang = translator.detect_language(user_input)
    translated_input = translator.translate_to_english(user_input)
    
    # Get relevant context
    context = pdf_handler.get_context(translated_input, top_k=3, pdf_files=st.session_state.uploaded_pdfs)
    response = chatbot.ask(translated_input, context)
    
    # Lazy image embedding (only if user asks for graphs/images)
    images_html = ""
    if any(keyword in user_input.lower() for keyword in ["graph", "image", "figure"]):
        relevant_images = []
        for pdf_file in st.session_state.uploaded_pdfs:
            try:
                pdf_images = pdf_handler.get_images_for_pdf(pdf_file)
                relevant_images.extend(pdf_images)
            except Exception as e:
                logging.warning(f"Failed to load images from {pdf_file}: {e}")
        for img_path in relevant_images:
            images_html += f"<br><img src='{img_to_base64(img_path)}' style='max-width:80%; display:block; margin:auto; border-radius:10px;'>"
    
    final_response = response + images_html
    if user_lang == "ar":
        final_response = translator.translate_to_arabic(final_response)
    
    st.session_state.history.append({
        "user": user_input,
        "bot": final_response,
        "user_lang": user_lang,
        "bot_lang": "ar" if user_lang == "ar" else "en",
        "bot_translated": None,
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# ----------------- Render Chat -----------------
render_chat()
