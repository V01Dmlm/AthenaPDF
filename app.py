# app.py (Streamlit Full Version + Dark Mode + PDF + Translation + Instant Language Switch)
import os
import base64
import logging
from datetime import datetime
import streamlit as st

from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler
from backend.translator import Translator

logging.basicConfig(level=logging.INFO)

# ----------------- Initialize Models -----------------
translator = Translator()
chatbot = ChatBot(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf", max_tokens=512)
pdf_handler = PDFHandler(max_workers=6)

# ----------------- Page Config -----------------
st.set_page_config(page_title="AthenaPDF", layout="wide", initial_sidebar_state="collapsed")

# ----------------- Page Header -----------------
st.markdown("""
<h1 style='text-align:center; color:#ffffff; font-family:sans-serif;'>
🤖 AthenaPDF – Your Smart Study Buddy
</h1>
""", unsafe_allow_html=True)

# ----------------- Sidebar -----------------
with st.sidebar.expander("⚙️ Settings", expanded=True):
    user_color = st.color_picker("User Bubble Color", "#1f77b4")
    bot_color = st.color_picker("Bot Bubble Color", "#ff7f0e")
    text_color = st.color_picker("Text Color", "#ffffff")
    chat_font_size = st.slider("Chat Font Size (px)", 12, 28, 14, 1)
    chat_height = st.slider("Chat Box Height (px)", 250, 1000, 600, 25)
    top_k_context = st.slider("Number of PDF Context Chunks", 1, 10, 3)

# ----------------- PDF Upload Section -----------------
st.sidebar.header("📄 Upload PDFs")
pdf_uploads = st.sidebar.file_uploader("Choose PDFs", type=["pdf"], accept_multiple_files=True)

# ----------------- Session State -----------------
if "history" not in st.session_state:
    st.session_state.history = []

if "uploaded_pdfs" not in st.session_state:
    st.session_state.uploaded_pdfs = []

# ----------------- Helpers -----------------
def img_to_base64(img_path):
    with open(img_path, "rb") as f:
        data = f.read()
    encoded = base64.b64encode(data).decode()
    ext = img_path.split('.')[-1]
    return f"data:image/{ext};base64,{encoded}"

# ----------------- Translation Toggle with Cache -----------------
def toggle_translation(chat):
    if chat.get("bot_translated_cache") is None:
        chat["bot_translated_cache"] = {}

    if chat['bot_translated'] is None:
        target_lang = "ar" if chat['bot_lang'] == "en" else "en"

        if target_lang in chat["bot_translated_cache"]:
            chat['bot_translated'] = chat["bot_translated_cache"][target_lang]
        else:
            if target_lang == "ar":
                translated = translator.translate_to_arabic(chat['bot'])
            else:
                translated = translator.translate_to_english(chat['bot'])
            chat["bot_translated_cache"][target_lang] = translated
            chat['bot_translated'] = translated
    else:
        chat['bot_translated'] = None

# ----------------- Render Chat -----------------
def render_chat():
    st.markdown(f"<div style='max-height:{chat_height}px; overflow-y:auto; padding:15px; border-radius:12px; background:#121212;'>", unsafe_allow_html=True)
    for i, chat in enumerate(st.session_state.history):
        # User bubble
        st.markdown(f"""
        <div style="
            background-color:{user_color};
            color:{text_color};
            padding:14px;
            margin:6px 0;
            border-radius:20px;
            text-align:{'right' if chat['user_lang']=='ar' else 'left'};
            direction:{'rtl' if chat['user_lang']=='ar' else 'ltr'};
            font-size:{chat_font_size}px;
            max-width:75%;
            word-break: break-word;
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        ">{chat['user']}</div>
        """, unsafe_allow_html=True)

        # Bot bubble
        bot_content = chat['bot_translated'] if chat['bot_translated'] else chat['bot']
        display_lang = "ar" if chat['bot_translated'] else chat['bot_lang']
        st.markdown(f"""
        <div style="
            background-color:{bot_color};
            color:{text_color};
            padding:14px;
            margin:6px 0;
            border-radius:20px;
            text-align:{'right' if display_lang=='ar' else 'left'};
            direction:{'rtl' if display_lang=='ar' else 'ltr'};
            font-size:{chat_font_size}px;
            max-width:75%;
            word-break: break-word;
            box-shadow: 0 6px 12px rgba(0,0,0,0.4);
        ">{bot_content}</div>
        """, unsafe_allow_html=True)

        toggle_key = f"toggle_{i}"
        if st.button("🌐 Switch Language", key=toggle_key):
            toggle_translation(chat)
    st.markdown("</div>", unsafe_allow_html=True)

# ----------------- Handle PDF Uploads -----------------
if pdf_uploads:
    for pdf_file in pdf_uploads:
        if pdf_file.name not in st.session_state.uploaded_pdfs:
            st.session_state.uploaded_pdfs.append(pdf_file.name)
            try:
                pdf_handler.save_pdf(pdf_file)
                st.success(f"✅ Uploaded: {pdf_file.name}")
            except Exception as e:
                st.error(f"❌ Failed to process PDF {pdf_file.name}: {e}")

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
    gap: 6px;
    padding: 10px;
    border-top: 1px solid #444;
    background: rgba(30,30,30,0.95);
    border-radius: 12px 12px 0 0;
    backdrop-filter: blur(6px);
    box-shadow: 0 -3px 12px rgba(0,0,0,0.5);
    z-index:1000;
}
.input-box {
    flex-grow: 1;
    padding: 14px;
    border-radius: 14px;
    border: 1px solid #555;
    outline: none;
    background: #2c2c2c;
    color: #ffffff;
}
.button-ask {
    padding: 14px 24px;
    border-radius: 14px;
    border: none;
    background-color: #ff7f0e;
    color: #ffffff;
    font-weight:bold;
    cursor: pointer;
    transition: all 0.15s ease-in-out;
}
.button-ask:hover { transform: scale(1.07); }
</style>
""", unsafe_allow_html=True)

# ----------------- Chat Form -----------------
with st.form(key="chat_form", clear_on_submit=False):
    user_input = st.text_input("Chat input", placeholder="Ask me anything about your PDFs…")
    ask_button_clicked = st.form_submit_button("🚀 Ask")

# ----------------- Handle Chat -----------------
def handle_chat(user_input_text):
    if not user_input_text.strip():
        return

    user_lang = translator.detect_language(user_input_text)
    translated_input = translator.translate_to_english(user_input_text)

    context = pdf_handler.get_context(translated_input, top_k_context, st.session_state.uploaded_pdfs)
    bot_response_text = chatbot.ask(translated_input, context, stream=False)
    if hasattr(bot_response_text, '__iter__') and not isinstance(bot_response_text, str):
        bot_response_text = ''.join(list(bot_response_text))
    else:
        bot_response_text = str(bot_response_text)

    images_html = ""
    if any(k in user_input_text.lower() for k in ["graph", "image", "figure"]):
        for pdf_file in st.session_state.uploaded_pdfs:
            pdf_images = pdf_handler.pdf_images.get(pdf_file, [])
            for img_path in pdf_images:
                images_html += f"<br><img src='{img_to_base64(img_path)}' style='max-width:80%; display:block; margin:auto; border-radius:12px;'>"

    final_response = bot_response_text + images_html

    if user_lang == "ar":
        final_response = translator.translate_to_arabic(final_response)

    st.session_state.history.append({
        "user": user_input_text,
        "bot": final_response,
        "user_lang": user_lang,
        "bot_lang": "ar" if user_lang == "ar" else "en",
        "bot_translated": None,
        "bot_translated_cache": {},
        "timestamp": datetime.now().strftime("%H:%M:%S")
    })

# ----------------- Trigger Chat -----------------
if ask_button_clicked:
    handle_chat(user_input)

# ----------------- Render Chat -----------------
render_chat()
