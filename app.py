# app.py (Gradio version)
import gradio as gr
from backend.chatbot import ChatBot
from backend.pdf_handler import PDFHandler
from backend.translator import Translator

translator = Translator()
chatbot = ChatBot(model_path="models/mistral-7b-instruct-v0.2.Q4_K_M.gguf")
pdf_handler = PDFHandler()

def upload_pdf(file):
    if file is not None:
        pdf_handler.save_pdf(file)
        return f"Uploaded: {file.name}"
    return "No file uploaded."

def summarize_pdf():
    context = pdf_handler.get_context("summary", top_k=5)
    summary = chatbot.summarize(context)
    return summary

def generate_quiz(num_questions):
    context = pdf_handler.get_context("quiz", top_k=5)
    quiz = chatbot.generate_quiz(context, num_questions=num_questions)
    return quiz

def chat(user_input, history):
    user_lang = translator.detect_language(user_input)
    translated_input = translator.translate_to_english(user_input)
    context = pdf_handler.get_context(translated_input)
    response = chatbot.ask(translated_input, context)
    if user_lang == "ar":
        final_response = translator.translate_to_arabic(response)
    else:
        final_response = response
    history = history or []
    history.append((user_input, final_response))
    return history, history

with gr.Blocks() as demo:
    gr.Markdown("# 🤖 AthenaPDF – Your Study Companion")

    with gr.Tab("Upload PDF"):
        pdf_file = gr.File(label="Upload a PDF", file_types=[".pdf"])
        upload_btn = gr.Button("Upload")
        upload_output = gr.Textbox(label="Upload Status", interactive=False)
        upload_btn.click(upload_pdf, inputs=pdf_file, outputs=upload_output)

    with gr.Tab("Summarize PDFs"):
        summarize_btn = gr.Button("Generate Summary")
        summary_output = gr.Textbox(label="Summary", lines=10)
        summarize_btn.click(summarize_pdf, outputs=summary_output)

    with gr.Tab("Generate Quiz"):
        num_questions = gr.Slider(label="Number of Questions per Chunk", minimum=3, maximum=10, value=5, step=1)
        quiz_btn = gr.Button("Generate Quiz")
        quiz_output = gr.Textbox(label="Quiz", lines=10)
        quiz_btn.click(generate_quiz, inputs=num_questions, outputs=quiz_output)

    with gr.Tab("Ask Questions"):
        chatbot_ui = gr.Chatbot(label="Chat History")
        user_input = gr.Textbox(label="Ask a question about your uploaded PDFs:")
        state = gr.State([])
        send_btn = gr.Button("Ask")
        send_btn.click(chat, [user_input, state], [chatbot_ui, state])

if __name__ == "__main__":
    demo.launch()
