# backend/chatbot.py
from ctransformers import AutoModelForCausalLM
import logging

logging.basicConfig(level=logging.WARNING)

class ChatBot:
    def __init__(self, model_path: str):
        """
        Initialize Mistral 7B Instruct GGUF model using CTransformers.
        """
        # Use from_pretrained method instead of passing argument directly
        self.llm = AutoModelForCausalLM.from_pretrained(model_path)

    def _call_model(self, prompt: str, max_tokens: int = 512, stop=None) -> str:
        try:
            output = self.llm(prompt, max_new_tokens=max_tokens, stop=stop)
            return output.strip()
        except Exception as e:
            logging.warning(f"LLM call failed: {e}")
            return "⚠️ Error generating response."

    def ask(self, query: str, context: str = "") -> str:
        max_context_len = 3500
        context = context[-max_context_len:]
        prompt = f"""
You are AthenaPDF, a helpful AI assistant for students.
Context: {context}
Question: {query}
Answer clearly and concisely:
"""
        return self._call_model(prompt, max_tokens=512, stop=["Question:"])

    def summarize(self, text: str, max_tokens=300) -> str:
        chunk_size = 1500
        if len(text) > chunk_size:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            summaries = [self._call_model(f"Summarize clearly for a student:\n\n{chunk}", max_tokens=max_tokens) for chunk in chunks]
            final_summary = " ".join(summaries)
            return self._call_model(f"Summarize the following concisely:\n\n{final_summary}", max_tokens=max_tokens)
        else:
            return self._call_model(f"Summarize clearly for a student:\n\n{text}", max_tokens=max_tokens)

    def generate_quiz(self, text: str, num_questions=5) -> str:
        chunk_size = 1500
        if len(text) > chunk_size:
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]
            quiz_parts = [
                self._call_model(
                    f"Generate {num_questions} multiple-choice questions with answers from this text:\n\n{chunk}",
                    max_tokens=500
                ) for chunk in chunks
            ]
            return "\n".join(quiz_parts)
        else:
            prompt = f"Generate {num_questions} multiple-choice questions (with 4 options each) from the following text:\n\n{text}\nProvide correct answers."
            return self._call_model(prompt, max_tokens=500)
