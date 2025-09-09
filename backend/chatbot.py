# backend/chatbot.py
from ctransformers import AutoModelForCausalLM
import logging
import os

logging.basicConfig(level=logging.WARNING)

class ChatBot:
    def __init__(self, model_path: str):
        """
        Initialize Mistral 7B Instruct GGUF model using CTransformers.
        Threading and device selection is controlled via environment variables.
        """
        try:
            # Optional: limit threads (default = all CPUs)
            os.environ["CT_THREADS"] = str(os.cpu_count())  # adjust if needed

            # Optional: force CPU (default automatically uses GPU if available)
            os.environ["CT_USE_CUDA"] = "0"  # "1" to use GPU

            # Load model
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral"  # required
            )
            logging.info("Model loaded successfully.")
        except Exception as e:
            logging.error(f"Failed to load model: {e}")
            raise

    def ask(self, query: str, context: str = "") -> str:
        """
        Generate an answer based on the user's query and optional context.
        """
        max_context_len = 700
        context = context[-max_context_len:]

        prompt = f"""
You are AthenaPDF, a helpful AI assistant for students.
Context: {context}
Question: {query}
Answer clearly and concisely:
"""
        try:
            output = self.llm(prompt, max_new_tokens=256, stop=["Question:"])
            if isinstance(output, list):
                return " ".join(output).strip()
            return str(output).strip()
        except Exception as e:
            logging.warning(f"LLM call failed: {e}")
            return "⚠️ Error generating response."
