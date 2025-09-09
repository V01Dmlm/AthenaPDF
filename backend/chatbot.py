import os
import logging
from ctransformers import AutoModelForCausalLM

logging.basicConfig(level=logging.WARNING)
logger = logging.getLogger(__name__)

class ChatBot:
    def __init__(self, model_path: str, max_tokens: int = 512):
        """
        GPU-optimized ChatBot for Streamlit with guaranteed synchronous output.
        """
        self.max_tokens = max_tokens

        # GPU + CPU threads
        os.environ.setdefault("CT_USE_CUDA", "1")
        os.environ.setdefault("CT_THREADS", str(max(1, os.cpu_count() - 1)))

        try:
            logger.info("Loading model...")
            self.llm = AutoModelForCausalLM.from_pretrained(
                model_path,
                model_type="mistral"
            )
            logger.info("Model loaded successfully.")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    def ask(self, query: str, context: str = "", stream: bool = False) -> str:
        """
        Generate a response synchronously. Returns a string for Streamlit.
        """
        try:
            # Limit context size
            max_context_tokens = 700
            try:
                tokens = self.llm.tokenize(context)
                tokens = tokens[-max_context_tokens:]
                context = self.llm.detokenize(tokens)
            except Exception:
                context = context[-3500:]

            # Build prompt
            prompt = "You are AthenaPDF, a helpful AI assistant for students.\n"
            if context.strip():
                prompt += f"Context: {context}\n"
            prompt += f"Question: {query}\nAnswer clearly and concisely:"

            # Generate response
            output = self.llm(prompt, max_new_tokens=self.max_tokens, stop=["\nQuestion:"])

            # cTransformers sometimes returns a generator or coroutine
            if hasattr(output, '__iter__') and not isinstance(output, str):
                text = "".join([str(chunk) for chunk in output])
            else:
                text = str(output)

            return text.strip()

        except Exception as e:
            logger.warning(f"LLM call failed: {e}")
            return "⚠️ Error generating response."
