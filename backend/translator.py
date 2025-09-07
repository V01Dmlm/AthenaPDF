# backend/translator.py
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import logging

# Fix random seed for consistent language detection
DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)

class Translator:
    def __init__(self):
        """
        Initialize translators for English ↔ Arabic.
        """
        self.to_english = GoogleTranslator(source='auto', target='en')
        self.to_arabic = GoogleTranslator(source='en', target='ar')

    def translate_to_english(self, text: str) -> str:
        """
        Translate text to English only if it's not already English.
        Returns original text on failure or if text is too short.
        """
        if not text or len(text.strip()) < 2:
            return text
        try:
            lang = detect(text)
            if lang != 'en':
                return self.to_english.translate(text)
            return text
        except Exception as e:
            logging.warning(f"⚠️ Translation to English failed: {e}")
            return text

    def translate_to_arabic(self, text: str) -> str:
        """
        Translate English text to Arabic.
        Returns original text on failure or if text is too short.
        """
        if not text or len(text.strip()) < 2:
            return text
        try:
            return self.to_arabic.translate(text)
        except Exception as e:
            logging.warning(f"⚠️ Translation to Arabic failed: {e}")
            return text

    def translate_auto(self, text: str, target_lang: str = "en") -> str:
        """
        Translate text automatically to the specified target language ('en' or 'ar').
        """
        if target_lang.lower() == "en":
            return self.translate_to_english(text)
        elif target_lang.lower() == "ar":
            return self.translate_to_arabic(text)
        else:
            logging.warning(f"⚠️ Unsupported target language: {target_lang}")
            return text
