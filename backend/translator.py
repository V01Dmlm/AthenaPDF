# backend/translator.py (Optimized Synchronous + Cache + Safe)
from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory
import logging

DetectorFactory.seed = 0
logging.basicConfig(level=logging.INFO)

class Translator:
    def __init__(self):
        self.to_english = GoogleTranslator(source='auto', target='en')
        self.to_arabic = GoogleTranslator(source='en', target='ar')
        self.cache = {}  # simple in-memory cache to speed up repeated translations

    def translate_to_english(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return text
        if text in self.cache.get('en', {}):
            return self.cache['en'][text]

        try:
            lang = detect(text)
            if lang != 'en':
                translated = self.to_english.translate(text)
            else:
                translated = text
            self.cache.setdefault('en', {})[text] = translated
            return translated
        except Exception as e:
            logging.warning(f"⚠️ Translation to English failed: {e}")
            return text

    def translate_to_arabic(self, text: str) -> str:
        if not text or len(text.strip()) < 2:
            return text
        if text in self.cache.get('ar', {}):
            return self.cache['ar'][text]

        try:
            translated = self.to_arabic.translate(text)
            self.cache.setdefault('ar', {})[text] = translated
            return translated
        except Exception as e:
            logging.warning(f"⚠️ Translation to Arabic failed: {e}")
            return text

    def detect_language(self, text: str) -> str:
        try:
            return detect(text)
        except Exception:
            return "en"  # default to English if detection fails
