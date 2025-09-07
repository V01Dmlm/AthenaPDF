# backend/translator.py

from deep_translator import GoogleTranslator
from langdetect import detect, DetectorFactory

# Fix random seed for consistent language detection
DetectorFactory.seed = 0

class Translator:
    def __init__(self):
        # Translators
        self.to_english = GoogleTranslator(source='auto', target='en')
        self.to_arabic = GoogleTranslator(source='en', target='ar')

    def translate_to_english(self, text: str) -> str:
        """
        Translate text to English only if it's not already in English.
        """
        try:
            lang = detect(text)
            if lang != 'en':
                return self.to_english.translate(text)
            return text
        except Exception as e:
            print(f"⚠️ Translation to English failed: {e}")
            return text

    def translate_to_arabic(self, text: str) -> str:
        """
        Translate English text to Arabic. Returns original text if fails.
        """
        try:
            return self.to_arabic.translate(text)
        except Exception as e:
            print(f"⚠️ Translation to Arabic failed: {e}")
            return text
