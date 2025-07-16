# src/core/utils/translator.py
from typing import Optional
from enum import Enum
from transformers import pipeline
from langdetect import detect
from src.core.config import settings
class Language(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"

class TextTranslator:
    def __init__(self):
        self.translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-mul-en",
            device=settings.DEVICE
        )
        
        self.reverse_translator = pipeline(
            "translation",
            model="Helsinki-NLP/opus-mt-en-mul",
            device=settings.DEVICE
        )
    def detect_language(self, text: str) -> Language:
        try:
            lang = detect(text)
            return Language(lang)
        except:
            return Language.ENGLISH

    def translate_to_english(self, text: str, source_lang: Optional[Language] = None) -> str:
        if not source_lang:
            source_lang = self.detect_language(text)
            
        if source_lang == Language.ENGLISH:
            return text
            
        return self.translator(
            text,
            src_lang=source_lang.value,
            tgt_lang=Language.ENGLISH.value
        )[0]['translation_text']

    def translate_from_english(self, text: str, target_lang: Language) -> str:
        if target_lang == Language.ENGLISH:
            return text
            
        return self.reverse_translator(
            text,
            src_lang=Language.ENGLISH.value,
            tgt_lang=target_lang.value
        )[0]['translation_text']