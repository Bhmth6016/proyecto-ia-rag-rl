# src/core/utils/translator.py
from typing import Optional, Dict, List, Any
from enum import Enum
from transformers import pipeline
from langdetect import detect
import re
from src.core.config import settings

class Language(str, Enum):
    ENGLISH = "en"
    SPANISH = "es"
    FRENCH = "fr"
    GERMAN = "de"
    ITALIAN = "it"

class TextTranslator:
    # Términos de dominio por categoría
    DOMAIN_TERMS = {
        # Belleza
        "crema": {"en": "cream (skincare)", "category": "beauty"},
        "labial": {"en": "lipstick (makeup)", "category": "beauty"},
        # Tecnología
        "inalámbrico": {"en": "wireless (tech)", "category": "tech"},
        "pulgadas": {"en": "inches (size)", "category": "tech"}
    }
    
    def __init__(self):
        self._init_pipelines()
        self.domain_context = {
            'beauty': {
                'crema': 'hydrating cream (skincare)',
                'labial': 'lipstick (makeup)',
                'serum': 'facial serum'
            },
            'tech': {
                'inalámbrico': 'wireless (tech)',
                'pulgadas': 'inches (display size)',
                'mAh': 'battery capacity'
            }
        }

    def _init_pipelines(self):
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

    def translate_for_rlhf(self, text: str, source_lang: Language) -> str:
        """Traducción especial para RLHF que preserva términos clave"""
        # Paso 1: Reemplazar términos de dominio
        for term_es, term_data in self.DOMAIN_TERMS.items():
            if term_es in text.lower():
                text = text.replace(term_es, term_data["en"])
        
        # Paso 2: Traducción normal
        if source_lang != Language.ENGLISH:
            text = self.translate_to_english(text, source_lang)
        
        return text.lower().strip()
    
    def extract_domain_terms(self, text: str) -> Dict[str, Dict[str, Any]]:
        """Identifica términos de dominio en el texto"""
        found_terms = {}
        text_lower = text.lower()
        
        for term_es, term_data in self.DOMAIN_TERMS.items():
            if term_es in text_lower:
                found_terms[term_es] = {
                    "en": term_data["en"],
                    "category": term_data["category"],
                    "positions": [m.start() for m in re.finditer(term_es, text_lower)]
                }
        
        return found_terms

    def translate_with_context(self, text: str, domain: str = None) -> str:
        """Traducción que preserva términos técnicos según dominio"""
        if not domain:
            domain = self._detect_domain(text)
        
        # Reemplazar términos específicos del dominio
        if domain in self.domain_context:
            for term_es, term_en in self.domain_context[domain].items():
                text = text.replace(term_es, term_en)
        
        # Traducción normal
        lang = self.detect_language(text)
        if lang != Language.ENGLISH:
            text = self.translate_to_english(text, lang)
        
        return text
    
    def _detect_domain(self, text: str) -> str:
        """Detecta el dominio principal del texto"""
        text_lower = text.lower()
        for domain, terms in self.domain_context.items():
            if any(term in text_lower for term in terms):
                return domain
        return 'general'