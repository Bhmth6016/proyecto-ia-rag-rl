# src/query/query_understanding.py
"""
Entendimiento de queries - NER + Zero-shot classification
SOLO para análisis de query, NO para retrieval
"""
import spacy
from typing import List, Dict, Any, Optional
import numpy as np
from transformers import pipeline
import logging

logger = logging.getLogger(__name__)


class QueryUnderstanding:
    """Analiza queries para extraer características"""
    
    def __init__(self):
        # Cargar modelo NER
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except:
            logger.warning("⚠️ spaCy modelo no encontrado, usando reglas simples")
            self.nlp = None
        
        # Cargar Zero-shot classifier
        try:
            self.zero_shot = pipeline(
                "zero-shot-classification",
                model="facebook/bart-large-mnli"
            )
        except Exception as e:
            logger.warning(f"⚠️ Zero-shot no disponible: {e}")
            self.zero_shot = None
        
        logger.info("✅ QueryUnderstanding inicializado")
    
    def analyze(self, query: str) -> Dict[str, Any]:
        """
        Analiza query y extrae características
        
        Returns:
            Dict con:
            - entities: Entidades NER
            - category: Categoría predicha
            - keywords: Palabras clave
            - intent: Intención (comprar, comparar, buscar)
        """
        analysis = {
            "query": query,
            "entities": [],
            "category": "General",
            "keywords": [],
            "intent": "search",
            "is_specific": False
        }
        
        # 1. NER
        if self.nlp:
            doc = self.nlp(query)
            entities = []
            for ent in doc.ents:
                entities.append({
                    "text": ent.text,
                    "label": ent.label_,
                    "start": ent.start_char,
                    "end": ent.end_char
                })
            
            analysis["entities"] = entities
        
        # 2. Extraer palabras clave (simplificado)
        words = query.lower().split()
        stop_words = {"the", "a", "an", "for", "and", "or", "but", "in", "on", "at", "to", "of"}
        keywords = [w for w in words if w not in stop_words and len(w) > 2]
        analysis["keywords"] = keywords
        
        # 3. Detectar intención
        query_lower = query.lower()
        if any(word in query_lower for word in ["buy", "purchase", "order"]):
            analysis["intent"] = "purchase"
        elif any(word in query_lower for word in ["compare", "vs", "versus"]):
            analysis["intent"] = "compare"
        elif any(word in query_lower for word in ["best", "top", "recommend"]):
            analysis["intent"] = "recommendation"
        
        # 4. Categoría via Zero-shot (si está disponible)
        if self.zero_shot:
            categories = [
                "Electronics", "Books", "Clothing", "Home & Kitchen",
                "Sports", "Beauty", "Toys", "Automotive", "Video Games"
            ]
            
            try:
                result = self.zero_shot(query, categories, multi_label=False)
                if result["scores"]:
                    best_idx = np.argmax(result["scores"])
                    analysis["category"] = result["labels"][best_idx]
            except Exception as e:
                logger.debug(f"Zero-shot error: {e}")
        
        # 5. Especificidad
        analysis["is_specific"] = len(keywords) > 2 or len(analysis["entities"]) > 0
        
        return analysis
    
    def extract_features(self, query: str, query_embedding: np.ndarray) -> Dict[str, float]:
        """Extrae características numéricas de la query"""
        analysis = self.analyze(query)
        
        return {
            "query_length": min(1.0, len(query) / 100),
            "num_keywords": min(1.0, len(analysis["keywords"]) / 10),
            "num_entities": min(1.0, len(analysis["entities"]) / 5),
            "is_specific": 1.0 if analysis["is_specific"] else 0.0,
            "intent_purchase": 1.0 if analysis["intent"] == "purchase" else 0.0,
            "intent_compare": 1.0 if analysis["intent"] == "compare" else 0.0,
            "intent_recommend": 1.0 if analysis["intent"] == "recommendation" else 0.0,
            # Embedding se usa directamente, no como feature escalar
        }