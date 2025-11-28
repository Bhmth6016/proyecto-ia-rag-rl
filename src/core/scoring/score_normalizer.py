# src/core/scoring/score_normalizer.py
import logging
from typing import Dict, Tuple, Any
import numpy as np

logger = logging.getLogger(__name__)

class ScoreNormalizer:
    """Normalizador robusto para scores del sistema"""
    
    def __init__(self):
        self.score_ranges = {
            'rag': (0.0, 1.0),
            'collaborative': (0.0, 1.0),
            'feedback_boost': (-3.0, 10.0),
            'final': (0.0, 1.0)
        }
    
    def normalize_rag_score(self, score: float) -> Tuple[float, float]:
        """Normaliza score RAG con confianza"""
        normalized = max(0.0, min(1.0, score))
        # RAG tiene confianza media (0.6-0.8)
        confidence = 0.7 + (normalized * 0.1)  # Mejor score → más confianza
        return normalized, min(confidence, 0.9)
    
    def normalize_collaborative_score(self, score: float, evidence_count: int) -> Tuple[float, float]:
        """Normaliza score colaborativo con confianza"""
        normalized = max(0.0, min(1.0, score))
        
        # Confianza basada en evidencia
        if evidence_count == 0:
            confidence = 0.0
        elif evidence_count == 1:
            confidence = 0.6
        elif evidence_count <= 3:
            confidence = 0.7
        elif evidence_count <= 5:
            confidence = 0.8
        else:
            confidence = 0.9
            
        return normalized, confidence
    
    def normalize_feedback_boost(self, boost: float) -> float:
        """Normaliza boost de feedback con clipping"""
        return max(-3.0, min(10.0, boost))
    
    def calculate_final_score(self, 
                            rag_score: float, 
                            collaborative_score: float,
                            feedback_boost: float,
                            rag_confidence: float,
                            collaborative_confidence: float) -> Tuple[float, float]:
        """Calcula score final ponderado por confianza"""
        # Aplicar boost de feedback
        base_rag = rag_score + (feedback_boost * 0.01)  # Efecto pequeño
        base_collab = collaborative_score + (feedback_boost * 0.02)  # Efecto moderado
        
        # Clipping
        base_rag = max(0.0, min(1.0, base_rag))
        base_collab = max(0.0, min(1.0, base_collab))
        
        # Combinar con confianzas
        total_confidence = rag_confidence + collaborative_confidence
        if total_confidence == 0:
            return 0.0, 0.0
            
        weighted_score = (
            (base_rag * rag_confidence) + 
            (base_collab * collaborative_confidence)
        ) / total_confidence
        
        final_confidence = total_confidence / 2  # Promedio de confianzas
        
        return weighted_score, final_confidence
    
    def softmax_scores(self, scores: Dict[str, float]) -> Dict[str, float]:
        """Aplica softmax para distribución de probabilidad"""
        if not scores:
            return {}
        
        try:
            score_values = list(scores.values())
            exp_scores = np.exp(score_values - np.max(score_values))  # Para estabilidad
            softmax_scores = exp_scores / np.sum(exp_scores)
            
            return {product_id: softmax_scores[i] 
                   for i, product_id in enumerate(scores.keys())}
        except Exception as e:
            logger.error(f"Error aplicando softmax: {e}")
            return scores