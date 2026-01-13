# src/__init__.py
"""
Paquete principal del sistema híbrido RAG + NER + RLHF
"""

__version__ = "2.0.0"
__author__ = "Yalim Villegas Polo"

# Importaciones principales para facilitar el uso
try:
    from .unified_system_v2 import UnifiedSystemV2
    from .ranking.rl_ranker_fixed import RLHFRankerFixed
    from .ranking.ner_enhanced_ranker import NEREnhancedRanker
    from .enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
except ImportError as e:
    print(f"Advertencia: No se pudieron importar todos los módulos: {e}")

# Lista de módulos disponibles
__all__ = [
    'UnifiedSystemV2',
    'RLHFRankerFixed', 
    'NEREnhancedRanker',
    'OptimizedNERExtractor'
]