# src/__init__.py
"""
Paquete principal del sistema híbrido RAG + NER + RLHF
"""
__version__ = "2.1.0"

# RLHFRankerFixed eliminado — era heurística de re-ranking, no RLHF real.
# El RLHF real está en src/rlhf/ (PolicyModel + RewardModel + PPO).

try:
    from .unified_system_v2 import UnifiedSystemV2
    from .ranking.ner_enhanced_ranker import NEREnhancedRanker
    from .enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
except ImportError as e:
    print(f"Advertencia: No se pudieron importar todos los módulos: {e}")

__all__ = [
    'UnifiedSystemV2',
    'NEREnhancedRanker',
    'OptimizedNERExtractor',
]