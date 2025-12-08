
# Configuración optimizada para WorkingRAGAgent
optimized_config = RAGConfig(
    max_retrieved=10,  # Más rápido
    max_final=5,
    enable_reranking=True,
    ml_enabled=True,
    use_ml_embeddings=True,
    ml_embedding_weight=0.3,
    local_llm_enabled=True,
    use_llm_for_reranking=False,  # Desactivar para velocidad
    semantic_weight=0.7,
    popularity_weight=0.2,
    diversity_weight=0.05,
    freshness_weight=0.05
)
