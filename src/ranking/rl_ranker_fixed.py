# src/ranking/rl_ranker_fixed.py
"""
ESTE ARCHIVO FUE VACIADO INTENCIONALMENTE.

RLHFRankerFixed fue eliminado porque NO era RLHF real.
Era un re-ranking heurístico basado en ajuste lineal de pesos por clicks.

Lo que hacía (INCORRECTO):
    si usuario hace click:
        feature_weights[feature] += reward * feature_value * learning_rate
    Eso es heurística, no RLHF.

RLHF real requiere (implementado en src/rlhf/):
    1. Reward Model explícito (red neuronal con Bradley-Terry loss)
    2. Policy optimizada con PPO
    3. Penalización KL contra política de referencia

Ver: src/rlhf/rlhf_pipeline.py para el ciclo correcto.
"""