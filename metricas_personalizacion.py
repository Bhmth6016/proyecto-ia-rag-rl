"""
Métricas que miden lo que RLHF realmente hace: personalización
"""
import numpy as np
from typing import List, Dict, Set

def calcular_metricas_personalizacion(
    baseline_ranking: List[str],
    rlhf_ranking: List[str],
    productos_historicos: Set[str]
) -> Dict[str, float]:
    """
    Calcula métricas de personalización
    
    Args:
        baseline_ranking: Ranking del baseline
        rlhf_ranking: Ranking del RLHF
        productos_historicos: Productos que el usuario clickeó históricamente
    
    Returns:
        Métricas de personalización
    """
    k = 10  # Top 10
    
    # 1. Cambio en ranking (cuánto difieren)
    cambios_top_k = sum(1 for i in range(k) 
                       if baseline_ranking[i] != rlhf_ranking[i])
    porcentaje_cambio = cambios_top_k / k
    
    # 2. Personalización hacia histórico
    baseline_historicos_en_top = sum(1 for pid in baseline_ranking[:k] 
                                    if pid in productos_historicos)
    rlhf_historicos_en_top = sum(1 for pid in rlhf_ranking[:k] 
                                if pid in productos_historicos)
    
    mejora_historicos = rlhf_historicos_en_top - baseline_historicos_en_top
    
    # 3. Diversidad del ranking (cuántos productos nuevos aparecen)
    productos_nuevos_rlhf = len(set(rlhf_ranking[:k]) - set(baseline_ranking[:k]))
    
    # 4. Estabilidad del primer lugar
    cambio_primer_lugar = 1.0 if baseline_ranking[0] != rlhf_ranking[0] else 0.0
    
    # 5. Score de personalización compuesto
    score_personalizacion = (
        porcentaje_cambio * 0.3 +  # Cambio es bueno
        (mejora_historicos / k) * 0.4 +  # Más productos históricos es bueno
        (productos_nuevos_rlhf / k) * 0.2 +  # Novedad controlada es buena
        (1 - cambio_primer_lugar) * 0.1  # Estabilidad en top 1
    )
    
    return {
        'porcentaje_cambio_top_k': porcentaje_cambio,
        'mejora_productos_historicos': mejora_historicos,
        'productos_nuevos_introducidos': productos_nuevos_rlhf,
        'cambio_primer_lugar': cambio_primer_lugar,
        'score_personalizacion': score_personalizacion
    }

def analizar_adaptacion_rlhf(system, ground_truth, interacciones):
    """
    Analiza cómo RLHF se adapta a preferencias
    """
    resultados = []
    
    for query, relevant_ids in ground_truth.items():
        # Obtener productos históricos del usuario
        productos_historicos = obtener_productos_historicos(query, interacciones)
        
        # Rankings
        baseline_ranking = obtener_baseline_ranking(system, query)
        rlhf_ranking = obtener_rlhf_ranking(system, query)
        
        # Métricas clásicas
        p5_baseline = precision_at_k(baseline_ranking, relevant_ids, 5)
        p5_rlhf = precision_at_k(rlhf_ranking, relevant_ids, 5)
        
        # Métricas de personalización
        metricas_pers = calcular_metricas_personalizacion(
            baseline_ranking[:10], 
            rlhf_ranking[:10], 
            productos_historicos
        )
        
        resultados.append({
            'query': query,
            'p5_baseline': p5_baseline,
            'p5_rlhf': p5_rlhf,
            **metricas_pers,
            'tiene_historial': len(productos_historicos) > 0
        })
    
    return resultados