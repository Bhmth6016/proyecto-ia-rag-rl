# debug_metrics.py - VERSIÓN CORREGIDA
import sys
import os
sys.path.append(os.getcwd())

from deepeval import build_queries_and_gts, Retriever, WorkingAgent

def debug_metrics():
    """Diagnóstico de por qué las métricas son cero - VERSIÓN CORREGIDA"""
    print("🔍 DIAGNÓSTICO DE MÉTRICAS CERO")
    
    # Cargar queries y ground truth
    queries, gt_sets = build_queries_and_gts(n=10)
    
    print(f"📊 {len(queries)} consultas, {len(gt_sets)} ground truth sets")
    
    # Probar algunas consultas
    retriever = Retriever()
    agent = WorkingAgent()
    
    for i, (query, gt) in enumerate(zip(queries[:3], gt_sets[:3])):
        print(f"\n🔍 Query {i+1}: '{query}'")
        print(f"   Ground Truth: {gt}")
        
        # Recuperación directa
        retrieved = retriever.retrieve(query, k=5)
        print(f"   Retriever: {len(retrieved)} productos - {retrieved[:3]}")
        
        # Respuesta del agente
        response = agent.process_query(query)
        recommended = response.recommended_ids
        
        # 🔥 CORRECCIÓN: Extraer IDs de productos en lugar de objetos Product
        recommended_ids = []
        for item in recommended:
            if hasattr(item, 'id'):
                recommended_ids.append(item.id)  # Es un objeto Product
            else:
                recommended_ids.append(str(item))  # Es un string ID
        
        print(f"   Agent: {len(recommended_ids)} recomendados - {recommended_ids}")
        
        # Coincidencias
        matches = set(recommended_ids) & gt
        print(f"   Coincidencias: {len(matches)} - {matches}")

if __name__ == "__main__":
    debug_metrics()