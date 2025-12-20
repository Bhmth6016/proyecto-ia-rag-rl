# save_interactions.py
import json
from pathlib import Path
from datetime import datetime

def save_interactions_from_interactive():
    """Lee el archivo de interacciones del sistema interactivo"""
    interactions_file = Path("data/interactions/real_interactions.json")
    
    if not interactions_file.exists():
        print("❌ No se encontró archivo de interacciones")
        print("💡 Ejecuta primero el sistema interactivo y haz clicks")
        return []
    
    try:
        with open(interactions_file, 'r', encoding='utf-8') as f:
            interactions = json.load(f)
        
        print(f"✅ {len(interactions)} interacciones cargadas")
        return interactions
        
    except Exception as e:
        print(f"❌ Error cargando interacciones: {e}")
        return []

def analyze_interactions(interactions):
    """Analiza las interacciones para extraer ground truth"""
    relevance = {}
    
    for interaction in interactions:
        if interaction.get('interaction_type') == 'click':
            query = interaction.get('context', {}).get('query')
            product_id = interaction.get('context', {}).get('product_id')
            
            if query and product_id:
                if query not in relevance:
                    relevance[query] = []
                relevance[query].append(product_id)
    
    return relevance

def create_mock_relevance_if_needed():
    """Crea datos de prueba si no hay interacciones reales"""
    return {
        "racing steering wheel ps4": ["meta_Video_Games_10000_1382", "meta_Video_Games_10000_1450"],
        "car parts": ["meta_Automotive_10000_123", "meta_Automotive_10000_456", "meta_Automotive_10000_789"],
        "wireless headphones": ["meta_Electronics_10000_234", "meta_Electronics_10000_567"],
        "laptop for programming": ["meta_Electronics_10000_891"],
        "running shoes": ["meta_Clothing_Shoes_and_Jewelry_10000_345", "meta_Clothing_Shoes_and_Jewelry_10000_678"]
    }

if __name__ == "__main__":
    print("="*80)
    print("📊 PROCESADOR DE INTERACCIONES")
    print("="*80)
    
    # Intentar cargar interacciones reales
    interactions = save_interactions_from_interactive()
    
    if interactions:
        relevance = analyze_interactions(interactions)
        print(f"\n📝 {len(relevance)} queries con clicks identificadas:")
        for query, products in relevance.items():
            print(f"   • '{query}': {len(products)} productos relevantes")
    else:
        print("\n⚠️  Usando datos de prueba para la evaluación")
        relevance = create_mock_relevance_if_needed()
        print(f"📝 {len(relevance)} queries de prueba creadas")
    
    # Guardar para el evaluador
    output_file = Path("data/interactions/relevance_labels.json")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(relevance, f, indent=2, ensure_ascii=False)
    
    print(f"\n💾 Ground truth guardado en: {output_file}")
    print("🎯 Ahora ejecuta el evaluador con este archivo")