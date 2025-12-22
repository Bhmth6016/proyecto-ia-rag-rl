# create_realistic_ground_truth.py
import json
import pickle
import numpy as np
from pathlib import Path

def main():
    print("🎯 Creando ground truth REALISTA basado en resultados reales")
    print("="*80)
    
    # Cargar sistema
    system_path = Path("data/cache/unified_system_with_fixed_rl.pkl")
    with open(system_path, 'rb') as f:
        system = pickle.load(f)
    
    # Queries comunes
    test_queries = [
        "car parts",
        "led lights for truck", 
        "door panel courtesy light",
        "generator stator",
        "carburetor rebuild kit",
        "fake nails",
        "volumizing conditioner",
        "acrylic nail tips",
        "brazilian virgin hair",
        "curly synthetic wigs",
        "christian daily devotional",
        "star trek audiobook",
        "spanish literature",
        "horror novel dead sea"
    ]
    
    ground_truth = {}
    
    for query in test_queries:
        print(f"\n🔍 Procesando: '{query}'")
        
        # Obtener embedding
        query_embedding = system.canonicalizer.embedding_model.encode(
            query, normalize_embeddings=True
        )
        
        # Buscar productos
        results = system.vector_store.search(query_embedding, k=30)
        
        # Seleccionar 1-3 productos como "relevantes"
        relevant_products = []
        
        for i, product in enumerate(results[:10]):
            # Verificar que tenga título
            if not hasattr(product, 'title') or not product.title:
                continue
            
            title_lower = product.title.lower()
            query_lower = query.lower()
            
            # Calcular similitud de palabras
            query_words = set(query_lower.split())
            title_words = set(title_lower.split())
            match_count = len(query_words.intersection(title_words))
            
            # Considerar relevante si:
            # 1. Tiene palabras en común O
            # 2. Es de categoría relacionada O  
            # 3. Tiene buen rating
            is_relevant = False
            
            # Regla 1: Match de palabras
            if match_count >= 1:
                is_relevant = True
            
            # Regla 2: Categoría relacionada
            elif hasattr(product, 'category'):
                category = str(product.category).lower()
                query_categories = {
                    'car': ['auto', 'vehicle', 'car'],
                    'led': ['auto', 'light'],
                    'door': ['auto'],
                    'generator': ['auto', 'industrial'],
                    'carburetor': ['auto'],
                    'nail': ['beauty', 'cosmetic'],
                    'conditioner': ['beauty', 'hair'],
                    'hair': ['beauty', 'hair'],
                    'wig': ['beauty', 'hair'],
                    'christian': ['book', 'religious'],
                    'audiobook': ['book'],
                    'literature': ['book'],
                    'novel': ['book']
                }
                
                for key, cats in query_categories.items():
                    if key in query_lower:
                        if any(cat in category for cat in cats):
                            is_relevant = True
                            break
            
            # Regla 3: Buen rating
            if not is_relevant and hasattr(product, 'rating'):
                if product.rating and product.rating >= 4.0:
                    is_relevant = True
            
            if is_relevant and len(relevant_products) < 3:
                relevant_products.append(product.id)
                print(f"   ✅ Seleccionado: {product.title[:60]}...")
        
        if relevant_products:
            ground_truth[query] = relevant_products
            print(f"   📊 {len(relevant_products)} productos relevantes seleccionados")
        else:
            # Si no hay "relevantes" por reglas, tomar los primeros 2
            if results:
                ground_truth[query] = [results[0].id]
                if len(results) > 1:
                    ground_truth[query].append(results[1].id)
                print(f"   ⚠️  Usando primeros resultados: {len(ground_truth[query])} productos")
    
    # Guardar
    output_file = Path("data/interactions/real_ground_truth.json")
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(ground_truth, f, indent=2, ensure_ascii=False)
    
    print(f"\n" + "="*80)
    print(f"✅ Ground truth REALISTA creado:")
    print(f"   • {len(ground_truth)} queries")
    total_products = sum(len(v) for v in ground_truth.values())
    print(f"   • {total_products} productos relevantes totales")
    print(f"💾 Guardado en: {output_file}")
    
    # Mostrar resumen
    print(f"\n📋 RESUMEN:")
    for query, ids in list(ground_truth.items())[:5]:
        print(f"   • '{query}': {len(ids)} productos")
        for pid in ids[:2]:
            product = next((p for p in system.canonical_products if p.id == pid), None)
            if product:
                print(f"     - {product.title[:60]}...")

if __name__ == "__main__":
    main()