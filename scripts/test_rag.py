# scripts/test_rag.py - VERSIÓN MEJORADA

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.core.rag.basic.retriever import Retriever
from src.core.data.loader import DataLoader
from src.core.config import settings

def test_rag_queries():
    """Test mejorado con consultas específicas y verificación de categorías"""
    
    print("🧪 TEST DE RAG CON CONSULTAS VARIADAS")
    print("=" * 60)
    
    # Cargar productos
    loader = DataLoader()
    products = loader.load_data()[:5000]  # Usar solo los primeros 5000 para prueba rápida
    
    # Inicializar retriever
    retriever = Retriever(
        index_path=settings.VECTOR_INDEX_PATH,
        embedding_model=settings.EMBEDDING_MODEL,
        use_product_embeddings=settings.ML_ENABLED
    )
    
    # Construir índice si no existe
    if not retriever.index_exists():
        print("🔧 Construyendo índice...")
        retriever.build_index(products[:1000])
    
    # Consultas de prueba mejoradas
    test_queries = [
        ("juegos de nintendo switch", "Video Games"),
        ("laptop gaming para programar", "Electronics"),
        ("zapatos deportivos running", "Clothing"),
        ("sofá cama para sala", "Home & Kitchen"),
        ("libro de ciencia ficción asimov", "Books"),
        ("crema facial hidratante", "Beauty"),
        ("bicicleta de montaña profesional", "Sports & Outdoors"),
        ("herramientas para mecánica automotriz", "Automotive"),
        ("juego de mesa monopoly", "Toys & Games"),
        ("impresora laser para oficina", "Office Products"),
        ("vitaminas para el sistema inmunológico", "Health & Personal Care"),
        ("auriculares bluetooth inalámbricos", "Electronics"),
        ("vestido de fiesta elegante", "Clothing"),
        ("cocina de inducción 4 zonas", "Home & Kitchen"),
        ("balón de fútbol profesional", "Sports & Outdoors")
    ]
    
    total_tests = len(test_queries)
    correct_tests = 0
    
    for query, expected_category in test_queries:
        print(f"\n📝 Consulta: '{query}'")
        print(f"   Categoría esperada: {expected_category}")
        
        try:
            results = retriever.search(query, k=3)
            
            if results:
                print(f"   ✅ Encontrados: {len(results)} productos")
                
                found_expected = False
                for i, product in enumerate(results[:3], 1):
                    category = getattr(product, 'main_category', 'Unknown')
                    predicted = getattr(product, 'predicted_category', category)
                    
                    # Verificar si encontramos la categoría esperada
                    if expected_category in [category, predicted]:
                        found_expected = True
                    
                    print(f"      {i}. {product.title[:60]}... [{category}]")
                
                if found_expected:
                    print("   🎯 ¡Categoría correcta encontrada!")
                    correct_tests += 1
                else:
                    print("   ⚠️  Categoría esperada no encontrada")
            else:
                print("   ❌ No se encontraron productos")
                
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    # Estadísticas finales
    accuracy = (correct_tests / total_tests) * 100
    print(f"\n📊 RESULTADOS FINALES:")
    print(f"   • Pruebas realizadas: {total_tests}")
    print(f"   • Pruebas exitosas: {correct_tests}")
    print(f"   • Precisión: {accuracy:.1f}%")
    
    if accuracy < 60:
        print("   ⚠️  La precisión es baja, necesita mejoras en el balanceo de categorías")
    elif accuracy < 80:
        print("   ⚠️  La precisión es moderada, considere mejorar los embeddings")
    else:
        print("   ✅ La precisión es buena")

if __name__ == "__main__":
    test_rag_queries()