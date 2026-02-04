from src.unified_system_v2 import UnifiedSystemV2
from pathlib import Path
import pickle

def verificar_cobertura_ner():
    print("🔍 VERIFICANDO COBERTURA NER")
    print("="*60)
    
    system = UnifiedSystemV2.load_from_cache()
    if not system:
        print("❌ Sistema no encontrado")
        return
    
    total = len(system.canonical_products)
    print(f"Total productos: {total:,}")
    
    # Revisar primeros 2,000 productos
    con_ner = 0
    sin_ner = 0
    con_atributos = 0
    
    for product in system.canonical_products[:2000]:
        if hasattr(product, 'ner_attributes'):
            con_ner += 1
            if product.ner_attributes:
                con_atributos += 1
        else:
            sin_ner += 1
    
    print(f"\n📊 MUESTRA (2,000 productos):")
    print(f"  • Con campo ner_attributes: {con_ner:,}")
    print(f"  • Sin campo ner_attributes: {sin_ner:,}")
    print(f"  • Con atributos NO vacíos: {con_atributos:,}")
    print(f"  • Cobertura estimada: {con_atributos/2000*100:.1f}%")
    
    # Verificar cache
    cache_path = Path("data/cache/ner_cache_incremental.pkl")
    if cache_path.exists():
        with open(cache_path, 'rb') as f:
            cache = pickle.load(f)
        print(f"\n💾 Cache NER: {len(cache):,} productos")
        print(f"  Cobertura: {len(cache)/total*100:.1f}%")
    else:
        print("\n❌ No existe cache NER")
    
    print("\n" + "="*60)
    if con_atributos < 200:  # < 10% de muestra
        print("🚨 PROBLEMA CRÍTICO: NER casi inexistente")
        print("✨ Solución: python extraer_ner_incremental.py")
    else:
        print("✅ NER parece funcional")

if __name__ == "__main__":
    verificar_cobertura_ner()