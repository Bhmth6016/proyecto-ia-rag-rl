# extraer_ner_incremental.py
"""
Extracción NER INCREMENTAL con cache inteligente
Solo procesa productos SIN atributos NER
"""

from src.unified_system_v2 import UnifiedSystemV2
from src.enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
from tqdm import tqdm
import pickle
from pathlib import Path
import hashlib

# Constantes
CACHE_PATH = Path("data/cache/ner_cache_incremental.pkl")
CHECKPOINT_INTERVAL = 5000  # Guardar cada 5K productos

def cargar_cache_ner():
    """Carga cache NER existente"""
    if CACHE_PATH.exists():
        print(f"📂 Cargando cache NER existente...")
        with open(CACHE_PATH, 'rb') as f:
            cache = pickle.load(f)
        print(f"   ✅ Cache cargado: {len(cache):,} productos\n")
        return cache
    else:
        print(f"📂 No existe cache previo, creando nuevo...\n")
        return {}

def guardar_cache_ner(cache):
    """Guarda cache NER"""
    CACHE_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(CACHE_PATH, 'wb') as f:
        pickle.dump(cache, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"💾 Cache guardado: {CACHE_PATH}")

def hash_title(title):
    """Hash del título para detectar cambios"""
    return hashlib.md5(str(title).encode()).hexdigest()[:8]

def extraer_ner_incremental():
    print("=" * 70)
    print("🚀 EXTRACCIÓN NER INCREMENTAL CON CACHE")
    print("=" * 70)
    print()
    
    # 1. Cargar sistema
    print("📂 Cargando sistema...")
    system = UnifiedSystemV2.load_from_cache()
    
    if system is None:
        print("❌ No se pudo cargar el sistema")
        return
    
    total_productos = len(system.canonical_products)
    print(f"✅ Sistema cargado: {total_productos:,} productos\n")
    
    # 2. Cargar cache NER
    ner_cache = cargar_cache_ner()
    
    # 3. Inicializar extractor
    print("🎯 Inicializando extractor NER...")
    print("   Modo: Keywords (rápido, sin GPU)")
    ner_extractor = OptimizedNERExtractor(use_zero_shot=False)
    print("✅ Extractor listo\n")
    
    # 4. Determinar qué productos procesar
    print("🔍 Analizando productos...")
    productos_pendientes = []
    productos_con_cache = 0
    
    for i, producto in enumerate(system.canonical_products):
        prod_id = getattr(producto, 'id', f'prod_{i}')
        title = getattr(producto, 'title', '')
        
        if not title:
            continue
        
        # Verificar cache
        if prod_id in ner_cache:
            cached = ner_cache[prod_id]
            title_hash = hash_title(title)
            
            # Si el título no cambió, usar cache
            if cached.get('title_hash') == title_hash:
                producto.ner_attributes = cached.get('ner_attributes', {})
                producto.enriched_text = cached.get('enriched_text', title)
                productos_con_cache += 1
                continue
        
        # Producto pendiente de procesar
        productos_pendientes.append((i, producto))
    
    print(f"   ✅ Con cache: {productos_con_cache:,}")
    print(f"   🔄 Pendientes: {len(productos_pendientes):,}")
    print()
    
    if len(productos_pendientes) == 0:
        print("✅ ¡Todos los productos ya tienen NER!")
        print("   No hay nada que procesar.")
        return
    
    # 5. Procesar productos pendientes
    print(f"📦 Procesando {len(productos_pendientes):,} productos pendientes...")
    print(f"   Tiempo estimado: {len(productos_pendientes) / 7500:.1f} minutos\n")
    
    procesados = 0
    con_atributos = 0
    errores = 0
    
    for idx, (i, producto) in enumerate(tqdm(productos_pendientes, 
                                             desc="Extrayendo NER")):
        try:
            prod_id = getattr(producto, 'id', f'prod_{i}')
            title = getattr(producto, 'title', '')
            category = getattr(producto, 'main_category', '') or \
                      getattr(producto, 'category', '')
            
            # Extraer atributos
            atributos = ner_extractor.extract_attributes(title, category)
            
            # Asignar al producto
            producto.ner_attributes = atributos
            
            # Crear texto enriquecido
            if atributos:
                enriched_parts = [title]
                for attr_type, values in atributos.items():
                    if values:
                        enriched_parts.append(
                            f"{attr_type}:{','.join(values[:2])}"
                        )
                producto.enriched_text = " | ".join(enriched_parts)
                con_atributos += 1
            else:
                producto.enriched_text = title
            
            # Actualizar cache
            ner_cache[prod_id] = {
                'ner_attributes': atributos,
                'enriched_text': producto.enriched_text,
                'title_hash': hash_title(title)
            }
            
            procesados += 1
            
            # Checkpoint periódico
            if procesados % CHECKPOINT_INTERVAL == 0:
                print(f"\n  💾 Checkpoint: {procesados:,}/{len(productos_pendientes):,}")
                guardar_cache_ner(ner_cache)
                system.save_to_cache()
            
        except Exception as e:
            errores += 1
            producto.ner_attributes = {}
            producto.enriched_text = getattr(producto, 'title', '')
    
    # 6. Guardar todo
    print(f"\n💾 Guardando sistema y cache final...")
    guardar_cache_ner(ner_cache)
    system.save_to_cache()
    print(f"✅ Sistema guardado\n")
    
    # 7. Estadísticas finales
    print("=" * 70)
    print("📊 RESUMEN DE EXTRACCIÓN INCREMENTAL")
    print("=" * 70)
    print(f"Total productos:          {total_productos:,}")
    print(f"Ya en cache:              {productos_con_cache:,}")
    print(f"Procesados ahora:         {procesados:,}")
    print(f"Con atributos extraídos:  {con_atributos:,} ({con_atributos/procesados*100 if procesados > 0 else 0:.1f}%)")
    print(f"Sin atributos:            {procesados - con_atributos:,}")
    print(f"Errores:                  {errores:,}")
    
    # Verificar cobertura global
    total_con_ner = productos_con_cache + con_atributos
    print(f"\n🎯 COBERTURA GLOBAL:")
    print(f"   {total_con_ner:,}/{total_productos:,} productos con NER ({total_con_ner/total_productos*100:.1f}%)")
    
    # Mostrar ejemplos
    if con_atributos > 0:
        print("\n📝 EJEMPLOS DE ATRIBUTOS EXTRAÍDOS:")
        ejemplos = 0
        for _, prod in productos_pendientes:
            if hasattr(prod, 'ner_attributes') and prod.ner_attributes:
                title = getattr(prod, 'title', 'N/A')[:60]
                attrs = prod.ner_attributes
                print(f"\n  {ejemplos + 1}. {title}")
                for attr_type, values in attrs.items():
                    print(f"     → {attr_type}: {', '.join(values)}")
                ejemplos += 1
                if ejemplos >= 3:
                    break
    
    print("\n" + "=" * 70)
    print("✅ ¡EXTRACCIÓN INCREMENTAL COMPLETADA!")
    print("\n🎯 PRÓXIMOS PASOS:")
    print("   1. python debug_ner_detail.py      (verificar NER)")
    print("   2. python main.py experimento      (evaluar)")
    print("\n💡 VENTAJAS DEL CACHE:")
    print("   • Re-ejecutar este script es instantáneo (usa cache)")
    print("   • Agregar productos nuevos: solo procesa los nuevos")
    print("   • Cambiar títulos: re-procesa solo los modificados")
    print("=" * 70)

if __name__ == "__main__":
    try:
        extraer_ner_incremental()
    except KeyboardInterrupt:
        print("\n\n⚠️  Proceso interrumpido")
        print("   El cache se guardó en el último checkpoint")
    except Exception as e:
        print("\n" + "=" * 70)
        print("💥 ERROR CRÍTICO")
        print("=" * 70)
        print(f"\n{e}")
        import traceback
        traceback.print_exc()