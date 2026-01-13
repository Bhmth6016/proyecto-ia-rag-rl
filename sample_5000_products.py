#!/usr/bin/env python3
"""
sample_10000_products.py - Toma archivos JSONL de raw, selecciona 10000 productos aleatorios
                          y los guarda con sufijo _10000 en raw
"""

import os
import sys
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def sample_jsonl_files():
    """
    Procesa archivos JSONL del directorio raw, selecciona 10000 productos aleatorios
    y guarda con sufijo _10000 en el mismo directorio.
    """
    # Directorios
    BASE_DIR = Path(__file__).parent.parent if "src" in str(Path(__file__)) else Path(__file__).parent
    RAW_DIR = BASE_DIR / "data" / "raw"
    
    if not RAW_DIR.exists():
        logger.error(f"❌ Directorio RAW no encontrado: {RAW_DIR}")
        RAW_DIR.mkdir(parents=True, exist_ok=True)
        logger.info(f"📁 Creado directorio: {RAW_DIR}")
        return False
    
    # Listar todos los .jsonl en raw
    files = [f for f in os.listdir(RAW_DIR) if f.endswith(".jsonl")]
    
    if not files:
        logger.error(f"❌ No se encontraron archivos .jsonl en {RAW_DIR}")
        return False
    
    logger.info(f"📂 Encontrados {len(files)} archivos .jsonl en {RAW_DIR}")
    
    # Número de productos a muestrear
    SAMPLE_SIZE = 10000
    
    for file in files:
        input_file = RAW_DIR / file
        output_name = file.replace(".jsonl", f"_10000.jsonl")
        output_file = RAW_DIR / output_name
        
        # Si ya existe el archivo _10000, saltar
        if output_file.exists():
            logger.info(f"⚠️  Archivo {output_name} ya existe, omitiendo...")
            continue
        
        print(f"\n{'='*60}")
        print(f"📊 Procesando: {file}")
        print(f"{'='*60}")
        
        dataset = []
        
        try:
            # Leer JSONL línea por línea
            with open(input_file, "r", encoding="utf-8") as f:
                lines_read = 0
                for line in f:
                    try:
                        dataset.append(json.loads(line.strip()))
                        lines_read += 1
                    except json.JSONDecodeError:
                        logger.warning(f"   ⚠️  Línea {lines_read+1} inválida, omitiendo...")
                        continue
                    except Exception as e:
                        logger.warning(f"   ⚠️  Error línea {lines_read+1}: {e}")
                        continue
            
            print(f"   📄 Líneas leídas: {lines_read}")
            print(f"   ✅ Productos cargados: {len(dataset)}")
            
            if not dataset:
                logger.warning(f"   ⚠️  Archivo vacío o sin productos válidos: {file}")
                continue
            
            # Limitar a 10000 si hay suficiente contenido
            if len(dataset) > SAMPLE_SIZE:
                print(f"   🎲 Seleccionando {SAMPLE_SIZE} productos aleatorios de {len(dataset)}...")
                sampled_data = random.sample(dataset, SAMPLE_SIZE)
                print(f"   ✅ Muestreo completado: {len(sampled_data)} productos")
            else:
                print(f"   ℹ️  Archivo tiene {len(dataset)} productos (< {SAMPLE_SIZE}), se mantiene completo")
                sampled_data = dataset
            
            # Guardar en nuevo archivo JSONL en raw
            with open(output_file, "w", encoding="utf-8") as f:
                for item in sampled_data:
                    f.write(json.dumps(item, ensure_ascii=False) + "\n")
            
            print(f"   💾 Guardado como: {output_name}")
            print(f"   📊 Productos guardados: {len(sampled_data)}")
            
            # Mostrar estadísticas del archivo guardado
            _show_file_stats(sampled_data, output_file)
            
        except Exception as e:
            logger.error(f"❌ Error procesando {file}: {e}")
            import traceback
            traceback.print_exc()
            continue
    
    print(f"\n{'='*60}")
    print("✅ Proceso completado")
    print(f"{'='*60}")
    
    return True

def _show_file_stats(data: List[Dict[str, Any]], output_file: Path):
    """Muestra estadísticas del archivo procesado."""
    if not data:
        return
    
    try:
        # Estadísticas básicas
        total_products = len(data)
        products_with_price = sum(1 for p in data if p.get('price') and float(p.get('price', 0)) > 0)
        products_with_title = sum(1 for p in data if p.get('title'))
        
        # Categorías únicas
        categories = set()
        for product in data:
            category = product.get('main_category') or product.get('category') or product.get('categories')
            if category:
                if isinstance(category, list):
                    categories.update(category)
                else:
                    categories.add(str(category))
        
        # Precios (si existen)
        prices = []
        for product in data:
            price = product.get('price')
            if price:
                try:
                    prices.append(float(price))
                except (ValueError, TypeError):
                    pass
        
        print(f"\n   📈 ESTADÍSTICAS DEL ARCHIVO:")
        print(f"   {'─'*40}")
        print(f"   • Total productos: {total_products}")
        print(f"   • Con precio definido: {products_with_price} ({products_with_price/total_products*100:.1f}%)")
        print(f"   • Con título: {products_with_title} ({products_with_title/total_products*100:.1f}%)")
        
        if categories:
            print(f"   • Categorías únicas: {len(categories)}")
            print(f"   • Ejemplo categorías: {', '.join(list(categories)[:3])}")
        
        if prices:
            print(f"   • Rango de precios: ${min(prices):.2f} - ${max(prices):.2f}")
            print(f"   • Precio promedio: ${sum(prices)/len(prices):.2f}")
        
        # Tamaño del archivo
        file_size = output_file.stat().st_size / 1024  # KB
        print(f"   • Tamaño archivo: {file_size:.1f} KB")
        
    except Exception as e:
        logger.debug(f"Error calculando estadísticas: {e}")

def main():
    """Función principal."""
    print("\n" + "="*60)
    print("🎲 SAMPLER 10000 - Selecciona 10000 productos aleatorios")
    print("="*60)
    print("📂 Directorio: data/raw/")
    print("📝 Formato: nombre_original_10000.jsonl")
    print("="*60)
    
    try:
        success = sample_jsonl_files()
        
        if success:
            print("\n💡 Archivos creados en data/raw/ con sufijo _10000")
            print("\n🎉 ¡Proceso completado exitosamente!")
        else:
            print("\n⚠️  No se pudieron procesar archivos")
            print("   Verifica que existan archivos .jsonl en data/raw/")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Proceso interrumpido por el usuario")
    except Exception as e:
        print(f"\n❌ Error inesperado: {e}")
        import traceback
        traceback.print_exc()
    
    print("\n" + "="*60)

if __name__ == "__main__":
    main()