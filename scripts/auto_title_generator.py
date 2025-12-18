#!/usr/bin/env python3
# scripts/auto_title_generator.py

"""
Generador automático de títulos para productos sin título.
Usa NLP para crear títulos basados en descripción, categoría, etc.
"""

import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional
import sys

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def auto_generate_titles(input_file: Path, output_file: Optional[Path] = None):
    """
    Genera títulos automáticamente para productos que no tienen título.
    
    Args:
        input_file: Archivo JSON con productos
        output_file: Archivo de salida (opcional, sobrescribe input si es None)
    """
    if not input_file.exists():
        logger.error(f"❌ Archivo no encontrado: {input_file}")
        return
    
    if output_file is None:
        output_file = input_file
    
    logger.info(f"📂 Procesando: {input_file}")
    
    # Cargar productos
    try:
        with open(input_file, 'r', encoding='utf-8') as f:
            products = json.load(f)
    except json.JSONDecodeError as e:
        logger.error(f"❌ Error leyendo JSON: {e}")
        return
    except Exception as e:
        logger.error(f"❌ Error cargando archivo: {e}")
        return
    
    if not isinstance(products, list):
        logger.error("❌ El archivo debe contener una lista de productos")
        return
    
    total_products = len(products)
    products_without_title = 0
    titles_generated = 0
    
    logger.info(f"📊 Total de productos a procesar: {total_products}")
    
    # Procesar cada producto
    for i, product in enumerate(products):
        if i % 100 == 0 and i > 0:
            logger.info(f"📊 Progreso: {i}/{total_products} productos")
        
        # Verificar si es un diccionario válido
        if not isinstance(product, dict):
            logger.debug(f"⚠️ Producto {i} no es un diccionario, saltando...")
            continue
        
        # Verificar si necesita título
        title = product.get('title', '')
        if not title or not str(title).strip() or title == "Unknown Product":
            products_without_title += 1
            
            try:
                # Generar título automáticamente
                generated_title = generate_title_for_product(product)
                
                if generated_title and generated_title != "Producto sin nombre":
                    old_title = product.get('title', '')
                    product['title'] = generated_title
                    product['title_generated'] = True
                    product['title_source'] = 'auto_generated'
                    titles_generated += 1
                    
                    if i % 50 == 0:  # Log cada 50 productos
                        logger.info(f"✅ Producto {i}: '{old_title}' → '{generated_title[:50]}...'")
                else:
                    product['title'] = "Producto sin nombre"
                    product['title_generated'] = False
                    logger.debug(f"⚠️ Producto {i}: No se pudo generar título")
                    
            except Exception as e:
                logger.warning(f"⚠️ Error generando título para producto {i}: {e}")
                product['title'] = "Producto sin nombre"
    
    # Guardar productos actualizados
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(products, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.error(f"❌ Error guardando archivo: {e}")
        return
    
    logger.info(f"✅ Proceso completado!")
    logger.info(f"📊 Estadísticas:")
    logger.info(f"   • Productos totales: {total_products}")
    logger.info(f"   • Productos sin título: {products_without_title}")
    logger.info(f"   • Títulos generados: {titles_generated}")
    logger.info(f"   • Guardado en: {output_file}")
    
    return {
        "total_products": total_products,
        "products_without_title": products_without_title,
        "titles_generated": titles_generated,
        "output_file": str(output_file)
    }

def generate_title_for_product(product_data: Dict[str, Any]) -> str:
    """
    Genera un título para un producto basado en sus datos.
    
    Args:
        product_data: Datos del producto
        
    Returns:
        Título generado
    """
    # Extraer datos básicos
    main_category = product_data.get('main_category', '')
    description = product_data.get('description', '')
    brand = product_data.get('brand', '')
    product_type = product_data.get('product_type', '')
    
    # Si hay muy pocos datos, usar fallback inmediatamente
    if not description and not main_category and not brand and not product_type:
        return generate_fallback_title(product_data)
    
    # Primero intentar con NLP si está disponible
    try:
        from src.core.nlp.enrichment import NLPEnricher
        
        # Inicializar NLPEnricher
        nlp_enricher = NLPEnricher(use_small_models=True)
        nlp_enricher.initialize()
        
        # Preparar texto para análisis
        text_parts = []
        
        if main_category:
            text_parts.append(f"Categoría: {main_category}")
        
        if description:
            # Limitar descripción para evitar textos muy largos
            text_parts.append(description[:500])
        
        if brand:
            text_parts.append(f"Marca: {brand}")
        
        text = " ".join(text_parts)
        
        if not text:
            return generate_fallback_title(product_data)
        
        # Extraer entidades
        entities = nlp_enricher.extract_entities(text)
        
        # Construir título
        title_components = []
        
        # 1. Añadir marca (de entidades o de product_data)
        if entities.get("BRAND"):
            brands = entities["BRAND"]
            if brands:
                # Seleccionar la marca con mayor confianza
                sorted_brands = sorted(brands, key=lambda x: x.get('confidence', 0), reverse=True)
                title_components.append(sorted_brands[0]["name"])
        elif brand:
            title_components.append(brand)
        
        # 2. Añadir tipo de producto
        if entities.get("PRODUCT"):
            products = entities["PRODUCT"]
            if products:
                # Seleccionar el producto con mayor confianza
                sorted_products = sorted(products, key=lambda x: x.get('confidence', 0), reverse=True)
                title_components.append(sorted_products[0]["name"])
        elif product_type:
            title_components.append(product_type)
        elif main_category:
            # Convertir categoría a tipo de producto
            category_to_type = {
                'Electronics': 'Electrónico',
                'Books': 'Libro',
                'Clothing': 'Prenda',
                'Home & Kitchen': 'Hogar',
                'Sports & Outdoors': 'Deportivo',
                'Beauty': 'Belleza',
                'Toys': 'Juguete',
                'Toys & Games': 'Juguete',
                'Automotive': 'Automotriz',
                'Office Products': 'Oficina',
                'Video Games': 'Videojuego',
                'Health': 'Salud',
                'General': 'Producto'
            }
            product_type_name = category_to_type.get(main_category, main_category)
            title_components.append(product_type_name)
        
        # 3. Añadir características clave
        if entities.get("ATTRIBUTE"):
            attributes = entities["ATTRIBUTE"]
            if attributes:
                # Seleccionar atributo con mayor confianza
                sorted_attrs = sorted(attributes, key=lambda x: x.get('confidence', 0), reverse=True)
                title_components.append(sorted_attrs[0]["name"])
        
        # Si tenemos al menos 2 componentes, construir título
        if len(title_components) >= 2:
            generated_title = " ".join(title_components[:3])  # Limitar a 3 componentes
        else:
            # Usar fallback
            generated_title = generate_fallback_title(product_data)
        
        # Capitalizar adecuadamente
        if generated_title:
            words = generated_title.split()
            if len(words) > 0:
                # Capitalizar primera palabra
                words[0] = words[0].capitalize()
                generated_title = " ".join(words)
        
        # Limpiar memoria
        nlp_enricher.cleanup_memory()
        
        return generated_title[:120]  # Limitar longitud
        
    except ImportError as e:
        logger.debug(f"NLP no disponible: {e}")
        return generate_fallback_title(product_data)
    except Exception as e:
        logger.debug(f"Error usando NLP: {e}")
        return generate_fallback_title(product_data)

def generate_fallback_title(product_data: Dict[str, Any]) -> str:
    """Genera título de fallback usando datos básicos."""
    title_parts = []
    
    # Añadir categoría
    main_category = product_data.get('main_category', '')
    if main_category and main_category != "General":
        cat_map = {
            'Electronics': 'Producto Electrónico',
            'Books': 'Libro',
            'Clothing': 'Prenda de Ropa',
            'Home & Kitchen': 'Artículo para el Hogar',
            'Sports & Outdoors': 'Equipo Deportivo',
            'Beauty': 'Producto de Belleza',
            'Toys': 'Juguete',
            'Toys & Games': 'Juguete',
            'Automotive': 'Producto Automotriz',
            'Office Products': 'Artículo de Oficina',
            'Video Games': 'Videojuego',
            'Health': 'Producto para la Salud',
            'Video Games': 'Videojuego de'
        }
        readable_cat = cat_map.get(main_category, f"Producto de {main_category}")
        title_parts.append(readable_cat)
    else:
        title_parts.append("Producto")
    
    # Añadir palabras clave de la descripción
    description = product_data.get('description', '')
    if description:
        # Extraer palabras clave simples
        import re
        words = re.findall(r'\b[a-záéíóúñ]{4,}\b', description.lower())
        
        # Filtrar palabras comunes
        common_words = {
            'producto', 'productos', 'calidad', 'excelente', 'mejor', 
            'nuevo', 'nueva', 'nuevos', 'nuevas', 'usado', 'usada',
            'gran', 'buen', 'buena', 'excelente', 'calidad'
        }
        filtered_words = [w for w in words if w not in common_words]
        
        if filtered_words:
            # Tomar palabras únicas
            unique_words = []
            for word in filtered_words[:2]:
                if word not in unique_words:
                    unique_words.append(word)
            
            if unique_words:
                keywords = " ".join(unique_words).title()
                title_parts.append(f"({keywords})")
    
    # Unir partes
    generated_title = " ".join(title_parts).strip()
    
    # Capitalizar primera letra
    if generated_title:
        generated_title = generated_title[0].upper() + generated_title[1:]
    
    return generated_title[:150]

def main():
    """Función principal."""
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Generador automático de títulos para productos sin título"
    )
    
    parser.add_argument(
        'input_file',
        type=Path,
        help='Archivo JSON de entrada con productos'
    )
    
    parser.add_argument(
        '-o', '--output',
        type=Path,
        help='Archivo de salida (opcional, sobrescribe input por defecto)'
    )
    
    parser.add_argument(
        '--verbose', '-v',
        action='store_true',
        help='Mostrar información detallada'
    )
    
    parser.add_argument(
        '--no-backup',
        action='store_true',
        help='No crear copia de respaldo'
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logger.setLevel(logging.DEBUG)
    
    # Verificar que el archivo existe
    if not args.input_file.exists():
        print(f"❌ Error: El archivo {args.input_file} no existe")
        sys.exit(1)
    
    # Crear copia de respaldo si se solicita
    if not args.no_backup and args.output is None:
        import shutil
        import time
        backup_file = args.input_file.parent / f"{args.input_file.stem}_backup_{time.strftime('%Y%m%d_%H%M%S')}{args.input_file.suffix}"
        try:
            shutil.copy2(args.input_file, backup_file)
            print(f"📋 Copia de respaldo creada: {backup_file}")
        except Exception as e:
            print(f"⚠️ No se pudo crear copia de respaldo: {e}")
    
    # Ejecutar generación de títulos
    results = auto_generate_titles(args.input_file, args.output)
    
    print("\n" + "="*60)
    print("✅ PROCESO COMPLETADO")
    print("="*60)
    if results:
        print(f"📊 Estadísticas:")
        print(f"   • Productos totales: {results['total_products']}")
        print(f"   • Productos sin título: {results['products_without_title']}")
        print(f"   • Títulos generados: {results['titles_generated']}")
    print(f"📄 Archivo procesado: {args.input_file}")
    
    if args.output:
        print(f"💾 Archivo de salida: {args.output}")

if __name__ == "__main__":
    main()