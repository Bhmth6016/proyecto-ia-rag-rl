# test_price_extractor.py
import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from src.core.data.product import Product

def test_new_price_extractor():
    """Prueba el nuevo extractor de precios con casos reales"""
    print("💰 === PRUEBA DEL NUEVO EXTRACTOR DE PRECIOS ===\n")
    
    test_cases = [
        # Casos básicos
        {"title": "Basic price", "price": "$19.99"},
        {"title": "Price without symbol", "price": "29.99"},
        {"title": "European format", "price": "19,99"},
        {"title": "With thousand separator", "price": "$1,299.99"},
        
        # Casos de ecommerce reales
        {"title": "Price range", "price": "$19.99 - $29.99"},
        {"title": "USD explicit", "price": "19.99 USD"},
        {"title": "Euro symbol", "price": "€19.99"},
        {"title": "Pound symbol", "price": "£19.99"},
        {"title": "Price with text", "price": "Price: $19.99"},
        {"title": "Complex format", "price": "US $19.99"},
        
        # Casos que deben fallar
        {"title": "Unavailable", "price": "Currently unavailable"},
        {"title": "See price", "price": "See price in cart"},
        {"title": "Contact us", "price": "Contact for price"},
        {"title": "Free", "price": "Free"},
        {"title": "Out of stock", "price": "Out of stock"},
        {"title": "Not available", "price": "Not available"},
        
        # Casos edge
        {"title": "Empty string", "price": ""},
        {"title": "None value", "price": None},
        {"title": "Number direct", "price": 19.99},
        {"title": "List price", "price": ["$19.99"]},
    ]
    
    success_count = 0
    total_cases = len(test_cases)
    
    for i, case in enumerate(test_cases, 1):
        try:
            product = Product.from_dict(case)
            extracted = getattr(product, 'price', 'NOT_EXTRACTED')
            original = case['price']
            
            # Determinar si es éxito
            if case['title'] in ['Unavailable', 'See price', 'Contact us', 'Free', 'Out of stock', 'Not available']:
                # Estos DEBEN fallar (devolver None o 0.0)
                if extracted in [None, 0.0, 'NOT_EXTRACTED']:
                    status = "✅ CORRECTO (rechazado)"
                    success_count += 1
                else:
                    status = f"❌ ERROR (debería rechazar)"
            else:
                # Estos DEBEN extraer precio
                if isinstance(extracted, float) and extracted > 0:
                    status = f"✅ EXITOSO: ${extracted:.2f}"
                    success_count += 1
                else:
                    status = f"❌ FALLÓ: {extracted}"
            
            print(f"{i:2d}. {case['title'][:30]:30} | {str(original)[:20]:20} → {status}")
            
        except Exception as e:
            print(f"{i:2d}. {case['title'][:30]:30} | {str(original)[:20]:20} → ❌ ERROR: {e}")
    
    success_rate = (success_count / total_cases) * 100
    print(f"\n📊 RESULTADO: {success_count}/{total_cases} casos exitosos ({success_rate:.1f}%)")
    
    if success_rate >= 80:
        print("🎉 ¡Extractor de precios funcionando EXCELENTE!")
    elif success_rate >= 60:
        print("👍 ¡Extractor de precios funcionando BIEN!")
    else:
        print("⚠️  El extractor necesita mejoras")

def test_with_real_data_sample():
    """Prueba con una muestra de datos reales del archivo"""
    print("\n📊 === PRUEBA CON DATOS REALES (muestra) ===\n")
    
    from pathlib import Path
    import json
    
    raw_file = Path("./data/raw/meta_Video_Games.jsonl")
    if not raw_file.exists():
        print("❌ No se encuentra el archivo de datos")
        return
    
    price_samples = []
    with open(raw_file, 'r', encoding='utf-8') as f:
        for i, line in enumerate(f):
            if i >= 15:  # Solo 15 muestras
                break
            try:
                data = json.loads(line)
                price_field = data.get('price')
                if price_field:
                    price_samples.append({
                        'title': data.get('title', 'No title'),
                        'original_price': price_field
                    })
            except:
                continue
    
    print(f"Analizando {len(price_samples)} productos con precios...\n")
    
    for i, sample in enumerate(price_samples, 1):
        try:
            product = Product.from_dict({'title': sample['title'], 'price': sample['original_price']})
            extracted = getattr(product, 'price', 'NOT_EXTRACTED')
            
            status = "✅" if isinstance(extracted, float) and extracted > 0 else "❌"
            print(f"{status} {i:2d}. {sample['title'][:40]:40}")
            print(f"      Original: '{sample['original_price']}'")
            print(f"      Extraído: {extracted}")
            print()
            
        except Exception as e:
            print(f"❌ {i:2d}. ERROR: {e}")

if __name__ == "__main__":
    test_new_price_extractor()
    test_with_real_data_sample()