# diagnosticar_datos.py
import json
from pathlib import Path

def diagnosticar_problema():
    print("🔍 DIAGNÓSTICO COMPLETO DE DATOS")
    
    success_log = Path("data/feedback/success_queries.log")
    failed_log = Path("data/feedback/failed_queries.log")
    
    for archivo, nombre in [(success_log, "SUCCESS"), (failed_log, "FAILED")]:
        print(f"\n📁 {nombre}_LOG:")
        if archivo.exists():
            with open(archivo, 'r', encoding='utf-8') as f:
                lineas = f.readlines()
                print(f"   Total líneas: {len(lineas)}")
                
                if lineas:
                    primera = json.loads(lineas[0])
                    print(f"   Campos en primera línea: {list(primera.keys())}")
                    print(f"   Tiene 'query': {'query' in primera}")
                    print(f"   Tiene 'response': {'response' in primera}")
                    
                    # Contar muestras válidas
                    validas = 0
                    for linea in lineas[:5]:  # Solo primeras 5 para diagnóstico
                        try:
                            data = json.loads(linea)
                            if data.get('query') and data.get('response'):
                                validas += 1
                        except:
                            pass
                    print(f"   Muestras válidas (primeras 5): {validas}/5")
        else:
            print(f"   ❌ Archivo no existe")

if __name__ == "__main__":
    diagnosticar_problema()