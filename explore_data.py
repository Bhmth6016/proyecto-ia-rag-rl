# Archivo: explore_data.py
import json
import os
from pathlib import Path
import pandas as pd
from collections import Counter

def explore_amazon_data():
    """Explora los archivos JSONL de Amazon"""
    raw_dir = Path("data/raw")
    
    print("📁 Explorando archivos en data/raw:")
    print("="*60)
    
    jsonl_files = list(raw_dir.glob("*.jsonl"))
    if not jsonl_files:
        print("❌ No se encontraron archivos .jsonl")
        return
    
    for file_path in jsonl_files:
        print(f"\n📊 Analizando: {file_path.name}")
        print("-"*40)
        
        # Leer primeras 5 líneas para entender estructura
        samples = []
        with open(file_path, 'r', encoding='utf-8') as f:
            for i, line in enumerate(f):
                if i >= 5:
                    break
                try:
                    samples.append(json.loads(line))
                except:
                    continue
        
        if samples:
            print(f"Primera muestra:")
            print(json.dumps(samples[0], indent=2, ensure_ascii=False)[:500] + "...")
            
            # Mostrar campos disponibles
            print(f"\nCampos disponibles en el primer registro:")
            for key in samples[0].keys():
                print(f"  • {key}")
        
        # Contar líneas totales
        line_count = 0
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line_count += 1
        
        print(f"\n📈 Total de registros: {line_count:,}")

if __name__ == "__main__":
    explore_amazon_data()