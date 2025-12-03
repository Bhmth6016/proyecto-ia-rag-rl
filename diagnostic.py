# diagnostic.py
import torch
from datasets import Dataset
import numpy as np
from pathlib import Path
import json

def check_dataset_compatibility():
    """Verifica la compatibilidad completa del dataset"""
    print("=== DIAGNÓSTICO COMPLETO DEL DATASET ===")
    
    # 1. Verificar archivos de logs
    failed_log = Path("data/feedback/failed_queries.log")
    success_log = Path("data/feedback/success_queries.log")
    
    print(f"1. Verificando archivos de logs:")
    print(f"   - failed_queries.log: {failed_log.exists()}")
    print(f"   - success_queries.log: {success_log.exists()}")
    
    # 2. Leer muestras de logs
    def read_sample_logs(filepath, n=3):
        samples = []
        if filepath.exists():
            with open(filepath, 'r', encoding='utf-8') as f:
                for i, line in enumerate(f):
                    if i >= n:
                        break
                    line = line.strip()
                    if line:
                        try:
                            data = json.loads(line)
                            samples.append(data)
                        except:
                            continue
        return samples
    
    print(f"\n2. Muestras de success_queries.log:")
    success_samples = read_sample_logs(success_log)
    for i, sample in enumerate(success_samples):
        print(f"   {i+1}. Keys: {list(sample.keys())}")
        if 'query' in sample:
            print(f"      Query: {sample['query'][:50]}...")
    
    print(f"\n3. Muestras de failed_queries.log:")
    failed_samples = read_sample_logs(failed_log)
    for i, sample in enumerate(failed_samples):
        print(f"   {i+1}. Keys: {list(sample.keys())}")
        if 'query' in sample:
            print(f"      Query: {sample['query'][:50]}...")
    
    # 3. Verificar columnas esperadas
    print(f"\n4. Columnas esperadas en dataset:")
    print(f"   - Obligatorias: query, response")
    print(f"   - Opcionales: answer, labels, score")
    
    return True

def test_tokenization():
    """Prueba la tokenización"""
    from transformers import AutoTokenizer
    
    print("\n=== PRUEBA DE TOKENIZACIÓN ===")
    
    tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
    
    # Ejemplo de texto
    sample_text = "Query: juegos multiplayer para pc Response: Te recomiendo juegos como Counter-Strike: Global Offensive, Fortnite, y Valorant para PC."
    
    print(f"Texto de ejemplo: {sample_text[:100]}...")
    
    # Tokenizar
    tokens = tokenizer(
        sample_text,
        padding="max_length",
        truncation=True,
        max_length=256,
        return_tensors="pt"
    )
    
    print(f"Tokenización exitosa:")
    print(f"  input_ids shape: {tokens['input_ids'].shape}")
    print(f"  attention_mask shape: {tokens['attention_mask'].shape}")
    
    return True

if __name__ == "__main__":
    print("🚀 DIAGNÓSTICO DEL SISTEMA RLHF")
    print("=" * 50)
    
    # Ejecutar diagnósticos
    check_dataset_compatibility()
    test_tokenization()
    
    print("\n" + "=" * 50)
    print("✅ Diagnóstico completado")