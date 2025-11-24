#!/usr/bin/env python3
"""
Script para forzar reentrenamiento creando feedback artificial
"""

import json
from pathlib import Path
from datetime import datetime

def create_training_data():
    """Crea datos de entrenamiento balanceados"""
    
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(parents=True, exist_ok=True)
    
    # Ejemplos POSITIVOS (rating 4-5)
    positive_examples = [
        {
            "query": "juegos nintendo switch para niños",
            "response": "Recomiendo Lego Incredibles y Splatoon 2 para Nintendo Switch",
            "rating": 5,
            "selected_product_id": "B07DPS6K36"
        },
        {
            "query": "videojuegos de acción ps4", 
            "response": "Te recomiendo Call of Duty y Battlefield para PS4",
            "rating": 4,
            "selected_product_id": "B08H75RTZ8"
        },
        {
            "query": "juegos rpg mundo abierto",
            "response": "The Witcher 3 y Zelda Breath of the Wild son excelentes RPGs",
            "rating": 5,
            "selected_product_id": "B082XY23D3"
        }
    ]
    
    # Ejemplos NEGATIVOS (rating 1-2)
    negative_examples = [
        {
            "query": "juegos de deportes xbox",
            "response": "No encontré juegos de deportes específicos",
            "rating": 2,
            "failure_reason": "product_not_found"
        },
        {
            "query": "nuevos lanzamientos pc",
            "response": "Error procesando tu solicitud",
            "rating": 1, 
            "failure_reason": "system_error"
        }
    ]
    
    # Escribir a success_queries.log
    success_log = feedback_dir / "success_queries.log"
    with open(success_log, 'w', encoding='utf-8') as f:
        for example in positive_examples:
            example["timestamp"] = datetime.now().isoformat()
            example["processed"] = False
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    # Escribir a failed_queries.log  
    failed_log = feedback_dir / "failed_queries.log"
    with open(failed_log, 'w', encoding='utf-8') as f:
        for example in negative_examples:
            example["timestamp"] = datetime.now().isoformat()
            example["processed"] = False
            f.write(json.dumps(example, ensure_ascii=False) + "\n")
    
    print(f"✅ Creados {len(positive_examples)} ejemplos positivos")
    print(f"✅ Creados {len(negative_examples)} ejemplos negativos")
    print("📊 Total ejemplos para entrenamiento:", len(positive_examples) + len(negative_examples))

if __name__ == "__main__":
    create_training_data()