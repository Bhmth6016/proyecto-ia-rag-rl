# verify_rlhf_data.py
import json
from pathlib import Path

def verify_rlhf_data():
    """Verifica que los datos de RLHF estén correctamente formateados"""
    
    print("🔍 VERIFICANDO DATOS DE RLHF...")
    
    # Verificar archivo de success_queries.log
    success_file = Path("data/feedback/success_queries.log")
    if success_file.exists():
        print(f"✅ Archivo encontrado: {success_file}")
        with open(success_file, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            print(f"   Líneas encontradas: {len(lines)}")
            
            for i, line in enumerate(lines[:3]):  # Mostrar primeras 3 líneas
                try:
                    data = json.loads(line.strip())
                    print(f"   Línea {i+1}: {data.get('query', 'Sin query')}")
                except Exception as e:
                    print(f"   ❌ Error en línea {i+1}: {e}")
    else:
        print(f"❌ Archivo NO encontrado: {success_file}")
        
    # Verificar si el trainer puede cargar los datos
    try:
        from src.core.rag.advanced.trainer import RLHFTrainer
        trainer = RLHFTrainer()
        
        # Probar carga de datos
        failed_log = Path("data/feedback/failed_queries.log")
        success_log = Path("data/feedback/success_queries.log")
        
        # Crear archivos si no existen
        if not failed_log.exists():
            failed_log.parent.mkdir(parents=True, exist_ok=True)
            failed_log.touch()
            
        dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
        print(f"✅ Trainer puede cargar datos: {len(dataset.get('train', []))} ejemplos de entrenamiento")
        
    except Exception as e:
        print(f"❌ Error en trainer: {e}")

if __name__ == "__main__":
    verify_rlhf_data()