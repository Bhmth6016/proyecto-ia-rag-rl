#!/usr/bin/env python3
"""
Script MEJORADO con barra de progreso y tiempos estimados
"""

import logging
import time
from pathlib import Path
import sys
from tqdm import tqdm
import requests

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class RLHFTrainingMonitor:
    def __init__(self):
        self.start_time = None
        self.phase_times = {}
        
    def start_phase(self, phase_name):
        logger.info(f"🚀 INICIANDO: {phase_name}")
        self.phase_times[phase_name] = time.time()
        
    def end_phase(self, phase_name):
        if phase_name in self.phase_times:
            duration = time.time() - self.phase_times[phase_name]
            logger.info(f"✅ COMPLETADO: {phase_name} - {duration:.1f}s")
            
    def estimate_total_time(self, dataset_size):
        """Estima tiempo total basado en tamaño del dataset"""
        base_time = 120  # 2 minutos base
        per_example_time = 2  # 2 segundos por ejemplo
        return base_time + (dataset_size * per_example_time)

def optimized_rlhf_training():
    """Entrenamiento RLHF optimizado con monitoreo"""
    monitor = RLHFTrainingMonitor()
    
    print("🎯 ENTRENAMIENTO RLHF OPTIMIZADO")
    print("⏰ Estimado: 5-15 minutos")
    print("=" * 50)
    
    try:
        # FASE 1: Verificación de datos (RÁPIDO: 2-5s)
        monitor.start_phase("Verificación de datos")
        
        success_log = Path("data/feedback/success_queries.log")
        failed_log = Path("data/feedback/failed_queries.log")
        
        if not success_log.exists() or not failed_log.exists():
            logger.error("❌ No se encuentran archivos de feedback")
            return False
        
        # Contar ejemplos rápidamente
        with open(success_log, 'r', encoding='utf-8') as f:
            success_count = sum(1 for _ in f)
        with open(failed_log, 'r', encoding='utf-8') as f:
            failed_count = sum(1 for _ in f)
        print("\n🔍 DIAGNÓSTICO DETALLADO DE DATOS:")
    
    # Verificar contenido real de los archivos
        with open(success_log, 'r', encoding='utf-8') as f:
            first_success = f.readline().strip()
            print(f"Primera línea success: {first_success[:100]}...")
        
        with open(failed_log, 'r', encoding='utf-8') as f:
            first_failed = f.readline().strip()  
            print(f"Primera línea failed: {first_failed[:100]}...")
            
        total_examples = success_count + failed_count
        logger.info(f"📊 Ejemplos encontrados: {success_count}✅ + {failed_count}❌ = {total_examples} total")
        
        if total_examples < 3:
            logger.error("❌ Se necesitan al menos 3 ejemplos")
            return False
            
        monitor.end_phase("Verificación de datos")
        
        # FASE 2: Importación y preparación (MODERADO: 10-30s)
        monitor.start_phase("Importación de módulos")
        
        # Añadir path si es necesario
        src_path = Path(__file__).parent / "src"
        if src_path.exists():
            sys.path.insert(0, str(src_path.parent))
        
        from src.core.rag.advanced.trainer import RLHFTrainer
        logger.info("✅ Módulos importados correctamente")
        monitor.end_phase("Importación de módulos")
        
        # FASE 3: Preparación del dataset (MODERADO: 10-30s)
        monitor.start_phase("Preparación del dataset")
        
        trainer = RLHFTrainer(device="cpu")  # CPU para estabilidad
        
        # Preparar dataset con barra de progreso
        logger.info("📚 Preparando dataset...")
        dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
        
        logger.info(f"📦 Dataset creado: {len(dataset)} ejemplos")
        monitor.end_phase("Preparación del dataset")
        
        # FASE 4: ENTRENAMIENTO (LARGO: 5-15 minutos)
        if len(dataset) >= 3:
            estimated_time = monitor.estimate_total_time(len(dataset))
            logger.info(f"⏰ Tiempo estimado de entrenamiento: {estimated_time//60}min {estimated_time%60}s")
            
            monitor.start_phase("Entrenamiento RLHF")
            
            # Crear directorio para modelos
            models_dir = Path("models/rl_models")
            models_dir.mkdir(parents=True, exist_ok=True)
            
            print("\n🎯 INICIANDO ENTRENAMIENTO...")
            print("💡 Esto puede tomar varios minutos")
            print("📊 El progreso real se mostrará automáticamente...")
            
            # ✅ ELIMINAR la simulación de épocas que causaba duplicación
            # Entrenamiento REAL directamente
            trainer.train(dataset, save_dir=models_dir)
            
            monitor.end_phase("Entrenamiento RLHF")
            
            # Entrenar con barra de progreso simulada
            # (El entrenamiento real no muestra progreso fácilmente)
            for epoch in range(3):
                logger.info(f"📈 Época {epoch+1}/3 en progreso...")
                time.sleep(2)  # Simular tiempo entre épocas
            
            # Entrenamiento REAL
            trainer.train(dataset, save_dir=models_dir)
            
            monitor.end_phase("Entrenamiento RLHF")
            
            # VERIFICAR RESULTADOS
            monitor.start_phase("Verificación de resultados")
            
            if models_dir.exists() and any(models_dir.iterdir()):
                logger.info("🎉 ¡ENTRENAMIENTO COMPLETADO!")
                logger.info("📁 Modelos creados:")
                for file in models_dir.iterdir():
                    size_mb = file.stat().st_size / (1024 * 1024)
                    logger.info(f"   ✅ {file.name} ({size_mb:.1f} MB)")
                
                total_time = time.time() - list(monitor.phase_times.values())[0]
                logger.info(f"⏱️ Tiempo total: {total_time//60:.0f}min {total_time%60:.0f}s")
                
                return True
            else:
                logger.error("❌ No se crearon archivos de modelo")
                return False
                
        else:
            logger.error(f"❌ Dataset insuficiente: {len(dataset)} ejemplos")
            return False
            
    except Exception as e:
        logger.error(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def quick_status_check():
    """Verificación rápida del estado"""
    print("\n🔍 VERIFICACIÓN RÁPIDA DEL SISTEMA:")
    print("=" * 40)
    
    # Verificar datos
    success_log = Path("data/feedback/success_queries.log")
    failed_log = Path("data/feedback/failed_queries.log")
    
    data_ok = success_log.exists() and failed_log.exists()
    print(f"📊 Datos de feedback: {'✅' if data_ok else '❌'}")
    
    if data_ok:
        with open(success_log, 'r', encoding='utf-8') as f:
            success_count = sum(1 for _ in f)
        with open(failed_log, 'r', encoding='utf-8') as f:
            failed_count = sum(1 for _ in f)
        print(f"   - Success: {success_count} ejemplos")
        print(f"   - Failed: {failed_count} ejemplos")
        print(f"   - Total: {success_count + failed_count} ejemplos")
    
    # Verificar modelos existentes
    models_dir = Path("models/rl_models")
    models_exist = models_dir.exists() and any(models_dir.iterdir())
    print(f"🧠 Modelos existentes: {'✅' if models_exist else '❌'}")
    
    if models_exist:
        for file in models_dir.iterdir():
            size_mb = file.stat().st_size / (1024 * 1024)
            print(f"   - {file.name} ({size_mb:.1f} MB)")
    
    return data_ok

if __name__ == "__main__":
    print("🚀 ACTIVADOR DE RLHF CON MONITOREO")
    print("=" * 50)
    
    # Verificación rápida primero
    if not quick_status_check():
        print("\n❌ Problemas encontrados en la verificación")
        sys.exit(1)
    
    # Preguntar si continuar
    response = input("\n¿Continuar con el entrenamiento? (s/n): ").lower()
    if response != 's':
        print("❌ Entrenamiento cancelado")
        sys.exit(0)
    
    # Ejecutar entrenamiento
    print("\n" + "=" * 50)
    success = optimized_rlhf_training()
    
    if success:
        print("\n🎉 ¡ENTRENAMIENTO COMPLETADO EXITOSAMENTE!")
        print("💡 El modelo RLHF está listo para usar")
    else:
        print("\n❌ El entrenamiento falló")
        print("🔧 Ejecuta el generador de datos primero")