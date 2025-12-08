#!/usr/bin/env python3
"""
FIX_AND_TRAIN_RLHF.py -
"""

import os
import sys
import json
import logging
from pathlib import Path
from datetime import datetime

# Configurar logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def fix_trainer_issue():
    """Corrige el error en trainer.py"""
    trainer_path = Path("src/core/rag/advanced/trainer.py")
    
    if not trainer_path.exists():
        logger.error(f"❌ No se encuentra trainer.py en {trainer_path}")
        return False
    
    try:
        with open(trainer_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Buscar y corregir el problema
        if "remove_columns=dataset.column_names" in content:
            logger.info("🔧 Encontrado problema en trainer.py, corrigiendo...")
            
            # Reemplazar la línea problemática
            old_line = "remove_columns=dataset.column_names"
            new_line = "remove_columns=[col for col in dataset.column_names if col != 'labels']"
            
            content = content.replace(old_line, new_line)
            
            # También buscar y corregir el problema de añadir labels
            if "def add_labels(examples):" in content:
                # Encontrar y reemplazar la función add_labels
                lines = content.split('\n')
                fixed_lines = []
                in_add_labels = False
                skip_until_empty = False
                
                for i, line in enumerate(lines):
                    if "def add_labels(examples):" in line:
                        in_add_labels = True
                        skip_until_empty = True
                        fixed_lines.append(line)
                        # Añadir función corregida
                        fixed_lines.append("    # Asegurar que labels esté presente")
                        fixed_lines.append("    if 'labels' not in examples:")
                        fixed_lines.append("        examples['labels'] = [0.5] * len(examples['input_ids'])")
                        fixed_lines.append("    return examples")
                        continue
                    
                    if skip_until_empty and line.strip() == "":
                        skip_until_empty = False
                        fixed_lines.append(line)
                    elif not skip_until_empty:
                        fixed_lines.append(line)
                
                content = '\n'.join(fixed_lines)
            
            # Hacer backup del original
            backup_path = trainer_path.with_suffix('.py.backup')
            with open(backup_path, 'w', encoding='utf-8') as f:
                trainer_path.open('r', encoding='utf-8').seek(0)
                original_content = trainer_path.open('r', encoding='utf-8').read()
                f.write(original_content)
            
            # Guardar corregido
            with open(trainer_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info(f"✅ Trainer corregido. Backup en: {backup_path}")
            return True
            
        else:
            logger.info("✅ Trainer ya parece estar corregido")
            return True
            
    except Exception as e:
        logger.error(f"❌ Error corrigiendo trainer: {e}")
        return False

def create_simple_trainer_fix():
    """Crea un trainer simplificado que funciona"""
    simple_trainer_content = '''#!/usr/bin/env python3
"""
RLHFTrainer SIMPLIFICADO - Versión que funciona
"""

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments
from datasets import Dataset as HFDataset
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import logging

logger = logging.getLogger(__name__)

class SimpleRLHFTrainer:
    """Trainer RLHF simplificado y funcional"""
    
    def __init__(self, device="cpu"):
        self.device = device
        self.tokenizer = AutoTokenizer.from_pretrained("distilbert-base-uncased")
        if not self.tokenizer.pad_token:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        self.model = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased",
            num_labels=1
        ).to(device)
        
        logger.info(f"🤖 SimpleRLHFTrainer inicializado en {device}")
    
    def prepare_rlhf_dataset_from_logs(self, failed_log_path: Path, success_log_path: Path) -> Optional[HFDataset]:
        """Prepara dataset desde logs de forma robusta"""
        import pandas as pd
        
        data = []
        
        # Cargar éxito
        if success_log_path.exists():
            with open(success_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            data.append({
                                'query': str(item.get('query', '')),
                                'response': str(item.get('response', '')),
                                'labels': 1.0  # Éxito
                            })
                        except:
                            continue
        
        # Cargar fracaso
        if failed_log_path.exists():
            with open(failed_log_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if line.strip():
                        try:
                            item = json.loads(line)
                            data.append({
                                'query': str(item.get('query', '')),
                                'response': str(item.get('response', '')),
                                'labels': 0.0  # Fracaso
                            })
                        except:
                            continue
        
        if not data:
            logger.error("❌ No se pudieron cargar datos")
            return None
        
        logger.info(f"📊 Dataset preparado: {len(data)} ejemplos")
        
        df = pd.DataFrame(data)
        return HFDataset.from_pandas(df)
    
    def train(self, dataset: HFDataset, save_dir: Path) -> Dict[str, Any]:
        """Entrenamiento simplificado y funcional"""
        from transformers import DataCollatorWithPadding
        
        try:
            # 1. Tokenización CORREGIDA
            def tokenize_function(examples):
                # Crear textos combinados
                texts = [
                    f"Query: {q} Response: {r}"
                    for q, r in zip(examples['query'], examples['response'])
                ]
                
                # Tokenizar
                tokenized = self.tokenizer(
                    texts,
                    padding="max_length",
                    truncation=True,
                    max_length=128,
                    return_tensors=None  # No devolver tensores todavía
                )
                
                # Añadir labels si no están
                if 'labels' in examples:
                    tokenized['labels'] = examples['labels']
                
                return tokenized
            
            logger.info("🔧 Tokenizando dataset...")
            tokenized_dataset = dataset.map(
                tokenize_function,
                batched=True,
                remove_columns=['query', 'response']  # Solo eliminar estas columnas
            )
            
            logger.info(f"✅ Dataset tokenizado: {len(tokenized_dataset)} ejemplos")
            logger.info(f"📊 Columnas después de tokenizar: {tokenized_dataset.column_names}")
            
            # 2. Verificar que tenemos labels
            if 'labels' not in tokenized_dataset.column_names:
                logger.error("❌ No se encontraron labels después de tokenizar")
                return {'error': 'no_labels_after_tokenization'}
            
            # 3. Configuración de entrenamiento mínima
            training_args = TrainingArguments(
                output_dir=str(save_dir),
                num_train_epochs=2,
                per_device_train_batch_size=4,
                learning_rate=2e-5,
                warmup_steps=50,
                weight_decay=0.01,
                logging_dir=str(save_dir / "logs"),
                logging_steps=10,
                save_strategy="no",
                report_to="none",
                remove_unused_columns=False  # IMPORTANTE: no eliminar columnas automáticamente
            )
            
            # 4. Data collator
            data_collator = DataCollatorWithPadding(tokenizer=self.tokenizer)
            
            # 5. Trainer
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset,
                data_collator=data_collator,
                tokenizer=self.tokenizer
            )
            
            # 6. Entrenar
            logger.info("🚀 Iniciando entrenamiento...")
            train_result = trainer.train()
            
            # 7. Guardar
            trainer.save_model(str(save_dir))
            self.tokenizer.save_pretrained(str(save_dir))
            
            # 8. Reporte
            results = {
                'success': True,
                'samples': len(dataset),
                'train_loss': train_result.training_loss,
                'training_time': train_result.metrics.get('train_runtime', 0)
            }
            
            logger.info(f"✅ Entrenamiento completado: loss={results['train_loss']:.4f}")
            return results
            
        except Exception as e:
            logger.error(f"❌ Error en entrenamiento: {e}")
            import traceback
            traceback.print_exc()
            return {'error': str(e)}
'''

    # Guardar el trainer simplificado
    trainer_dir = Path("src/core/rag/advanced")
    trainer_dir.mkdir(parents=True, exist_ok=True)
    
    fixed_trainer_path = trainer_dir / "trainer_fixed.py"
    
    with open(fixed_trainer_path, 'w', encoding='utf-8') as f:
        f.write(simple_trainer_content)
    
    logger.info(f"✅ Trainer simplificado creado en: {fixed_trainer_path}")
    
    # También crear un __init__.py si no existe
    init_file = trainer_dir / "__init__.py"
    if not init_file.exists():
        init_file.write_text('''"""
Módulo de entrenamiento RLHF
"""

from .trainer_fixed import SimpleRLHFTrainer

RLHFTrainer = SimpleRLHFTrainer

__all__ = ['RLHFTrainer', 'SimpleRLHFTrainer']
''')
    
    return fixed_trainer_path

def run_training_with_fix():
    """Ejecuta el entrenamiento con el fix aplicado"""
    
    print("🤖 ENTRENAMIENTO RLHF CON FIX APLICADO")
    print("=" * 60)
    
    # Opción 1: Intentar corregir el trainer existente
    if not fix_trainer_issue():
        print("⚠️  No se pudo corregir trainer existente, creando uno nuevo...")
        create_simple_trainer_fix()
    
    # Verificar datos
    success_log = Path("data/feedback/success_queries.log")
    failed_log = Path("data/feedback/failed_queries.log")
    
    if not success_log.exists() or not failed_log.exists():
        print("❌ No se encuentran archivos de feedback")
        return False
    
    with open(success_log, 'r', encoding='utf-8') as f:
        success_count = sum(1 for _ in f)
    with open(failed_log, 'r', encoding='utf-8') as f:
        failed_count = sum(1 for _ in f)
    
    total = success_count + failed_count
    print(f"📊 Datos disponibles: {success_count}✅ + {failed_count}❌ = {total} total")
    
    if total < 5:
        print("❌ Muy pocos datos para entrenar")
        return False
    
    # Intentar importar y entrenar
    try:
        # Añadir path
        sys.path.insert(0, str(Path(__file__).parent))
        
        # Intentar importar desde trainer_fixed primero
        try:
            from src.core.rag.advanced.trainer_fixed import SimpleRLHFTrainer
            trainer_class = SimpleRLHFTrainer
            print("✅ Usando trainer_fixed")
        except ImportError as e:
            print(f"⚠️  No se pudo importar trainer_fixed: {e}")
            # Intentar importar el original (ya corregido)
            try:
                from src.core.rag.advanced.trainer import RLHFTrainer
                trainer_class = RLHFTrainer
                print("✅ Usando trainer original (corregido)")
            except ImportError as e2:
                print(f"❌ No se pudo importar ningún trainer: {e2}")
                return False
        
        # Crear trainer y entrenar
        trainer = trainer_class(device="cpu")
        
        print("📚 Preparando dataset...")
        dataset = trainer.prepare_rlhf_dataset_from_logs(failed_log, success_log)
        
        if dataset is None or len(dataset) == 0:
            print("❌ No se pudo crear dataset")
            return False
        
        print(f"📦 Dataset: {len(dataset)} ejemplos")
        if hasattr(dataset, 'column_names'):
            print(f"📊 Columnas: {dataset.column_names}")
        
        # Entrenar
        models_dir = Path("models/rl_models")
        models_dir.mkdir(parents=True, exist_ok=True)
        
        print("\n🎯 INICIANDO ENTRENAMIENTO...")
        print("⏰ Esto tomará varios minutos...")
        
        import time
        start_time = time.time()
        
        results = trainer.train(dataset, save_dir=models_dir)
        
        total_time = time.time() - start_time
        
        if results and results.get('success', False):
            print(f"\n🎉 ¡ENTRENAMIENTO COMPLETADO!")
            print(f"⏱️  Tiempo: {total_time//60:.0f}min {total_time%60:.0f}s")
            print(f"📊 Pérdida: {results.get('train_loss', 'N/A')}")
            print(f"📦 Ejemplos: {results.get('samples', 0)}")
            
            # Verificar archivos creados
            model_files = list(models_dir.glob("*"))
            if model_files:
                print("\n📁 Archivos creados:")
                for file in model_files:
                    size_mb = file.stat().st_size / (1024 * 1024)
                    print(f"   ✅ {file.name} ({size_mb:.1f} MB)")
            
            # Guardar metadata
            metadata = {
                'trained_at': datetime.now().isoformat(),
                'duration_seconds': total_time,
                'dataset_size': len(dataset),
                'results': results
            }
            
            with open(models_dir / "training_report.json", 'w') as f:
                json.dump(metadata, f, indent=2)
            
            print(f"\n💡 Modelo guardado en: {models_dir}")
            return True
        else:
            print(f"\n❌ Entrenamiento falló: {results.get('error', 'Error desconocido')}")
            return False
            
    except Exception as e:
        print(f"❌ Error durante el entrenamiento: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """Función principal"""
    print("🔧 CORRECTOR Y ENTRENADOR RLHF")
    print("=" * 60)
    
    # Verificar que tenemos directorios
    Path("models/rl_models").mkdir(parents=True, exist_ok=True)
    
    # Ejecutar entrenamiento con fix
    success = run_training_with_fix()
    
    if success:
        print("\n" + "=" * 60)
        print("🎉 ¡PROCESO COMPLETADO EXITOSAMENTE!")
        print("=" * 60)
        print("\n💡 El modelo RLHF está listo para usar:")
        print("   python main.py rag --mode enhanced")
        print("\n📁 Modelos en: models/rl_models/")
    else:
        print("\n❌ El proceso falló")
        print("💡 Intenta estas opciones:")
        print("   1. Verifica que tienes datos en data/feedback/")
        print("   2. Ejecuta: python generate_feedback_data.py")
        print("   3. Intenta de nuevo")

if __name__ == "__main__":
    main()