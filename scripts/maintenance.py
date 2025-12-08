# scripts/maintenance.py
#!/usr/bin/env python3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent        # << reconoce tu proyecto
sys.path.append(str(ROOT))

import schedule
import time
import logging
from datetime import datetime


logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def retrain_rlhf_model():
    """Reentrena modelo RLHF con nuevo feedback"""
    try:
        from src.core.rag.advanced.train_pipeline import RLHFTrainingPipeline
        
        pipeline = RLHFTrainingPipeline()
        result = pipeline.train_from_feedback(min_samples=20)
        
        if result:
            logger.info(f"✅ RLHF reentrenado: {result}")
        else:
            logger.info("⚠️ RLHF: No hay suficiente feedback nuevo")
            
    except Exception as e:
        logger.error(f"❌ Error reentrenando RLHF: {e}")

def update_collaborative_embeddings():
    """Actualiza embeddings del Collaborative Filter"""
    try:
        from src.core.rag.advanced.collaborative_filter import CollaborativeFilter
        from src.core.data.user_manager import UserManager
        from src.core.data.loader import DataLoader
        
        # Cargar productos
        loader = DataLoader()
        products = loader.load_data()
        
        # Crear servicio simple
        class ProductService:
            def __init__(self, products):
                self.products = {p.id: p for p in products if hasattr(p, 'id')}
            
            def get_product(self, product_id):
                return self.products.get(product_id)
            
            def get_popular_products(self, limit=100):
                sorted_products = sorted(
                    self.products.values(),
                    key=lambda p: getattr(p, 'rating_count', 0),
                    reverse=True
                )
                return sorted_products[:limit]
        
        product_service = ProductService(products)
        user_manager = UserManager()
        
        # Actualizar filtro
        cf = CollaborativeFilter(
            user_manager=user_manager,
            product_service=product_service,
            use_ml_features=True
        )
        
        # Forzar recarga de embeddings
        cf._preload_embeddings()
        
        logger.info("✅ Embeddings colaborativos actualizados")
        
    except Exception as e:
        logger.error(f"❌ Error actualizando embeddings: {e}")

def cleanup_logs():
    """Limpia logs antiguos"""
    try:
        feedback_dir = Path("data/feedback")
        cutoff_date = datetime.now().timestamp() - (30 * 24 * 3600)  # 30 días
        
        for log_file in feedback_dir.glob("*.log"):
            if log_file.stat().st_mtime < cutoff_date:
                log_file.unlink()
                logger.info(f"🗑️ Eliminado: {log_file.name}")
                
    except Exception as e:
        logger.error(f"❌ Error limpiando logs: {e}")

def main():
    """Programar tareas de mantenimiento"""
    print("🔄 Sistema de Mantenimiento Automático")
    print("="*50)
    
    # Programar tareas
    schedule.every().day.at("02:00").do(retrain_rlhf_model)
    schedule.every().day.at("02:30").do(update_collaborative_embeddings)
    schedule.every().sunday.at("03:00").do(cleanup_logs)
    
    print("📅 Tareas programadas:")
    print("  • 02:00 - Reentrenar RLHF")
    print("  • 02:30 - Actualizar embeddings colaborativos")
    print("  • Domingo 03:00 - Limpiar logs antiguos")
    print("\n⏳ Ejecutando en segundo plano...")
    
    # Ejecutar inmediatamente primera vez
    retrain_rlhf_model()
    update_collaborative_embeddings()
    
    # Mantener en ejecución
    while True:
        schedule.run_pending()
        time.sleep(60)

if __name__ == "__main__":
    main()