# initialize_ml.py
import argparse
import logging
from src.core.data.product import Product, get_system_metrics

def initialize_ml_system(preload_models: bool = True, fit_tfidf: bool = False):
    """
    Inicializa el sistema ML de manera controlada.
    
    Args:
        preload_models: Pre-cargar modelos ML (usa más memoria)
        fit_tfidf: Entrenar TF-IDF con datos de ejemplo
    """
    logger = logging.getLogger(__name__)
    
    print("🚀 Initializing ML System...")
    
    # 1. Verificar config
    metrics = get_system_metrics()
    if not metrics["product_model"]["ml_enabled"]:
        print("❌ ML is disabled in configuration")
        return False
    
    print("✅ ML is enabled in configuration")
    
    # 2. Pre-cargar modelos (opcional)
    if preload_models:
        print("📦 Pre-loading ML models...")
        try:
            # Forzar carga de modelos
            from src.core.data.ml_processor import ProductDataPreprocessor
            preprocessor = ProductDataPreprocessor()
            
            # Lazy loading se activará cuando se usen
            _ = preprocessor.zero_shot_classifier
            _ = preprocessor.ner_pipeline
            _ = preprocessor.embedding_model
            
            print("✅ ML models pre-loaded successfully")
        except Exception as e:
            print(f"⚠️ Warning: Failed to pre-load models: {e}")
    
    # 3. Entrenar TF-IDF con datos de ejemplo (opcional)
    if fit_tfidf:
        print("🔧 Training TF-IDF with sample data...")
        try:
            sample_descriptions = [
                "wireless bluetooth headphones noise cancelling",
                "laptop computer intel core i7 16gb ram",
                "smartphone android 128gb storage camera",
                "book python programming language guide",
                "kitchen blender stainless steel appliance"
            ]
            
            from src.core.data.product import MLProductEnricher
            MLProductEnricher.fit_tfidf(sample_descriptions)
            print("✅ TF-IDF trained with sample data")
        except Exception as e:
            print(f"⚠️ Warning: TF-IDF training failed: {e}")
    
    print("🎉 ML System Initialization Complete")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--preload", action="store_true", help="Pre-load ML models")
    parser.add_argument("--fit-tfidf", action="store_true", help="Train TF-IDF with samples")
    
    args = parser.parse_args()
    initialize_ml_system(args.preload, args.fit_tfidf)