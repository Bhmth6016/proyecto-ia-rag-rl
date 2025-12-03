# scripts/initialize_ml.py
#!/usr/bin/env python3
"""
Script para inicializar el sistema ML de manera controlada.
Uso: python initialize_ml.py [--preload] [--fit-tfidf] [--health-check]
"""

import argparse
import logging
import sys
from pathlib import Path

# Añadir src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

def setup_logging():
    """Configura logging básico"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    return logging.getLogger(__name__)

def initialize_ml_system(preload_models: bool = True, fit_tfidf: bool = False) -> bool:
    """
    Inicializa el sistema ML de manera controlada.
    
    Args:
        preload_models: Pre-cargar modelos ML (usa más memoria)
        fit_tfidf: Entrenar TF-IDF con datos de ejemplo
        
    Returns:
        True si la inicialización fue exitosa
    """
    logger = setup_logging()
    
    print("🚀 Initializing ML System...")
    print("="*50)
    
    # 1. Verificar config y dependencias
    try:
        from src.core.config import settings
        from src.core.data.product import Product, get_system_metrics
        
        ml_enabled = getattr(settings, "ML_ENABLED", False)
        if not ml_enabled:
            print("❌ ML is disabled in configuration")
            print("   Set ML_ENABLED=True in .env or use --ml-enabled flag")
            return False
        
        print("✅ ML is enabled in configuration")
        
    except ImportError as e:
        print(f"❌ Cannot import core modules: {e}")
        return False
    
    # 2. Verificar dependencias ML
    print("\n📦 Checking ML dependencies...")
    try:
        import transformers
        import sentence_transformers
        import torch
        print(f"✅ transformers: {transformers.__version__}")
        print(f"✅ sentence-transformers: {sentence_transformers.__version__}")
        print(f"✅ torch: {torch.__version__}")
        print(f"✅ Device: {torch.device('cuda' if torch.cuda.is_available() else 'cpu')}")
    except ImportError as e:
        print(f"❌ Missing ML dependency: {e}")
        print("   Install with: pip install transformers sentence-transformers torch")
        return False
    
    # 3. Pre-cargar modelos (opcional)
    if preload_models:
        print("\n📦 Pre-loading ML models...")
        try:
            from src.core.data.ml_processor import ProductDataPreprocessor
            
            # Configurar modelo ligero para inicialización rápida
            preprocessor = ProductDataPreprocessor(
                embedding_model='sentence-transformers/all-MiniLM-L6-v2',  # Modelo ligero
                use_gpu=False  # Usar CPU para inicialización
            )
            
            # Activar lazy loading
            _ = preprocessor.zero_shot_classifier
            _ = preprocessor.ner_pipeline
            _ = preprocessor.embedding_model
            
            print("✅ ML models pre-loaded successfully")
            print(f"   Embedding model: {preprocessor.embedding_model_name}")
            print(f"   Categories: {len(preprocessor.categories)}")
            
        except Exception as e:
            print(f"⚠️ Warning: Failed to pre-load models: {e}")
            print("   Models will load on first use (lazy loading)")
    
    # 4. Entrenar TF-IDF con datos de ejemplo (opcional)
    if fit_tfidf:
        print("\n🔧 Training TF-IDF with sample data...")
        try:
            from src.core.data.product import MLProductEnricher
            
            sample_descriptions = [
                "wireless bluetooth headphones noise cancelling",
                "laptop computer intel core i7 16gb ram",
                "smartphone android 128gb storage camera",
                "book python programming language guide",
                "kitchen blender stainless steel appliance",
                "gaming console playstation video game",
                "smart watch fitness tracker heart rate",
                "coffee maker espresso machine automatic",
                "running shoes athletic sport footwear",
                "backpack laptop bag travel waterproof"
            ]
            
            success = MLProductEnricher.fit_tfidf(sample_descriptions)
            if success:
                print("✅ TF-IDF trained with sample data")
                print(f"   Sample size: {len(sample_descriptions)} descriptions")
            else:
                print("⚠️ TF-IDF training may have failed")
                
        except Exception as e:
            print(f"⚠️ Warning: TF-IDF training failed: {e}")
    
    # 5. Configurar Product para usar ML
    print("\n⚙️ Configuring Product class for ML...")
    try:
        from src.core.data.product import Product
        
        # Obtener configuración
        ml_features = getattr(settings, "ML_FEATURES", ["category", "entities"])
        ml_categories = getattr(settings, "ML_CATEGORIES", None)
        
        Product.configure_ml(
            enabled=True,
            features=ml_features,
            categories=ml_categories
        )
        
        print(f"✅ Product class configured")
        print(f"   Features: {', '.join(ml_features)}")
        
    except Exception as e:
        print(f"⚠️ Warning: Product configuration failed: {e}")
    
    # 6. Verificar métricas finales
    print("\n📊 Final system check...")
    try:
        metrics = get_system_metrics()
        
        if metrics.get("ml_system", {}).get("ml_enabled", False):
            print("✅ ML system is ready")
            print(f"   Status: {metrics['ml_system'].get('preprocessor_loaded', False)}")
            
            # Mostrar modelos cargados
            models = metrics.get("ml_system", {}).get("models_loaded", {})
            if models:
                print("   Models loaded:")
                for model_name, loaded in models.items():
                    status = "✅" if loaded else "❌"
                    print(f"     {status} {model_name}")
        
        print("\n🎉 ML System Initialization Complete!")
        print("="*50)
        
        return True
        
    except Exception as e:
        print(f"❌ Final check failed: {e}")
        return False

def main():
    parser = argparse.ArgumentParser(
        description="Initialize ML system for Amazon Recommendation"
    )
    parser.add_argument("--preload", action="store_true", 
                       help="Pre-load ML models (uses more memory)")
    parser.add_argument("--fit-tfidf", action="store_true", 
                       help="Train TF-IDF with sample data")
    parser.add_argument("--health-check", action="store_true",
                       help="Run health check after initialization")
    parser.add_argument("--verbose", "-v", action="store_true",
                       help="Verbose output")
    
    args = parser.parse_args()
    
    # Ejecutar inicialización
    success = initialize_ml_system(args.preload, args.fit_tfidf)
    
    # Health check opcional
    if args.health_check and success:
        print("\n" + "="*50)
        print("🩺 Running Health Check...")
        try:
            from src.core.utils.health_check import print_health_report
            print_health_report()
        except ImportError:
            print("⚠️ Health check module not available")
    
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()