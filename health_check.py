# health_check.py
def check_ml_system_health():
    """Verifica que el sistema ML esté funcionando correctamente"""
    from src.core.data.product import get_system_metrics
    
    metrics = get_system_metrics()
    
    health_report = {
        "ml_enabled": metrics["product_model"]["ml_enabled"],
        "status": "UNKNOWN",
        "issues": []
    }
    
    if not health_report["ml_enabled"]:
        health_report["status"] = "DISABLED"
        health_report["issues"].append("ML features are disabled in config")
        return health_report
    
    # Verificar módulo ML
    if "ml_system" in metrics:
        ml_metrics = metrics["ml_system"]
        
        if not ml_metrics.get("preprocessor_loaded", False):
            health_report["status"] = "ERROR"
            health_report["issues"].append("ML preprocessor failed to load")
        elif not ml_metrics.get("embedding_model_loaded", False):
            health_report["status"] = "WARNING"
            health_report["issues"].append("Embedding model not loaded")
        else:
            health_report["status"] = "HEALTHY"
            health_report["details"] = {
                "models_loaded": ml_metrics.get("models_loaded", {}),
                "cache_size": ml_metrics.get("embedding_cache_size", 0)
            }
    else:
        health_report["status"] = "ERROR"
        health_report["issues"].append("ML system metrics not available")
    
    return health_report