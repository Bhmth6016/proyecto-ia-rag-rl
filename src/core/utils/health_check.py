# src/core/utils/health_check.py
import logging
from typing import Dict, Any, List
from pathlib import Path

logger = logging.getLogger(__name__)

class MLHealthChecker:
    """Verificador de salud del sistema ML"""
    
    def __init__(self):
        self.checks = []
    
    def check_ml_system_health(self) -> Dict[str, Any]:
        """Verifica que el sistema ML esté funcionando correctamente"""
        health_report = {
            "overall_status": "UNKNOWN",
            "ml_enabled": False,
            "checks_passed": 0,
            "checks_total": 0,
            "issues": [],
            "warnings": [],
            "details": {}
        }
        
        try:
            from src.core.data.product import get_system_metrics
            metrics = get_system_metrics()
        except ImportError as e:
            health_report["issues"].append(f"Cannot import system metrics: {e}")
            health_report["overall_status"] = "ERROR"
            return health_report
        
        # Check 1: ML enabled in config
        health_report["checks_total"] += 1
        ml_enabled = metrics.get("product_model", {}).get("ml_enabled", False)
        health_report["ml_enabled"] = ml_enabled
        
        if not ml_enabled:
            health_report["overall_status"] = "DISABLED"
            health_report["warnings"].append("ML features are disabled in configuration")
            return health_report
        
        health_report["checks_passed"] += 1
        
        # Check 2: ML system metrics available
        health_report["checks_total"] += 1
        if "ml_system" not in metrics:
            health_report["issues"].append("ML system metrics not available")
            health_report["overall_status"] = "ERROR"
            return health_report
        
        health_report["checks_passed"] += 1
        ml_metrics = metrics["ml_system"]
        
        # Check 3: Preprocessor loaded
        health_report["checks_total"] += 1
        if not ml_metrics.get("preprocessor_loaded", False):
            health_report["issues"].append("ML preprocessor failed to load")
            health_report["overall_status"] = "ERROR"
        else:
            health_report["checks_passed"] += 1
        
        # Check 4: Embedding model
        health_report["checks_total"] += 1
        if not ml_metrics.get("models_loaded", {}).get("embedding_model", False):
            health_report["warnings"].append("Embedding model not loaded")
        else:
            health_report["checks_passed"] += 1
        
        # Check 5: NER model
        health_report["checks_total"] += 1
        if not ml_metrics.get("models_loaded", {}).get("ner_model", False):
            health_report["warnings"].append("NER model not loaded")
        else:
            health_report["checks_passed"] += 1
        
        # Determine overall status
        if len(health_report["issues"]) > 0:
            health_report["overall_status"] = "ERROR"
        elif len(health_report["warnings"]) > 0:
            health_report["overall_status"] = "WARNING"
        else:
            health_report["overall_status"] = "HEALTHY"
        
        # Add details
        health_report["details"] = {
            "models_loaded": ml_metrics.get("models_loaded", {}),
            "embedding_cache_size": ml_metrics.get("embedding_cache_size", 0),
            "tfidf_fitted": ml_metrics.get("tfidf_fitted", False),
            "config": metrics.get("product_model", {}).get("ml_config", {})
        }
        
        return health_report
    
    def check_data_pipeline_health(self) -> Dict[str, Any]:
        """Verifica la salud del pipeline de datos"""
        report = {
            "data_directories": {},
            "index_status": "UNKNOWN",
            "product_count": 0
        }
        
        # Check directories
        directories = [
            ("raw_data", Path("data/raw")),
            ("processed_data", Path("data/processed")),
            ("chroma_index", Path("data/processed/chroma_db")),
            ("feedback", Path("data/feedback")),
            ("users", Path("data/users"))
        ]
        
        for name, path in directories:
            exists = path.exists()
            report["data_directories"][name] = {
                "exists": exists,
                "path": str(path)
            }
            if exists and path.is_dir():
                try:
                    if name == "raw_data":
                        json_files = list(path.glob("*.json"))
                        jsonl_files = list(path.glob("*.jsonl"))
                        report["data_directories"][name]["file_count"] = len(json_files) + len(jsonl_files)
                    elif name == "processed_data":
                        products_file = path / "products.json"
                        if products_file.exists():
                            try:
                                import json
                                with open(products_file, 'r') as f:
                                    data = json.load(f)
                                report["product_count"] = len(data) if isinstance(data, list) else 1
                            except:
                                pass
                except Exception as e:
                    logger.debug(f"Error checking directory {name}: {e}")
        
        return report

# Función de conveniencia
def get_health_report() -> Dict[str, Any]:
    """Obtiene reporte completo de salud del sistema"""
    checker = MLHealthChecker()
    
    report = {
        "timestamp": __import__('datetime').datetime.now().isoformat(),
        "ml_system": checker.check_ml_system_health(),
        "data_pipeline": checker.check_data_pipeline_health(),
        "system_info": {
            "python_version": __import__('sys').version,
            "platform": __import__('platform').platform()
        }
    }
    
    return report

def print_health_report():
    """Imprime reporte de salud formateado"""
    report = get_health_report()
    
    print("\n" + "="*60)
    print("🩺 SYSTEM HEALTH REPORT")
    print("="*60)
    
    # ML System Status
    ml_status = report["ml_system"]["overall_status"]
    status_emoji = {
        "HEALTHY": "✅",
        "WARNING": "⚠️",
        "ERROR": "❌",
        "DISABLED": "⏸️",
        "UNKNOWN": "❓"
    }.get(ml_status, "❓")
    
    print(f"\n🤖 ML SYSTEM: {status_emoji} {ml_status}")
    print(f"   Enabled: {'Yes' if report['ml_system']['ml_enabled'] else 'No'}")
    print(f"   Checks: {report['ml_system']['checks_passed']}/{report['ml_system']['checks_total']} passed")
    
    if report["ml_system"]["issues"]:
        print(f"   Issues: {', '.join(report['ml_system']['issues'])}")
    if report["ml_system"]["warnings"]:
        print(f"   Warnings: {', '.join(report['ml_system']['warnings'])}")
    
    # Data Pipeline
    print(f"\n📊 DATA PIPELINE:")
    print(f"   Products loaded: {report['data_pipeline']['product_count']}")
    
    for dir_name, dir_info in report["data_pipeline"]["data_directories"].items():
        status = "✅" if dir_info["exists"] else "❌"
        print(f"   {dir_name}: {status} {dir_info['path']}")
    
    print("\n" + "="*60)