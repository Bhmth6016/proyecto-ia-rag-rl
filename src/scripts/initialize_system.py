#!/usr/bin/env python3
# scripts/initialize_system.py

import sys
from pathlib import Path

# Agregar src al path
src_path = Path(__file__).parent.parent / "src"
sys.path.insert(0, str(src_path))

from src.core.data.loader import DataLoader
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

def initialize_complete_system():
    """Inicialización completa del sistema"""
    print("🚀 INICIALIZACIÓN COMPLETA DEL SISTEMA")
    print("=" * 50)
    
    # 1. Cargar datos
    print("📦 Paso 1: Cargando productos...")
    loader = DataLoader()
    products = loader.load_data()
    print(f"✅ {len(products)} productos cargados")
    
    # 2. Construir índice
    print("🔍 Paso 2: Construyendo índice vectorial...")
    retriever = Retriever()
    
    if retriever.index_exists():
        print("ℹ️  Índice ya existe, omitiendo construcción")
    else:
        retriever.build_index(products)
        print("✅ Índice construido exitosamente")
    
    # 3. Verificar componentes
    print("🧪 Paso 3: Verificando componentes...")
    
    # Verificar RAGAgent
    try:
        from src.core.rag.advanced.WorkingRAGAgent import RAGAgent
        agent = RAGAgent(products=products)
        print("✅ RAGAgent inicializado correctamente")
    except Exception as e:
        print(f"❌ Error en RAGAgent: {e}")
    
    # Verificar FeedbackProcessor
    try:
        from src.core.rag.advanced.feedback_processor import FeedbackProcessor
        processor = FeedbackProcessor()
        print("✅ FeedbackProcessor inicializado correctamente")
    except Exception as e:
        print(f"❌ Error en FeedbackProcessor: {e}")
    
    print("=" * 50)
    print("🎉 SISTEMA INICIALIZADO EXITOSAMENTE")
    return True

if __name__ == "__main__":
    initialize_complete_system()