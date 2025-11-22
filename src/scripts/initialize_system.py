# scripts/initialize_system.py
from src.core.init import initialize_system
from src.core.data.loader import FastDataLoader
from src.core.data.chroma_builder import OptimizedChromaBuilder
from src.core.rag.basic.retriever import Retriever

def setup_complete_system():
    """Inicializa todo el sistema para RAGAgent"""
    print("🚀 Inicializando sistema completo...")
    
    # 1. Cargar productos
    loader = FastDataLoader()
    products = loader.load_data()
    print(f"✅ {len(products)} productos cargados")
    
    # 2. Construir índice Chroma
    chroma_builder = OptimizedChromaBuilder()
    chroma_index = chroma_builder.build_index()
    print("✅ Índice Chroma construido")
    
    # 3. Inicializar retriever
    retriever = Retriever()
    print("✅ Retriever inicializado")
    
    # 4. Configurar sistema global
    initialize_system(products=products, retriever=retriever)
    print("✅ Sistema global configurado")
    
    return products, retriever

if __name__ == "__main__":
    setup_complete_system()