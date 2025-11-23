from src.core.rag.advanced import RAGAgent
from src.core.utils.parsers import parse_binary_score

print("✅ Importaciones funcionando correctamente")
print("✅ RAGAgent disponible con métodos:", [m for m in dir(RAGAgent) if not m.startswith('_')])