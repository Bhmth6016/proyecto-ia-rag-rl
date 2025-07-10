import os
from dotenv import load_dotenv

load_dotenv()

# 🛠️ Configuración de rutas
PERSIST_DIR = os.getenv("PERSIST_DIR", r"C:\Users\x\Downloads\rag\chroma_index")

# 🤖 Modelos
EVAL_MODEL = os.getenv("EVAL_MODEL", "google/flan-t5-large")
LOCAL_MODEL = os.getenv("LOCAL_MODEL", "google/flan-t5-base")
EMBEDDING_MODEL = os.getenv("EMBEDDING_MODEL", "sentence-transformers/all-MiniLM-L6-v2")
