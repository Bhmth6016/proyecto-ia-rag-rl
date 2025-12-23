Sistema de Recomendación Amazon con ML Local
Un sistema de recomendación inteligente que utiliza procesamiento de lenguaje natural (NLP) y aprendizaje automático (ML) 100% local, sin dependencia de servicios en la nube.

🎯 Características Principales
RAG (Retrieval-Augmented Generation): Búsqueda semántica de productos

ML Local: Procesamiento completo en tu máquina

ProductReference: Sistema unificado de manejo de productos

RLHF (Reinforcement Learning from Human Feedback): Mejora continua con feedback

Filtro Colaborativo: Recomendaciones basadas en usuarios similares

NLP Avanzado: Extracción de entidades y clasificación Zero-Shot

📁 Estructura del Proyecto

amazon-recommendation-system/
├── src/
│   ├── core/
│   │   ├── config.py                 # Configuración centralizada
│   │   ├── data/
│   │   │   ├── product.py           # Modelo principal de producto
│   │   │   ├── product_reference.py # Sistema unificado de referencia
│   │   │   ├── loader.py           # Cargador optimizado de datos
│   │   │   ├── ml_processor.py     # Procesador ML con gestión de memoria
│   │   │   └── user_manager.py     # Gestión de perfiles de usuario
│   │   ├── rag/
│   │   │   ├── advanced/
│   │   │   │   ├── WorkingRAGAgent.py # Agente RAG principal
│   │   │   │   ├── collaborative_filter.py # Filtro colaborativo
│   │   │   │   └── trainer.py      # Entrenamiento RLHF
│   │   │   └── basic/
│   │   │       └── retriever.py    # Búsqueda semántica
│   │   ├── nlp/
│   │   │   └── enrichment.py       # Procesamiento NLP
│   │   └── llm/
│   │       └── local_llm.py        # LLM local (Ollama)
│   ├── scripts/
│   │   ├── verify_system.py        # Verificación del sistema
│   │   ├── fix_categories.py       # Reparación de categorías
│   │   └── maintenance.py          # Mantenimiento automático
│   └── models/
│       └── rl_models/              # Modelos RLHF entrenados
├── data/
│   ├── raw/                       # Datos crudos
│   ├── processed/                 # Datos procesados
│   └── feedback/                  # Logs de feedback
├── models/
│   └── sentence_transformers/     # Modelos de embeddings
├── main.py                       # Punto de entrada principal
├── FIX_AND_TRAIN_RLHF.py         # Corrector y entrenador RLHF
├── evaluate_4_points_final.py    # Evaluador de 4 puntos
└── requirements.txt              # Dependencias

🚀 Guía de Inicio Rápido
1. Prerrequisitos
Python 3.9+

8GB+ RAM (recomendado 16GB para ML)

5GB+ espacio en disco para modelos

Ollama (opcional, para LLM local)

2. Instalación
# 1. Clonar el repositorio
git clone <tu-repositorio>
cd amazon-recommendation-system

# 2. Crear entorno virtual (recomendado)
python -m venv venv
source venv/bin/activate  # Linux/Mac
# o
venv\Scripts\activate  # Windows

# 3. Instalar dependencias
pip install -r requirements.txt

# 4. Descargar modelos (automático en primera ejecución)
# Los modelos se descargarán automáticamente al ejecutar el sistema

Preparar Datos
## Descargar dataset:
1. Visita: https://amazon-reviews-2023.github.io
2. Descarga `meta_Video_Games.json.gz`
3. Descomprímelo en `data/raw/`

## Formato esperado:
```json
{
  "asin": "B001234567",
  "title": "Nombre del producto",
  "description": "Descripción detallada",
  "price": 29.99,
  "main_category": "Video Games",
  "categories": ["Video Games", "Accessories"]
}

# 1. Colocar datos de productos en data/raw/
# Formato: JSON o JSONL con productos de Amazon

# 2. Procesar datos
python main.py index

# 3. Verificar sistema
python main.py verify

4. Modos de Uso
Modo Básico (sin ML)

python main.py rag --mode basic

* Solo búsqueda semántica

* Más rápido, menos recursos

Modo Mejorado (ML completo)

python main.py rag --mode enhanced

* NLP (NER + Zero-Shot)

* ML embeddings

* Recomendaciones inteligentes

Modo Balanceado

python main.py rag --mode balanced

* ML básico sin NLP

* Buen equilibrio rendimiento/calidad

🛠️ Comandos Principales
Construir Índice

python main.py index

Sistema RAG Interactivo

python main.py rag --mode enhanced

Entrenar RLHF

python main.py train rlhf

Verificar Sistema

python main.py verify

Reparar Categorías

python main.py ml repair

Ver Estadísticas ML

python main.py ml

🔧 Configuración Avanzada

Archivo de Configuración

El sistema usa src/core/config.py para toda la configuración:


# Habilitar/deshabilitar características
ML_ENABLED = True
NLP_ENABLED = True
LOCAL_LLM_ENABLED = False  # Requiere Ollama

# Modelos
ML_EMBEDDING_MODEL = "all-MiniLM-L6-v2"
LOCAL_LLM_MODEL = "llama2"

ProductReference

Sistema unificado para manejo de productos:

from src.core.data.product_reference import ProductReference

# Crear referencia desde producto
ref = ProductReference.from_product(product)

# Acceder a información ML
if ref.is_ml_processed:
    embedding = ref.embedding
    category = ref.predicted_category

📊 Evaluación del Sistema

Evaluación de 4 Puntos

python evaluate_4_points_final.py

Evalúa:

1. Base sin entrenar (sin NER/Zero-shot)

2. Base sin entrenar (con NER/Zero-shot)

3. Entrenado (sin NER/Zero-shot)

4. Entrenado (con NER/Zero-shot)

Métricas RLHF

El sistema aprende de:

* data/feedback/success_queries.log

* data/feedback/failed_queries.log

🔄 Mantenimiento

Tareas Automáticas

python scripts/maintenance.py

Programa:

* Reentrenamiento RLHF cada 24h

* Actualización embeddings colaborativos

* Limpieza de logs antiguos

Reparación de Embeddings

python scripts/repair_ml_embeddings.py

🧠 Componentes ML

1. Procesador ML

from src.core.data.ml_processor import ProductDataPreprocessor

processor = ProductDataPreprocessor()
producto_ml = processor.preprocess_product(producto_data)

2. NLP Enricher

from src.core.nlp.enrichment import NLPEnricher

enricher = NLPEnricher()
producto_nlp = enricher.enrich_product(producto_data)

3. Filtro Colaborativo

from src.core.rag.advanced.collaborative_filter import CollaborativeFilter

filter = CollaborativeFilter()
recomendaciones = filter.get_collaborative_scores(usuario_id, productos)

🚨 Solución de Problemas

Error: "ProductReference no configurado"

python main.py test product-ref

Error: Serialización de embeddings

python main.py test serialization

Error: ML Processor

python main.py test ml-processor

Limpiar Memoria


# En tu código
from src.core.data.ml_processor import cleanup_memory
cleanup_memory()

📈 Mejores Prácticas

1. Gestión de Memoria

* Usa batch_size adecuado (100-1000)

* Limpia memoria periódicamente: cleanup_memory()

* Monitorea uso: python main.py ml

2. Calidad de Datos

* Verifica categorías: python main.py ml repair

* Valida embeddings: python main.py test serialization

* Limpia datos antes de indexar

3. Feedback

* Califica respuestas (s/n)

* El sistema aprende automáticamente

* Revisa logs en data/feedback/

4. Rendimiento

* Modo basic para pruebas rápidas

* Modo enhanced para producción

* Ajusta batch_size según RAM disponible

🎮 Ejemplos de Uso

Consulta Simple

python main.py rag --mode basic

# > 🔍 Tu consulta: "nintendo switch juegos de aventura"
# > 🤖 Encontré 5 productos...

Entrenamiento Personalizado

# 1. Generar feedback interactivo
python main.py rag --mode enhanced

# 2. Entrenar con feedback
python main.py train rlhf

# 3. Evaluar mejora
python evaluate_4_points_final.py --points 3,4

Sistema de Producción

# 1. Construir índice optimizado
python main.py index

# 2. Verificar todo el sistema
python main.py verify

# 3. Iniciar servicio
python main.py rag --mode enhanced --verbose

📚 Recursos Adicionales

Modelos Disponibles

* Embeddings: all-MiniLM-L6-v2, paraphrase-multilingual-MiniLM-L12-v2

* NLP: dslim/bert-base-NER, facebook/bart-large-mnli

* LLM: Cualquier modelo Ollama compatible

Estructura de Datos

json
{
  "id": "product_123",
  "title": "Nintendo Switch OLED",
  "description": "Consola de videojuegos...",
  "price": 349.99,
  "main_category": "Video Games",
  "categories": ["Electronics", "Gaming"],
  "features": ["Pantalla OLED", "Joy-Con", "Portátil"]
}


🎉 ¡Listo para Usar!
El sistema de recomendación Amazon con ML local está configurado. Comienza con:

python main.py verify
python main.py rag --mode enhanced