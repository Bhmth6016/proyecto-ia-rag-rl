# Amazon Recommendation Hybrid System

Un sistema avanzado de recomendación de productos de Amazon que combina búsqueda semántica y aprendizaje por refuerzo

# Características Principales


# Tabla de Contenidos
## Configuración Inicial

## Estructura del Proyecto

## Instalación

## Configuración

## Uso

## Desarrollo

## Arquitectura

## Contribución
_________________________________________________________________________________________________________________________________________________________
_________________________________________________________________________________________________________________________________________________________
## Configuración Inicial
Requisitos Previos
Python 3.8+

pip 20.0+

Microsoft C++ Build Tools (solo Windows)

8GB+ RAM (recomendado para grandes datasets)

Configuración del Entorno

### Crear entorno virtual (Python 3.8+ recomendado)
python -m venv venv

### Activar el entorno (Windows)
venv\Scripts\activate

### Activar el entorno (Linux/Mac)
source venv/bin/activate

### Actualizar pip
pip install --upgrade pip

## Estructura del Proyecto


## Instalación
Modelos que deben instalarse aparte:
python -m spacy download en_core_web_sm
python -m nltk.downloader averaged_perceptron_tagger

## Configuración

Copia el archivo .env.example a .env y completa tus credenciales:
### .env
DATA_DIR=./data/raw

VECTOR_INDEX_PATH=./data/vector
GEMINI_API_KEY=AIzaSyBnXA2lIP6xfyMICg77XctxmninUOdrzLQ
CHROMA_DB_PATH=./data/processed/chroma_db
CHROMA_DB_COLLECTION=amazon_products
LOG_LEVEL=INFO
LOG_FILE=./logs/amazon_recommendations.log
DEVICE=cpu
MAX_PRODUCTS_TO_LOAD=1000000
MAX_QUERY_LENGTH=20000
MAX_QUERY_RESULTS=5
VECTOR_INDEX_PATH=./data/processed/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
VECTOR_BACKEND=chroma
CACHE_ENABLED=true
ANONYMIZED_TELEMETRY=false

Coloca tus archivos de datos en data/raw/ (formato JSON o JSONL)
https://amazon-reviews-2023.github.io
solo descargar los archivos tipo meta
### Inicializa la estructura del proyecto?????
python setup.py

## Uso
Modos de Operación

### Modo RAG (Recomendación Inteligente)
python main.py rag

Interfaz conversacional para búsquedas semánticas

Ejemplo: "auriculares inalámbricos con cancelación de ruido bajo $100"
as
# Modo Categoría (Navegación Manual)
python main.py category

Explora productos por categorías jerárquicas

Filtra por precio, rating y marcas

# Modo Indexación
python main.py index [--force]

Reconstruye el índice vectorial

Usa --force para reindexar completamente


Ejemplos de Uso

# Iniciar con interfaz de categorías
python main.py category --category "Electrónicos"

# Búsqueda semántica con feedback
python cli.py rag --top-k 5

# Reindexar completamente
python main.py index --force

Desarrollo
Estructura del Código
Los componentes principales están en src/:

agent.py: Clase principal del agente RAG

category_tree.py: Manejo de categorías y filtros

chroma_builder.py: Construcción del índice vectorial

retriever.py: Búsqueda semántica de productos

rlhf.py: Fine-tuning con feedback humano (opcional)
