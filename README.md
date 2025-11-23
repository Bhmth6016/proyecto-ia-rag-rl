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

### Inicializa la estructura del proyecto?????
python setup.py

## Uso
Modos de Operación

### Modo RAG (Recomendación Inteligente)
python main.py rag

Interfaz conversacional para búsquedas semánticas

Ejemplo: "auriculares inalámbricos con cancelación de ruido bajo $100"
assa
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
