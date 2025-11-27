# Amazon Recommendation Hybrid System
Un sistema avanzado de recomendación de productos que combina RAG tradicional (40%) con filtrado colaborativo (60%) y aprendizaje por refuerzo con feedback humano (RLHF).

# 🚀 Características Principales
# 🔥 Sistema Híbrido Inteligente
60% Filtrado Colaborativo: Recomendaciones basadas en usuarios similares

40% RAG Tradicional: Búsqueda semántica en base de productos

Personalización Demográfica: Edad, género y país del usuario

# 🧠 Aprendizaje Automático
RLHF Integrado: Mejora continua con feedback de usuarios

Reentrenamiento Automático: Cuando se acumula suficiente feedback

Embeddings Avanzados: Modelos SentenceTransformer optimizados

# 👥 Gestión de Usuarios
Perfiles Persistente: Historial de búsquedas y preferencias

Similitud de Usuarios: Encuentra usuarios con gustos similares

Datos Demográficos: Age, gender, country para personalización

# ⚡ Optimizaciones Técnicas
ChromaDB Optimizado: Índices vectoriales de alto rendimiento

Procesamiento por Lotes: Manejo eficiente de grandes datasets

Caché Inteligente: Reducción de tiempos de respuesta

# 📋 Tabla de Contenidos
Configuración Inicial

Estructura del Proyecto

Instalación

Configuración

Uso

Desarrollo

Arquitectura

Contribución

# 🛠 Configuración Inicial
Requisitos Previos
Python 3.8+

pip 20.0+

8GB+ RAM (recomendado para grandes datasets)

Conexión a internet (para descargar modelos)

Configuración del Entorno
bash
# Crear entorno virtual (Python 3.8+ recomendado)
python -m venv venv

# Activar el entorno (Windows)
venv\Scripts\activate

# Activar el entorno (Linux/Mac)
source venv/bin/activate

# Actualizar pip
pip install --upgrade pip
# 📁 Estructura del Proyecto
text
amazon-recommendation-system/
├── src/
│   ├── core/
│   │   ├── data/                 # Gestión de datos
│   │   │   ├── loader.py         # Cargador optimizado
│   │   │   ├── product.py        # Modelos de producto
│   │   │   ├── user_manager.py   # Gestión de usuarios
│   │   │   └── user_models.py    # Modelos de usuario
│   │   ├── rag/
│   │   │   ├── basic/            # RAG básico
│   │   │   │   └── retriever.py  # Sistema de recuperación
│   │   │   └── advanced/         # RAG avanzado
│   │   │       ├── WorkingRAGAgent.py    # Agente principal
│   │   │       ├── collaborative_filter.py # Filtro colaborativo
│   │   │       ├── trainer.py    # Entrenamiento RLHF
│   │   │       └── RLHFMonitor.py # Monitoreo RLHF
│   │   ├── config.py             # Configuración
│   │   └── init.py               # Inicialización del sistema
│   └── interfaces/
│       └── cli.py                # Interfaz de línea de comandos
├── data/
│   ├── raw/                      # Datos brutos
│   ├── processed/                # Datos procesados  
│   └── users/                    # Perfiles de usuarios
├── main.py                       # Punto de entrada principal
└── requirements.txt              # Dependencias
# 📦 Instalación
Clonar el repositorio

bash
git clone <repository-url>
cd amazon-recommendation-system
Instalar dependencias

bash
pip install -r requirements.txt
Configurar variables de entorno

bash
cp .env.example .env
# Editar .env con tus configuraciones
# ⚙️ Configuración
Archivo .env
env
# API Configuration
GEMINI_API_KEY=your_gemini_api_key_here
MODEL_NAME=gemini-2.5-flash

# Data Paths
DATA_DIR=./data
RAW_DIR=./data/raw
PROC_DIR=./data/processed

# Vector Store
VECTOR_INDEX_PATH=./data/processed/chroma_db
EMBEDDING_MODEL=sentence-transformers/all-MiniLM-L6-v2
DEVICE=cpu  # o cuda si tienes GPU

# System Limits
MAX_PRODUCTS_TO_LOAD=10000
LOG_LEVEL=INFO
Inicialización del Sistema
bash
# Cargar datos y construir índices
python main.py index --force
# 🎯 Uso
Modo RAG (Sistema Híbrido)
bash
python main.py rag --user-age 25 --user-gender male --user-country Spain
Ejemplo de interacción:

text
🧑 You: juegos de acción para PlayStation 5

🤖 **Recomendaciones de videojuegos para 'juegos de acción para PlayStation 5'**

📀 **PlayStation**
  1. **Call of Duty: Modern Warfare III**
     💰 $69.99 | ⭐ 4.5/5
  2. **Spider-Man 2: Miles Morales**  
     💰 $59.99 | ⭐ 4.8/5

📊 System Info: 2 productos | Quality: 0.85
Gestión de Usuarios
bash
# Listar usuarios
python main.py users --list

# Ver estadísticas
python main.py users --stats
Reindexación
bash
# Reconstruir índice completo
python main.py index --force

# Reindexar con parámetros específicos
python cli.py index --batch-size 4000 --workers 4
# 🔧 Desarrollo
Componentes Principales
WorkingRAGAgent
Procesamiento híbrido: Combina RAG y filtrado colaborativo

Gestión de memoria: Mantiene contexto de conversación

RLHF integrado: Aprendizaje con feedback de usuarios

CollaborativeFilter
Búsqueda de similares: Encuentra usuarios con preferencias similares

Fallback inteligente: Usa categorías cuando no hay datos colaborativos

Ponderación temporal: Feedback reciente tiene más peso

UserManager
Perfiles persistentes: Almacena historial de usuarios

Estadísticas demográficas: Análisis de base de usuarios

Búsqueda de similitudes: Algoritmos de matching entre usuarios

Flujo de Datos
Carga de Productos → FastDataLoader

Indexación Vectorial → OptimizedChromaBuilder

Procesamiento de Consulta → WorkingRAGAgent

Recuperación Híbrida → Retriever + CollaborativeFilter

Generación de Respuesta → Templates optimizados

Procesamiento de Feedback → FeedbackProcessor + RLHF

# 🏗 Arquitectura
Sistema Híbrido de Recomendación
text
Consulta Usuario
    ↓
[WorkingRAGAgent]
    ├── RAG Tradicional (40%) → ChromaDB + Embeddings
    └── Colaborativo (60%) → UserManager + CollaborativeFilter
    ↓
Fusión de Scores
    ↓  
Respuesta Personalizada
Pipeline de RLHF
text
Feedback Usuario (1-5)
    ↓
[FeedbackProcessor]
    ↓
Almacenamiento en Logs
    ↓
Verificación Umbral (min_feedback)
    ↓
[RLHFTrainer] - Fine-tuning
    ↓  
[RLHFMonitor] - Tracking Métricas
# 🤝 Contribución
Estructura de Desarrollo
Nuevas Características: Crear branch feature/nueva-funcionalidad

Bug Fixes: Crear branch fix/descripcion-bug

Documentación: Actualizar README y comentarios de código

Guías de Estilo
Código: Seguir PEP 8

Documentación: Google-style docstrings

Commits: Conventional commits

Tests: Incluir tests unitarios para nuevas funcionalidades

Proceso de PR
Fork del repositorio

Crear branch de feature

Commit de cambios

Push al branch

Crear Pull Request

# 📊 Monitoreo y Métricas
El sistema incluye:

RLHF Monitor: Tracking de mejoras en el entrenamiento

User Analytics: Estadísticas demográficas y de uso

Performance Metrics: Tiempos de respuesta y calidad de recomendaciones

# 🚀 Despliegue
Requisitos de Producción
RAM: 16GB+ recomendado

Almacenamiento: 10GB+ para índices vectoriales

CPU: 4+ cores para procesamiento paralelo

GPU: Opcional para aceleración de embeddings

Escalabilidad
ChromaDB: Soporte para millones de productos

Procesamiento por Lotes: Manejo eficiente de datos grandes

Caché Distribuido: Posibilidad de integración con Redis