# *Título tentativo del proyecto*

**Sistema Híbrido de Recomendación Inteligente basado en RAG, Redes Neuronales y Aprendizaje por Refuerzo**

## 1. Breve descripción del problema o aplicación a abordar

Los sistemas modernos de recomendación enfrentan el desafío de procesar catálogos extensos de productos y adaptarse a las preferencias cambiantes de los usuarios. Los métodos tradicionales, como el filtrado colaborativo o la búsqueda por palabras clave, presentan limitaciones en precisión, escalabilidad y personalización.

Esta propuesta consiste en el desarrollo de un sistema de recomendación híbrido que integre:

* Retrieval-Augmented Generation (RAG) para búsqueda semántica mediante modelos Transformer.

* Redes neuronales para generar representaciones densas de productos a partir de metadatos.

* Aprendizaje por refuerzo contextual para personalizar recomendaciones basándose en las interacciones en tiempo real del usuario.

El objetivo principal es mejorar la relevancia, diversidad y adaptabilidad de las recomendaciones en comparación con los modelos clásicos.

## 2. Objetivos del proyecto
### Objetivo general

Desarrollar un sistema de recomendación inteligente basado en búsqueda semántica, redes neuronales y aprendizaje por refuerzo, capaz de aprender y mejorar continuamente a partir de la interacción del usuario.

### Objetivos específicos

* Implementar un módulo de búsqueda semántica utilizando modelos Transformer para generar embeddings de productos.

* Construir un recomendador basado en metadatos mediante redes neuronales (Autoencoder o MLP).

* Diseñar un agente de aprendizaje por refuerzo neuronal que aprenda de las interacciones del usuario.

* Integrar los componentes en un pipeline híbrido tipo RAG.

* Evaluar el sistema con métricas específicas de recomendación y aprendizaje profundo.

* Establecer un flujo de aprendizaje continuo basado en feedback.

## 3. Descripción del conjunto de datos a utilizar

Para este proyecto se utilizará un dataset construido a partir de archivos JSONL con productos de Amazon obtenidos desde fuentes públicas.

El dataset será procesado mediante un módulo personalizado (DataLoader), el cual se encargará de:

* Estandarizar y limpiar campos como título, descripción, rating y precio.

* Inferir y normalizar metadatos como categoría, etiquetas, tipo de producto y especificaciones.

* Generar un dataset final en formato JSON o PKL adecuado para tareas de NLP, RL y Deep Learning.

Variables principales del dataset:

* Texto: título, descripción, características.

* Metadatos: categoría, tags inferidos, especificaciones.

* Numéricas: precio, rating.

* Representaciones profundas: embeddings generados por Transformers.

Este dataset es adecuado para búsqueda semántica, recomendación híbrida y aprendizaje por refuerzo.

## 4. Arquitecturas que se planean implementar

El proyecto incorporará diversas arquitecturas neuronales modernas:

### 4.1 Transformers (para RAG y búsqueda semántica)

* Modelos: MiniLM, DistilBERT o E5-small.

* Uso: Generación de embeddings densos para consultas y productos.

* Entrenamiento: Posible fine-tuning con Triplet Loss o Cosine Similarity Loss.

### 4.2 Autoencoder o MLP (para recomendación basada en metadatos)

Autoencoder:

* Entrada: vectores TF-IDF o metadatos concatenados.

* Representación latente en el cuello de botella.

* Uso para similitud entre productos.

MLP:

* Múltiples capas densas.

* Aprendizaje de relaciones no lineales entre metadatos del producto.

### 4.3 Neural Contextual Bandit (Aprendizaje por Refuerzo)

* Modelo: MLP para predicción de la recompensa esperada.

* Entrada: contexto del usuario más embedding del producto.

* Política: epsilon-greedy o Thompson Sampling neuronal.

* Alternativa: DQN simplificado para tareas de selección de acciones.

### 4.4 Pipeline híbrido RAG

* Indexación vectorial mediante FAISS.

* Recuperación semántica basada en embeddings neuronales.

* Fusión de señales mediante ponderación (weighted fusion) y el módulo de RL.

## 5. DETALLES DE LOS DATOS DEL PROYECTO
### 5.1. DATOS DE PRODUCTOS
Fuente:
python
# data/raw/*.jsonl - Formato original
{
  "id": "B0D12C7Y5N",
  "title": "Nintendo Switch OLED - Mario Red Edition", 
  "main_category": "consoles",
  "price": 349.99,
  "average_rating": 4.8,
  "description": "Consola Nintendo Switch OLED edición especial Mario Rojo",
  "categories": ["consoles", "electronics", "gaming"],
  "rating_count": 500,
  "tags": ["gaming", "electronics", "amazon_choice"],
  "details": {
    "features": ["Feature 1", "Feature 2"],
    "specifications": {
      "brand": "Nintendo",
      "color": "red", 
      "weight": "2 kg"
    }
  }
}
Procesamiento (loader.py):
python
# data/processed/products.json - Formato procesado
Product.from_dict() → Normalización automática:
- Limpieza de títulos y descripciones
- Parseo inteligente de precios
- Extracción automática de características
- Generación de content_hash para duplicados
Para qué sirven:
Embeddings: Para búsqueda semántica en ChromaDB

Metadata: Para filtrado y scoring

Presentación: Para generar respuestas al usuario

### 5.2. DATOS DE USUARIOS
Fuente:
python
# data/users/*.json - Perfiles persistentes
{
  "user_id": "user_001",
  "age": 25,
  "gender": "male", 
  "country": "Spain",
  "preferred_categories": ["games", "consoles"],
  "preferred_brands": ["Sony", "Nintendo"],
  "feedback_history": [
    {
      "query": "juegos nintendo switch",
      "rating": 5,
      "selected_product": "B0D12C7Y5N",
      "timestamp": "2024-01-15T10:30:00"
    }
  ]
}
Para qué sirven:
Filtro colaborativo: Encontrar usuarios similares

Personalización: Adaptar recomendaciones

RLHF: Entrenar con preferencias reales

### 5.3. DATOS DE FEEDBACK
Fuentes múltiples:
python
# data/feedback/success_queries.log - Éxitos (rating 4-5)
# data/feedback/failed_queries.log - Fallos (rating 1-3)  
# data/feedback/feedback_*.jsonl - Tiempo real
# data/processed/historial/conversation_*.json - Histórico
Para qué sirven:
RLHF: Dataset de entrenamiento

Evaluación: Métricas de calidad

Mejora continua: Aprendizaje automático

### 5.4. MÉTRICAS RLHF
Fuente:
python
# data/feedback/rlhf_metrics/training_metrics.jsonl
{
  "timestamp": "2024-01-15T10:30:00",
  "examples_used": 150,
  "previous_accuracy": 0.65,
  "new_accuracy": 0.72,
  "improvement": 0.07,
  "training_time_seconds": 1200
}
## 6. Métricas de evaluación y metodología de ajuste de hiperparámetros
### 6.1 Métricas para modelos semánticos

* Cosine Similarity.

* Triplet Loss.

* Recall@k para evaluar recuperación de productos relevantes.

### 6.2 Métricas para recomendación

* Precision@k.

* Recall@k.

* NDCG@k.

* MAP (Mean Average Precision).

* MSE (en caso de usar autoencoders).

### 6.3 Métricas para Aprendizaje por Refuerzo

* Recompensa promedio.

* Regret acumulado.

* CTR (Click-Through Rate).

* Conversion Rate.

* Relación exploración/explotación.

### 6.4 Ajuste de hiperparámetros

* Grid Search o Random Search.

* Early Stopping.

* Regularización (Dropout, Weight Decay).

* Ajuste de learning rate, batch size y optimizador (Adam).

## 7. Resultados esperados

* Generación de embeddings semánticos precisos gracias al uso de modelos Transformer.

* Representaciones densas mediante autoencoder o MLP que aumenten la relevancia y diversidad de recomendaciones.

* Un agente de RL capaz de aprender de las interacciones del usuario en tiempo real.

* Un sistema híbrido (RAG + RL + Redes Neuronales) con mejoras significativas en métricas como Precision@k, NDCG y CTR respecto a sistemas tradicionales.

* Incremento progresivo en la recompensa promedio del agente conforme acumula experiencia, demostrando aprendizaje continuo.

# Comentarios

Describir los detalles de los datos en la presentación del avance.
