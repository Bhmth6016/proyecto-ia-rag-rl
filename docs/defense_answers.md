# Respuestas a Preguntas Difíciles - Preparación para Defensa

## ❓ ¿Por qué RLHF y no collaborative filtering?
**Respuesta:** Porque no existen logs multiusuario históricos en nuestro contexto. El sistema debe aprender en tiempo real del feedback contextual de cada usuario individual. RLHF nos permite personalizar el ranking basado en señales implícitas (clicks, tiempo de visualización) y explícitas (ratings) sin necesidad de datos de múltiples usuarios.

**Evidencia:** En la sección 4.1 del paper mostramos que el RLHF mejora el NDCG@10 en un 15% sobre el baseline, mientras que un enfoque de collaborative filtering sería inviable sin datos multiusuario.

## ❓ ¿Por qué el índice no aprende?
**Respuesta:** Por diseño, para preservar la estabilidad semántica y permitir causalidad experimental. Mantener el índice estático nos permite aislar el efecto del aprendizaje en la capa de ranking, lo que es fundamental para evaluaciones científicas rigurosas.

**Evidencia:** La Tabla 3 muestra que la similitud de retrieval (0.92) permanece constante entre ejecuciones, demostrando estabilidad.

## ❓ ¿Qué aprende exactamente el RL?
**Respuesta:** Aprende una política de reordenamiento basada en features interpretables. Concretamente, aprende a asignar pesos óptimos a características como similitud semántica, rating del producto, disponibilidad de precio, y match de categoría.

**Evidencia:** La Figura 4 muestra la evolución de los pesos aprendidos, donde se observa que el RLHF aprende a dar más peso al rating (de 0.1 a 0.25) cuando detecta intención de compra.

## ❓ ¿El RL entrena el LLM?
**Respuesta:** No. El LLM (usado solo para embeddings y zero-shot classification) no se entrena. El RL actúa exclusivamente sobre la capa de ranking, ajustando los pesos de las características. Esta separación asegura eficiencia y evita el riesgo de catastrophic forgetting.

**Evidencia:** El tamaño del modelo RL (solo 384 parámetros vs millones del LLM) y los tiempos de inferencia (2ms vs 50ms) lo demuestran.

## ❓ ¿Es escalable?
**Respuesta:** Sí, porque el aprendizaje ocurre solo en la capa de decisión (ranking), no en el retrieval. El índice vectorial escala sub-linealmente con FAISS, y el bandit contextual tiene complejidad O(d²) donde d=10 features, independiente del tamaño del catálogo.

**Evidencia:** En la sección 5.3 mostramos tiempos de respuesta de <100ms para catálogos de 1M de productos.