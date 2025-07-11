# src/core/rag/advanced/prompts/init.py

from langchain_core.prompts import ChatPromptTemplate

# Prompt principal para generación RAG
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an Amazon product recommender. Recommend products matching:

User Query: {question}
Max Price: $30
Category: Beauty

Context:
{context}

Format each recommendation with:
1. 🏷️ Product: [Name]
2. 💵 Price: [Price]
3. ⭐ Rating: [Rating]/5
4. 📝 Why Recommended: [Explanation]
""")

# Prompt para reescritura de preguntas
QUERY_REWRITE_SYSTEM = """Eres un especialista en mejorar consultas de búsqueda para Amazon. Realiza:

- Expansión de abreviaturas/acrónimos.
- Adición de contexto implícito.
- Clarificación de términos ambiguos.
- Inclusión de sinónimos relevantes.
- Preservación de la intención original.

Devuelve SOLO la consulta mejorada, sin explicaciones."""
QUERY_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
    ("system", QUERY_REWRITE_SYSTEM),
    ("human", "Consulta original: {query}\nConsulta mejorada:")
])

# Prompt para validación de respuestas
VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evalúa si la respuesta está completamente soportada por el contexto.

Responde EXCLUSIVAMENTE con:
- 'yes' si toda la información está respaldada.
- 'no' seguido de los fragmentos no soportados entre paréntesis si hay alucinaciones."""
    ),
    (
        "human",
        """Contexto:
{context}

Respuesta a evaluar:
{answer}

¿Está completamente soportada? Responde (yes/no):"""
    )
])

# Prompt para evaluación de relevancia
RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evalúa la relevancia del documento para responder la pregunta.

Formato:
Evaluación: [yes/no] (breve explicación)

Ejemplo:
Documento: "Batería de 4000mAh"
Pregunta: "¿Cuánto dura la batería?"
Evaluación: yes (menciona capacidad de batería)"""
    ),
    (
        "human",
        """Documento:
{document}

Pregunta:
{question}

Evaluación:"""
    )
])

# Prompt para detección de alucinaciones
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Identifica cualquier información en la respuesta que NO esté respaldada por los documentos.

Responde con:
- 'yes' si está completamente respaldada.
- 'no' seguido de los fragmentos no soportados.

Ejemplo:
Documentos: "Pantalla de 6.1 pulgadas"
Respuesta: "Tiene pantalla de 6.1 pulgadas y resistencia al agua"
Evaluación: no (resistencia al agua)"""
    ),
    (
        "human",
        """Documentos de referencia:
{documents}

Respuesta a evaluar:
{generation}

Evaluación:"""
    )
])

# Prompt para evaluación de calidad de respuestas
ANSWER_QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evalúa la calidad de la respuesta (1 a 5) basado en:

- Precisión (basada en hechos)
- Completitud (cubre todos los aspectos)
- Claridad (fácil de entender)
- Utilidad (resuelve la pregunta)

Formato:
Puntuación: [1-5]
Explicación: [breve justificación]
Mejoras: [sugerencias si aplica; solo si la puntuación es <4]"""
    ),
    (
        "human",
        """Pregunta original:
{question}

Respuesta evaluada:
{answer}

Por favor evalúa:"""
    )
])

# Exportación controlada
__all__ = [
    "RAG_PROMPT_TEMPLATE",
    "QUERY_REWRITE_PROMPT",
    "VALIDATION_PROMPT",
    "RELEVANCE_PROMPT",
    "HALLUCINATION_PROMPT",
    "ANSWER_QUALITY_PROMPT"
]
