from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
# ============================================================================
# PROMPTS PARA EVALUATOR - RESPETAR ESTOS NOMBRES
# ============================================================================

RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""
Evalúa si el documento es relevante para responder la pregunta.

Documento: {document}
Pregunta: {question}

¿El documento es relevante para responder la pregunta?
Responde EXACTAMENTE: "yes" o "no" seguido de una explicación breve entre paréntesis.

Ejemplo:
yes (el documento menciona características del producto solicitado)
no (el documento no contiene información relacionada con la pregunta)

Tu evaluación:
""")

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""
Verifica si la respuesta está completamente basada en los documentos proporcionados.

Documentos de referencia:
{documents}

Respuesta a evaluar:
{generation}

¿Toda la información en la respuesta está respaldada por los documentos?
Responde EXACTAMENTE: "yes" o "no" seguido de los elementos no respaldados entre paréntesis.

Ejemplo:
yes (toda la información está en los documentos)
no (la respuesta menciona 'color azul' pero los documentos no mencionan colores)

Tu evaluación:
""")

ANSWER_QUALITY_PROMPT = ChatPromptTemplate.from_template("""
Evalúa la calidad de la respuesta en escala 1-5.

Pregunta original: {question}
Respuesta evaluada: {answer}

Criterios:
1. Muy pobre — No responde la pregunta
2. Pobre — Respuesta vaga o incompleta
3. Aceptable — Responde, pero con deficiencias
4. Buena — Respuesta clara y útil
5. Excelente — Respuesta completa, precisa y bien estructurada

Formato esperado:
Calificación: [1-5]
Explicación: [breve justificación]
Mejoras: [sugerencias si la calificación es menor a 4]

Ejemplo:
Calificación: 4
Explicación: La respuesta es clara y menciona productos relevantes
Mejoras: Podría incluir más detalles sobre precios

Tu evaluación:
""")

# ============================================================================
# PROMPTS ADICIONALES PARA WORKING RAG AGENT
# ============================================================================

RAG_RESPONSE_PROMPT = ChatPromptTemplate.from_template("""
Eres un asistente especializado en recomendaciones de productos de Amazon. 

Contexto de productos disponibles:
{context}

Pregunta del usuario: {question}

Proporciona una respuesta útil, clara y precisa basándote únicamente en los productos del contexto.
Si no hay productos relevantes, sugiere alternativas o justifica por qué no puedes recomendar nada específico.

Mantén un tono profesional y amigable.
""")

QUERY_ENRICHMENT_PROMPT = ChatPromptTemplate.from_template("""
Mejora la siguiente consulta de búsqueda para productos de Amazon:

Consulta original: {query}

Objetivos:
- Expandir términos técnicos o abreviaturas
- Añadir sinónimos relevantes
- Clarificar ambigüedades
- Mantener la intención original

Devuelve SOLO la consulta mejorada sin explicaciones:
""")
def load_evaluator_llm():
    """
    Carga un modelo rápido para las evaluaciones:
    - Gemini 1.5 Flash (rápido, barato, ideal para evaluator)
    """
    return ChatGoogleGenerativeAI(
        model="gemini-1.5-flash",
        temperature=0
    )