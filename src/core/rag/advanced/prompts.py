# src/core/rag/advanced/prompts.py
from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# PROMPTS COMPATIBLES CON evaluator.py - FORMATO EXACTO REQUERIDO
# ============================================================================

RELEVANCE_PROMPT = ChatPromptTemplate.from_template("""Evalúa si el documento es relevante para responder la pregunta.

Documento: {document}
Pregunta: {question}

¿El documento es relevante para responder la pregunta?
Responde EXACTAMENTE: "yes" (sí) o "no" (no) seguido de una breve explicación entre paréntesis.

Ejemplo: 
yes (el documento menciona características del producto solicitado)
no (el documento no contiene información relacionada con la pregunta)

Tu evaluación:""")

HALLUCINATION_PROMPT = ChatPromptTemplate.from_template("""Verifica si la respuesta está completamente basada en los documentos proporcionados.

Documentos de referencia:
{documents}

Respuesta a evaluar:
{generation}

¿Toda la información en la respuesta está respaldada por los documentos?
Responde EXACTAMENTE: "yes" (sí) o "no" (no) seguido de los elementos no respaldados entre paréntesis.

Ejemplo:
yes (toda la información está en los documentos)
no (la respuesta menciona "color azul" pero los documentos no especifican color)

Tu evaluación:""")

ANSWER_QUALITY_PROMPT = ChatPromptTemplate.from_template("""Evalúa la calidad de la respuesta en escala 1-5.

Pregunta original: {question}
Respuesta evaluada: {answer}

Criterios:
1. Muy pobre - No responde la pregunta
2. Pobre - Respuesta vaga o incompleta  
3. Aceptable - Responde pero con deficiencias
4. Buena - Respuesta clara y útil
5. Excelente - Respuesta completa, precisa y bien estructurada

Calificación: [1-5]
Explicación: [breve justificación]
Mejoras: [sugerencias si la calificación es menor a 4]

Ejemplo:
Calificación: 4
Explicación: La respuesta es clara y menciona productos relevantes
Mejoras: Podría incluir más detalles sobre precios

Tu evaluación:""")

# ============================================================================
# PROMPTS ADICIONALES PARA RAG (opcionales)
# ============================================================================

RAG_RESPONSE_PROMPT = ChatPromptTemplate.from_template("""Eres un asistente especializado en recomendaciones de productos de Amazon. 

Contexto de productos disponibles:
{context}

Pregunta del usuario: {question}

Proporciona una respuesta útil y precisa basándote únicamente en los productos del contexto. 
Si no hay productos relevantes, sugiere alternativas o explica por qué no puedes recomendar nada específico.

Formato de respuesta:
- Recomendaciones claras con productos específicos
- Precios y características cuando estén disponibles
- Explicación breve de por qué son relevantes

Mantén un tono amigable y profesional.""")