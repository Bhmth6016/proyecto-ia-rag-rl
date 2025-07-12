# src/core/rag/advanced/prompts/__init__.py

from langchain_core.prompts import ChatPromptTemplate

# ============================================================================
# 1. PROMPTS DE GENERACIÓN
# ============================================================================

# 1.1 Prompt principal para generación RAG (atributos dinámicos)
RAG_PROMPT_TEMPLATE = ChatPromptTemplate.from_template("""
You are an Amazon product recommender. Recommend products matching:

User Query: {question}

Context:
{context}

Format each recommendation with:
1. 🏷️ **Product**: [Name]
2. 💵 **Price**: [Price]
3. ⭐ **Rating**: [Rating]/5
{dynamic_attributes}

**Why Recommended**: Explain how this product matches the user's criteria (be specific about price, rating, material, brand, etc.). Only include attributes present in the context.
""")

# 1.2 Prompt para extracción dinámica de atributos presentes en el contexto
DYNAMIC_ATTRIBUTES_EXTRACTOR = ChatPromptTemplate.from_template("""
Given the context below, return a comma-separated list of available product attributes EXCLUDING name, price, and rating.

Context:
{context}

Example output: material,color,size,brand,discount
Only return the attributes list, no extra text.
""")

# ============================================================================
# 2. PROMPTS DE PRE-PROCESAMIENTO
# ============================================================================

# 2.1 Reescritura de preguntas
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

# 2.2 Extracción de constraints estructurados
QUERY_CONSTRAINT_EXTRACTION_PROMPT = ChatPromptTemplate.from_template("""
Extract structured constraints from the user's Amazon search query.

Query: "{query}"

Return valid JSON with:
{{
  "category": "<general category if mentioned>",
  "attributes": {{"<key>": "<value>"}},
  "price_range": [min, max] | null,
  "min_rating": <number> | null,
  "max_rating": <number> | null,
  "brand": "<brand name>" | null,
  "sort_preference": "price_asc" | "price_desc" | "rating_desc" | "relevance" | null
}}

Only include fields present in the query. Do not invent data.
""")

# 2.3 Normalización multilingüe (opcional)
MULTILINGUAL_NORMALIZATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a multilingual e-commerce query normalizer. Detect the language and translate the query to English while preserving:
- Product specifications
- Brand names
- Technical terms
- Numerical values

Return ONLY the English translation."""
    ),
    ("human", "Query: {query}")
])

# ============================================================================
# 3. PROMPTS DE VALIDACIÓN Y CONTROL DE CALIDAD
# ============================================================================

# 3.1 Validación de respuestas
VALIDATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate if the response is fully supported by the context.

Respond EXCLUSIVELY with:
- 'yes' if all information is backed.
- 'no' followed by unsupported fragments in parentheses if hallucinations exist."""
    ),
    (
        "human",
        """Context:
{context}

Response to evaluate:
{answer}

Is fully supported? Answer (yes/no):"""
    )
])

# 3.2 Evaluación de relevancia
RELEVANCE_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate document relevance for answering the question.

Format:
Evaluation: [yes/no] (brief explanation)

Example:
Document: "4000mAh battery"
Question: "How long does the battery last?"
Evaluation: yes (mentions battery capacity)"""
    ),
    (
        "human",
        """Document:
{document}

Question:
{question}

Evaluation:"""
    )
])

# 3.3 Detección de alucinaciones
HALLUCINATION_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Identify any information in the response NOT supported by the documents.

Respond with:
- 'yes' if fully supported.
- 'no' followed by unsupported fragments.

Example:
Documents: "6.1 inch screen"
Response: "Has 6.1 inch screen and water resistance"
Evaluation: no (water resistance)"""
    ),
    (
        "human",
        """Reference documents:
{documents}

Response to evaluate:
{generation}

Evaluation:"""
    )
])

# 3.4 Evaluación de calidad de respuestas
ANSWER_QUALITY_PROMPT = ChatPromptTemplate.from_messages([
    (
        "system",
        """Evaluate response quality (1-5) based on:

- Accuracy (fact-based)
- Completeness (covers all aspects)
- Clarity (easy to understand)
- Usefulness (solves the question)
- Specificity (addresses user criteria)

Format:
Score: [1-5]
Explanation: [brief justification]
Improvements: [suggestions if score <4]"""
    ),
    (
        "human",
        """Original question:
{question}

Evaluated response:
{answer}

Please evaluate:"""
    )
])

# ============================================================================
# 4. EXPORTACIÓN
# ============================================================================
__all__ = [
    "RAG_PROMPT_TEMPLATE",
    "DYNAMIC_ATTRIBUTES_EXTRACTOR",
    "QUERY_REWRITE_PROMPT",
    "QUERY_CONSTRAINT_EXTRACTION_PROMPT",
    "MULTILINGUAL_NORMALIZATION_PROMPT",
    "VALIDATION_PROMPT",
    "RELEVANCE_PROMPT",
    "HALLUCINATION_PROMPT",
    "ANSWER_QUALITY_PROMPT"
]