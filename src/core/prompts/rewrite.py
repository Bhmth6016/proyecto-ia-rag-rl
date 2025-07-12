# src/core/prompts/rewrite.py
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class ProductPrompts:
    """
    Prompt templates specialized for a **general-purpose Amazon catalog**.
    Products span every category (electronics, home, sports, fashion, etc.) and are
    loaded from `.pkl` files located at:
    C:\Users\evill\OneDrive\Documentos\Github\proyecto-ia-rag-rl\proyecto-ia-rag-rl\data\processed
    """

    # ------------------------------------------------------------------
    # Core system prompt
    # ------------------------------------------------------------------
    SYSTEM_PROMPT = """You are an expert Amazon product assistant with deep knowledge of:
- Product specifications and features across all categories
- Price comparisons and value assessment
- User review analysis and sentiment extraction
- Category-specific recommendations and use-cases
- Cross-category compatibility (e.g., accessories for devices, replacement parts)

Guidelines:
1. Always verify claims against the provided context
2. Keep answers concise yet comprehensive
3. Highlight key differentiators, pros/cons, and best-use scenarios
4. Mention price ranges or value statements when available
5. Note compatibility, required accessories, or bundle opportunities"""

    # ------------------------------------------------------------------
    # 1. Main Q&A prompt (default)
    # ------------------------------------------------------------------
    QA_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Structure your answer as:\n"
            "- Summary: 1–2 sentences\n"
            "- Key Features: bullet list\n"
            "- Price/Value: if context provides it\n"
            "- Best For: user profile or use-case"
        )
    ])

    # ------------------------------------------------------------------
    # 2. Product comparison
    # ------------------------------------------------------------------
    COMPARE_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nCompare the products below objectively."
        ),
        HumanMessagePromptTemplate.from_template(
            "Products:\n{product_list}\n\n"
            "Focus on: {criteria}\n\n"
            "Return a markdown table:\n"
            "| Feature | Product A | Product B |\n"
            "|---------|-----------|-----------|"
        )
    ])

    # ------------------------------------------------------------------
    # 3. Review summarization
    # ------------------------------------------------------------------
    REVIEW_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nSummarize customer reviews concisely."
        ),
        HumanMessagePromptTemplate.from_template(
            "Product: {product_name}\nReviews:\n{reviews}\n\n"
            "Output:\n"
            "- Positives: bullet points\n"
            "- Negatives: bullet points\n"
            "- Overall Sentiment: positive / neutral / negative"
        )
    ])

    # ------------------------------------------------------------------
    # 4. Personalized recommendations
    # ------------------------------------------------------------------
    RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nRecommend the best 3 products from the list."
        ),
        HumanMessagePromptTemplate.from_template(
            "User Needs: {user_needs}\n"
            "Budget: {budget}\n"
            "Preferred Brands: {brands}\n\n"
            "Available Products:\n{products}\n\n"
            "Return:\n"
            "1. Product Name - Price\n   Reason: …\n   Best For: …\n"
            "2. Product Name - Price\n   Reason: …\n   Best For: …\n"
            "3. Product Name - Price\n   Reason: …\n   Best For: …"
        )
    ])

    # ------------------------------------------------------------------
    # 5. Query rewrite for vector retrieval
    # ------------------------------------------------------------------
    QUERY_REWRITE_PROMPT = PromptTemplate(
        input_variables=["query"],
        template="""Improve this Amazon search query for vector retrieval:
Original: {query}

Actions:
1. Expand abbreviations
2. Add implicit category context
3. Clarify ambiguous terms

Rewritten Query:"""
    )

    # ------------------------------------------------------------------
    # 6. Dynamic filtering on PKL metadata
    # ------------------------------------------------------------------
    FILTER_PROMPT = PromptTemplate(
        input_variables=["question", "metadata_fields"],
        template="""Given the question: "{question}"
Which metadata fields should be filtered?
Available Fields: {metadata_fields}

Respond in JSON:
{{
  "filter_field": "field_name",
  "filter_value": ["value1", "value2"] OR "value"
}}"""
    )

    # ------------------------------------------------------------------
    # 7. Legacy simple prompt (backward compatibility)
    # ------------------------------------------------------------------
    SIMPLE_PROMPT_TEMPLATE = """You are an expert Amazon product assistant.

User Question:
{question}

Relevant Products:
{context}

Provide a concise answer highlighting the key product(s)."""

    # ------------------------------------------------------------------
    # Prompt selector
    # ------------------------------------------------------------------
    @classmethod
    def get_prompt(cls, prompt_type: str = "qa") -> ChatPromptTemplate:
        """Return the requested prompt template."""
        prompts = {
            "qa": cls.QA_PROMPT,
            "compare": cls.COMPARE_PROMPT,
            "review": cls.REVIEW_PROMPT,
            "recommend": cls.RECOMMEND_PROMPT,
            "rewrite": cls.QUERY_REWRITE_PROMPT,
            "filter": cls.FILTER_PROMPT,
            "simple": PromptTemplate.from_template(cls.SIMPLE_PROMPT_TEMPLATE),
        }
        return prompts.get(prompt_type.lower(), cls.QA_PROMPT)