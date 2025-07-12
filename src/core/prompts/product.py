# src/core/prompts/product.py
from langchain_core.prompts import (
    ChatPromptTemplate,
    PromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)

class ProductPrompts:
    """
    Prompt templates specialized for a general Amazon product catalog.
    Products span multiple categories (tech, home, sports, etc.) and are
    loaded from .pkl files located at:
    C:\Users\evill\OneDrive\Documentos\Github\proyecto-ia-rag-rl\proyecto-ia-rag-rl\data\processed
    """

    SYSTEM_PROMPT = """You are an expert Amazon product assistant with deep knowledge of:
- Product specifications and features
- Price comparisons and value assessment
- User review analysis
- Category-specific recommendations
- Cross-category compatibility (e.g., accessories for laptops, filters for coffee machines)

Guidelines:
1. Always verify information against provided sources
2. Be concise but comprehensive
3. Highlight key product differentiators
4. Mention price/value when relevant
5. Indicate compatibility or required accessories if applicable"""

    # Main QA prompt
    QA_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template(
            "Context:\n{context}\n\n"
            "Question: {question}\n\n"
            "Answer in this format:\n"
            "- Summary: [1-2 sentence overview]\n"
            "- Key Features: [bulleted list]\n"
            "- Price: [if mentioned in context]\n"
            "- Best For: [type of users]"
        )
    ])

    # Product comparison
    COMPARE_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nCompare these products objectively:"
        ),
        HumanMessagePromptTemplate.from_template(
            "Products:\n{product_list}\n\n"
            "Comparison Criteria: {criteria}\n\n"
            "Provide detailed comparison in markdown table:\n"
            "| Feature | Product A | Product B |"
        )
    ])

    # Review summarization
    REVIEW_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nSummarize these reviews:"
        ),
        HumanMessagePromptTemplate.from_template(
            "Product: {product_name}\nReviews:\n{reviews}\n\n"
            "Summarize:\n"
            "- Positives: [bullets]\n"
            "- Negatives: [bullets]\n"
            "- Overall Sentiment: [positive/neutral/negative]"
        )
    ])

    # Personalized recommendations
    RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(
            SYSTEM_PROMPT + "\nRecommend products based on:"
        ),
        HumanMessagePromptTemplate.from_template(
            "User Needs: {user_needs}\n"
            "Budget: {budget}\n"
            "Preferred Brands: {brands}\n\n"
            "Available Products:\n{products}\n\n"
            "Recommend 3 options formatted as:\n"
            "1. [Product Name] - [Price]\n   - Why Recommended: [reason]\n   - Best For: [type of user]"
        )
    ])

    # Query rewrite for better retrieval
    QUERY_REWRITE_PROMPT = PromptTemplate(
        input_variables=["query"],
        template="""Analyze this Amazon product search query and rewrite it for better vector retrieval:
Original: {query}

Consider:
1. Expanding abbreviations
2. Adding implicit category context
3. Clarifying ambiguous terms

Rewritten Query:"""
    )

    # Dynamic filtering on PKL fields
    FILTER_PROMPT = PromptTemplate(
        input_variables=["question", "metadata_fields"],
        template="""Based on this question: "{question}"
Which of these metadata fields should be filtered?
Available Fields: {metadata_fields}

Respond with JSON format:
{{
  "filter_field": "field_name",
  "filter_value": ["value1", "value2"] OR "value"
}}"""
    )

    # Legacy simple prompt (kept for backward compatibility)
    SIMPLE_PROMPT_TEMPLATE = """You are an expert Amazon product assistant.

User Question:
{question}

Relevant Products:
{context}

Provide a concise answer highlighting the key product(s)."""

    @classmethod
    def get_prompt(cls, prompt_type: str = "qa") -> ChatPromptTemplate:
        """Get a prompt template by type."""
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