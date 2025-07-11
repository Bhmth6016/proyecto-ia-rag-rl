# src/core/prompts/product.py
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain.prompts import SystemMessagePromptTemplate, HumanMessagePromptTemplate

class ProductPrompts:
    """Prompt templates specialized for Amazon product RAG system"""
    
    # System message defining the AI's role
    SYSTEM_PROMPT = """You are an expert Amazon product assistant with deep knowledge of:
- Product specifications and features
- Price comparisons and value assessment
- User review analysis
- Category-specific recommendations

Guidelines:
1. Always verify information against provided sources
2. Be concise but comprehensive
3. Highlight key product differentiators
4. Mention price/value when relevant"""

    # Main QA prompt template
    QA_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT),
        HumanMessagePromptTemplate.from_template("""Context:
{context}

Question: {question}
Answer in this format:
- Summary: [1-2 sentence overview]
- Key Features: [bulleted list]
- Price: [if mentioned in context]
- Best For: [type of users]""")
    ])

    # Product comparison prompt
    COMPARE_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT + "\nCompare these products objectively:"),
        HumanMessagePromptTemplate.from_template("""Products:
{product_list}

Comparison Criteria: {criteria}
Provide detailed comparison in table format with columns: Feature, Product A, Product B""")
    ])

    # Review summarization prompt
    REVIEW_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT + "\nSummarize these reviews:"),
        HumanMessagePromptTemplate.from_template("""Product: {product_name}
Reviews:
{reviews}

Summarize in this format:
- Positives: [bullets]
- Negatives: [bullets]
- Overall Sentiment: [positive/neutral/negative]""")
    ])

    # Recommendation prompt
    RECOMMEND_PROMPT = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template(SYSTEM_PROMPT + "\nRecommend products based on:"),
        HumanMessagePromptTemplate.from_template("""User Needs: {user_needs}
Budget: {budget}
Preferred Brands: {brands}

Available Products:
{products}

Recommend 3 options formatted as:
1. [Product Name] - [Price]
   - Why Recommended: [reason]
   - Best For: [type of user]""")
    ])

    # Query understanding prompt
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

    # Metadata filtering prompt
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

    @classmethod
    def get_prompt(cls, prompt_type: str) -> ChatPromptTemplate:
        """Get a prompt template by type"""
        prompts = {
            "qa": cls.QA_PROMPT,
            "compare": cls.COMPARE_PROMPT,
            "review": cls.REVIEW_PROMPT,
            "recommend": cls.RECOMMEND_PROMPT,
            "rewrite": cls.QUERY_REWRITE_PROMPT,
            "filter": cls.FILTER_PROMPT
        }
        return prompts.get(prompt_type.lower(), cls.QA_PROMPT)