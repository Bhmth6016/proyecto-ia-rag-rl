# src/core/prompts/rewrite.py
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from typing import List, Dict, Optional
import re
from src.core.utils.logger import get_logger

logger = get_logger(__name__)

class QueryRewriter:
    """Handles query rewriting for better product search retrieval"""
    
    # Main rewrite prompt template
    REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """You are a search query specialist for Amazon products. Improve queries by:
1. Expanding abbreviations and acronyms
2. Adding implicit category context
3. Clarifying ambiguous terms
4. Including common synonyms
5. Preserving the original intent

Return ONLY the improved query, no additional text."""),
        ("human", "Original query: {query}\nImproved query:")
    ])

    # Contextual rewrite template
    CONTEXTUAL_REWRITE_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Improve this product search query considering the conversation history.
Keep it concise while adding necessary context.

Conversation History:
{history}

Guidelines:
- Maintain the current search intent
- Resolve pronouns (it, they) to actual product names
- Add only relevant context"""),
        ("human", "Original query: {query}\nContextual query:")
    ])

    # Spelling correction template
    SPELLING_PROMPT = ChatPromptTemplate.from_messages([
        ("system", """Correct any spelling mistakes in this Amazon product search query.
Return ONLY the corrected query, no explanations.

Common corrections:
- 'hedphones' → 'headphones'
- 'lapop' → 'laptop'
- 'samsng' → 'samsung'"""),
        ("human", "Query: {query}\nCorrected query:")
    ])

    def __init__(self, llm):
        self.llm = llm
        self.rewrite_chain = self.REWRITE_PROMPT | self.llm | StrOutputParser()
        self.contextual_chain = self.CONTEXTUAL_REWRITE_PROMPT | self.llm | StrOutputParser()
        self.spelling_chain = self.SPELLING_PROMPT | self.llm | StrOutputParser()

    def rewrite_query(
        self,
        query: str,
        history: Optional[List[Dict]] = None,
        correct_spelling: bool = True
    ) -> str:
        """
        Improve a product search query for better retrieval
        
        Args:
            query: Original search query
            history: Conversation history for context
            correct_spelling: Whether to apply spelling correction
            
        Returns:
            Rewritten query optimized for vector search
        """
        try:
            # Step 1: Spelling correction
            if correct_spelling:
                query = self.spelling_chain.invoke({"query": query}).strip()
            
            # Step 2: Contextual or basic rewrite
            if history and len(history) > 0:
                formatted_history = "\n".join(
                    f"{msg['role']}: {msg['content']}" 
                    for msg in history
                )
                rewritten = self.contextual_chain.invoke({
                    "query": query,
                    "history": formatted_history
                })
            else:
                rewritten = self.rewrite_chain.invoke({"query": query})
            
            # Clean and validate
            rewritten = self._clean_query(rewritten)
            if not self._validate_rewrite(query, rewritten):
                logger.warning(f"Rewrite validation failed, using original: {query}")
                return query
                
            return rewritten
            
        except Exception as e:
            logger.error(f"Query rewrite failed: {str(e)}")
            return query

    def _clean_query(self, query: str) -> str:
        """Remove artifacts from LLM output"""
        # Remove quotes if present
        query = re.sub(r'^["\']|["\']$', '', query.strip())
        # Remove any explanatory text
        query = query.split("\n")[0].split(":")[-1].strip()
        return query

    def _validate_rewrite(self, original: str, rewritten: str) -> bool:
        """Basic validation that rewrite preserves intent"""
        # Check for empty or extremely short queries
        if len(rewritten) < 3 or len(rewritten) > 200:
            return False
            
        # Check if rewrite dropped key terms
        original_terms = set(original.lower().split())
        rewritten_terms = set(rewritten.lower().split())
        return not original_terms.isdisjoint(rewritten_terms)

    @staticmethod
    def add_category_context(query: str, category: str) -> str:
        """Add explicit category context when needed"""
        ambiguous_terms = {
            "adapter", "cable", "case", "cover", "stand", 
            "charger", "battery", "protector"
        }
        
        query_terms = set(query.lower().split())
        if (not query_terms.intersection({"for", "in"}) and 
            query_terms.intersection(ambiguous_terms)):
            return f"{query} for {category}"
        return query

# Example usage (would be in __main__ or tests)
if __name__ == "__main__":
    from langchain_community.llms import FakeListLLM
    
    # Test with mock LLM
    mock_llm = FakeListLLM(responses=[
        "wireless headphones with noise cancellation under $100"
    ])
    
    rewriter = QueryRewriter(mock_llm)
    test_query = "noise cancelling buds under 100 bucks"
    print(f"Original: {test_query}")
    print(f"Rewritten: {rewriter.rewrite_query(test_query)}")