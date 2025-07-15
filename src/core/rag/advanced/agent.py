# src/core/rag/advanced/agent.py
"""
Unified RAG agent for the Amazon catalog with multilingual support
& synonym-aware response templates.

Features
--------
• Category-aware retrieval (CategoryTree)
• Runtime filtering (ProductFilter)
• Context-rich feedback (FeedbackProcessor)
• Automatic query/response translation
• Smart fallback when no / partial results
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

# Internal imports
from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.advanced.prompts import (
    RAG_PROMPT_TEMPLATE as SIMPLE_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    NO_RESULTS_TEMPLATE,
    PARTIAL_RESULTS_TEMPLATE,
)
from src.core.config import settings
from src.core.utils.translator import TextTranslator, Language
from src.core.rag.basic.retriever import Retriever  # synonym-aware retriever

# ------------------------------------------------------------------
# Logging
# ------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)

# ------------------------------------------------------------------
# Agent
# ------------------------------------------------------------------
class RAGAgent:
    """
    Enhanced RAG agent with multilingual support and smart fallback.
    """

    def __init__(
        self,
        *,
        products: List[Dict],
        enable_translation: bool = True,
    ):
        self.products = products
        self.enable_translation = enable_translation

        # Translation
        self.translator = TextTranslator() if enable_translation else None

        # Category tree & filters
        self.tree = CategoryTree(products).build_tree()
        self.active_filter = ProductFilter()

        # Vector store via synonym-aware retriever
        self.retriever = Retriever()

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GOOGLE_API_KEY,
            temperature=0.3,
        )

        # LangChain memory & chain
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
        )
        self.chain = self._build_chain()

        # Feedback
        self.feedback = FeedbackProcessor(
            feedback_dir=str(settings.DATA_DIR / "feedback")
        )

    # ------------------------------------------------------------------
    # Build chain
    # ------------------------------------------------------------------
    def _build_chain(self) -> ConversationalRetrievalChain:
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context below. "
            "If you don't know the answer, say you don't know. "
            "Be concise and helpful.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.store.as_retriever(
                search_kwargs={"k": 5, "filter": self.active_filter.apply}
            ),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
        )

    # ------------------------------------------------------------------
    # Query / response helpers
    # ------------------------------------------------------------------
    def _process_query(self, query: str) -> Tuple[str, Optional[Language]]:
        """Translate to English if needed."""
        if not self.enable_translation:
            return query, None

        source_lang = self.translator.detect_language(query)
        if source_lang == Language.ENGLISH:
            return query, None

        english_query = self.translator.translate_to_english(query, source_lang)
        logger.debug("Translated query: %s -> %s", query, english_query)
        return english_query, source_lang

    def _process_response(self, answer: str, target_lang: Optional[Language]) -> str:
        """Translate back if needed."""
        if not target_lang or not self.enable_translation:
            return answer
        return self.translator.translate_from_english(answer, target_lang)

    # ------------------------------------------------------------------
    # Ask
    # ------------------------------------------------------------------
    def ask(self, query: str) -> str:
        """
        1. Normalise + expand synonyms
        2. Retrieve
        3. Generate answer with templates if no / partial results
        """
        processed_query, source_lang = self._process_query(query)

        # Retrieve products
        products = self.retriever.retrieve(query=processed_query, k=5)

        # --- Fallback & templates ---
        if not products:
            # fallback with original query
            products = self.retriever.retrieve(query=query, k=5)

        if not products:
            suggestions = ", ".join(
                self.retriever._expand_query(query)[:3]
            ) or "backpack, headphones, speaker"
            return NO_RESULTS_TEMPLATE.format(query=query, suggestions=suggestions)

        # partial results
        exact = [
            p for p in products
            if self.retriever._score(query, p) > 0
        ]
        if not exact and products:
            titles = [p.title for p in products[:3]]
            return PARTIAL_RESULTS_TEMPLATE.format(
                query=query,
                similar_products="\n".join(f"- {t}" for t in titles)
            )

        # Build context
        context = "\n\n".join(p.to_document()["page_content"] for p in products)

        # Generate answer
        llm_answer = self.chain.invoke({"question": processed_query, "context": context})["answer"]
        final_answer = self._process_response(llm_answer, source_lang)

        # Feedback
        self.feedback.save_feedback(
            query=query,
            answer=final_answer,
            rating=0,
            retrieved_docs=[p.title for p in products],
            category_tree=self.tree,
            active_filter=self.active_filter,
            extra_meta={
                "english_query": processed_query if source_lang else None,
                "detected_language": source_lang.value if source_lang else "en",
            },
        )
        return final_answer

    # ------------------------------------------------------------------
    # Runtime filters
    # ------------------------------------------------------------------
    def set_filters(
        self,
        *,
        price_range: Optional[tuple] = None,
        rating_range: Optional[tuple] = None,
        brands: Optional[list] = None,
    ) -> None:
        self.active_filter.clear_filters()
        if price_range:
            self.active_filter.add_price_filter(*price_range)
        if rating_range:
            self.active_filter.add_rating_filter(*rating_range)
        if brands:
            self.active_filter.add_brand_filter(brands)

    # ------------------------------------------------------------------
    # Chat loop
    # ------------------------------------------------------------------
    def chat_loop(self) -> None:
        print("\n🌍 Multilingual RAG Agent (type 'exit' to quit)\n")
        while True:
            try:
                query = input("🧑 User: ").strip()
                if query.lower() in {"exit", "quit", "q"}:
                    print("👋 Goodbye!")
                    break

                answer = self.ask(query)
                print(f"\n🤖 Assistant:\n{answer}\n")

                rating = input("Rating (1-5): ").strip()
                while rating not in {"1", "2", "3", "4", "5"}:
                    rating = input("Please rate 1-5: ").strip()

                self.feedback.save_feedback(
                    query=query,
                    answer=answer,
                    rating=int(rating),
                    extra_meta={"user_rating": rating},
                )

            except KeyboardInterrupt:
                print("\n🛑 Session ended")
                break
            except Exception as e:
                logger.error("Chat error: %s", e)
                print("⚠️ Error occurred, please try again.")

    # ------------------------------------------------------------------
    # Factory
    # ------------------------------------------------------------------
    @classmethod
    def from_pickle_dir(
        cls,
        pickle_dir: Path = settings.PROC_DIR,
        enable_translation: bool = True,
    ) -> "RAGAgent":
        from src.core.data.loader import DataLoader

        products = DataLoader().load_data(
            raw_dir=settings.RAW_DIR,
            processed_dir=pickle_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        if not products:
            raise RuntimeError("No products found")
        return cls(products=products, enable_translation=enable_translation)

# ------------------------------------------------------------------
# CLI
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = RAGAgent.from_pickle_dir(enable_translation=True)
    agent.chat_loop()