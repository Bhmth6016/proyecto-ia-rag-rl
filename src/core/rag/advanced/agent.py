# src/core/rag/advanced/agent.py
"""
Unified RAG agent for the Amazon catalog with multilingual support.

Features
--------
• Category-aware retrieval (CategoryTree)
• Runtime filtering (ProductFilter)
• Context-rich feedback (FeedbackProcessor)
• Optional RLHF fine-tuned model (LoRA checkpoint)
• Automatic query/response translation
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple

import torch
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain.prompts import PromptTemplate

# Internal imports
from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.llms.local_llm import local_llm
from src.core.rag.advanced.prompts import (
    RAG_PROMPT_TEMPLATE as SIMPLE_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    QUERY_CONSTRAINT_EXTRACTION_PROMPT
)
from src.core.config import settings
from src.core.utils.translator import TextTranslator, Language

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
    Enhanced RAG agent with multilingual support.
    Handles automatic translation of queries/responses.
    """

    def __init__(
        self,
        *,
        products: List[Dict],
        lora_checkpoint: Optional[Path] = None,
        enable_translation: bool = True
    ):
        self.products = products
        self.enable_translation = enable_translation
        self.embedder = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

        # Translation setup
        self.translator = TextTranslator() if enable_translation else None

        # Build category tree + filters
        self.tree = CategoryTree(products).build_tree()
        self.active_filter = ProductFilter()

        # Vector store
        self.vector_store = self._build_vector_store()

        # LLM (LoRA-aware)
        self.llm = local_llm(
            base_model_name=settings.BASE_LLM,
            lora_checkpoint=lora_checkpoint,
        )

        # LangChain memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
        )

        # LangChain chain
        self.chain = self._build_chain()

        # Feedback
        self.feedback = FeedbackProcessor(feedback_dir=str(settings.DATA_DIR / "feedback"))

    def _build_vector_store(self) -> FAISS:
        texts = [
            f"{p.get('title', '')} {' '.join(f'{k}:{v}' for k, v in p.get('details', {}).items())}"
            for p in self.products
        ]
        return FAISS.from_texts(texts=texts, embedding=self.embedder)

    def _build_chain(self) -> ConversationalRetrievalChain:
        prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=SIMPLE_PROMPT_TEMPLATE,
        )
        retriever = self.vector_store.as_retriever(
            search_kwargs={"k": 5, "filter": self.active_filter.apply}
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
        )

    def _process_query(self, query: str) -> Tuple[str, Optional[Language]]:
        """Handle query translation if enabled."""
        if not self.enable_translation:
            return query, None

        source_lang = self.translator.detect_language(query)
        if source_lang == Language.ENGLISH:
            return query, None

        english_query = self.translator.translate_to_english(query, source_lang)
        logger.debug(f"Translated query from {source_lang} to English: {english_query}")
        return english_query, source_lang

    def _process_response(self, answer: str, target_lang: Optional[Language]) -> str:
        """Handle response translation if needed."""
        if not target_lang or not self.enable_translation:
            return answer

        translated = self.translator.translate_from_english(answer, target_lang)
        logger.debug(f"Translated response back to {target_lang}: {translated}")
        return translated

    def ask(self, query: str) -> str:
        """
        Process query with optional translation:
        1. Detect input language
        2. Translate to English if needed
        3. Get LLM response
        4. Translate back to original language
        """
        processed_query, source_lang = self._process_query(query)
        english_answer = self.chain({"question": processed_query})["answer"]
        final_answer = self._process_response(english_answer, source_lang)

        # Store feedback with both original and translated versions
        retrieved_titles = [
            doc.metadata.get("title", "No title")
            for doc in self.vector_store.similarity_search(processed_query, k=5)
        ]
        self.feedback.save_feedback(
            query=query,
            answer=final_answer,
            rating=0,
            retrieved_docs=retrieved_titles,
            category_tree=self.tree,
            active_filter=self.active_filter,
            metadata={
                "english_query": processed_query if source_lang else None,
                "english_answer": english_answer if source_lang else None,
                "detected_language": source_lang.value if source_lang else "en"
            }
        )
        return final_answer

    def set_filters(
        self,
        *,
        price_range: Optional[tuple] = None,
        rating_range: Optional[tuple] = None,
        brands: Optional[list] = None,
    ) -> None:
        """Runtime filter update (hot-swapped in retriever)."""
        self.active_filter.clear_filters()
        if price_range:
            self.active_filter.add_price_filter(*price_range)
        if rating_range:
            self.active_filter.add_rating_filter(*rating_range)
        if brands:
            self.active_filter.add_brand_filter(brands)

    def chat_loop(self) -> None:
        """Interactive multilingual chat interface."""
        print("\n🌍 Multilingual RAG Agent (type 'exit' to quit)\n")

        while True:
            try:
                query = input("🧑 User: ").strip()
                if query.lower() in {"exit", "quit", "q"}:
                    print("👋 Goodbye!")
                    break

                answer = self.ask(query)
                print(f"\n🤖 Assistant:\n{answer}\n")

                # Feedback collection
                rating = input("Rating (1-5): ").strip()
                while rating not in {"1", "2", "3", "4", "5"}:
                    rating = input("Please rate 1-5: ").strip()
                
                self.feedback.save_feedback(
                    query=query,
                    answer=answer,
                    rating=int(rating),
                    extra_meta={
                        "user_rating": rating,
                        "translation_enabled": self.enable_translation
                    }
                )

            except KeyboardInterrupt:
                print("\n🛑 Session ended")
                break
            except Exception as e:
                logger.error("Chat error: %s", str(e))
                print("⚠️ Error occurred, please try again.")

    @classmethod
    def from_pickle_dir(
        cls,
        pickle_dir: Path = settings.PROC_DIR,
        lora_checkpoint: Optional[Path] = settings.RLHF_CHECKPOINT,
        enable_translation: bool = True
    ) -> "RAGAgent":
        """Factory method with translation option."""
        from src.core.data.loader import DataLoader

        products = DataLoader().load_data(
            raw_dir=settings.RAW_DIR,
            processed_dir=pickle_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        if not products:
            raise RuntimeError("No products found")
        return cls(
            products=products,
            lora_checkpoint=lora_checkpoint,
            enable_translation=enable_translation
        )

if __name__ == "__main__":
    agent = RAGAgent.from_pickle_dir(enable_translation=True)
    agent.chat_loop()
