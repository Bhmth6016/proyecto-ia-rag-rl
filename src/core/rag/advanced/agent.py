# src/core/rag/advanced/agent.py
"""
Unified RAG agent for the Amazon catalog.

Features
--------
• Category-aware retrieval (CategoryTree)
• Runtime filtering (ProductFilter)
• Context-rich feedback (FeedbackProcessor)
• Optional RLHF fine-tuned model (LoRA checkpoint)
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional

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
from src.core.prompts.product import ProductPrompts
from src.core.config import settings

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
    One-stop RAG agent that is aware of:
    • product categories
    • user filters
    • feedback collection
    • RLHF checkpoints
    """

    def __init__(
        self,
        *,
        products: List[Dict],
        lora_checkpoint: Optional[Path] = None,
    ):
        self.products = products
        self.embedder = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )

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

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------
    def _build_vector_store(self) -> FAISS:
        texts = [
            f"{p.get('title', '')} {' '.join(f'{k}:{v}' for k, v in p.get('details', {}).items())}"
            for p in self.products
        ]
        return FAISS.from_texts(texts=texts, embedding=self.embedder)

    def _build_chain(self) -> ConversationalRetrievalChain:
        prompt = PromptTemplate(
            input_variables=["chat_history", "question", "context"],
            template=ProductPrompts.SIMPLE_PROMPT_TEMPLATE,
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

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------
    def ask(self, query: str) -> str:
        """
        Run retrieval + generation and store feedback skeleton.
        """
        answer = self.chain({"question": query})["answer"]

        # Fire-and-forget feedback record
        retrieved_titles = [
            doc.metadata.get("title", "No title")
            for doc in self.vector_store.similarity_search(query, k=5)
        ]
        self.feedback.save_feedback(
            query=query,
            answer=answer,
            rating=0,  # placeholder; user will update via CLI
            retrieved_docs=retrieved_titles,
            category_tree=self.tree,
            active_filter=self.active_filter,
        )
        return answer

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

    # ------------------------------------------------------------------
    # CLI
    # ------------------------------------------------------------------
    def chat_loop(self) -> None:
        """Interactive CLI exactly like the old rag_agent.py"""
        print("\n🧠 Agente listo. Escribe 'salir' para terminar.\n")

        while True:
            try:
                query = input("🧑 Tú: ").strip()
                if query.lower() in {"salir", "exit", "q"}:
                    print("👋 ¡Hasta luego!")
                    break

                answer = self.ask(query)
                print(f"\n🤖 Asistente:\n{answer}\n")

                # Ask user for rating
                rating = None
                while rating not in {"1", "2", "3", "4", "5"}:
                    rating = input("¿Qué tan útil? (1-5): ").strip()
                comment = input("¿Comentarios? (opcional): ").strip()

                self.feedback.save_feedback(
                    query=query,
                    answer=answer,
                    rating=int(rating),
                    extra_meta={"comment": comment or None},
                )

            except KeyboardInterrupt:
                print("\n🛑 Cancelado")
                break
            except Exception as e:
                logger.error("Error in chat loop: %s", e, exc_info=True)
                print("⚠️ Error, inténtalo de nuevo.")

    # ------------------------------------------------------------------
    # Entry-point
    # ------------------------------------------------------------------
    @classmethod
    def from_pickle_dir(
        cls,
        pickle_dir: Path = settings.PROC_DIR,
        lora_checkpoint: Optional[Path] = settings.RLHF_CHECKPOINT,
    ) -> "RAGAgent":
        """Factory that loads products from the canonical .pkl files."""
        from src.core.data.loader import DataLoader

        products = DataLoader().load_data(
            raw_dir=settings.RAW_DIR,
            processed_dir=pickle_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        if not products:
            raise RuntimeError("No products found")
        return cls(products=products, lora_checkpoint=lora_checkpoint)


# ------------------------------------------------------------------
# Main
# ------------------------------------------------------------------
if __name__ == "__main__":
    agent = RAGAgent.from_pickle_dir()
    agent.chat_loop()
