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
import psutil
import faiss
from tqdm import tqdm 
import logging
import sys
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import os
from langchain_google_genai import ChatGoogleGenerativeAI
import torch
from langchain_community.vectorstores import FAISS
from langchain_huggingface import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate, PromptTemplate
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
        print("Initializing RAGAgent...")  # Debug
        self.products = products
        print(f"Loaded {len(products)} products")  # Debug
        self.enable_translation = enable_translation

        self.embedder = HuggingFaceEmbeddings(
            model_name=settings.EMBEDDING_MODEL,
            model_kwargs={"device": "cpu"},
            encode_kwargs={"normalize_embeddings": True},
        )
        print("Embedder initialized")  # Debug

        # Translation setup
        self.translator = TextTranslator() if enable_translation else None
        print("Translator ready")  # Debug

        # Build category tree + filters
        self.tree = CategoryTree(products).build_tree()
        print("Category tree built")  # Debug
        self.active_filter = ProductFilter()
        print("ProductFilter ready")  # Debug

        # Vector store
        print("Building vector store...")  # Debug
        self.vector_store = self._build_vector_store()
        print("Vector store ready")  # Debug

        # LLM (LoRA-aware)
        print("Loading LLM...")  # Debug
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
            temperature=0.3
        )
        print("LLM loaded")  # Debug

        # LangChain memory
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
        )
        print("Memory initialized")  # Debug

        # LangChain chain
        print("Building chain...")  # Debug
        self.chain = self._build_chain()
        print("Chain ready")  # Debug

        # Feedback
        self.feedback = FeedbackProcessor(
            feedback_dir=str(settings.DATA_DIR / "feedback")
        )
        print("RAGAgent initialization complete")  # Debug

    def _build_vector_store(self) -> FAISS:
        """Build vector store with memory monitoring"""
        import psutil
        
        def log_memory_usage(stage=""):
            process = psutil.Process()
            memory_mb = process.memory_info().rss / 1024 / 1024
            print(f"💾 [Memory] {stage}: {memory_mb:.1f} MB")
        
        log_memory_usage("Before processing")
        
        # Reduce further for initial testing
        max_products = min(10000, len(self.products))
        products_to_process = self.products[:max_products]
        
        print(f"🔄 Processing {len(products_to_process)} products...")
        
        # Create texts
        texts = []
        for p in products_to_process:
            brand = p.details.brand if p.details and p.details.brand else ""
            model = p.details.model if p.details and p.details.model else ""
            text = f"{p.title} {brand} {model}".strip()
            if text:  # Only add non-empty texts
                texts.append(text)
            specs = []
            if p.details and p.details.specifications:
                for k, v in p.details.specifications.items():
                    if any(keyword in k.lower() for keyword in ['size', 'dimension', 'inch']):
                        specs.append(f"{k}: {v}")
            
            text = f"{p.title} - Brand: {brand} - Model: {model} - {' '.join(specs)}"
            if text.strip():
                texts.append(text)
        
        log_memory_usage("Before vector store creation")
        
        # Use simple FAISS creation (fixed)
        vector_store = FAISS.from_texts(
            texts=texts,
            embedding=self.embedder
        )
        
        log_memory_usage("After vector store creation")
        return vector_store

    def _build_chain(self) -> ConversationalRetrievalChain:
        """Build the LangChain conversation chain."""
        # Update the prompt template to use ChatPromptTemplate
        prompt = ChatPromptTemplate.from_template(
            "Answer the question based on the context below. "
            "If you don't know the answer, say you don't know. "
            "Be concise and helpful.\n\n"
            "Context: {context}\n\n"
            "Question: {question}\n"
            "Answer:"
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
        english_answer = self.chain.invoke({"question": processed_query})["answer"]
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
            extra_meta={  # ✅ parámetro correcto
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
        )[:50000]
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