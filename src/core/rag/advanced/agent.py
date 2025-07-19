# src/core/rag/agent.py
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple

from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_core.prompts import ChatPromptTemplate

from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.advanced.prompts import (
    RAG_PROMPT_TEMPLATE as SIMPLE_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    NO_RESULTS_TEMPLATE,
    PARTIAL_RESULTS_TEMPLATE,
)
from src.core.utils.translator import TextTranslator, Language
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.init import get_system  # Nueva importación

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)


class RAGAgent:
    def __init__(self, products: Optional[List[Dict]] = None, enable_translation: bool = True):
        print("Inicializando RAGAgent - Paso 1/4: Cargando productos")
        system = get_system()
        self.products = products or system.products
        print(f"DEBUG - Productos cargados: {len(self.products)}")

        print("Paso 2/4: Inicializando retriever")
        max_retries = 3
        retry_delay = 2  # segundos

        for attempt in range(max_retries):
            try:
                self.retriever = system.retriever
                if not self.retriever.store:
                    print("Construyendo índice...")
                    self.retriever.build_index(self.products)
                break
            except Exception as e:
                if attempt == max_retries - 1:
                    raise RuntimeError(f"Failed to initialize retriever after {max_retries} attempts: {e}")
                logger.warning(f"Retriever initialization failed (attempt {attempt + 1}), retrying in {retry_delay}s...")
                time.sleep(retry_delay)

        print("Paso 3/4: Configurando traducción")
        self.enable_translation = enable_translation
        self.translator = TextTranslator() if enable_translation else None

        print("Paso 4/4: Creando cadena de conversación")
        self.tree = CategoryTree(self.products).build_tree()
        self.active_filter = ProductFilter()

        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
            temperature=0.3,
        )

        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            k=5,
            return_messages=True,
        )
        self.chain = self._build_chain()

        self.feedback = FeedbackProcessor(
            feedback_dir=str(settings.DATA_DIR / "feedback")
        )

    def _build_chain(self) -> ConversationalRetrievalChain:
        # Espera activa por el store (máx 10 segundos)
        start_time = time.time()
        while not hasattr(self.retriever, 'store') or self.retriever.store is None:
            if time.time() - start_time > 10:
                raise RuntimeError("Timeout waiting for retriever store to initialize")
            time.sleep(0.5)
        if not self.retriever.store:
            raise ValueError("Retriever store is not initialized. Please check the index path and ensure the index is built.")

        prompt = ChatPromptTemplate.from_template(
            "Responde la pregunta basándote en el contexto siguiente. "
            "Si no sabes la respuesta, di que no sabes. "
            "Sé conciso y útil.\n\n"
            "Contexto: {context}\n\n"
            "Pregunta: {question}\n"
            "Respuesta:"
        )
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever.store.as_retriever(search_kwargs={"k": 5}),
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=False,
        )

    def _process_query(self, query: str) -> Tuple[str, Optional[Language]]:
        if not self.enable_translation:
            return query, None

        source_lang = self.translator.detect_language(query)
        if source_lang == Language.ENGLISH:
            return query, None

        english_query = self.translator.translate_to_english(query, source_lang)
        logger.debug("Translated query: %s -> %s", query, english_query)
        return english_query, source_lang

    def _process_response(self, answer: str, target_lang: Optional[Language]) -> str:
        if not target_lang or not self.enable_translation:
            return answer
        return self.translator.translate_from_english(answer, target_lang)

    def _detect_category(self, query: str) -> Optional[str]:
        # Simple category detection based on keywords
        for category in self.tree.categories:
            if category.lower() in query.lower():
                return category
        return None

    def ask(self, query: str) -> str:
        print(f"DEBUG - Original query: {query}")  # Debug line
        processed_query, source_lang = self._process_query(query)
        print(f"DEBUG - Processed query: {processed_query}")  # Debug line

        products = []
        try:
            products = self.retriever.retrieve(query=processed_query, k=5)
            print(f"DEBUG - Retrieved products: {[p.title for p in products]}")  # Debug line
        except Exception as e:
            logger.error(f"Error retrieving products: {e}")
            products = []

        if not products:
            # Try with simpler query
            simple_query = " ".join(query.split()[:3])  # Use first 3 words
            products = self.retriever.retrieve(query=simple_query, k=5)

            if not products:
                # Fallback to category-based retrieval
                category = self._detect_category(query)
                if category:
                    products = self.retriever.retrieve_by_category(category, k=5)

        if not products:
            try:
                expanded = self.retriever._expand_query(processed_query)
                suggestions = ", ".join(expanded[:3]) if expanded else "backpack, headphones, speaker"
            except Exception:
                suggestions = "backpack, headphones, speaker"

            return NO_RESULTS_TEMPLATE.format(query=query, suggestions=suggestions)

        # Filtrar productos inválidos
        valid_products = [p for p in products if p and p.title != 'Unknown Product']
        if not valid_products:
            return "No encontré productos válidos para tu búsqueda."

        exact = [p for p in valid_products if self.retriever._score(query, p) > 0]
        if not exact and valid_products:
            titles = [p.title for p in valid_products[:3]]
            return PARTIAL_RESULTS_TEMPLATE.format(
                query=query,
                similar_products="\n".join(f"- {t}" for t in titles)
            )

        context = "\n\n".join(p.to_document()["page_content"] for p in valid_products)

        llm_answer = self.chain.invoke({"question": processed_query, "context": context})["answer"]
        final_answer = self._process_response(llm_answer, source_lang)

        self.feedback.save_feedback(
            query=query,
            answer=final_answer,
            rating=0,
            retrieved_docs=[p.title for p in valid_products],
            category_tree=self.tree,
            active_filter=self.active_filter,
            extra_meta={
                "english_query": processed_query if source_lang else None,
                "detected_language": source_lang.value if source_lang else "en",
            },
        )
        return final_answer

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

    @classmethod
    def from_pickle_dir(
        cls,
        pickle_dir: Path = settings.PROC_DIR,
        enable_translation: bool = True,
    ) -> "RAGAgent":
        logging.warning("Deprecated: Use SystemInitializer instead")
        from src.core.data.loader import DataLoader

        products = DataLoader().load_data(
            raw_dir=settings.RAW_DIR,
            processed_dir=pickle_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        if not products:
            raise RuntimeError("No products found")
        return cls(products=products, enable_translation=enable_translation)