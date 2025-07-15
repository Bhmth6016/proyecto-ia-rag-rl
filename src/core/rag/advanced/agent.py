from __future__ import annotations

import logging
import sys
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

logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)

class RAGAgent:
    def __init__(
        self,
        products: List[Dict],
        lora_checkpoint: Optional[str] = None,
        enable_translation: bool = True,
    ):
        self.products = products
        self.enable_translation = enable_translation
        self.lora_checkpoint = lora_checkpoint

        # Translation
        self.translator = TextTranslator() if enable_translation else None

        # Category tree & filters
        self.tree = CategoryTree(products).build_tree()
        self.active_filter = ProductFilter()

        # Vector store via synonym-aware retriever
        try:
            self.retriever = Retriever(
                index_path=settings.VECTOR_INDEX_PATH,
                vectorstore_type=settings.VECTOR_BACKEND,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Retriever: {e}")
            raise RuntimeError(
                f"Index not found at {settings.VECTOR_INDEX_PATH}. Please run the following command to build the index:\n"
                f"python src/interfaces/cli.py index"
            )

        # LLM
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            google_api_key=settings.GEMINI_API_KEY,
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

    def _build_chain(self) -> ConversationalRetrievalChain:
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

    def ask(self, query: str) -> str:
        processed_query, source_lang = self._process_query(query)

        products = self.retriever.retrieve(query=processed_query, k=5)

        if not products:
            products = self.retriever.retrieve(query=query, k=5)

        if not products:
            # Intentar sugerencias basadas en expansión de consulta
            try:
                expanded = self._expand_query(processed_query)
                suggestions = ", ".join(expanded[:3]) if expanded else "backpack, headphones, speaker"
            except Exception:
                suggestions = "backpack, headphones, speaker"

            return NO_RESULTS_TEMPLATE.format(query=query, suggestions=suggestions)

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

        context = "\n\n".join(p.to_document()["page_content"] for p in products)

        llm_answer = self.chain.invoke({"question": processed_query, "context": context})["answer"]
        final_answer = self._process_response(llm_answer, source_lang)

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
        from src.core.data.loader import DataLoader

        products = DataLoader().load_data(
            raw_dir=settings.RAW_DIR,
            processed_dir=pickle_dir,
            cache_enabled=settings.CACHE_ENABLED,
        )
        if not products:
            raise RuntimeError("No products found")
        return cls(products=products, enable_translation=enable_translation)