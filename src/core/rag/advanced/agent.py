from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple

# Imports estándar
import time
import sys

# Imports de terceros
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Imports locales
from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.advanced.prompts import (
    QUERY_CONSTRAINT_EXTRACTION_PROMPT,
    NO_RESULTS_TEMPLATE,
    PARTIAL_RESULTS_TEMPLATE,
    QUERY_REWRITE_PROMPT,
)
from src.core.utils.translator import TextTranslator, Language
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.init import get_system

# Configuración de logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)


class RAGAgent:
    """
    Agente conversacional basado en recuperación aumentada (RAG)
    para recomendar productos según consultas de lenguaje natural.
    """

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

        self.constraint_extractor = LLMChain(
            llm=self.llm,
            prompt=QUERY_CONSTRAINT_EXTRACTION_PROMPT
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
        logger.debug(f"Original query: {query}")
        
        # 1. Traducción y preprocesamiento
        processed_query, source_lang = self._process_query(query)
        logger.debug(f"Processed query: {processed_query}")

        # 2. Extraer constraints desde el prompt
        try:
            constraint_data = self.constraint_extractor.invoke({"query": processed_query})
            logger.debug(f"Constraints extracted: {constraint_data}")
        except Exception as e:
            logger.warning(f"Failed to extract constraints: {e}")
            constraint_data = {}

        # 3. Determinar cantidad de resultados deseados
        quantity = 3
        for word in query.split():
            if word.isdigit():
                quantity = max(1, min(int(word), 5))  # Entre 1 y 5

        # 4. Recuperar productos
        try:
            products = self.retriever.retrieve(
                query=processed_query,
                k=quantity + 3,
                filters=constraint_data.get("attributes", {}),
                category=constraint_data.get("category"),
                price_range=constraint_data.get("price_range"),
                min_rating=constraint_data.get("min_rating"),
                max_rating=constraint_data.get("max_rating"),
            )
            logger.debug(f"Retrieved products: {[p.title for p in products]}")
        except Exception as e:
            logger.error(f"Retrieval failed: {e}")
            products = []

        # 5. Fallback: consulta simplificada
        if not products:
            fallback_query = " ".join(query.split()[:3])
            logger.debug(f"Trying simplified query: {fallback_query}")
            try:
                products = self.retriever.retrieve(query=fallback_query, k=quantity + 3)
            except Exception as e:
                logger.error(f"Simplified retrieval failed: {e}")
                products = []

        # 6. Fallback por categoría detectada
        if not products:
            category = self._detect_category(query)
            logger.debug(f"Detected category: {category}")
            if category:
                try:
                    products = self.retriever.retrieve_by_category(category, k=quantity + 3)
                except Exception as e:
                    logger.error(f"Error retrieving by category: {e}")
                    products = []

        # 7. Si no hay resultados
        if not products:
            try:
                expanded = self.retriever._expand_query(processed_query)
                suggestions = ", ".join(expanded[:3]) if expanded else "headphones, backpack, charger"
            except Exception:
                suggestions = "headphones, backpack, charger"

            return NO_RESULTS_TEMPLATE.format(query=query, suggestions=suggestions)

        # 8. Filtrar productos válidos
        valid_products = [p for p in products if p and p.title and p.title.lower() != "unknown product"]
        if not valid_products:
            return "No encontré productos válidos para tu búsqueda."

        # 9. Generar contexto para LLM
        context = "\n\n".join(p.to_document()["page_content"] for p in valid_products[:quantity])

        try:
            llm_answer = self.chain.invoke({"question": processed_query, "context": context})["answer"]
            final_answer = self._process_response(llm_answer, source_lang)
        except Exception as e:
            logger.error(f"LLM generation failed: {e}")
            final_answer = "Aquí tienes algunos productos que podrían interesarte:"

        # 10. Respuesta final
        response_parts = []
        for i, product in enumerate(valid_products[:quantity], 1):
            response_parts.append(
                f"{i}. {product.title}\n"
                f"   💵 Price: {product.price if product.price else 'Not specified'}\n"
                f"   ⭐ Rating: {product.average_rating if product.average_rating else 'No ratings'}\n"
                f"   📝 Description: {product.description[:150] + '...' if product.description else 'No description available'}"
            )

        full_response = f"{final_answer}\n\nHere are some recommendations:\n" + "\n".join(response_parts)

        # 11. Guardar retroalimentación
        self._log_feedback(
            query=query,
            answer=full_response,
            rating=0,
            retrieved_docs=[p.title for p in valid_products],
            category_tree=self.tree,
            active_filter=self.active_filter,
            extra_meta={
                "english_query": processed_query if source_lang else None,
                "detected_language": source_lang.value if source_lang else "en",
                "extracted_constraints": constraint_data
            },
        )

        return full_response

    def _log_feedback(self, query: str, answer: str, rating: int, retrieved_docs: List[str], category_tree: CategoryTree, active_filter: ProductFilter, extra_meta: Dict):
        self.feedback.save_feedback(
            query=query,
            answer=answer,
            rating=rating,
            retrieved_docs=retrieved_docs,
            category_tree=category_tree,
            active_filter=active_filter,
            extra_meta=extra_meta,
        )

    def chat_loop(self) -> None:
        print("\n=== Amazon RAG ===\nType 'exit' to quit\n")
        
        while True:
            try:
                # Get user input
                query = input("🧑 You: ").strip()
                
                # Check for exit command
                if query.lower() in {"exit", "quit", "q"}:
                    print("\n🤖 Goodbye! Have a nice day!")
                    break
                
                # Process query and get response
                print("\n🤖 Processing your request...")
                answer = self.ask(query)
                
                # Display response
                print("\n🤖 Recommendation:")
                print("=" * 50)
                print(answer)
                print("=" * 50)
                
                # Get feedback
                while True:
                    feedback = input("\n🤖 Was this helpful? (1-5, 'skip', or 'more' for alternatives): ").strip().lower()
                    
                    if feedback in {'1', '2', '3', '4', '5'}:
                        self._log_feedback(
                            query=query,
                            answer=answer,
                            rating=int(feedback),
                            extra_meta={"user_rating": feedback},
                        )
                        print("Thank you for your feedback!")
                        break
                    elif feedback == 'skip':
                        break
                    elif feedback == 'more':
                        # Show additional options
                        answer = self._get_more_options(query)
                        print("\n🤖 Additional Options:")
                        print("-" * 50)
                        print(answer)
                        print("-" * 50)
                    else:
                        print("Please enter 1-5, 'skip', or 'more'")
                        
            except KeyboardInterrupt:
                print("\n🛑 Session ended by user")
                break
            except Exception as e:
                logger.error(f"Error in chat loop: {str(e)}")
                print("⚠️ An error occurred. Please try again.")

    def _get_more_options(self, query: str) -> str:
        try:
            products = self.retriever.retrieve(query=query, k=10)  # Get more results
            if len(products) <= 5:
                return "No additional options available."
                
            response = ["Here are some additional options:"]
            for i, product in enumerate(products[5:8], 6):  # Show next 3 results
                product_info = [
                    f"{i}. {product.title}",
                    f"   💵 Price: {product.price if product.price else 'Not specified'}",
                    f"   ⭐ Rating: {product.average_rating if product.average_rating else 'No ratings yet'}",
                ]
                response.extend(product_info)
                
            return "\n".join(response)
        except Exception as e:
            logger.error(f"Error getting more options: {str(e)}")
            return "Couldn't retrieve additional options at this time."
        

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