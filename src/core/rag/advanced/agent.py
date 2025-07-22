from __future__ import annotations
import re
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid
import time
import sys
import torch
from sklearn.metrics.pairwise import cosine_similarity
import openai

# Third-party imports
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.chat_models import ChatOpenAI
from langchain_core.memory import BaseMemory
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.messages import HumanMessage, AIMessage
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from chromadb.config import Settings as ChromaSettings
from chromadb import PersistentClient

# Local imports
from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.advanced.prompts import (
    QUERY_CONSTRAINT_EXTRACTION_PROMPT,
    NO_RESULTS_TEMPLATE,
    PARTIAL_RESULTS_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_PROMPT
)
from src.core.utils.translator import TextTranslator, Language
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.init import get_system
from src.core.rag.advanced.trainer import RLHFTrainer
from src.core.rag.advanced.evaluator import RAGEvaluator, load_llm
from src.core.rag.advanced.rlhf import (
    _setup_rlhf_pipeline,
    _retrain_rlhf_model,
    _load_feedback_memory,
    _find_similar_low_rated,
    _generate_alternative_response,
    _should_evaluate_response,
    _evaluate_response,
    _improve_response,
    _log_feedback,
    _get_more_options,
    evaluate_model_performance
)

# Configure logging
logger = logging.getLogger(__name__)
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logging.getLogger("transformers").setLevel(logging.WARNING)


class RAGAgent:
    def __init__(self, products: Optional[List[Dict]] = None, enable_translation: bool = True):
        logger.info("Initializing RAGAgent - Step 1/4: Loading products")
        system = get_system()
        self.products = products or system.products
        
        # Configure history directory
        self.history_dir = settings.PROC_DIR / "historial"
        self.history_dir.mkdir(exist_ok=True)
        
        logger.info("Step 2/4: Initializing retriever")
        self.retriever = system.retriever
        
        if not self.retriever.index_exists():
            logger.info("Building new index...")
            self.retriever.build_index(self.products)
        else:
            logger.info("Loading existing index...")
            try:
                client_settings = ChromaSettings(
                    chroma_db_impl="duckdb+parquet",
                    persist_directory=str(settings.VECTOR_INDEX_PATH),
                    anonymized_telemetry=False
                )
                
                self.retriever.store = Chroma(
                    persist_directory=str(settings.VECTOR_INDEX_PATH),
                    embedding_function=self.retriever.embedder,
                    client_settings=client_settings
                )
                
                # Verify index is loaded
                if not self.retriever.store._collection.count():
                    logger.warning("Index exists but is empty - rebuilding")
                    raise ValueError("Empty index")
                    
            except Exception as e:
                logger.warning(f"Failed to load index: {str(e)} - rebuilding")
                self.retriever.build_index(self.products)

        logger.info("Step 3/4: Configuring translation")
        self.enable_translation = enable_translation
        # Use GPU if available
        device = "cuda" if torch.cuda.is_available() else "cpu"
        self.translator = TextTranslator(device=device) if enable_translation else None

        logger.info("Step 4/4: Configuring conversation chain")
        self.llm = ChatOpenAI(
            model_name=settings.OPENAI_MODEL_NAME,
            openai_api_key=settings.OPENAI_API_KEY,
            temperature=0.3,
        )
        
        # Initialize memory with proper message history
        self.message_history = ChatMessageHistory()
        self.memory = ConversationBufferWindowMemory(
            memory_key="chat_history",
            chat_memory=self.message_history,
            k=5,
            return_messages=True,
        )
        
        self.chain = self._build_chain()
        
        self.feedback_processor = FeedbackProcessor()
        self.last_rlhf_check = datetime.now()
        self._setup_rlhf_pipeline()
        
        self.feedback_memory = self._load_feedback_memory()

    def index_is_compatible(self) -> bool:
        """Check if existing index is compatible with current version"""
        try:
            return hasattr(self.retriever.store, '_client') and isinstance(self.retriever.store._client, PersistentClient)
        except:
            return False

    def _build_chain(self) -> ConversationalRetrievalChain:
        # Wait for store to initialize (max 10 seconds)
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
            return_source_documents=True,
            verbose=True
        )

    def _save_conversation(self, query: str, response: str, feedback: Optional[str] = None):
        """Guarda la interacción en un archivo JSON"""
        timestamp = datetime.now().isoformat()
        session_id = str(uuid.uuid4())
        
        data = {
            "session_id": session_id,
            "timestamp": timestamp,
            "query": query,
            "response": response,
            "feedback": feedback,
            "translation_used": self.enable_translation
        }
        
        filename = self.history_dir / f"conversation_{timestamp[:10]}.json"
        
        try:
            # Si el archivo ya existe, cargamos y agregamos
            if filename.exists():
                with open(filename, 'r', encoding='utf-8') as f:
                    existing = json.load(f)
                if not isinstance(existing, list):
                    existing = [existing]
                existing.append(data)
            else:
                existing = [data]
                
            with open(filename, 'w', encoding='utf-8') as f:
                json.dump(existing, f, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.error(f"Error saving conversation: {str(e)}")

    def ask(self, query: str) -> str:
        try:
            logger.debug("=== INICIO DE ASK ===")
            
            # Debug del estado del agente
            logger.debug(f"Retriever: {hasattr(self, 'retriever')}")
            logger.debug(f"Translator: {hasattr(self, 'translator')}")
            logger.debug(f"LLM: {hasattr(self, 'llm')}")
            
            # 1. Verificar feedback memory
            if hasattr(self, 'feedback_memory'):
                logger.debug(f"Feedback memory items: {len(self.feedback_memory)}")
            
            # 2. Procesar consulta
            logger.debug("Procesando traducción...")
            processed_query, source_lang = self._process_query(query)
            logger.debug(f"Query procesada: {processed_query}, lang: {source_lang}")
            
            # 3. Extraer filtros
            filters = self._extract_filters_from_query(query)
            logger.debug(f"Filtros extraídos: {filters}")
            
            # 4. Recuperar productos
            logger.debug("Iniciando recuperación de productos...")
            products = []
            attempts = [
                (processed_query, filters, 0.3),
                (processed_query, None, 0.2),
                (self._generalize_query(processed_query), None, 0.2),
            ]

            for q, f, min_sim in attempts:
                logger.debug(f"Intento con query: '{q}', filtros: {f}, sim_min: {min_sim}")
                try:
                    new_products = self.retriever.retrieve(
                        query=q,
                        k=10,
                        min_similarity=min_sim,
                        filters=f
                    )
                    logger.debug(f"Productos recuperados: {len(new_products)}")
                    products.extend(p for p in new_products if p not in products)
                except Exception as e:
                    logger.error(f"Error en recuperación: {str(e)}", exc_info=True)

            if not products:
                logger.debug("No se encontraron productos")
                return NO_RESULTS_TEMPLATE.format(...)

            # 5. Formatear respuesta
            logger.debug("Formateando respuesta...")
            response = self._format_response(query, products[:3], source_lang)
            logger.debug("Respuesta formateada")
            
            # 6. Guardar conversación
            self._save_conversation(query, response)
            
            # 7. Use OpenAI to generate a response
            openai_response = openai.ChatCompletion.create(
                model=settings.OPENAI_MODEL_NAME,
                messages=[
                    {"role": "system", "content": "Eres un asistente de recomendaciones de Amazon."},
                    {"role": "user", "content": query}
                ]
            )
            openai_response_content = openai_response.choices[0].message.content
            
            return openai_response_content

        except Exception as e:
            logger.error(f"ERROR GLOBAL EN ASK: {str(e)}", exc_info=True)
            return "Ocurrió un error inesperado. Por favor intenta de nuevo."

    def _extract_filters_from_query(self, query: str) -> Dict:
        """Extract filters from natural language query"""
        filters = {}
        
        # Price filters
        price_patterns = [
            (r'precio\s*menor\s*a\s*(\d+)', 'max'),
            (r'precio\s*mayor\s*a\s*(\d+)', 'min'),
            (r'price\s*under\s*(\d+)', 'max'),
            (r'price\s*over\s*(\d+)', 'min'),
        ]
        
        for pattern, filter_type in price_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                filters.setdefault('price_range', {})[filter_type] = float(matches[0])
        
        # Feature filters
        feature_patterns = [
            ('inalámbrico|wireless', 'wireless', True),
            ('bluetooth', 'bluetooth', True),
            ('color\s+(rojo|azul|verde|negro|blanco)', 'color', None),
            ('peso\s*menor\s*a\s*(\d+)\s*(g|gramos|kg|kilos)?', 'max_weight', None),
        ]
        
        for pattern, filter_key, filter_value in feature_patterns:
            matches = re.findall(pattern, query, re.IGNORECASE)
            if matches:
                if filter_value is not None:
                    filters[filter_key] = filter_value
                else:
                    filters[filter_key] = matches[0][0] if matches[0] else matches[0]
        
        return filters

    def _process_query(self, query: str) -> Tuple[str, Optional[Language]]:
        try:
            logger.debug(f"Procesando query: {query}")
            if not self.enable_translation:
                logger.debug("Traducción desactivada")
                return query, None

            logger.debug("Detectando idioma...")
            source_lang = self.translator.detect_language(query)
            logger.debug(f"Idioma detectado: {source_lang}")

            if source_lang == Language.ENGLISH:
                logger.debug("Idioma es inglés, no se traduce")
                return query, None

            logger.debug("Traduciendo a inglés...")
            english_query = self.translator.translate_to_english(query, source_lang)
            logger.debug(f"Query traducida: {english_query}")
            return english_query, source_lang

        except Exception as e:
            logger.error(f"Error en _process_query: {str(e)}", exc_info=True)
            raise

    def _format_response(self, original_query: str, products: List, target_lang: Optional[Language]) -> str:
        logger.info(f"Formatting response for {len(products)} products")
        if not products:
            logger.warning("No products to format")
            return NO_RESULTS_TEMPLATE.format(...)
        
        try:
            # Debug del primer producto
            if products:
                logger.debug(f"First product structure: {dir(products[0])}")
                logger.debug(f"First product title: {getattr(products[0], 'title', 'N/A')}")
        except Exception as e:
            logger.error(f"Debug failed: {str(e)}")
        
        # Ensure we have unique products
        unique_products = []
        seen_ids = set()
        
        for product in products:
            try:
                # Try different ways to get a unique identifier
                product_id = getattr(product, 'id', None) or getattr(product, 'product_id', None) or str(getattr(product, 'title', ''))
                if product_id and product_id not in seen_ids:
                    seen_ids.add(product_id)
                    unique_products.append(product)
            except Exception as e:
                logger.warning(f"Error processing product: {str(e)}")
        
        if not unique_products:
            return "No se encontraron productos válidos para mostrar."
        
        response = [f"🔍 Resultados para '{original_query}':"]
        
        for i, product in enumerate(unique_products[:3], 1):
            try:
                title = getattr(product, 'title', 'Producto sin nombre')
                price = getattr(product, 'price', None)
                rating = getattr(product, 'average_rating', None)
                
                price_str = f"${price:.2f}" if price is not None else "Precio no disponible"
                rating_str = f"{rating}/5" if rating is not None else "Sin calificaciones"
                
                response.append(
                    f"{i}. 🏷️ **{title}** (⭐ {rating_str})\n"
                    f"   💵 {price_str} | 📱 {self._get_key_features(product)}\n"
                    f"   📝 {self._get_product_summary(product)}"
                )
            except Exception as e:
                logger.warning(f"Error formatting product {i}: {str(e)}")
                response.append(f"{i}. [Error mostrando este producto]")
        
        return "\n\n".join(response)

    def _get_key_features(self, product) -> str:
        """Extrae características clave del producto con manejo robusto"""
        try:
            features = getattr(product, 'features', [])
            if isinstance(features, list):
                return " | ".join(str(f) for f in features[:3]) if features else "Características no especificadas"
            elif isinstance(features, str):
                return features[:100] + ("..." if len(features) > 100 else "")
            return "Características no especificadas"
        except Exception as e:
            logger.warning(f"Error getting features: {str(e)}")
            return "Características no disponibles"

    def _get_product_summary(self, product) -> str:
        """Genera un resumen breve del producto con manejo robusto"""
        try:
            desc = getattr(product, 'description', None)
            if not desc:
                desc = getattr(product, 'title', 'Descripción no disponible')
            return desc[:100] + ("..." if len(desc) > 100 else "")
        except Exception as e:
            logger.warning(f"Error getting description: {str(e)}")
            return "Descripción no disponible"

    def _generalize_query(self, query: str) -> str:
        """Generaliza la consulta para búsquedas más amplias"""
        beauty_terms = ["belleza", "beauty", "crema", "labial", "maquillaje", "skincare"]
        if any(term in query.lower() for term in beauty_terms):
            return "productos de belleza"
        return query.split()[0] if query else "producto"

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