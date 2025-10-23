from __future__ import annotations
# src/core/rag/advanced/rlhf.py
import re
import logging
import json
from datetime import datetime, timedelta
from pathlib import Path
from typing import List, Dict, Optional, Tuple
import uuid
import time
import sys
from sklearn.metrics.pairwise import cosine_similarity

# Imports estándar
import time
import sys

# Imports de terceros
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.memory import BaseMemory
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Imports locales
from src.core.rag.advanced.evaluator import load_llm_for_reward_model
from src.core.category_search.category_tree import CategoryTree, ProductFilter
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.advanced.prompts import (
    QUERY_CONSTRAINT_EXTRACTION_PROMPT,
    NO_RESULTS_TEMPLATE,
    PARTIAL_RESULTS_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    RELEVANCE_PROMPT  # Importar RELEVANCE_PROMPT
)
from src.core.utils.translator import TextTranslator, Language
from src.core.rag.basic.retriever import Retriever
from src.core.config import settings
from src.core.init import get_system
from src.core.rag.advanced.trainer import RLHFTrainer
from src.core.rag.advanced.evaluator import RAGEvaluator, load_llm
# Configuración de logging
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
        
        # Configurar directorio de historial
        self.history_dir = settings.PROC_DIR / "historial"
        self.history_dir.mkdir(exist_ok=True)
        
        print("Paso 2/4: Inicializando retriever")
        self.retriever = system.retriever
        
        if not self.retriever.index_exists():
            print("Construyendo nuevo índice...")
            self.retriever.build_index(self.products)
        else:
            print("Cargando índice existente...")
            try:
                from langchain_chroma import Chroma  # Versión actualizada
                self.retriever.store = Chroma(
                    persist_directory=str(settings.VECTOR_INDEX_PATH),
                    embedding_function=self.retriever.embedder
                )
            except Exception as e:
                print(f"Error cargando índice: {str(e)}")
                print("Reconstruyendo índice...")
                self.retriever.build_index(self.products)

        print("Paso 3/4: Configurando traducción")
        self.enable_translation = enable_translation
        self.translator = TextTranslator() if enable_translation else None

        print("Paso 4/4: Configurando cadena de conversación")
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
        
        # Nueva cadena con el prompt mejorado
        self.chain = self._build_chain()
        
        self.feedback_processor = FeedbackProcessor()
        self._setup_rlhf_pipeline()
        
        # Inicializar feedback_memory solo si existe el directorio
        self.feedback_memory = []
        if Path("data/feedback").exists():
            try:
                self.feedback_memory = self._load_feedback_memory()
            except Exception as e:
                logger.error(f"Error loading feedback memory: {str(e)}")

        model_path = settings.MODELS_DIR / "rlhf_model"
        if model_path.exists():
            self.llm = load_llm(model_name=str(model_path))
            logger.info(f"Modelo RLHF cargado desde: {model_path}")

    def _setup_rlhf_pipeline(self):
        """Verifica si hay suficientes muestras en logs para reentrenar."""
        try:
            failed_log = Path("data/feedback/failed_queries.log")
            success_log = Path("data/feedback/success_queries.log")

            total_lines = 0
            if failed_log.exists():
                total_lines += sum(1 for _ in open(failed_log, "r", encoding="utf-8"))
            if success_log.exists():
                total_lines += sum(1 for _ in open(success_log, "r", encoding="utf-8"))

            if total_lines >= 10:
                self._retrain_rlhf_model(failed_log, success_log)
        except Exception as e:
            logger.error(f"Error verificando logs de feedback: {e}")


    def _retrain_rlhf_model(self, failed_log: Path, success_log: Path):
        print("🔄 Iniciando reentrenamiento RLHF desde logs...")

        save_dir = Path("models/rlhf_checkpoints")
        save_dir.mkdir(parents=True, exist_ok=True)

        try:
            trainer = RLHFTrainer()
            dataset = trainer.prepare_rlhf_dataset_from_logs(
                failed_log=failed_log,
                success_log=success_log,
                min_samples=10
            )
            trainer.train(dataset, save_dir)

            model_path = save_dir.resolve().as_posix()

            # ✅ Cargar modelo RLHF como reward model, no como generador
            self.reward_model = load_llm_for_reward_model(model_path)
            print(f"✅ Modelo RLHF (reward) cargado desde: {model_path}")

        except Exception as e:
            print(f"❌ Error en RLHF desde logs: {str(e)}")


    def _load_feedback_memory(self):
        """Carga feedbacks previos para evitar repetir respuestas mal evaluadas"""
        feedback_dir = Path("data/feedback")
        if not feedback_dir.exists():
            return []
        
        feedbacks = []
        for file in feedback_dir.glob("*.jsonl"):
            with open(file) as f:
                feedbacks.extend([json.loads(line) for line in f])
        
        # Filtrar feedbacks con rating bajo
        return [fb for fb in feedbacks if fb.get('rating', 5) <= 3]

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
            logger.info(f"Processing query: {query}")

            # 1. Verificar feedback negativo
            if hasattr(self, 'feedback_memory'):
                similar_low_rated = self._find_similar_low_rated(query)
                if similar_low_rated:
                    return self._generate_alternative_response(query, similar_low_rated)

            # 2. Procesar consulta (sin traducción por ahora)
            processed_query = query  # Usamos la consulta original directamente
            
            # 3. Recuperar productos (versión simplificada)
            try:
                products = self.retriever.retrieve(
                    query=processed_query,
                    k=5,
                    min_similarity=0.2
                )

                if not products:
                    return "No encontré resultados. Prueba con términos más específicos."

                # Formatear respuesta básica
                response = []
                for i, product in enumerate(products[:3], 1):
                    response.append(
                        f"{i}. {getattr(product, 'title', 'Producto sin nombre')}\n"
                        f"   Precio: {getattr(product, 'price', 'No disponible')}\n"
                        f"   Rating: {getattr(product, 'average_rating', 'Sin calificaciones')}"
                    )

                answer = "🔍 Resultados:\n" + "\n\n".join(response)

                # 🔽 Aquí agregas la validación con reward_model
                if hasattr(self, "reward_model"):
                    score = self.reward_model(f"{query} {answer}")
                    print(f"🎯 Puntuación RLHF: {score:.2f}")
                    if score < 0.3:
                        print("⚠️ Respuesta de baja calidad. Intentando alternativa...")
                        return self._generate_alternative_response(query, [])

                return answer

                
            except Exception as e:
                logger.error(f"Error en recuperación: {str(e)}")
                return f"Error al buscar productos. Por favor intenta de nuevo."

        except Exception as e:
            logger.exception(f"Error en ask(): {str(e)}")
            return "Ocurrió un error. Estamos trabajando para solucionarlo."

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
        if not self.enable_translation or not hasattr(self, 'translator'):
            return query, None

        try:
            source_lang = self.translator.detect_language(query)
            if source_lang == Language.ENGLISH:
                return query, None

            english_query = self.translator.translate_to_english(query, source_lang)
            logger.debug("Translated query: %s -> %s", query, english_query)
            return english_query, source_lang
        except Exception as e:
            logger.error(f"Translation error: {str(e)}")
            return query, None  # Fallback to original query

    def _format_response(self, original_query: str, products: List, target_lang: Optional[Language]) -> str:
        """Formatea exactamente 3 productos en un formato consistente"""
        if not products:
            return NO_RESULTS_TEMPLATE.format(query=original_query, suggestions="Prueba con otros términos")
        
        response = [f"🔍 Resultados para '{original_query}':"]
        
        for i, product in enumerate(products[:3], 1):
            # Asegurar que el producto tenga los atributos necesarios
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
        
        return "\n\n".join(response)

    def _get_key_features(self, product) -> str:
        """Extrae características clave del producto"""
        features = []
        if hasattr(product, 'features'):
            features = product.features[:3]  # Tomar máximo 3 características
        return " | ".join(features) if features else "Características no especificadas"

    def _get_product_summary(self, product) -> str:
        """Genera un resumen breve del producto"""
        desc = getattr(product, 'description', '') or ''
        return desc[:100] + "..." if len(desc) > 100 else desc or "Descripción no disponible"

    def _generalize_query(self, query: str) -> str:
        """Generaliza la consulta para búsquedas más amplias"""
        beauty_terms = ["belleza", "beauty", "crema", "labial", "maquillaje", "skincare"]
        if any(term in query.lower() for term in beauty_terms):
            return "productos de belleza"
        return query.split()[0] if query else "producto"

    def _find_similar_low_rated(self, query: str, threshold: float = 0.8):
        """Busca consultas similares con baja calificación"""
        # Get the embedding for the query
        query_embedding = self.retriever.embedder.embed_query(query)
        
        similar = []
        for fb in self.feedback_memory:
            # Get embedding for the feedback query
            fb_embedding = self.retriever.embedder.embed_query(fb['query_es'])
            
            # Calculate cosine similarity
            similarity = cosine_similarity([query_embedding], [fb_embedding])[0][0]
            if similarity > threshold:
                similar.append(fb)
        
        return similar

    def _generate_alternative_response(self, query: str, bad_feedbacks: list) -> str:
        """Genera una respuesta alternativa basada en productos reales."""
        try:
            products = self.retriever.retrieve(query=query, k=5, min_similarity=0.2)
            if not products:
                return "No encontré productos alternativos. Prueba con otras palabras."

            return self._format_response(query, products[:3], target_lang=None)
        except Exception as e:
            logger.error(f"Error generando respuesta alternativa: {str(e)}")
            return "Ocurrió un error buscando una mejor respuesta."


    def _should_evaluate_response(self, response: str) -> bool:
        """Determina si una respuesta debe ser evaluada"""
        # Evaluar solo respuestas largas o con ciertas características
        return len(response.split()) > 20

    def _evaluate_response(self, query: str, response: str) -> dict:
        """Evalúa la calidad de una respuesta"""
        evaluator = RAGEvaluator(llm=self.llm)
        return evaluator.evaluate_all(
            question=query,
            documents=[],  # Opcional: documentos de referencia
            answer=response
        )

    def _improve_response(self, query: str, response: str, evaluation: dict) -> str:
        """Mejora una respuesta basada en evaluación"""
        prompt = f"""
        La siguiente respuesta recibió una baja evaluación:
        Evaluación: {evaluation['explanation']}
        
        Respuesta original:
        {response}
        
        Por favor genera una versión mejorada para la pregunta:
        {query}
        """
        return self.llm(prompt)

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
                        
                        # Recargar feedback_memory para que el cambio sea inmediato
                        self.feedback_memory = self._load_feedback_memory()
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
        
    def _log_feedback(self, query, answer, rating, extra_meta=None):
        print(f"Guardando feedback: rating={rating}")
        self.feedback_processor.save_feedback(
            query=query,
            answer=answer,
            rating=rating,
            extra_meta=extra_meta,
        )

    def log_product_selection(self, user_query: str, product_id: str, response_text: str):
        """
        Guarda la selección exitosa de un producto por parte del usuario como feedback positivo.
        """
        self.feedback_processor.save_feedback(
            query=user_query,
            answer=response_text,
            rating=5,  # o permitir que el usuario califique explícitamente
            extra_meta={
                "selected_product_id": product_id,
                "selection_method": "user_click"
            }
        )
        logger.info(f"✅ Feedback registrado para selección de producto {product_id}")

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