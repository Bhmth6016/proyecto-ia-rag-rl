from __future__ import annotations
# src/core/rag/advanced/agent.py
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
from langchain_core.runnables import RunnablePassthrough
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.memory import BaseMemory
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
        self.last_rlhf_check = datetime.now()
        self._setup_rlhf_pipeline()
        
        # Inicializar feedback_memory solo si existe el directorio
        self.feedback_memory = []
        if Path("data/feedback").exists():
            try:
                self.feedback_memory = self._load_feedback_memory()
            except Exception as e:
                logger.error(f"Error loading feedback memory: {str(e)}")

    def _setup_rlhf_pipeline(self):
        """Verifica periódicamente si hay suficientes feedbacks para reentrenar"""
        # Verificar cada semana y si hay al menos 1000 muestras
        if datetime.now() - self.last_rlhf_check > timedelta(days=7):
            self.last_rlhf_check = datetime.now()
            feedback_dir = Path("data/feedback")
            if feedback_dir.exists():
                feedback_files = list(feedback_dir.glob("*.jsonl"))
                total_samples = sum(1 for f in feedback_files for _ in open(f))
                if total_samples >= 1000:
                    self._retrain_rlhf_model()

    def _retrain_rlhf_model(self):
        """Ejecuta el pipeline completo de RLHF"""
        print("Iniciando reentrenamiento RLHF...")
        trainer = RLHFTrainer(
            base_model_name=settings.MODEL_NAME,
            device=settings.DEVICE
        )
        
        # 1. Preparar dataset
        dataset = trainer.prepare_rlhf_dataset(
            feedback_dir=Path("data/feedback"),
            min_samples=1000
        )
        
        # 2. Entrenar modelo
        trainer.train(
            dataset=dataset,
            save_dir=Path("models/rlhf_checkpoints")
        )
        
        # 3. Actualizar modelo en producción
        self.llm = load_llm(model_name="models/rlhf_checkpoints/latest")
        print("Reentrenamiento RLHF completado y modelo actualizado")

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

            # 1. Verificar si hay feedback negativo para consultas similares
            if hasattr(self, 'feedback_memory'):
                similar_low_rated = self._find_similar_low_rated(query)
                if similar_low_rated:
                    return self._generate_alternative_response(query, similar_low_rated)

            # 1. Process query and extract filters
            processed_query, source_lang = self._process_query(query)
            if not processed_query:
                return "Error en traducción. Por favor intenta de nuevo."

            # 2. Extract filters from query
            filters = self._extract_filters_from_query(query)
            
            # 3. Retrieve products with filters
            try:
                products = self.retriever.retrieve(
                    query=processed_query,
                    k=10,
                    min_similarity=0.3,
                    filters=filters
                )

                if not products:
                    # Try without filters if no results
                    products = self.retriever.retrieve(
                        query=processed_query,
                        k=10,
                        min_similarity=0.2,
                        filters=None
                    )
                    
                    if not products:
                        return NO_RESULTS_TEMPLATE.format(
                            query=query,
                            suggestions="Prueba con términos más específicos o menos filtros"
                        )
                    else:
                        return "No encontré productos con esos filtros, pero aquí hay algunas opciones:\n\n" + \
                               self._format_response(query, products[:3], source_lang)

                # 4. Format response
                response = self._format_response(query, products[:3], source_lang)

                # Auto-evaluación de la respuesta
                if self._should_evaluate_response(response):
                    eval_result = self._evaluate_response(query, response)
                    if eval_result['score'] < 0.7:  # Umbral de calidad
                        response = self._improve_response(query, response, eval_result)

                # Save conversation
                self._save_conversation(query, response)

                return response

            except Exception as e:
                logger.error(f"Retrieval error: {str(e)}")
                return f"Error al buscar productos: {str(e)}"

        except Exception as e:
            logger.exception(f"Detailed error in ask(): {str(e)}")
            # Mostrar el error real para debugging
            return f"Ocurrió un error al procesar tu solicitud: {str(e)}. Por favor intenta de nuevo."

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
        query_embedding = self.retriever.embedder.encode(query)
        
        similar = []
        for fb in self.feedback_memory:
            fb_embedding = self.retriever.embedder.encode(fb['query_en'])
            similarity = cosine_similarity([query_embedding], [fb_embedding])[0][0]
            if similarity > threshold:
                similar.append(fb)
        
        return similar

    def _generate_alternative_response(self, query: str, bad_feedbacks: list):
        """Genera una respuesta alternativa basada en feedback negativo"""
        # Construir prompt con contexto de feedback negativo
        prompt = f"""
        El usuario preguntó: {query}
        
        Respuestas anteriores con baja calificación (evitar):
        {', '.join([fb['answer_en'] for fb in bad_feedbacks])}
        
        Por favor proporciona una respuesta alternativa diferente y más útil.
        """
        
        return self.llm(prompt)

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