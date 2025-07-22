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
from openai import OpenAI 
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
from langchain_openai import ChatOpenAI
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
    
    # 2. Subir el dataset y fine-tunear
    client = OpenAI(api_key=settings.OPENAI_API_KEY)
    response = client.files.create(file=open(dataset, "rb"), purpose="fine-tune")
    fine_tune_job = client.fine_tuning.jobs.create(training_file=response.id, model="gpt-3.5-turbo")
    
    # 3. Esperar a que el fine-tuning termine
    while True:
        job_status = client.fine_tuning.jobs.retrieve(fine_tune_job.id)
        if job_status.status == "succeeded":
            print("Fine-tuning completado")
            break
        elif job_status.status == "failed":
            print("Fine-tuning falló")
            return
        time.sleep(60)  # Esperar 1 minuto antes de verificar el estado nuevamente
    
    # 4. Actualizar modelo en producción
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

def _log_feedback(
    self,
    query: str,
    answer: str,
    rating: int,
    extra_meta: Optional[dict] = None,
    timestamp: Optional[str] = None,
) -> None:
    """Log user feedback to a JSONL file for RLHF training.
    
    Args:
        query: Original user query
        answer: Agent's response
        rating: User rating (1-5)
        extra_meta: Additional metadata to store
        timestamp: Optional timestamp for the feedback
    """
    feedback_dir = Path("data/feedback")
    feedback_dir.mkdir(exist_ok=True)
    
    timestamp = timestamp or datetime.now().isoformat()
    date_str = timestamp[:10]  # YYYY-MM-DD
    
    # Translate query and answer to English for consistent storage
    query_en = query
    answer_en = answer
    if self.enable_translation:
        try:
            source_lang = self.translator.detect_language(query)
            if source_lang != Language.ENGLISH:
                query_en = self.translator.translate(query, source_lang, Language.ENGLISH)
                answer_en = self.translator.translate(answer, source_lang, Language.ENGLISH)
        except Exception as e:
            logger.warning(f"Could not translate feedback: {str(e)}")
    
    feedback_data = {
        "timestamp": timestamp,
        "query": query,
        "query_en": query_en,
        "answer": answer,
        "answer_en": answer_en,
        "rating": rating,
        "metadata": extra_meta or {},
    }
    
    # Append to daily feedback file
    feedback_file = feedback_dir / f"feedback_{date_str}.jsonl"
    
    try:
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        
        # Update in-memory feedback for immediate use
        if rating <= 3:  # Only store negative feedbacks in memory
            self.feedback_memory.append(feedback_data)
        
        logger.info(f"Feedback logged (rating: {rating})")
    except Exception as e:
        logger.error(f"Error saving feedback: {str(e)}")

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

def evaluate_model_performance(
    self,
    test_dataset_path: Path,
    baseline_model: Optional[str] = None,
    num_samples: int = 100,
) -> Dict[str, float]:
    """Evaluate current model against a baseline on a test dataset.
    
    Args:
        test_dataset_path: Path to JSONL file with test queries
        baseline_model: Name of baseline model to compare against
        num_samples: Number of samples to evaluate (0 for all)
        
    Returns:
        Dictionary with evaluation metrics
    """
    # Load test dataset
    try:
        with open(test_dataset_path) as f:
            test_data = [json.loads(line) for line in f]
            
        if num_samples > 0:
            test_data = test_data[:num_samples]
            
        if not test_data:
            raise ValueError("Test dataset is empty")
    except Exception as e:
        logger.error(f"Error loading test dataset: {str(e)}")
        return {"error": str(e)}
    
    evaluator = RAGEvaluator(llm=self.llm)
    metrics = {
        "current_model": evaluator.evaluate_batch(test_data),
        "baseline": None,
    }
    
    # Compare with baseline if provided
    if baseline_model:
        try:
            baseline_llm = load_llm(model_name=baseline_model)
            baseline_evaluator = RAGEvaluator(llm=baseline_llm)
            metrics["baseline"] = baseline_evaluator.evaluate_batch(test_data)
            
            # Calculate improvement
            if metrics["baseline"] and metrics["current_model"]:
                for metric in ["accuracy", "relevance", "coherence"]:
                    improvement = metrics["current_model"][metric] - metrics["baseline"][metric]
                    metrics[f"{metric}_improvement"] = improvement
        except Exception as e:
            logger.error(f"Error evaluating baseline: {str(e)}")
            metrics["baseline_error"] = str(e)
    
    return metrics