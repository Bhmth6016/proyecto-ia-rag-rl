# src/core/rag/advanced/agent.py
from typing import Dict, Any, Optional, List
from tenacity import retry, stop_after_attempt, wait_exponential, retry_if_exception_type
import time
import logging
from google.api_core import exceptions as google_exceptions
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.core.utils.logger import configure_root_logger

from src.core.rag.advanced.prompts import (
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    VALIDATION_PROMPT,
    RELEVANCE_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_QUALITY_PROMPT
)

class RateLimiter:
    """Clase para manejar rate limiting"""
    def __init__(self, calls_per_minute: int = 60):
        self.calls_per_minute = calls_per_minute
        self.last_calls = []

    def wait_if_needed(self):
        now = time.time()
        # Eliminar llamadas antiguas (más de 1 minuto)
        self.last_calls = [t for t in self.last_calls if now - t < 60]
        
        if len(self.last_calls) >= self.calls_per_minute:
            sleep_time = 60 - (now - self.last_calls[0])
            time.sleep(max(0, sleep_time))
        
        self.last_calls.append(time.time())

class AdvancedRAGAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        memory: BaseMemory,
        feedback_processor: Optional[Any] = None,
        prompt_template: ChatPromptTemplate = RAG_PROMPT_TEMPLATE,
        enable_rewrite: bool = True,
        enable_validation: bool = True,
        top_k: int = 5,
        rate_limit: int = 60  # Llamadas por minuto
    ):
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.feedback_processor = feedback_processor
        self.prompt_template = prompt_template
        self.enable_rewrite = enable_rewrite
        self.enable_validation = enable_validation
        self.top_k = top_k
        self.rate_limiter = RateLimiter(calls_per_minute=rate_limit)
        self.output_parser = StrOutputParser()
        self.setup_chains()

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=4, max=10),
        retry=retry_if_exception_type(google_exceptions.ResourceExhausted)
    )
    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """Versión protegida contra rate limiting y con retry"""
        try:
            self.rate_limiter.wait_if_needed()
            return self._invoke_impl(question, **kwargs)
        except google_exceptions.ResourceExhausted as e:
            logger.error(f"Quota exceeded, retrying: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error: {str(e)}")
            return {
                "response": "Disculpa, estamos experimentando problemas técnicos. Por favor intenta nuevamente más tarde.",
                "metadata": {"error": str(e)}
            }

    def _invoke_impl(self, question: str, **kwargs) -> Dict[str, Any]:
        """Implementación principal del invoke"""
        inputs = {"question": question, **kwargs}
        
        # Get the effective query (rewritten if available, original otherwise)
        effective_query = question
        if self.enable_rewrite:
            try:
                inputs["rewritten_question"] = self.rewrite_query(question)
                effective_query = inputs["rewritten_question"]
            except Exception as e:
                logger.warning(f"Query rewriting failed: {str(e)}")
                effective_query = question
        
        # Retrieve documents with the effective query
        inputs["context"] = self.retrieve_documents(effective_query, k=self.top_k)
        
        # Load conversation history
        inputs["chat_history"] = self.load_memory()
        
        # Generate response
        response = self.rag_chain.invoke(inputs)
        
        # Validate response if enabled
        is_valid = True
        if self.enable_validation:
            context = "\n\n".join([doc.page_content for doc in inputs.get("context", [])])
            is_valid = self.validate_response(context, response)
            if not is_valid:
                response = "No puedo verificar completamente esta información. ¿Deseas que investigue más?"

        # Update memory
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        
        return {
            "response": response,
            "metadata": {
                "context": inputs.get("context", []),
                "is_valid": is_valid,
                "rewritten_question": inputs.get("rewritten_question", ""),
                "chat_history": inputs.get("chat_history", ""),
                "top_k": self.top_k
            }
        }

    def setup_chains(self) -> None:
        """Configura todas las cadenas de procesamiento."""
        # Cadena para reescritura de consultas
        self.rewrite_chain = (
            {"query": RunnablePassthrough()}
            | QUERY_REWRITE_PROMPT
            | self.llm
            | self.output_parser
        )
        
        # Cadena principal RAG
        self.rag_chain = (
            RunnablePassthrough.assign(
                context=lambda x: self.retrieve_documents(
                    x.get("rewritten_question", x["question"]), 
                    k=self.top_k
                ),
                chat_history=lambda x: self.load_memory()
            )
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        # Cadena de recuperación de documentos (ahora usa top_k)
        self.retrieval_chain = RunnableLambda(
            lambda x: self.retriever.retrieve(
                x.get("rewritten_question", x["question"]),
                k=self.top_k
            )
        )
        
        
        # Cadenas de validación
        self.validation_chain = VALIDATION_PROMPT | self.llm | self.output_parser
        self.relevance_chain = RELEVANCE_PROMPT | self.llm | self.output_parser
        self.hallucination_chain = HALLUCINATION_PROMPT | self.llm | self.output_parser
        self.quality_chain = ANSWER_QUALITY_PROMPT | self.llm | self.output_parser

    def rewrite_query(self, query: str) -> str:
        """Reescribe la consulta del usuario para mejor recuperación."""
        return self.rewrite_chain.invoke(query)

    def retrieve_documents(self, query: str, k: Optional[int] = None) -> List[Document]:
        """Recupera documentos relevantes para la consulta."""
        k = k or self.top_k
        return self.retriever.retrieve(query, k=k)  # Changed from get_relevant_documents to retrieve

    def load_memory(self) -> str:
        """Carga el historial de conversación desde la memoria."""
        return self.memory.load_memory_variables({}).get("chat_history", "")

    def validate_response(self, context: str, answer: str) -> bool:
        """Valida si la respuesta está soportada por el contexto."""
        validation = self.validation_chain.invoke({
            "context": context,
            "answer": answer
        })
        return "yes" in validation.lower()

    def process_feedback(self, question: str, response: str, feedback: Dict[str, Any]) -> None:
        """Procesa el feedback del usuario si está disponible el procesador."""
        if self.feedback_processor:
            self.feedback_processor.process(
                question=question,
                response=response,
                feedback=feedback
            )

    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo del agente.
        
        Args:
            question: Consulta del usuario
            **kwargs: Argumentos adicionales
            
        Returns:
            Dict con respuesta y metadatos
        """
        # Preparar y ejecutar cadena RAG
        inputs = {"question": question, **kwargs}
        response = self.rag_chain.invoke(inputs)
        
        # Validar respuesta si está habilitado
        is_valid = True
        if self.enable_validation:
            context = "\n\n".join([doc.page_content for doc in inputs.get("context", [])])
            is_valid = self.validate_response(context, response)
            if not is_valid:
                response = "No puedo verificar completamente esta información. ¿Deseas que investigue más?"

        # Actualizar memoria
        self.memory.save_context(
            {"input": question},
            {"output": response}
        )
        
        # Preparar metadatos
        metadata = {
            "context": inputs.get("context", []),
            "is_valid": is_valid,
            "rewritten_question": inputs.get("rewritten_question", ""),
            "chat_history": inputs.get("chat_history", ""),
            "top_k": self.top_k
        }
        
        return {
            "response": response,
            "metadata": metadata
        }

    async def ainvoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """Versión asíncrona del método invoke."""
        # Implementación básica - adaptar para operaciones async reales
        return self.invoke(question, **kwargs)

    def stream(self, question: str, **kwargs):
        """Versión streaming del método invoke."""
        result = self.invoke(question, **kwargs)
        yield result["response"]