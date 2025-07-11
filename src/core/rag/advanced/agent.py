from typing import Dict, Any, Optional, List
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import StrOutputParser
from langchain_core.documents import Document
from src.core.rag.advanced.prompts import (
    RAG_PROMPT_TEMPLATE,
    QUERY_REWRITE_PROMPT,
    VALIDATION_PROMPT,
    RELEVANCE_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_QUALITY_PROMPT
)

class AdvancedRAGAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        memory: BaseMemory,
        prompt_template: ChatPromptTemplate = RAG_PROMPT_TEMPLATE,
        enable_rewrite: bool = True,
        enable_validation: bool = True
    ):
        """
        Agente RAG avanzado con capacidades mejoradas.
        
        Args:
            llm: Modelo de lenguaje para generación
            retriever: Módulo de recuperación de documentos
            memory: Sistema de memoria para contexto conversacional
            prompt_template: Plantilla para el prompt principal
            enable_rewrite: Habilitar reescritura de consultas
            enable_validation: Habilitar validación de respuestas
        """
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompt_template = prompt_template
        self.enable_rewrite = enable_rewrite
        self.enable_validation = enable_validation
        self.output_parser = StrOutputParser()
        self.setup_chains()

    def setup_chains(self) -> None:
        """Configura todas las cadenas de procesamiento."""
        # Cadena para reescritura de consultas
        self.rewrite_chain = (
            {"query": RunnablePassthrough()}
            | QUERY_REWRITE_PROMPT
            | self.llm
            | self.output_parser
        )
        
        # Cadena de recuperación de documentos
        self.retrieval_chain = RunnableLambda(
            lambda x: self.retrieve_documents(
                x.get("rewritten_question", x["question"])
            )
        )
        
        # Cadena principal RAG
        self.rag_chain = (
            RunnablePassthrough.assign(
                rewritten_question=lambda x: self.rewrite_query(x["question"]) if self.enable_rewrite else x["question"],
                context=lambda x: self.retrieve_documents(x["rewritten_question"]),
                chat_history=lambda x: self.load_memory()
            )
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        # Cadenas de validación
        self.validation_chain = VALIDATION_PROMPT | self.llm | self.output_parser
        self.relevance_chain = RELEVANCE_PROMPT | self.llm | self.output_parser
        self.hallucination_chain = HALLUCINATION_PROMPT | self.llm | self.output_parser
        self.quality_chain = ANSWER_QUALITY_PROMPT | self.llm | self.output_parser

    def rewrite_query(self, query: str) -> str:
        """Reescribe la consulta del usuario para mejor recuperación."""
        return self.rewrite_chain.invoke(query)

    def retrieve_documents(self, query: str) -> List[Document]:
        """Recupera documentos relevantes para la consulta."""
        return self.retriever.get_relevant_documents(query)

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
            "chat_history": inputs.get("chat_history", "")
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