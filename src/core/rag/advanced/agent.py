# src/core/rag/advanced/agent.py
from typing import Dict, Any, Optional
from langchain_core.runnables import RunnablePassthrough
from langchain_core.language_models import BaseLLM
from langchain_core.retrievers import BaseRetriever
from langchain_core.memory import BaseMemory
from langchain_core.output_parsers import StrOutputParser
from src.core.rag.advanced.prompts import RAG_PROMPT_TEMPLATE

class AdvancedRAGAgent:
    def __init__(
        self,
        llm: BaseLLM,
        retriever: BaseRetriever,
        memory: BaseMemory,
        prompt_template: str = RAG_PROMPT_TEMPLATE,
        enable_rewrite: bool = True,
        enable_validation: bool = True
    ):
        """
        Agente RAG avanzado con reescritura de preguntas y validación de respuestas.
        
        Args:
            llm: Modelo de lenguaje (ej. OpenAI, Anthropic, etc.)
            retriever: Retriever para obtener documentos relevantes
            memory: Sistema de memoria para mantener el historial de chat
            prompt_template: Plantilla para el prompt RAG (opcional)
            enable_rewrite: Habilitar reescritura de preguntas (default: True)
            enable_validation: Habilitar validación de respuestas (default: True)
        """
        self.llm = llm
        self.retriever = retriever
        self.memory = memory
        self.prompt_template = prompt_template
        self.enable_rewrite = enable_rewrite
        self.enable_validation = enable_validation
        
        # Configurar componentes
        self.output_parser = StrOutputParser()
        self.setup_chains()

    def setup_chains(self):
        """Configura todas las cadenas de procesamiento."""
        # Cadena para reescribir preguntas
        self.rewrite_chain = (
            {"question": RunnablePassthrough()}
            | REWRITE_PROMPT
            | self.llm
            | self.output_parser
        )
        
        # Cadena para recuperar contexto
        self.retrieval_chain = (
            lambda x: self.retriever.get_relevant_documents(x["rewritten_question"] if self.enable_rewrite else x["question"]))
        
        # Cadena principal RAG
        self.rag_chain = (
            {"question": RunnablePassthrough()}
            | self._prepare_inputs
            | self.prompt_template
            | self.llm
            | self.output_parser
        )
        
        # Cadena de validación
        self.validation_chain = (
            VALIDATION_PROMPT
            | self.llm
            | self.output_parser
        )

    def _prepare_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Prepara los inputs para la cadena RAG."""
        question = inputs["question"]
        
        # Reescribir pregunta si está habilitado
        if self.enable_rewrite:
            inputs["rewritten_question"] = self.rewrite_chain.invoke(question)
        
        # Obtener documentos relevantes
        search_query = inputs.get("rewritten_question", question)
        inputs["context"] = self.retriever.get_relevant_documents(search_query)
        
        # Añadir historial de chat
        memory_vars = self.memory.load_memory_variables({})
        inputs["chat_history"] = memory_vars.get("chat_history", "")
        
        return inputs

    def _validate_response(self, context: str, answer: str) -> bool:
        """Valida si la respuesta está soportada por el contexto."""
        if not self.enable_validation:
            return True
            
        validation = self.validation_chain.invoke({
            "context": context,
            "answer": answer
        })
        return "yes" in validation.lower()

    def invoke(self, question: str, **kwargs) -> Dict[str, Any]:
        """
        Ejecuta el flujo completo del agente.
        
        Args:
            question: Pregunta del usuario
            **kwargs: Argumentos adicionales
            
        Returns:
            Dict con la respuesta y metadatos
        """
        # Preparar inputs
        inputs = {"question": question, **kwargs}
        
        # Ejecutar cadena RAG
        response = self.rag_chain.invoke(inputs)
        
        # Validar respuesta si está habilitado
        is_valid = True
        if self.enable_validation:
            context = "\n\n".join([doc.page_content for doc in inputs["context"]])
            is_valid = self._validate_response(context, response)
            
            if not is_valid:
                response = "No puedo verificar completamente esta información en mis fuentes. ¿Te gustaría que busque más detalles?"
        
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
        # Nota: Implementación básica - deberías adaptarla para operaciones async reales
        return self.invoke(question, **kwargs)

    def stream(self, question: str, **kwargs):
        """Versión streaming del método invoke."""
        # Implementación básica - adaptar según necesidades
        result = self.invoke(question, **kwargs)
        yield result["response"]