# src/core/rag/advanced/evaluators.py
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
from enum import Enum
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from src.core.rag.advanced.prompts import (
    RELEVANCE_PROMPT,
    HALLUCINATION_PROMPT,
    ANSWER_QUALITY_PROMPT
)

class EvaluationResult(Enum):
    PASS = "pass"
    FAIL = "fail"
    WARNING = "warning"

@dataclass
class EvaluationMetric:
    name: str
    score: float
    threshold: float = 0.7
    explanation: str = ""
    result: EvaluationResult = EvaluationResult.PASS

    def is_pass(self) -> bool:
        return self.result == EvaluationResult.PASS

class RAGEvaluator:
    def __init__(
        self,
        llm: BaseLLM,
        relevance_threshold: float = 0.7,
        quality_threshold: float = 3.5,
        strict_validation: bool = True
    ):
        """
        Evaluador completo para sistemas RAG con múltiples métricas.
        
        Args:
            llm: Modelo de lenguaje para realizar evaluaciones
            relevance_threshold: Umbral para considerar un documento relevante (0-1)
            quality_threshold: Umbral para considerar una respuesta de calidad (1-5)
            strict_validation: Si es True, falla con cualquier alucinación
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.quality_threshold = quality_threshold
        self.strict_validation = strict_validation
        self.output_parser = StrOutputParser()
        
        # Configurar cadenas de evaluación
        self._setup_evaluation_chains()

    def _setup_evaluation_chains(self):
        """Configura todas las cadenas de evaluación."""
        self.relevance_evaluator = (
            RELEVANCE_PROMPT 
            | self.llm 
            | self.output_parser
            | self._parse_relevance
        )
        
        self.hallucination_detector = (
            HALLUCINATION_PROMPT 
            | self.llm 
            | self.output_parser
            | self._parse_hallucination
        )
        
        self.quality_evaluator = (
            ANSWER_QUALITY_PROMPT 
            | self.llm 
            | self.output_parser
            | self._parse_quality
        )

    def evaluate_all(
        self,
        question: str,
        documents: List[str],
        answer: str
    ) -> Dict[str, Any]:
        """
        Ejecuta todas las evaluaciones disponibles.
        
        Returns:
            Dict con los resultados de todas las métricas
        """
        relevance = self.evaluate_relevance(documents[0], question) if documents else None
        hallucination = self.detect_hallucination(documents, answer) if documents else None
        quality = self.evaluate_answer_quality(question, answer)
        
        return {
            "relevance": relevance,
            "hallucination": hallucination,
            "quality": quality,
            "overall_pass": all([
                rel.is_pass() if rel else True for rel in [
                    relevance,
                    hallucination,
                    quality
                ]
            ])
        }

    def evaluate_relevance(self, document: str, question: str) -> EvaluationMetric:
        """
        Evalúa la relevancia de un documento para una pregunta dada.
        
        Returns:
            EvaluationMetric con score (0-1) y explicación
        """
        evaluation = self.relevance_evaluator.invoke({
            "document": document,
            "question": question
        })
        
        return EvaluationMetric(
            name="relevance",
            score=evaluation["score"],
            threshold=self.relevance_threshold,
            explanation=evaluation["explanation"],
            result=EvaluationResult.PASS if evaluation["score"] >= self.relevance_threshold 
                  else EvaluationResult.FAIL
        )

    def detect_hallucination(
        self, 
        documents: List[str], 
        generation: str
    ) -> EvaluationMetric:
        """
        Detecta alucinaciones en una generación comparando con los documentos.
        
        Returns:
            EvaluationMetric con score (0-1) y fragmentos no soportados
        """
        evaluation = self.hallucination_detector.invoke({
            "documents": "\n\n".join(documents),
            "generation": generation
        })
        
        has_hallucination = "no" in evaluation["raw_response"].lower()
        score = 0.0 if has_hallucination else 1.0
        
        return EvaluationMetric(
            name="hallucination",
            score=score,
            threshold=1.0 if self.strict_validation else 0.5,
            explanation=evaluation["explanation"],
            result=EvaluationResult.FAIL if has_hallucination and self.strict_validation
                  else EvaluationResult.WARNING if has_hallucination
                  else EvaluationResult.PASS
        )

    def evaluate_answer_quality(
        self, 
        question: str, 
        answer: str
    ) -> EvaluationMetric:
        """
        Evalúa la calidad de una respuesta (1-5) considerando:
        - Precisión
        - Completitud
        - Claridad
        - Utilidad
        
        Returns:
            EvaluationMetric con score (1-5) y sugerencias de mejora
        """
        evaluation = self.quality_evaluator.invoke({
            "question": question,
            "answer": answer
        })
        
        return EvaluationMetric(
            name="quality",
            score=evaluation["score"],
            threshold=self.quality_threshold,
            explanation=evaluation["explanation"],
            result=EvaluationResult.PASS if evaluation["score"] >= self.quality_threshold
                  else EvaluationResult.WARNING if evaluation["score"] >= 2.5
                  else EvaluationResult.FAIL
        )

    def _parse_relevance(self, response: str) -> Dict[str, Any]:
        """Parsea la respuesta de evaluación de relevancia."""
        parts = response.split("(", 1)
        score = 1.0 if "yes" in response.lower() else 0.0
        explanation = parts[1][:-1] if len(parts) > 1 else ""
        
        return {
            "raw_response": response,
            "score": score,
            "explanation": explanation
        }

    def _parse_hallucination(self, response: str) -> Dict[str, Any]:
        """Parsea la respuesta de detección de alucinaciones."""
        parts = response.split("(", 1)
        is_hallucination = "no" in response.lower()
        explanation = parts[1][:-1] if len(parts) > 1 else ""
        
        return {
            "raw_response": response,
            "is_hallucination": is_hallucination,
            "explanation": explanation
        }

    def _parse_quality(self, response: str) -> Dict[str, Any]:
        """Parsea la respuesta de evaluación de calidad."""
        # Extraer puntuación numérica (ej. "3", "4.5")
        score_str = ""
        for char in response:
            if char.isdigit() or char == ".":
                score_str += char
            elif score_str:
                break
                
        score = float(score_str) if score_str else 1.0
        
        # Extraer sugerencias de mejora
        improvements = ""
        if "Mejoras:" in response:
            improvements = response.split("Mejoras:")[1].strip()
        
        return {
            "raw_response": response,
            "score": min(max(score, 1.0), 5.0),  # Asegurar entre 1-5
            "explanation": improvements
        }

    async def aevaluate_all(
        self,
        question: str,
        documents: List[str],
        answer: str
    ) -> Dict[str, Any]:
        """Versión asíncrona de evaluate_all."""
        # Implementación básica - en producción usar llamadas async reales al LLM
        return self.evaluate_all(question, documents, answer)