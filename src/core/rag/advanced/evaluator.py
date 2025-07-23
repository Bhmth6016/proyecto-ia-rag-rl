# src/core/rag/advanced/evaluator.py

from typing import List, Dict, Any
from dataclasses import dataclass
from enum import Enum
from pydantic import BaseModel, Field 

from transformers import (
    AutoTokenizer,
    AutoModelForSeq2SeqLM,
    pipeline,
)
from langchain_huggingface import HuggingFacePipeline
from langchain_core.language_models import BaseLLM
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

from src.core.rag.advanced.prompts import (
    RELEVANCE_PROMPT,
    HALLUCINATION_PROMPT, 
    ANSWER_QUALITY_PROMPT
)

# ------------------------------------------------------------------
# Generic LLM loader
# ------------------------------------------------------------------
def load_llm(
    model_name: str = "google/flan-t5-base",
    max_new_tokens: int = 256,
    do_sample: bool = True,
    temperature: float = 0.7,
    device: int = -1,
) -> HuggingFacePipeline:
    """
    Load a Hugging Face model and wrap it in a LangChain HuggingFacePipeline.

    Parameters
    ----------
    model_name : str
        Hugging Face model identifier.
    max_new_tokens : int
        Maximum new tokens to generate.
    do_sample : bool
        Whether to use sampling (True) or greedy decoding (False).
    temperature : float
        Sampling temperature (only if do_sample=True).
    device : int
        Device index (-1 for CPU, 0/1/... for GPUs).

    Returns
    -------
    HuggingFacePipeline
        Ready-to-use LangChain pipeline.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=max_new_tokens,
        do_sample=do_sample,
        temperature=temperature if do_sample else 0.0,
        device=device,
    )
    return HuggingFacePipeline(pipeline=pipe)


# ------------------------------------------------------------------
# Deterministic evaluator LLM
# ------------------------------------------------------------------
def load_evaluator_llm(
    model_name: str = "google/flan-t5-large",
    max_new_tokens: int = 256,
    device: int = -1,
) -> HuggingFacePipeline:
    """
    Load a deterministic LLM for evaluation tasks (relevance, hallucination, etc.).
    """
    return load_llm(
        model_name=model_name,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        temperature=0.0,
        device=device,
    )


# ------------------------------------------------------------------
# Auxiliary binary-grader Pydantic models (kept for compatibility)
# ------------------------------------------------------------------
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")


class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")


class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")


# ------------------------------------------------------------------
# RAG evaluator
# ------------------------------------------------------------------
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
    """
    Full RAG evaluator with multiple metrics.
    """

    def __init__(
        self,
        llm: BaseLLM,
        relevance_threshold: float = 0.7,
        quality_threshold: float = 3.5,
        strict_validation: bool = True,
    ):
        """
        Args:
            llm: Language model for evaluations
            relevance_threshold: Threshold to deem a document relevant (0-1)
            quality_threshold: Threshold to deem an answer high-quality (1-5)
            strict_validation: If True, fail on any hallucination
        """
        self.llm = llm
        self.relevance_threshold = relevance_threshold
        self.quality_threshold = quality_threshold
        self.strict_validation = strict_validation
        self.output_parser = StrOutputParser()

        # Configure evaluation chains
        self._setup_evaluation_chains()

    # ------------------- Setup -------------------
    def _setup_evaluation_chains(self):
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

    # ------------------- Public API -------------------
    def evaluate_all(
        self,
        question: str,
        documents: List[str],
        answer: str,
    ) -> Dict[str, Any]:
        """
        Run all available evaluations.

        Returns
        -------
        Dict
            Results for every metric.
        """
        relevance = self.evaluate_relevance(documents[0], question) if documents else None
        hallucination = self.detect_hallucination(documents, answer) if documents else None
        quality = self.evaluate_answer_quality(question, answer)

        return {
            "relevance": relevance,
            "hallucination": hallucination,
            "quality": quality,
            "overall_pass": all(
                m.is_pass() if m else True
                for m in [relevance, hallucination, quality]
            ),
        }

    # ------------------- Individual evaluations -------------------
    def evaluate_relevance(self, document: str, question: str) -> EvaluationMetric:
        evaluation = self.relevance_evaluator.invoke({
            "document": document,
            "question": question,
        })

        return EvaluationMetric(
            name="relevance",
            score=evaluation["score"],
            threshold=self.relevance_threshold,
            explanation=evaluation["explanation"],
            result=EvaluationResult.PASS
            if evaluation["score"] >= self.relevance_threshold
            else EvaluationResult.FAIL,
        )

    def detect_hallucination(
        self,
        documents: List[str],
        generation: str,
    ) -> EvaluationMetric:
        evaluation = self.hallucination_detector.invoke({
            "documents": "\n\n".join(documents),
            "generation": generation,
        })

        has_hallucination = "no" in evaluation["raw_response"].lower()
        score = 0.0 if has_hallucination else 1.0

        return EvaluationMetric(
            name="hallucination",
            score=score,
            threshold=1.0 if self.strict_validation else 0.5,
            explanation=evaluation["explanation"],
            result=EvaluationResult.FAIL
            if has_hallucination and self.strict_validation
            else EvaluationResult.WARNING
            if has_hallucination
            else EvaluationResult.PASS,
        )

    def evaluate_answer_quality(
        self,
        question: str,
        answer: str,
    ) -> EvaluationMetric:
        evaluation = self.quality_evaluator.invoke({
            "question": question,
            "answer": answer,
        })

        return EvaluationMetric(
            name="quality",
            score=evaluation["score"],
            threshold=self.quality_threshold,
            explanation=evaluation["explanation"],
            result=EvaluationResult.PASS
            if evaluation["score"] >= self.quality_threshold
            else EvaluationResult.WARNING
            if evaluation["score"] >= 2.5
            else EvaluationResult.FAIL,
        )

    # ------------------- Parsing helpers -------------------
    def _parse_relevance(self, response: str) -> Dict[str, Any]:
        parts = response.split("(", 1)
        score = 1.0 if "yes" in response.lower() else 0.0
        explanation = parts[1][:-1] if len(parts) > 1 else ""
        return {
            "raw_response": response,
            "score": score,
            "explanation": explanation,
        }

    def _parse_hallucination(self, response: str) -> Dict[str, Any]:
        parts = response.split("(", 1)
        is_hallucination = "no" in response.lower()
        explanation = parts[1][:-1] if len(parts) > 1 else ""
        return {
            "raw_response": response,
            "is_hallucination": is_hallucination,
            "explanation": explanation,
        }

    def _parse_quality(self, response: str) -> Dict[str, Any]:
        score_str = ""
        for char in response:
            if char.isdigit() or char == ".":
                score_str += char
            elif score_str:
                break

        score = float(score_str) if score_str else 1.0
        improvements = ""
        if "Improvements:" in response:
            improvements = response.split("Improvements:")[1].strip()

        return {
            "raw_response": response,
            "score": min(max(score, 1.0), 5.0),
            "explanation": improvements,
        }

    # ------------------- Async stub -------------------
    async def aevaluate_all(
        self,
        question: str,
        documents: List[str],
        answer: str,
    ) -> Dict[str, Any]:
        # production: switch to real async LLM calls
        return self.evaluate_all(question, documents, answer)