from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from src.llms.evaluator import cargar_llm_evaluador

eval_llm = cargar_llm_evaluador()

# Relevancia del documento
class GradeDocuments(BaseModel):
    binary_score: str = Field(description="Documents are relevant to the question, 'yes' or 'no'")

relevance_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing relevance of a retrieved document to a user question. If the document contains keyword(s) or semantic meaning related to the user question, grade it as relevant. Give a binary score 'yes' or 'no'."),
    ("human", "Retrieved document: \n\n {document} \n\n User question: {question}")
])
retrieval_grader = relevance_prompt | eval_llm | StrOutputParser()

# Hallucination grader
class GradeHallucinations(BaseModel):
    binary_score: str = Field(description="Answer is grounded in the facts, 'yes' or 'no'")

hallucination_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an LLM generation is grounded in / supported by a set of retrieved facts. Give a binary score 'yes' or 'no'."),
    ("human", "Set of facts: \n\n {documents} \n\n LLM generation: {generation}")
])
hallucination_grader = hallucination_prompt | eval_llm | StrOutputParser()

# Answer grader
class GradeAnswer(BaseModel):
    binary_score: str = Field(description="Answer addresses the question, 'yes' or 'no'")

answer_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a grader assessing whether an answer addresses / resolves a question. Give a binary score 'yes' or 'no'."),
    ("human", "User question: \n\n {question} \n\n LLM generation: {generation}")
])
answer_grader = answer_prompt | eval_llm | StrOutputParser()