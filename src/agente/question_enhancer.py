from langchain_core.output_parsers import StrOutputParser
from src.prompts.rewrite_prompt import rewrite_prompt
from src.llms.evaluator import cargar_llm_evaluador

eval_llm = cargar_llm_evaluador()
question_rewriter = rewrite_prompt | eval_llm | StrOutputParser()

def mejorar_pregunta(pregunta: str) -> str:
    try:
        return question_rewriter.invoke({"question": pregunta}).strip()
    except Exception:
        return pregunta