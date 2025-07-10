from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_huggingface import HuggingFacePipeline

def cargar_llm_evaluador(model_name="google/flan-t5-large"):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
    pipe = pipeline(
        "text2text-generation",
        model=model,
        tokenizer=tokenizer,
        max_new_tokens=256,
        do_sample=False,
        temperature=0.0,
        device=-1
    )
    return HuggingFacePipeline(pipeline=pipe)