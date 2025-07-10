from src.agente.agent_loader import cargar_agente
from src.agente.question_enhancer import mejorar_pregunta
from src.llms.evaluators import hallucination_grader, answer_grader, retrieval_grader
from src.utils.documento import parse_binary_score

def evaluar_respuesta(question, context, respuesta):
    h = hallucination_grader.invoke({"documents": context, "generation": respuesta})
    a = answer_grader.invoke({"question": question, "generation": respuesta})
    return parse_binary_score(h) == "yes" and parse_binary_score(a) == "yes"

def filtrar_documentos(docs, question):
    relevantes = []
    for d in docs:
        score = retrieval_grader.invoke({"question": question, "document": d.page_content})
        if parse_binary_score(score) == "yes":
            relevantes.append(d)
    return relevantes

def main():
    persist_dir = r"C:\Users\x\OneDrive\Documents\GitHub\rag\chroma_index"
    agente = cargar_agente(persist_dir)

    print("🧠 Agente listo. Escribe 'salir' para terminar.\n")
    while True:
        texto = input("🧑 Tú: ").strip()
        if texto.lower() in {"salir", "exit", "q"}:
            print("👋 ¡Hasta luego!")
            break

        pregunta = mejorar_pregunta(texto)
        docs = agente.retriever.get_relevant_documents(pregunta)
        docs_filtrados = filtrar_documentos(docs, pregunta)
        contexto = "\n\n".join([d.page_content for d in docs_filtrados])

        resultado = agente.combine_docs_chain.invoke({
            "question": pregunta,
            "input_documents": docs_filtrados,
            "chat_history": []
        })
        respuesta = resultado.get("text", "No se pudo generar respuesta.").strip()

        if not evaluar_respuesta(pregunta, contexto, respuesta):
            respuesta += "\n⚠ Esta respuesta puede no estar completamente fundamentada."

        print(f"\n🤖 Asistente:\n{respuesta}\n")

if __name__ == "__main__":
    main()