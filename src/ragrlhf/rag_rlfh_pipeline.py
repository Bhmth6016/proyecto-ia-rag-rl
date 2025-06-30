import json
import numpy as np
import faiss
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain.embeddings import HuggingFaceEmbeddings
from src.data_loader import DataLoader

# 1. Embeddings con LangChain
def get_langchain_embedder(model_name="sentence-transformers/all-MiniLM-L6-v2"):
    return HuggingFaceEmbeddings(model_name=model_name)

def embed_products(products, embedder):
    texts = [p.get("title", "") + " " + str(p.get("details", "")) for p in products]
    return embedder.embed_documents(texts)

# 2. Índice FAISS
def build_faiss_index(embeddings):
    dim = embeddings[0].shape[0]
    index = faiss.IndexFlatL2(dim)
    index.add(np.array(embeddings).astype('float32'))
    return index

# 3. Recuperación de productos
def retrieve_products(query, embedder, index, products, k=5):
    query_emb = np.array(embedder.embed_query(query)).astype('float32')
    D, I = index.search(np.array([query_emb]), k)
    return [products[i] for i in I[0]]

# 4. Generación con DeepSeek
def generate_answer(query, retrieved_products, model, tokenizer):
    context = "\n".join([p["title"] + ": " + str(p.get("details", "")) for p in retrieved_products])
    prompt = f"Usuario: {query}\nProductos relevantes:\n{context}\nRespuesta:"
    inputs = tokenizer(prompt, return_tensors="pt")
    output = model.generate(**inputs, max_new_tokens=128)
    return tokenizer.decode(output[0], skip_special_tokens=True)

# 5. Recolección de feedback
def collect_feedback(query, retrieved, respuesta, feedback_file="feedback.jsonl"):
    print("\nRespuesta generada:\n", respuesta)
    rating = input("¿Qué tan útil fue esta respuesta? (1-5): ")
    comentario = input("¿Comentarios adicionales? (opcional): ")
    feedback_data = {
        "query": query,
        "retrieved_titles": [p["title"] for p in retrieved],
        "respuesta": respuesta,
        "rating": rating,
        "comentario": comentario
    }
    with open(feedback_file, "a", encoding="utf-8") as f:
        f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
    print("¡Gracias por tu feedback!\n")

# 6. Preparación del dataset para RLHF supervisado
def build_supervised_dataset(feedback_file="feedback.jsonl", output_file="rlhf_supervised.jsonl"):
    with open(feedback_file, encoding="utf-8") as f, open(output_file, "w", encoding="utf-8") as out:
        for line in f:
            entry = json.loads(line)
            if int(entry.get("rating", 0)) >= 4:
                context = "\n".join(entry["retrieved_titles"])
                prompt = f"Usuario: {entry['query']}\nProductos relevantes:\n{context}\nRespuesta:"
                data = {
                    "prompt": prompt,
                    "response": entry["respuesta"]
                }
                out.write(json.dumps(data, ensure_ascii=False) + "\n")

# 7. Pipeline principal
def main():
    # Carga productos
    loader = DataLoader()
    products = loader.load_data(use_cache=True)

    # Embeddings e índice
    embedder = get_langchain_embedder()
    embeddings = embed_products(products, embedder)
    index = build_faiss_index(embeddings)

    # Modelo DeepSeek
    tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
    model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

    # Loop de consulta y feedback
    while True:
        query = input("¿Qué producto buscas? (o escribe 'salir'): ")
        if query.lower() == "salir":
            break
        retrieved = retrieve_products(query, embedder, index, products, k=5)
        respuesta = generate_answer(query, retrieved, model, tokenizer)
        collect_feedback(query, retrieved, respuesta)

    # Preparar dataset RLHF (opcional, puedes llamarlo aparte)
    build_supervised_dataset()

if __name__ == "__main__":
    main()