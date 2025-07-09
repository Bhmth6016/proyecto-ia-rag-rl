import json
import time
import logging
import numpy as np
import faiss
from typing import List, Dict, Optional
from transformers import AutoTokenizer, AutoModelForCausalLM
from langchain_huggingface import HuggingFaceEmbeddings
from src.data_loader import DataLoader

# Configuración del logger
logger = logging.getLogger(__name__)

def configure_logging():
    """Configura el sistema de logging"""
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s [%(levelname)s] %(name)s - %(message)s',
        handlers=[
            logging.FileHandler("rag_agent.log"),
            logging.StreamHandler(),
        ]
    )
    # Reducir verbosidad de algunas librerías
    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("sentence_transformers").setLevel(logging.WARNING)

def validate_product(product: dict) -> bool:
    """Valida que un producto tenga la estructura mínima requerida"""
    if not isinstance(product, dict):
        return False
        
    # Requerimos al menos título o detalles
    has_title = isinstance(product.get('title'), str) and product['title'].strip()
    has_details = isinstance(product.get('details'), dict) and product['details']
    
    return has_title or has_details

def get_langchain_embedder(model_name: str = "sentence-transformers/all-MiniLM-L6-v2"):
    """Obtiene el embedder con configuración optimizada"""
    return HuggingFaceEmbeddings(
        model_name=model_name,
        model_kwargs={'device': 'cpu'},
        encode_kwargs={'normalize_embeddings': True}
    )

def embed_products(products: list, embedder, max_retries: int = 3) -> list:
    """
    Genera embeddings para productos con validación y manejo de errores
    
    Args:
        products: Lista de diccionarios con datos de productos
        embedder: Instancia del embedder de HuggingFace
        max_retries: Intentos máximos para generar embeddings
        
    Returns:
        Lista de embeddings
        
    Raises:
        RuntimeError: Si falla después de max_retries intentos
    """
    # Validación inicial
    if not products or not isinstance(products, list):
        logger.warning("Lista de productos vacía o inválida")
        return []
    
    # Preprocesamiento y limpieza
    texts = []
    invalid_count = 0
    
    for product in products:
        if not validate_product(product):
            invalid_count += 1
            continue
            
        title = str(product.get('title', '')).strip()
        details = ' '.join([f"{k}:{v}" for k, v in product.get('details', {}).items()])
        text = f"{title} {details}".strip()
        texts.append(text)
    
    if invalid_count > 0:
        logger.warning(f"{invalid_count} productos inválidos omitidos")
    
    # Generación de embeddings con reintentos
    attempts = 0
    last_error = None
    
    while attempts < max_retries:
        try:
            return embedder.embed_documents(texts)
        except Exception as e:
            attempts += 1
            last_error = e
            logger.warning(f"Intento {attempts} fallido: {str(e)}")
            time.sleep(1)  # Pequeña pausa entre intentos
    
    logger.error(f"Fallo al generar embeddings después de {max_retries} intentos")
    raise RuntimeError(f"No se pudieron generar embeddings: {str(last_error)}")

def batch_embed_products(products: list, embedder, batch_size: int = 500) -> list:
    """
    Genera embeddings por lotes para manejar grandes volúmenes de productos
    
    Args:
        products: Lista completa de productos
        embedder: Instancia del embedder
        batch_size: Número de productos por lote
        
    Returns:
        Lista completa de embeddings
    """
    if not products:
        return []
    
    total_products = len(products)
    embeddings = []
    processed = 0
    
    logger.info(f"Iniciando embedding de {total_products} productos en lotes de {batch_size}")
    
    while processed < total_products:
        batch = products[processed:processed + batch_size]
        
        try:
            batch_embeddings = embed_products(batch, embedder)
            embeddings.extend(batch_embeddings)
            processed += len(batch)
            
            logger.info(
                f"Progreso: {processed}/{total_products} "
                f"({(processed/total_products)*100:.1f}%)"
            )
        except Exception as e:
            logger.error(f"Error en lote {processed}-{processed+batch_size}: {str(e)}")
            raise
    
    logger.info("Embedding completado exitosamente")
    return embeddings

def validate_embeddings(embeddings: list) -> bool:
    """Valida que los embeddings tengan la forma correcta"""
    if not embeddings:
        return False
        
    first_dim = len(embeddings[0])
    return all(
        len(emb) == first_dim 
        for emb in embeddings
        if isinstance(emb, (list, np.ndarray))
    )

def build_faiss_index(embeddings: list) -> faiss.Index:
    """Construye índice FAISS con manejo de memoria"""
    if not embeddings:
        raise ValueError("Lista de embeddings vacía")
        
    if not validate_embeddings(embeddings):
        raise ValueError("Embeddings con dimensiones inconsistentes")
        
    dim = len(embeddings[0])
    index = faiss.IndexFlatL2(dim)
    
    # Convertir a array numpy en bloques para grandes datasets
    chunk_size = 50000
    for i in range(0, len(embeddings), chunk_size):
        chunk = embeddings[i:i + chunk_size]
        arr = np.array(chunk).astype('float32')
        index.add(arr)
        del arr  # Liberar memoria
        
    return index

def retrieve_products(query: str, embedder, index, products: list, k: int = 5) -> list:
    """Recupera productos relevantes con manejo de errores"""
    try:
        query_emb = np.array(embedder.embed_query(query)).astype('float32')
        D, I = index.search(np.array([query_emb]), k)
        return [products[i] for i in I[0]]
    except Exception as e:
        logger.error(f"Error en recuperación: {str(e)}")
        return []

def generate_answer(query: str, retrieved_products: list, model, tokenizer) -> str:
    """Genera respuesta con contexto de productos recuperados"""
    try:
        context = "\n".join([
            f"{p['title']}: {str(p.get('details', ''))[:200]}..." 
            for p in retrieved_products
            if isinstance(p, dict)
        ])
        prompt = f"Usuario: {query}\nProductos relevantes:\n{context}\nRespuesta:"
        inputs = tokenizer(prompt, return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=128)
        return tokenizer.decode(output[0], skip_special_tokens=True)
    except Exception as e:
        logger.error(f"Error generando respuesta: {str(e)}")
        return "Lo siento, ocurrió un error al generar la respuesta."

def collect_feedback(query: str, retrieved: list, respuesta: str, feedback_file: str = "feedback.jsonl"):
    """Recolecta feedback del usuario de manera robusta"""
    try:
        print("\nRespuesta generada:\n", respuesta)
        
        rating = None
        while rating not in {'1', '2', '3', '4', '5'}:
            rating = input("¿Qué tan útil fue esta respuesta? (1-5, donde 5 es muy útil): ").strip()
        
        comentario = input("¿Comentarios adicionales? (opcional): ").strip()
        
        feedback_data = {
            "query": query,
            "retrieved_titles": [p.get("title", "Sin título") for p in retrieved],
            "respuesta": respuesta,
            "rating": int(rating),
            "comentario": comentario if comentario else None,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S")
        }
        
        with open(feedback_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(feedback_data, ensure_ascii=False) + "\n")
        
        print("¡Gracias por tu feedback!\n")
    except Exception as e:
        logger.error(f"Error recolectando feedback: {str(e)}")

def build_supervised_dataset(feedback_file: str = "feedback.jsonl", output_file: str = "rlhf_supervised.jsonl"):
    """Prepara dataset para RLHF con validación de datos"""
    try:
        with open(feedback_file, "r", encoding="utf-8") as f, \
             open(output_file, "w", encoding="utf-8") as out:
            
            for line in f:
                try:
                    entry = json.loads(line)
                    if int(entry.get("rating", 0)) >= 4:
                        context = "\n".join(entry["retrieved_titles"])
                        prompt = f"Usuario: {entry['query']}\nProductos relevantes:\n{context}\nRespuesta:"
                        data = {
                            "prompt": prompt,
                            "response": entry["respuesta"],
                            "source_feedback": entry
                        }
                        out.write(json.dumps(data, ensure_ascii=False) + "\n")
                except json.JSONDecodeError:
                    logger.warning("Línea de feedback inválida omitida")
                    continue
                
        logger.info(f"Dataset RLHF generado en {output_file}")
    except Exception as e:
        logger.error(f"Error generando dataset RLHF: {str(e)}")
        raise

def main():
    """Función principal del agente RAG."""
    configure_logging()
    
    try:
        # Carga productos
        loader = DataLoader()
        products = loader.load_data(use_cache=True)
        
        # Validación inicial
        if not products:
            logger.error("No se pudieron cargar productos")
            return
            
        logger.info(f"Productos cargados: {len(products)}")
        
        # Embeddings e índice
        embedder = get_langchain_embedder()
        embeddings = batch_embed_products(products, embedder)
        
        # Verificar embeddings
        valid_products = [p for p in products if validate_product(p)]
        if len(embeddings) != len(valid_products):
            logger.error(f"Discrepancia en conteo: {len(embeddings)} embeddings vs {len(valid_products)} productos válidos")
            return
            
        index = build_faiss_index(embeddings)
        
        # Modelo de generación
        tokenizer = AutoTokenizer.from_pretrained("deepseek-ai/deepseek-llm-7b-base")
        model = AutoModelForCausalLM.from_pretrained("deepseek-ai/deepseek-llm-7b-base")

        print("\n🧠 Agente listo. Escribe 'salir' para terminar.\n")
        
        # Loop de consulta
        while True:
            try:
                query = input("🧑 Tú: ").strip()
                if query.lower() in {"salir", "exit", "q"}:
                    print("👋 ¡Hasta luego!")
                    break

                # Recuperación y generación
                retrieved = retrieve_products(query, embedder, index, valid_products)
                respuesta = generate_answer(query, retrieved, model, tokenizer)
                
                # Mostrar resultados
                print(f"\n🤖 Asistente:\n{respuesta}\n")
                print("📌 Productos relevantes:")
                for i, prod in enumerate(retrieved, 1):
                    print(f"  {i}. {prod.get('title', 'Sin título')}")
                
                # Feedback
                collect_feedback(query, retrieved, respuesta)
                
            except KeyboardInterrupt:
                print("\n🛑 Operación cancelada por el usuario")
                break
            except Exception as e:
                logger.error(f"Error en ciclo de consulta: {str(e)}")
                print("⚠️ Ocurrió un error. Por favor intenta de nuevo.")
                
        # Preparar dataset RLHF al finalizar
        build_supervised_dataset()

    except Exception as e:
        logger.error(f"Error en el flujo principal: {str(e)}", exc_info=True)
        print(f"❌ Error crítico: {e}")
        raise

if __name__ == "__main__":
    main()