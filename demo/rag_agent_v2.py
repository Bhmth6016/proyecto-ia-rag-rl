import os
import logging
from typing import Optional, List, Dict, Any
from dotenv import load_dotenv

from langchain.prompts import PromptTemplate
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFacePipeline
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, pipeline
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain.schema import Document

# Configuración de logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
    handlers=[
        logging.FileHandler("rag_agent_v2.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Cargar variables de entorno
load_dotenv()

# Configuración global
DEFAULT_PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_indexes")
DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_LLM_MODEL = "google/flan-t5-large"

class RAGAgentV2:
    def __init__(self, persist_directory: str = None, embedding_model: str = None, llm_model: str = None):
        """
        Inicializa el agente RAG mejorado.
        
        Args:
            persist_directory: Directorio con los índices Chroma
            embedding_model: Nombre del modelo de embeddings
            llm_model: Nombre del modelo LLM
        """
        self.persist_directory = persist_directory or DEFAULT_PERSIST_DIR
        self.embedding_model = embedding_model or DEFAULT_EMBEDDING_MODEL
        self.llm_model = llm_model or DEFAULT_LLM_MODEL
        
        self.vectordb = None
        self.retriever = None
        self.llm = None
        self.memory = None
        self.chain = None
        
        self._initialize_components()

    def _initialize_components(self):
        """Inicializa todos los componentes del agente."""
        try:
            # 1. Cargar vectorstore
            self.vectordb = self._load_vectorstore()
            
            # 2. Configurar retriever
            self.retriever = self.vectordb.as_retriever(search_kwargs={"k": 5})
            
            # 3. Cargar modelo LLM
            self.llm = self._load_llm()
            
            # 4. Configurar memoria
            self.memory = ConversationBufferWindowMemory(
                memory_key="chat_history",
                k=5,
                return_messages=True
            )
            
            # 5. Crear cadena conversacional
            self.chain = self._create_conversational_chain()
            
            logger.info("Agente RAG inicializado correctamente")
            
        except Exception as e:
            logger.error(f"Error inicializando el agente: {str(e)}")
            raise

    def _load_vectorstore(self) -> Chroma:
        """Carga el vectorstore desde el directorio persistente."""
        if not os.path.isdir(self.persist_directory):
            logger.error(f"Directorio no encontrado: {self.persist_directory}")
            raise FileNotFoundError(f"Directorio de índices no encontrado: {self.persist_directory}")

        embeddings = HuggingFaceEmbeddings(
            model_name=self.embedding_model,
            model_kwargs={"device": "cpu"}
        )
        
        return Chroma(
            persist_directory=self.persist_directory,
            embedding_function=embeddings
        )

    def _load_llm(self):
        """Carga el modelo de lenguaje."""
        try:
            tokenizer = AutoTokenizer.from_pretrained(self.llm_model)
            model = AutoModelForSeq2SeqLM.from_pretrained(self.llm_model)
            
            pipe = pipeline(
                "text2text-generation",
                model=model,
                tokenizer=tokenizer,
                max_new_tokens=256,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
                device="cpu"
            )
            
            return HuggingFacePipeline(pipeline=pipe)
        except Exception as e:
            logger.error(f"Error cargando el modelo LLM: {str(e)}")
            raise

    def _create_conversational_chain(self):
        """Crea la cadena conversacional con plantilla personalizada."""
        prompt_template = """
        Eres un asistente experto en productos para bebés. Sigue esta estructura:

        - Saludo breve
        - Recomendación clara y concisa basada en los productos relevantes
        - Cierre amable invitando a más preguntas

        Contexto:
        {context}

        Pregunta:
        {question}

        Respuesta útil:
        """
        
        prompt = PromptTemplate(
            template=prompt_template,
            input_variables=["context", "question"]
        )
        
        return ConversationalRetrievalChain.from_llm(
            llm=self.llm,
            retriever=self.retriever,
            memory=self.memory,
            combine_docs_chain_kwargs={"prompt": prompt},
            return_source_documents=True
        )

    def query(self, question: str) -> Dict[str, Any]:
        """
        Procesa una pregunta y devuelve la respuesta del agente.
        
        Args:
            question: Pregunta del usuario
            
        Returns:
            Dict con:
            - answer: Respuesta generada
            - source_documents: Documentos fuente relevantes
            - chat_history: Historial de conversación
        """
        try:
            # Mejorar la pregunta para mejor recuperación
            improved_question = self._improve_question(question)
            
            # Ejecutar la cadena
            result = self.chain({
                "question": improved_question,
                "chat_history": self.memory.chat_history
            })
            
            # Validar la respuesta
            is_valid = self._validate_response(
                question=improved_question,
                context="\n".join([doc.page_content for doc in result["source_documents"]]),
                answer=result["answer"]
            )
            
            if not is_valid:
                result["answer"] += "\n\n Advertencia: Esta respuesta puede contener información no verificada."
            
            return {
                "answer": result["answer"],
                "source_documents": result["source_documents"],
                "chat_history": self.memory.chat_history
            }
            
        except Exception as e:
            logger.error(f"Error procesando pregunta: {str(e)}")
            return {
                "answer": "Lo siento, ocurrió un error al procesar tu pregunta.",
                "source_documents": [],
                "chat_history": self.memory.chat_history
            }

    def _improve_question(self, question: str) -> str:
        """Mejora la pregunta para una mejor recuperación."""
        rewrite_system = """
        Eres un reescribidor de preguntas que convierte una pregunta de entrada en una 
        versión mejorada optimizada para recuperación de documentos. Analiza la entrada 
        e intenta deducir la intención semántica subyacente.
        """
        
        rewrite_prompt = ChatPromptTemplate.from_messages([
            ("system", rewrite_system),
            ("human", "Aquí está la pregunta inicial: \n\n {question} \n Formula una pregunta mejorada.")
        ])
        
        rewriter = rewrite_prompt | self.llm | StrOutputParser()
        return rewriter.invoke({"question": question}).strip()

    def _validate_response(self, question: str, context: str, answer: str) -> bool:
        """Evalúa si la respuesta es adecuada."""
        class GradeResponse(BaseModel):
            binary_score: str = Field(description="'yes' si la respuesta es válida, 'no' si no")
        
        validation_prompt = ChatPromptTemplate.from_messages([
            ("system", """Eres un evaluador de respuestas. Determina si:
            1. La respuesta está fundamentada en el contexto
            2. Responde adecuadamente a la pregunta
            Responde solo con 'yes' o 'no'."""),
            ("human", f"Pregunta: {question}\nContexto: {context}\nRespuesta: {answer}")
        ])
        
        grader = validation_prompt | self.llm | StrOutputParser()
        score = grader.invoke({}).strip().lower()
        return score == 'yes'

    def reset_conversation(self):
        """Reinicia el historial de conversación."""
        self.memory.clear()
        logger.info("Historial de conversación reiniciado")

def initialize_rag_agent_v2(persist_directory: str = None) -> RAGAgentV2:
    """
    Inicializa y retorna una instancia del agente RAG mejorado.
    
    Args:
        persist_directory: Directorio con los índices Chroma (opcional)
    
    Returns:
        Instancia de RAGAgentV2
    """
    try:
        return RAGAgentV2(persist_directory=persist_directory)
    except Exception as e:
        logger.error(f"Error inicializando RAGAgentV2: {str(e)}")
        raise

def run_rag_agent_v2_interactive(persist_directory: str = None):
    """
    Ejecuta el agente RAG en modo interactivo (para uso desde main.py).
    
    Args:
        persist_directory: Directorio con los índices Chroma (opcional)
    """
    try:
        agent = initialize_rag_agent_v2(persist_directory)
        print("\n Agente RAG Mejorado listo. Escribe 'salir' para terminar.\n")
        
        while True:
            try:
                question = input(" Tú: ").strip()
                if question.lower() in {"salir", "exit", "q"}:
                    print(" ¡Hasta luego!")
                    break
                
                response = agent.query(question)
                print(f"\n Asistente:\n{response['answer']}\n")
                
                if response['source_documents']:
                    print(" Documentos relevantes:")
                    for i, doc in enumerate(response['source_documents'], 1):
                        print(f"{i}. {doc.metadata.get('title', 'Sin título')}")
                    print()
                
            except KeyboardInterrupt:
                print("\n Operación cancelada por el usuario")
                break
            except Exception as e:
                logger.error(f"Error en ciclo de conversación: {str(e)}")
                print(" Ocurrió un error. Por favor intenta de nuevo.")
                
    except Exception as e:
        logger.error(f"Error en run_rag_agent_v2_interactive: {str(e)}")
        print(f" Error crítico: {e}")

if __name__ == "__main__":
    # Para pruebas directas
    run_rag_agent_v2_interactive()