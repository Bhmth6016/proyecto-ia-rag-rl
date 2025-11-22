# src/core/rag/advanced/RAGAgent.py
from __future__ import annotations
import json
import logging
import uuid
import time
from pathlib import Path
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

# External imports (asegúrate de tener estas dependencias)
from langchain_core.prompts import ChatPromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.memory import ConversationBufferWindowMemory
from langchain.chains import ConversationalRetrievalChain

# Local imports
from src.core.rag.advanced.evaluator import RAGEvaluator, load_llm_for_reward_model
from src.core.rag.advanced.feedback_processor import FeedbackProcessor
from src.core.rag.basic.retriever import Retriever
from src.core.data.product import Product
from src.core.config import settings
from src.core.init import get_system

try:
    from src.core.rag.advanced.trainer import RLHFTrainer
    HAS_RLHF_TRAINER = True
except Exception:
    RLHFTrainer = None
    HAS_RLHF_TRAINER = False

# Logger
try:
    from src.core.utils.logger import get_logger
    logger = get_logger(__name__)
except Exception:
    logger = logging.getLogger(__name__)
    logging.basicConfig(level=logging.INFO)


# ============================================================
# RL CONFIG / SIMPLE RL TRAINER
# ============================================================

@dataclass
class RLConfig:
    min_samples_for_training: int = 10
    models_dir: Path = Path("models/rl_models")
    reward_model_name: str = "reward_model.pkl"


class SimpleRLTrainer:

    def __init__(self, config: RLConfig = RLConfig()):
        self.config = config
        self.config.models_dir.mkdir(parents=True, exist_ok=True)

    def collect_logs(self, feedback_dir: Path) -> List[Dict]:
        feedbacks = []
        if not feedback_dir.exists():
            return feedbacks

        for path in sorted(feedback_dir.glob("*.jsonl")):
            try:
                with open(path, "r", encoding="utf-8") as f:
                    for line in f:
                        if not line.strip():
                            continue
                        try:
                            feedbacks.append(json.loads(line))
                        except json.JSONDecodeError:
                            logger.debug("Skipping invalid json line in %s", path)
            except Exception as e:
                logger.debug("Error reading %s: %s", path, e)
        return feedbacks

    def prepare_dataset(self, feedbacks: List[Dict]) -> List[Dict]:
        dataset = []
        for fb in feedbacks:
            rating = fb.get("rating", None)
            if rating is None:
                continue
            dataset.append({
                "query": fb.get("query", ""),
                "answer": fb.get("answer", ""),
                "rating": rating,
                "user_id": fb.get("extra_meta", {}).get("user_id", "unknown"),
                "selected_products": fb.get("extra_meta", {}).get("selected_product_id", None),
            })
        return dataset

    def train(self, dataset: List[Dict]) -> Optional[Path]:
        if not dataset or len(dataset) < self.config.min_samples_for_training:
            logger.info("Insufficient samples for RL training.")
            return None

        # Advanced trainer available?
        if HAS_RLHF_TRAINER and RLHFTrainer is not None:
            try:
                logger.info("Training with RLHFTrainer...")
                trainer = RLHFTrainer()
                prepared = trainer.prepare_rlhf_dataset_from_records(dataset)
                save_dir = self.config.models_dir / f"rlhf_{int(time.time())}"
                save_dir.mkdir(parents=True, exist_ok=True)
                trainer.train(prepared, save_dir)
                logger.info("RLHFTrainer model saved at %s", save_dir)
                return save_dir
            except Exception as e:
                logger.exception("RLHF trainer failed: %s", e)

        # Simple fallback artifact
        try:
            summary = {
                "trained_at": datetime.now().isoformat(),
                "num_samples": len(dataset),
                "avg_rating": sum(d["rating"] for d in dataset) / len(dataset)
            }
            path = self.config.models_dir / f"simple_reward_{int(time.time())}.json"
            with open(path, "w", encoding="utf-8") as fh:
                json.dump(summary, fh, indent=2)
            logger.info("Simple reward model saved at %s", path)
            return path
        except Exception as e:
            logger.exception("Failed to save simple reward artifact: %s", e)
            return None

    def get_latest_model_path(self) -> Optional[Path]:
        files = sorted(self.config.models_dir.glob("*"), key=lambda p: p.stat().st_mtime)
        return files[-1] if files else None


# ============================================================
# RAG AGENT (SIN TRADUCCIÓN)
# ============================================================

class RAGAgent:

    # ========================================================
    # 🔥 NUEVO __init__ (SIN TRADUCCIÓN)
    # ========================================================
    def __init__(self, products: Optional[List[Product]] = None, user_id: str = "default"):
        """RAGAgent optimizado manteniendo enriquecimiento de datos"""
        print("🧠 Inicializando RAGAgent (SIN TRADUCCIÓN)")

        system = get_system()
        self.products = products or getattr(system, "products", None)
        self.user_id = user_id

        # paths
        self.history_dir = Path(settings.PROC_DIR) / "historial"
        self.history_dir.mkdir(parents=True, exist_ok=True)
        self.feedback_dir = Path("data/feedback")
        self.feedback_dir.mkdir(parents=True, exist_ok=True)

        # retriever
        self.retriever = getattr(system, "retriever", None) or Retriever()
        try:
            if not self.retriever.index_exists():
                if self.products:
                    logger.info("Building retriever index...")
                    self.retriever.build_index(self.products)
                else:
                    logger.warning("No products provided and retriever index missing")
            else:
                try:
                    self.retriever._ensure_store_loaded()
                except Exception:
                    pass
        except Exception as e:
            logger.exception("Retriever initialization failed: %s", e)

        # LLM
        self.llm = self._initialize_llm()

        # memory & chain
        self.memory = ConversationBufferWindowMemory(memory_key="chat_history", k=5, return_messages=True)
        self.chain = self._build_chain()

        # feedback & evaluator
        self.feedback_processor = FeedbackProcessor()
        self.evaluator = RAGEvaluator(llm=self.llm)

        # perfil
        self.profile_manager = self._initialize_profile_manager()
        self.user_profile = self.profile_manager.get_or_create_profile(self.user_id)

        # RL
        self.rl_config = RLConfig(min_samples_for_training=getattr(settings, "RL_MIN_SAMPLES", 10))
        self.rl_trainer = SimpleRLTrainer(self.rl_config)
        self.reward_model = None
        self._try_load_reward_model_on_startup()

        # cargar feedback memory
        try:
            self.feedback_memory = self._load_feedback_memory()
        except Exception:
            self.feedback_memory = []

        logger.info("RAGAgent inicializado correctamente (sin traducción).")

    # ====================================================
    # LLM INIT
    # ====================================================
    def _initialize_llm(self):
        try:
            if getattr(settings, "GEMINI_API_KEY", None):
                logger.info("Initializing Gemini LLM")
                return ChatGoogleGenerativeAI(
                    model="gemini-1.5-flash",
                    google_api_key=settings.GEMINI_API_KEY,
                    temperature=0.3,
                )
        except Exception as e:
            logger.debug("Gemini init failed: %s", e)

        try:
            from langchain_huggingface import HuggingFacePipeline
            logger.info("Initializing HF pipeline LLM fallback")
            return HuggingFacePipeline.from_model_id(
                model_id=getattr(settings, "HF_FALLBACK_MODEL", "google/flan-t5-small"),
                task="text2text-generation",
                model_kwargs={"temperature": 0.3, "max_length": 512},
            )
        except Exception as e:
            logger.debug("HF fallback unavailable: %s", e)

        # Mock LLM
        class _MockLLM:
            def __call__(self, prompt: str):
                return "Respuesta simulada por MockLLM. Instala Gemini o HF para mejores resultados."
        logger.warning("Using MockLLM. Install Gemini or HF for production usage.")
        return _MockLLM()

    # ====================================================
    # BUILD CHAIN
    # ====================================================
    def _build_chain(self):
        start = time.time()
        while not hasattr(self.retriever, "store") or self.retriever.store is None:
            if time.time() - start > 10:
                logger.warning("Timeout waiting for retriever.store - continuing without chain")
                break
            time.sleep(0.2)

        prompt = ChatPromptTemplate.from_template(
            "Eres un asistente de productos. Usa el contexto proporcionado para contestar la pregunta.\n\n"
            "Contexto: {context}\n\n"
            "Pregunta: {question}\n\n"
            "Responde de forma concisa y útil."
        )

        try:
            return ConversationalRetrievalChain.from_llm(
                llm=self.llm,
                retriever=self.retriever.store.as_retriever(search_kwargs={"k": 5}) if hasattr(self.retriever, "store") and self.retriever.store else self.retriever.as_retriever(),
                memory=self.memory,
                combine_docs_chain_kwargs={"prompt": prompt},
                return_source_documents=False,
            )
        except Exception as e:
            logger.debug("Could not build ConversationalRetrievalChain: %s", e)
            return None

    # ====================================================
    # PROFILES
    # ====================================================
    def _initialize_profile_manager(self):

        class ProfileManager:

            def __init__(self, path: Path):
                self.path = path
                self.path.mkdir(parents=True, exist_ok=True)
                self._profiles: Dict[str, Dict] = {}
                self._load()

            def _load(self):
                try:
                    f = self.path / "profiles.json"
                    if f.exists():
                        with open(f, "r", encoding="utf-8") as fh:
                            self._profiles = json.load(fh)
                    else:
                        self._profiles = {}
                except Exception as e:
                    logger.debug("Could not load profiles: %s", e)
                    self._profiles = {}

            def _save(self):
                try:
                    with open(self.path / "profiles.json", "w", encoding="utf-8") as fh:
                        json.dump(self._profiles, fh, indent=2)
                except Exception as e:
                    logger.debug("Could not save profiles: %s", e)

            def get_or_create_profile(self, user_id: str) -> Dict:
                if user_id not in self._profiles:
                    self._profiles[user_id] = {
                        "user_id": user_id,
                        "preferred_categories": [],
                        "search_history": [],
                        "purchase_history": []
                    }
                    self._save()
                return self._profiles[user_id]

            def update_user_activity(self, user_id: str, query: str, selected_products: List[str]):
                profile = self.get_or_create_profile(user_id)
                profile["search_history"].append({
                    "query": query,
                    "selected_products": selected_products,
                    "timestamp": datetime.now().isoformat()
                })
                # Keep recent history size bounded
                if len(profile["search_history"]) > 200:
                    profile["search_history"] = profile["search_history"][-200:]
                self._save()

            def add_purchase(self, user_id: str, product_id: str, product_data: Dict):
                profile = self.get_or_create_profile(user_id)
                profile["purchase_history"].append({
                    "product_id": product_id,
                    "product_data": product_data,
                    "timestamp": datetime.now().isoformat()
                })
                # Update preferred categories heuristically
                cats = [product_data.get("main_category")] if product_data.get("main_category") else []
                profile["preferred_categories"] = list({*profile.get("preferred_categories", []), *cats})[:5]
                self._save()

        return ProfileManager(Path(settings.PROC_DIR) / "profiles")

    # ====================================================
    # LOAD REWARD MODEL
    # ====================================================
    def _try_load_reward_model_on_startup(self):
        try:
            latest = self.rl_trainer.get_latest_model_path()
            if latest:
                logger.info("Attempting to load reward model from %s", latest)
                # Si es carpeta creada por RLHFTrainer, usar load_llm_for_reward_model
                if latest.is_dir():
                    try:
                        self.reward_model = load_llm_for_reward_model(str(latest))
                        logger.info("Reward model loaded from dir %s", latest)
                        return
                    except Exception as e:
                        logger.debug("load_llm_for_reward_model failed: %s", e)
                else:
                    # If it's a JSON artifact, we keep it as metadata
                    try:
                        with open(latest, "r", encoding="utf-8") as fh:
                            meta = json.load(fh)
                        logger.info("Loaded simple reward artifact metadata: %s", latest)
                        # create a simple scoring function based on avg_rating
                        avg = meta.get("avg_rating", 0.0)
                        def simple_reward_scoring(text: str) -> float:
                            # heurística: texts with length > threshold get slightly higher score
                            length_factor = min(len(text.split()) / 100.0, 1.0)
                            return float(min(1.0, avg/5.0 + 0.1 * length_factor))
                        self.reward_model = simple_reward_scoring
                        return
                    except Exception:
                        logger.debug("Not a json artifact or failed to read: %s", latest)
        except Exception as e:
            logger.debug("Error loading reward model at startup: %s", e)

    # ====================================================
    # FEEDBACK / LOGS
    # ====================================================
    def _load_feedback_memory(self) -> List[Dict]:
        """Carga feedbacks con rating bajo para evitar repetir malas respuestas"""
        feedbacks = []
        try:
            for path in sorted(self.feedback_dir.glob("*.jsonl")):
                with open(path, "r", encoding="utf-8") as fh:
                    for line in fh:
                        try:
                            obj = json.loads(line.strip())
                            if obj.get("rating", 5) <= 3:
                                feedbacks.append(obj)
                        except Exception:
                            continue
        except Exception as e:
            logger.debug("Error loading feedback memory: %s", e)
        return feedbacks

    def _save_conversation(self, query: str, response: str, feedback: Optional[str] = None):
        """Guarda la conversación diaria en historial"""
        try:
            ts = datetime.now().isoformat()
            file = self.history_dir / f"conversation_{ts[:10]}.json"
            item = {
                "session_id": str(uuid.uuid4()),
                "timestamp": ts,
                "user_id": self.user_id,
                "query": query,
                "response": response,
                "feedback": feedback
            }
            # append to existing day file
            if file.exists():
                try:
                    with open(file, "r", encoding="utf-8") as fh:
                        data = json.load(fh)
                    if isinstance(data, list):
                        data.append(item)
                    else:
                        data = [data, item]
                except Exception:
                    data = [item]
            else:
                data = [item]
            with open(file, "w", encoding="utf-8") as fh:
                json.dump(data, fh, indent=2, ensure_ascii=False)
        except Exception as e:
            logger.debug("Error saving conversation: %s", e)

    # ====================================================
    # CORE QUERY HANDLING
    # ====================================================
    def ask(self, query: str) -> str:
        """
        Flujo principal:
        1. Personalizar consulta por perfil
        2. Recuperar con retriever
        3. Rankear personalizado
        4. Formatear y evaluar con reward_model si existe
        5. Guardar interacción y feedback
        """
        try:
            logger.info("Ask called for user=%s query=%s", self.user_id, query)
            # 1. Personalize
            personalized_query = self._personalize_query(query)

            # 2. Retrieve
            products = self.retriever.retrieve(query=personalized_query, k=10, min_similarity=0.05)
            if not products:
                return self._handle_no_results(query)

            # 3. Rank
            ranked = self._rank_products_for_user(products, query)
            if not ranked:
                return self._handle_no_results(query)

            # 4. Format
            response = self._format_personalized_response(query, ranked[:5])

            # 5. Reward model evaluation (if available)
            if self.reward_model:
                try:
                    # reward_model can be callable or LLM wrapper
                    if callable(self.reward_model):
                        score = float(self.reward_model(f"{query} {response}"))
                    else:
                        out = self.reward_model(f"{query} {response}")
                        try:
                            score = float(out)
                        except Exception:
                            score = float(out.get("score", 0.0)) if isinstance(out, dict) else 0.0
                    logger.info("Reward score: %.3f", score)
                    if score < 0.25:
                        logger.info("Low reward score - generating fallback")
                        return self._generate_fallback_response(query)
                except Exception as e:
                    logger.debug("Reward model evaluation failed: %s", e)

            # 6. Save conversation & record for RL training
            self._save_conversation(query, response)
            self._save_interaction_for_training(query, [p.id for p in ranked[:3]])

            return response

        except Exception as e:
            logger.exception("Error in ask(): %s", e)
            return "Ocurrió un error interno al procesar tu consulta."

    def _personalize_query(self, query: str) -> str:
        """Adjunta categorías preferidas para mejorar recuperación"""
        try:
            prefs = self.user_profile.get("preferred_categories", [])
            if prefs:
                return f"{query} " + " ".join(prefs)
        except Exception:
            pass
        return query

    def _rank_products_for_user(self, products: List[Product], query: str) -> List[Product]:
        """Ranking personalizado usando scores del retriever + boosts"""
        scored = []
        for p in products:
            try:
                base = getattr(p, "score", None)
                if base is None:
                    base = self.retriever._score(query, p) if hasattr(self.retriever, "_score") else 0.5
                boost = 1.0
                # category boost
                if p.main_category and p.main_category in self.user_profile.get("preferred_categories", []):
                    boost *= 1.4
                # purchase similarity boost
                if any(ph.get("product_data", {}).get("main_category") == p.main_category for ph in self.user_profile.get("purchase_history", [])[-3:]):
                    boost *= 1.2
                final = base * boost
                scored.append((final, p))
            except Exception:
                continue
        scored = [t for t in scored if t[0] > 0.01]
        scored.sort(key=lambda x: x[0], reverse=True)
        return [p for _, p in scored]

    def _format_personalized_response(self, query: str, products: List[Product]) -> str:
        """Formatea top-3 (o top-n) productos en respuesta amigable"""
        if not products:
            return self._handle_no_results(query)

        lines = [f"🎯 Recomendaciones para '{query}':"]

        for i, p in enumerate(products[:3], 1):
            title = getattr(p, "title", "Producto sin nombre")
            price = getattr(p, "price", None)
            rating = getattr(p, "average_rating", None)

            price_str = f"${price:.2f}" if price else "Precio no disponible"
            rating_str = f"{rating}/5" if rating else "Sin calificaciones"
            cat = getattr(p, "main_category", "Categoría")

            # 🔥 CORRECCIÓN: Acceso seguro a features
            try:
                details_obj = getattr(p, "details", None)
                if details_obj and hasattr(details_obj, "features"):
                    features = details_obj.features or []
                else:
                    features = []
            except Exception:
                features = []

            feat = " | ".join(features[:2]) if features else "Características no disponibles"
            personal_tag = " 🎯" if cat in self.user_profile.get("preferred_categories", []) else ""

            lines.append(f"{i}. {title}{personal_tag}")
            lines.append(f"   {price_str} | {rating_str} | {cat}")
            lines.append(f"   {feat}")

        if self.user_profile.get("preferred_categories"):
            lines.append(
                f"\n💡 Basado en tus preferencias: "
                f"{', '.join(self.user_profile.get('preferred_categories', []))}"
            )
            
        return "\n".join(lines)


    def _handle_no_results(self, query: str) -> str:
        return (
            f"🔎 No encontré productos para '{query}'.\n"
            "Prueba con términos más generales o revisa la ortografía.\n"
            "Sugerencias: buscar por categoría (electrónica, hogar, belleza) o usar palabras como 'inalámbrico', 'económico'."
        )

    def _generate_fallback_response(self, query: str) -> str:
        return (
            f"Lo siento — tengo dificultades en encontrar buenos resultados para '{query}'.\n"
            "Intenta usar términos más generales o explora las categorías principales."
        )

    # ====================================================
    # TRAINING & RL lifecycle
    # ====================================================
    def _prepare_training_data_from_feedback(self) -> List[Dict]:
        """Agrupa feedbacks y prepara dataset para trainer"""
        all_feedbacks = self.rl_trainer.collect_logs(self.feedback_dir)
        dataset = self.rl_trainer.prepare_dataset(all_feedbacks)
        return dataset

    def trigger_retrain_if_needed(self) -> Optional[Path]:
        """Valida si hay suficientes logs y entrena nueva versión del reward model"""
        try:
            dataset = self._prepare_training_data_from_feedback()
            if len(dataset) >= self.rl_config.min_samples_for_training:
                logger.info("Triggering RL retrain with %d samples", len(dataset))
                model_path = self.rl_trainer.train(dataset)
                if model_path:
                    # Si trainer dejó carpeta con modelo, intentar cargar con load_llm_for_reward_model
                    if model_path.is_dir():
                        try:
                            self.reward_model = load_llm_for_reward_model(str(model_path))
                            logger.info("Loaded reward model from %s", model_path)
                        except Exception as e:
                            logger.exception("Failed to load reward model from %s: %s", model_path, e)
                    else:
                        # if JSON artifact, create simple scoring as before
                        try:
                            with open(model_path, "r", encoding="utf-8") as fh:
                                meta = json.load(fh)
                            avg = meta.get("avg_rating", 0.0)
                            def simple_reward_scoring(text: str) -> float:
                                length_factor = min(len(text.split()) / 100.0, 1.0)
                                return float(min(1.0, avg/5.0 + 0.1 * length_factor))
                            self.reward_model = simple_reward_scoring
                        except Exception as e:
                            logger.debug("Could not create simple scoring from artifact: %s", e)
                return model_path
            else:
                logger.info("Not enough data to retrain RL: %d/%d", len(dataset), self.rl_config.min_samples_for_training)
                return None
        except Exception as e:
            logger.exception("Error triggering retrain: %s", e)
            return None

    def _save_interaction_for_training(self, query: str, selected_product_ids: List[str]):
        try:
            self.profile_manager.update_user_activity(self.user_id, query, selected_product_ids)
        except Exception as e:
            logger.debug("Could not save interaction for training: %s", e)

    # ====================================================
    # FEEDBACK API
    # ====================================================
    def _log_feedback(self, query: str, answer: str, rating: int, extra_meta: Optional[Dict] = None):
        """Guardado robusto de feedback en formato jsonl"""
        try:
            entry = {
                "timestamp": datetime.now().isoformat(),
                "query": query,
                "answer": answer,
                "rating": rating,
                "extra_meta": extra_meta or {"user_id": self.user_id}
            }
            fname = self.feedback_dir / f"feedback_{datetime.now().strftime('%Y%m%d')}.jsonl"
            with open(fname, "a", encoding="utf-8") as fh:
                fh.write(json.dumps(entry, ensure_ascii=False) + "\n")
            # también dejar pasar por feedback_processor si existe
            try:
                self.feedback_processor.save_feedback(query=query, answer=answer, rating=rating, extra_meta=entry["extra_meta"])
            except Exception:
                pass
            # Al juntar nuevos datos, considerar retrain asíncrono en reinicio (no aquí)
        except Exception as e:
            logger.debug("Failed to log feedback: %s", e)

    # ====================================================
    # Utilities & CLI loop
    # ====================================================
    def product_selected(self, product_id: str, product_data: Dict):
        """Registrar compra/selección (call desde UI)"""
        try:
            self.profile_manager.add_purchase(self.user_id, product_id, product_data)
            # Guardar feedback implícito como positivo
            self._log_feedback(product_data.get("title", ""), f"Selected product {product_id}", rating=5,
                               extra_meta={"user_id": self.user_id, "selected_product_id": product_id})
            logger.info("Product selection saved for user %s product %s", self.user_id, product_id)
        except Exception as e:
            logger.debug("Error saving product selection: %s", e)

    def get_recommendations(self, k: int = 5) -> List[Product]:
        """Recomendaciones rápidas basadas en último query / categorías preferidas"""
        try:
            prefs = self.user_profile.get("preferred_categories", [])
            last_query = (self.user_profile.get("search_history") or [])[-1]["query"] if self.user_profile.get("search_history") else ""
            query = last_query or " ".join(prefs) or "productos populares"
            products = self.retriever.retrieve(query=query, k=k*2)
            ranked = self._rank_products_for_user(products, query)
            return ranked[:k]
        except Exception as e:
            logger.debug("Error in get_recommendations: %s", e)
            return []

    def chat_loop(self) -> None:
        """Bucle CLI para testing rápido"""
        print("\n=== RAG Agent (with RL) ===\nType 'exit' to quit\n")
        while True:
            try:
                q = input("You: ").strip()
                if q.lower() in {"exit", "quit", "q"}:
                    break
                if not q:
                    continue
                ans = self.ask(q)
                print("\n" + "-"*60)
                print(ans)
                print("-"*60 + "\n")
                # feedback
                fb = input("Was that helpful? (1-5 / skip): ").strip().lower()
                if fb in {"1","2","3","4","5"}:
                    self._log_feedback(q, ans, int(fb), extra_meta={"user_id": self.user_id})
                    print("Thanks! Feedback registered.")
                elif fb == "skip":
                    continue
            except KeyboardInterrupt:
                print("\nSession ended by user.")
                break
            except Exception as e:
                logger.exception("Error in chat loop: %s", e)
                print("Internal error. See logs.")

# Convenience factory
def create_rag_agent(products: List[Product] = None, user_id: str = "default") -> RAGAgent:
    return RAGAgent(products=products, user_id=user_id)
