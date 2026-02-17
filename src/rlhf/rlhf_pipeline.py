"""
RLHF Pipeline — Orquesta los 5 componentes del ciclo RLHF completo

El ciclo completo que implementa este módulo:

    ┌─────────────────────────────────────────────────────┐
    │  1. FAISS retrieval → candidatos (base)             │
    │  2. PolicyModel genera 2 rankings alternativos      │
    │  3. Humano compara A vs B → PreferenceCollector     │
    │  4. RewardModel.train_on_preferences(...)           │
    │  5. PPOTrainer.collect_and_store(...)               │
    │  6. PPOTrainer.update()  ← EL RL REAL              │
    │  7. Volver a 2 con política mejorada                │
    └─────────────────────────────────────────────────────┘

La diferencia con el sistema anterior:
    ANTES: ajuste de pesos con regla manual (bandit lineal)
    AHORA: reward model aprendido + PPO con KL constraint
"""

import json
import pickle
import logging
from pathlib import Path
from typing import List, Dict, Optional, Tuple, Any
import numpy as np
import torch

from .policy_model import PolicyModel
from .reward_model import RewardModel
from .ppo_trainer import PPOTrainer, RankingExperience
from .preference_collector import PreferenceCollector
from .tensor_utils import ProductTensorizer

logger = logging.getLogger(__name__)


class RLHFPipeline:
    """
    Orquesta el ciclo RLHF completo para ranking de productos.

    Uso:
        pipeline = RLHFPipeline(system_v2)
        pipeline.initialize()

        # Paso 1: recolectar preferencias A-vs-B
        pipeline.run_preference_collection_session(queries)

        # Paso 2: entrenar el Reward Model con las preferencias
        pipeline.train_reward_model()

        # Paso 3: optimizar la Policy con PPO
        pipeline.run_ppo_training_cycle(queries)

        # Usar la política entrenada para ranking
        ranked = pipeline.rank_products(query, products)
    """

    def __init__(
        self,
        base_system=None,
        embedding_dim: int = 384,
        feature_dim: int = 8,
        hidden_dim: int = 128,
        max_products: int = 20,
        device: str = "cpu",
        cache_dir: str = "data/cache/rlhf",
    ):
        self.base_system = base_system
        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_products = max_products
        self.device = device
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Componentes (se inicializan en initialize())
        self.policy: Optional[PolicyModel] = None
        self.reward_model: Optional[RewardModel] = None
        self.ppo_trainer: Optional[PPOTrainer] = None
        self.preference_collector: Optional[PreferenceCollector] = None
        self.tensorizer: Optional[ProductTensorizer] = None

        # Estado del pipeline
        self.reward_model_trained: bool = False
        self.policy_trained: bool = False
        self.ppo_cycles_completed: int = 0

        logger.info(f"RLHFPipeline creado — device={device}")

    # ─────────────────────────────────────────────────────────────────────
    # Inicialización
    # ─────────────────────────────────────────────────────────────────────

    def initialize(self, load_checkpoint: bool = True):
        """
        Inicializa todos los componentes del pipeline.
        Si existen checkpoints, los carga automáticamente.
        """
        logger.info("Inicializando RLHFPipeline completo...")

        # Tensorizer (bridge entre objetos Python y tensores)
        self.tensorizer = ProductTensorizer(
            embedding_dim=self.embedding_dim,
            max_products=self.max_products,
        )

        # Policy Model
        self.policy = PolicyModel(
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
        ).to(self.device)

        # Reward Model
        self.reward_model = RewardModel(
            embedding_dim=self.embedding_dim,
            feature_dim=self.feature_dim,
            hidden_dim=self.hidden_dim,
            max_products=self.max_products,
        ).to(self.device)

        # PPO Trainer
        self.ppo_trainer = PPOTrainer(
            policy_model=self.policy,
            reward_model=self.reward_model,
            device=self.device,
        )

        # Preference Collector
        self.preference_collector = PreferenceCollector(
            output_file="data/preferences/preferences.jsonl"
        )

        # Cargar checkpoints si existen
        if load_checkpoint:
            self._try_load_checkpoints()

        params = self.policy.count_parameters()
        logger.info(
            f"Pipeline inicializado:\n"
            f"  PolicyModel:     {params:,} parámetros\n"
            f"  RewardModel entrenado: {self.reward_model_trained}\n"
            f"  Policy entrenada:      {self.policy_trained}\n"
            f"  PPO cycles:            {self.ppo_cycles_completed}\n"
            f"  Preferencias:          {self.preference_collector.total_preferences}"
        )

    def _try_load_checkpoints(self):
        policy_ck = self.cache_dir / "policy.pt"
        reward_ck = self.cache_dir / "reward_model.pt"
        ppo_ck = self.cache_dir / "ppo_trainer.pt"
        state_ck = self.cache_dir / "pipeline_state.json"

        if reward_ck.exists():
            try:
                self.reward_model = RewardModel.load(str(reward_ck), self.device)
                self.reward_model_trained = True
                logger.info(f"RewardModel cargado desde checkpoint")
            except Exception as e:
                logger.warning(f"No se pudo cargar RewardModel: {e}")

        if ppo_ck.exists():
            try:
                self.ppo_trainer.load(str(ppo_ck))
                self.policy_trained = self.ppo_trainer.total_updates > 0
                logger.info(f"PPOTrainer cargado (updates={self.ppo_trainer.total_updates})")
            except Exception as e:
                logger.warning(f"No se pudo cargar PPOTrainer: {e}")

        if state_ck.exists():
            try:
                with open(state_ck) as f:
                    state = json.load(f)
                self.ppo_cycles_completed = state.get("ppo_cycles_completed", 0)
            except Exception:
                pass

    # ─────────────────────────────────────────────────────────────────────
    # Paso 1 — Retrieval base (FAISS)
    # ─────────────────────────────────────────────────────────────────────

    def retrieve_candidates(
        self, query: str, k: int = 20
    ) -> Tuple[list, np.ndarray, np.ndarray]:
        """
        Recupera candidatos con FAISS del sistema base.

        Returns:
            products:         lista de productos canónicos
            query_emb:        (emb_dim,) ndarray
            baseline_scores:  (k,) ndarray de cosine similarities
        """
        if self.base_system is None:
            logger.error("base_system no configurado")
            return [], np.zeros(self.embedding_dim), np.array([])

        try:
            canonicalizer = self.base_system.canonicalizer
            vector_store = self.base_system.vector_store

            query_emb = canonicalizer.embedding_model.encode(
                query, normalize_embeddings=True
            )

            products = vector_store.search(query_emb, k=k)

            # Calcular cosine similarities
            scores = []
            for p in products:
                if hasattr(p, "content_embedding") and p.content_embedding is not None:
                    p_norm = p.content_embedding / (np.linalg.norm(p.content_embedding) + 1e-8)
                    q_norm = query_emb / (np.linalg.norm(query_emb) + 1e-8)
                    scores.append(float(np.dot(q_norm, p_norm)))
                else:
                    scores.append(0.0)

            return products, query_emb, np.array(scores)

        except Exception as e:
            logger.error(f"Error en retrieval: {e}")
            return [], np.zeros(self.embedding_dim), np.array([])

    # ─────────────────────────────────────────────────────────────────────
    # Paso 2 — Generar rankings alternativos con la Policy
    # ─────────────────────────────────────────────────────────────────────

    def generate_two_rankings(
        self,
        query: str,
        products: list,
        query_emb_np: np.ndarray,
    ) -> Tuple[List[Dict], List[Dict], torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Genera dos rankings alternativos usando la política:
            - Ranking A: temperatura baja (más determinista)
            - Ranking B: temperatura alta + ruido (más exploratorio)

        Returns:
            display_a, display_b:            listas de dicts para UI
            ranking_a, ranking_b:            tensores de índices
            log_prob_a, log_prob_b:          log-probabilidades bajo policy
        """
        # Convertir a tensores
        query_emb_t = self.tensorizer.query_to_tensor(query_emb_np).to(self.device)
        prod_embs, prod_feats = self.tensorizer.products_to_tensors(
            products, query, query_emb_np
        )
        prod_embs = prod_embs.to(self.device)
        prod_feats = prod_feats.to(self.device)

        # Ranking A: determinista (temperatura = 0.5)
        with torch.no_grad():
            rank_a, lp_a = self.ppo_trainer.generate_ranking(
                query_emb_t, prod_embs, prod_feats,
                temperature=0.5, noise_scale=0.0,
            )
            # Ranking B: exploratorio (temperatura = 1.5 + ruido)
            rank_b, lp_b = self.ppo_trainer.generate_ranking(
                query_emb_t, prod_embs, prod_feats,
                temperature=1.5, noise_scale=0.3,
            )

        # Asegurar que A y B sean diferentes
        if (rank_a == rank_b).all():
            # Si son iguales, forzar más ruido en B
            with torch.no_grad():
                rank_b, lp_b = self.ppo_trainer.generate_ranking(
                    query_emb_t, prod_embs, prod_feats,
                    temperature=2.0, noise_scale=0.8,
                )

        display_a = self.tensorizer.ranking_to_display(products, rank_a.cpu())
        display_b = self.tensorizer.ranking_to_display(products, rank_b.cpu())

        return display_a, display_b, rank_a, rank_b, lp_a, lp_b

    # ─────────────────────────────────────────────────────────────────────
    # Paso 3 — Sesión de recolección de preferencias
    # ─────────────────────────────────────────────────────────────────────

    def run_preference_collection_session(self, queries: List[str], k: int = 10):
        """
        Sesión completa A-vs-B con una lista de queries.
        Para cada query genera dos rankings y pide preferencia al usuario.
        """
        if self.policy is None:
            self.initialize()

        comparisons = []

        print(f"\nPreparando {len(queries)} comparaciones...")

        for i, query in enumerate(queries, 1):
            print(f"  [{i}/{len(queries)}] Generando rankings para: '{query}'")

            products, query_emb_np, _ = self.retrieve_candidates(query, k=k)
            if not products:
                logger.warning(f"Sin resultados para: {query}")
                continue

            display_a, display_b, _, _, _, _ = self.generate_two_rankings(
                query, products, query_emb_np
            )

            comparisons.append((query, display_a, display_b))

        if not comparisons:
            print("No hay comparaciones para mostrar.")
            return {}

        stats = self.preference_collector.run_comparison_session(comparisons)

        print(f"\nSesión completada: {self.preference_collector.total_preferences} "
              f"preferencias totales")

        return stats

    # ─────────────────────────────────────────────────────────────────────
    # Paso 4 — Entrenar el Reward Model
    # ─────────────────────────────────────────────────────────────────────

    def train_reward_model(
        self,
        epochs: int = 30,
        lr: float = 1e-4,
        batch_size: int = 8,
    ) -> Dict:
        """
        Entrena el RewardModel con las preferencias recolectadas.
        Este es el componente que APRENDE qué rankings son mejores.
        """
        if self.preference_collector is None:
            self.initialize()

        prefs = self.preference_collector.load_preferences()

        # Solo entrenar con preferencias explícitas A o B (no empates)
        prefs_ab = [p for p in prefs if p.get("preference") in ("A", "B")]

        if len(prefs_ab) < 5:
            logger.error(
                f"Insuficientes preferencias: {len(prefs_ab)} "
                f"(mínimo 5, recomendado 20+)"
            )
            return {"error": "insufficient_data", "count": len(prefs_ab)}

        logger.info(f"Entrenando RewardModel con {len(prefs_ab)} preferencias...")

        # Construir dataset de entrenamiento
        all_products = getattr(self.base_system, "canonical_products", [])
        id_map = {getattr(p, "id", ""): p for p in all_products if hasattr(p, "id")}

        dataset = []
        skipped = 0

        for pref in prefs_ab:
            query = pref.get("query", "")
            preference = pref.get("preference")

            # Determinar cuál fue preferido y cuál rechazado
            if preference == "A":
                preferred_ids = pref.get("ranking_a_ids", [])
                rejected_ids  = pref.get("ranking_b_ids", [])
            else:  # "B"
                preferred_ids = pref.get("ranking_b_ids", [])
                rejected_ids  = pref.get("ranking_a_ids", [])

            # Obtener embeddings del query
            try:
                if self.base_system and hasattr(self.base_system, "canonicalizer"):
                    q_emb_np = self.base_system.canonicalizer.embedding_model.encode(
                        query, normalize_embeddings=True
                    )
                else:
                    q_emb_np = np.zeros(self.embedding_dim)
                q_emb_t = self.tensorizer.query_to_tensor(q_emb_np)
            except Exception as e:
                logger.warning(f"Error encodeando query '{query}': {e}")
                skipped += 1
                continue

            sample = self.tensorizer.preference_to_training_sample(
                query_emb_tensor=q_emb_t,
                all_products=all_products,
                preferred_ids=preferred_ids,
                rejected_ids=rejected_ids,
                query=query,
                query_embedding_np=q_emb_np,
            )

            if sample is None:
                skipped += 1
                continue

            dataset.append(sample)

        if not dataset:
            logger.error("No se pudo construir dataset de entrenamiento")
            return {"error": "empty_dataset", "skipped": skipped}

        logger.info(
            f"Dataset: {len(dataset)} muestras "
            f"({skipped} omitidas por IDs no encontrados)"
        )

        # Entrenar
        result = self.reward_model.train_on_preferences(
            preference_dataset=dataset,
            epochs=epochs,
            lr=lr,
            batch_size=batch_size,
            device=self.device,
        )

        self.reward_model_trained = True

        # Guardar checkpoint
        self.reward_model.save(str(self.cache_dir / "reward_model.pt"))

        logger.info(
            f"RewardModel entrenado — "
            f"accuracy final: {result.get('best_accuracy', 0):.4f}"
        )

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Paso 5+6 — Ciclo PPO (collect + update)
    # ─────────────────────────────────────────────────────────────────────

    def run_ppo_training_cycle(
        self,
        queries: List[str],
        k: int = 15,
        rankings_per_query: int = 3,
        ppo_updates_per_cycle: int = 1,
    ) -> Dict:
        """
        Ciclo PPO completo:
          1. Para cada query, genera N rankings con la política actual
          2. Los puntúa con el RewardModel
          3. Ejecuta update PPO para mejorar la política

        Retorna métricas del ciclo.
        """
        if not self.reward_model_trained:
            logger.error("El RewardModel debe entrenarse antes de PPO")
            return {"error": "reward_model_not_trained"}

        if self.ppo_trainer is None:
            self.initialize()

        logger.info(
            f"Iniciando ciclo PPO #{self.ppo_cycles_completed + 1} — "
            f"{len(queries)} queries, {rankings_per_query} rankings/query"
        )

        cycle_rewards = []

        for query in queries:
            products, query_emb_np, _ = self.retrieve_candidates(query, k=k)
            if not products:
                continue

            n = min(len(products), self.max_products)
            products = products[:n]

            query_emb_t = self.tensorizer.query_to_tensor(query_emb_np)
            prod_embs, prod_feats = self.tensorizer.products_to_tensors(
                products, query, query_emb_np
            )

            # Generar varios rankings con temperatura variable para explorar
            temperatures = [0.8, 1.2, 1.5][:rankings_per_query]

            for temp in temperatures:
                self.ppo_trainer.collect_and_store(
                    query_embedding=query_emb_t,
                    product_embeddings=prod_embs,
                    product_features=prod_feats,
                    query_text=query,
                    temperature=temp,
                    noise_scale=0.1 * (temp - 0.5),
                )

        # Ejecutar update PPO si hay suficiente data en el buffer
        all_metrics = []
        for _ in range(ppo_updates_per_cycle):
            metrics = self.ppo_trainer.update()
            if metrics:
                all_metrics.append(metrics)
                cycle_rewards.append(metrics["mean_reward"])

        if not all_metrics:
            logger.warning("PPO update no ejecutado — buffer insuficiente")
            return {"warning": "insufficient_buffer"}

        self.ppo_cycles_completed += 1
        self.policy_trained = True

        # Guardar estado
        self.ppo_trainer.save(str(self.cache_dir / "ppo_trainer.pt"))
        self._save_pipeline_state()

        summary = {
            "cycle": self.ppo_cycles_completed,
            "queries_processed": len(queries),
            "mean_reward": float(np.mean(cycle_rewards)) if cycle_rewards else 0,
            "ppo_updates": len(all_metrics),
            "last_update": all_metrics[-1] if all_metrics else {},
        }

        logger.info(
            f"Ciclo PPO #{self.ppo_cycles_completed} completado — "
            f"reward={summary['mean_reward']:.4f}"
        )

        return summary

    # ─────────────────────────────────────────────────────────────────────
    # Inferencia — usar la política entrenada
    # ─────────────────────────────────────────────────────────────────────

    def rank_products(
        self,
        query: str,
        products: list,
        query_emb_np: Optional[np.ndarray] = None,
    ) -> list:
        """
        Usa la política entrenada para ordenar una lista de productos.
        Si la política no está entrenada, devuelve el orden original.
        """
        if not self.policy_trained or self.policy is None:
            logger.debug("Política no entrenada, devolviendo orden original")
            return products

        if query_emb_np is None and self.base_system:
            try:
                query_emb_np = self.base_system.canonicalizer.embedding_model.encode(
                    query, normalize_embeddings=True
                )
            except Exception:
                return products

        n = min(len(products), self.max_products)
        products_slice = products[:n]

        try:
            query_emb_t = self.tensorizer.query_to_tensor(query_emb_np).to(self.device)
            prod_embs, prod_feats = self.tensorizer.products_to_tensors(
                products_slice, query, query_emb_np
            )
            prod_embs = prod_embs.to(self.device)
            prod_feats = prod_feats.to(self.device)

            self.policy.eval()
            with torch.no_grad():
                ranking, _ = self.ppo_trainer.generate_ranking(
                    query_emb_t, prod_embs, prod_feats,
                    temperature=0.5, noise_scale=0.0,
                )

            ranked_products = [products_slice[i] for i in ranking.cpu().tolist()]
            # Añadir resto si había más de max_products
            ranked_products.extend(products[n:])
            return ranked_products

        except Exception as e:
            logger.error(f"Error en rank_products: {e}")
            return products

    # ─────────────────────────────────────────────────────────────────────
    # Sesión interactiva completa (flujo recomendado)
    # ─────────────────────────────────────────────────────────────────────

    def run_full_rlhf_session(
        self,
        queries: List[str],
        n_cycles: int = 3,
        preferences_per_cycle: int = 10,
    ):
        """
        Ejecuta el flujo RLHF completo en modo interactivo:
          Para cada ciclo:
            1. Recolectar preferencias A-vs-B
            2. Entrenar Reward Model
            3. Ciclo PPO
            4. Mostrar mejora

        Este es el "loop de entrenamiento" que convierte feedback humano
        en mejoras reales de la política.
        """
        if self.policy is None:
            self.initialize()

        print("\n" + "═" * 80)
        print("  CICLO RLHF COMPLETO")
        print(f"  {n_cycles} ciclos × ~{preferences_per_cycle} preferencias/ciclo")
        print("═" * 80)

        for cycle in range(1, n_cycles + 1):
            print(f"\n{'─'*80}")
            print(f"  CICLO {cycle}/{n_cycles}")
            print(f"{'─'*80}")

            # Tomar queries para este ciclo
            import random
            cycle_queries = random.sample(queries, min(preferences_per_cycle, len(queries)))

            # Paso 1: Preferencias
            print(f"\n[1/3] Recolectando preferencias ({len(cycle_queries)} comparaciones)...")
            self.run_preference_collection_session(cycle_queries)

            # Paso 2: Reward Model
            print(f"\n[2/3] Entrenando Reward Model...")
            result = self.train_reward_model(epochs=20)
            if "error" in result:
                print(f"  ⚠ No se pudo entrenar Reward Model: {result['error']}")
                continue
            print(f"  ✓ Accuracy: {result.get('best_accuracy', 0):.4f}")

            # Paso 3: PPO
            print(f"\n[3/3] Optimizando Policy con PPO...")
            ppo_result = self.run_ppo_training_cycle(cycle_queries)
            if "error" not in ppo_result:
                print(f"  ✓ Reward medio: {ppo_result.get('mean_reward', 0):.4f}")

        print("\n" + "═" * 80)
        print(f"  RLHF completado — {self.ppo_cycles_completed} ciclos")
        print(f"  Preferencias totales: {self.preference_collector.total_preferences}")
        print("═" * 80)

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────

    def _save_pipeline_state(self):
        state = {
            "ppo_cycles_completed": self.ppo_cycles_completed,
            "reward_model_trained": self.reward_model_trained,
            "policy_trained": self.policy_trained,
        }
        with open(self.cache_dir / "pipeline_state.json", "w") as f:
            json.dump(state, f, indent=2)

    def get_stats(self) -> Dict:
        stats = {
            "reward_model_trained": self.reward_model_trained,
            "policy_trained": self.policy_trained,
            "ppo_cycles_completed": self.ppo_cycles_completed,
        }

        if self.preference_collector:
            stats["preferences"] = self.preference_collector.get_stats()

        if self.ppo_trainer:
            stats["ppo"] = self.ppo_trainer.get_stats()

        if self.policy:
            stats["policy_parameters"] = self.policy.count_parameters()

        return stats