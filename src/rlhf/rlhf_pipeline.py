"""
rlhf_pipeline.py
Orquestación del ciclo RLHF con PointwiseRewardModel + PPO.

Responsabilidades de este módulo:
    - Mantener estado del reward model y policy model
    - Ejecutar ciclos PPO (_ppo_step)
    - Inferencia final (rank_products)

Lo que ya NO hace este módulo (movido a scripts externos):
    - Entrenar el reward model  →  train_pointwise_reward.py
    - Validar el reward         →  evaluate_methods.py
"""

import torch
import numpy as np
import json
import random
import logging
from pathlib import Path
from typing import List, Dict, Optional

from .pointwise_reward_model import PointwiseRewardModel
from .preference_collector import PreferenceCollector
from .policy_model import PolicyModel
from .ppo_trainer import PPOTrainer

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/rlhf_checkpoints")
POLICY_CKPT    = CHECKPOINT_DIR / "policy_model.pt"
REWARD_CKPT    = CHECKPOINT_DIR / "reward_model.pt"
STATS_FILE     = CHECKPOINT_DIR / "training_stats.json"


class RLHFPipeline:
    """
    Orquesta el ciclo RLHF con PointwiseRewardModel + PPO.

    Args:
        embedding_model:  sentence-transformer (ya cargado)
        product_index:    Dict[product_id -> np.ndarray(emb_dim)]
        vector_store:     FAISS store para retrieval baseline
        emb_dim:          Dimensión embeddings (default 384)
        top_k_ranking:    Productos por ranking (default 10)
    """

    def __init__(
        self,
        embedding_model,
        product_index: Dict[str, np.ndarray],
        vector_store,
        emb_dim: int = 384,
        top_k_ranking: int = 10,
    ):
        self.emb_model     = embedding_model
        self.product_index = product_index
        self.vector_store  = vector_store
        self.emb_dim       = emb_dim
        self.top_k         = top_k_ranking
        self.device        = 'cuda' if torch.cuda.is_available() else 'cpu'

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        # Reward model (pointwise — siempre)
        self.reward_model = PointwiseRewardModel(emb_dim=emb_dim).to(self.device)
        self.reward_trained = False

        # Preference collector (para recolección A/B, no para entrenar aquí)
        self.preference_collector = PreferenceCollector(
            embedding_model=embedding_model,
            product_index=product_index,
            top_k=top_k_ranking,
        )

        # Policy model + PPO
        self.policy_model = PolicyModel(embedding_dim=emb_dim).to(self.device)
        self.ppo_trainer  = PPOTrainer(self.policy_model)
        self.policy_trained = False

        self.training_stats = {
            'ppo_rewards': [],
            'ppo_kl':      [],
        }

        logger.info(f"RLHFPipeline: device={self.device}, top_k={top_k_ranking}")

    # ------------------------------------------------------------------
    # Inicialización / carga de checkpoints
    # ------------------------------------------------------------------

    def initialize(self, load_checkpoint: bool = True):
        """Carga checkpoints de reward y policy si existen."""
        if not load_checkpoint:
            return

        if REWARD_CKPT.exists():
            try:
                ckpt  = torch.load(REWARD_CKPT, map_location=self.device)
                # Soporta formato con 'model_state' y state_dict directo
                state = ckpt.get('model_state', ckpt)
                self.reward_model.load_state_dict(state)
                self.reward_model.eval()
                self.reward_trained = True
                logger.info(f"  Reward model cargado: {REWARD_CKPT.name}")
            except Exception as e:
                logger.warning(f"  No se pudo cargar reward: {e}")

        if POLICY_CKPT.exists():
            try:
                self.policy_model.load_state_dict(
                    torch.load(POLICY_CKPT, map_location=self.device)
                )
                self.policy_trained = True
                logger.info(f"  Policy model cargado: {POLICY_CKPT.name}")
            except Exception as e:
                logger.warning(f"  No se pudo cargar policy: {e}")

        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    self.training_stats = json.load(f)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # Utilidad: score DCG de un ranking completo con modelo pointwise
    # ------------------------------------------------------------------

    def _dcg_reward(self, q_emb: torch.Tensor, prod_embs: torch.Tensor) -> float:
        """
        Score de un ranking completo usando el PointwiseRewardModel.

        Cada producto recibe un score pointwise; se pondera con pesos DCG
        según su posición en el ranking. Así el reward es sensible al orden.

        Args:
            q_emb:      [emb_dim]          — embedding de la query
            prod_embs:  [n_prod, emb_dim]  — embeddings en orden del ranking
        Returns:
            float — score escalar del ranking completo
        """
        n = prod_embs.size(0)
        dcg_w = torch.tensor(
            [1.0 / np.log2(k + 2) for k in range(n)],
            dtype=torch.float32,
            device=self.device,
        )
        dcg_w = dcg_w / dcg_w.sum()

        q_exp = q_emb.unsqueeze(0).expand(n, -1)  # [n, emb_dim]

        self.reward_model.eval()
        with torch.no_grad():
            scores = self.reward_model(q_exp, prod_embs)  # [n]

        return (scores * dcg_w).sum().item()

    # ------------------------------------------------------------------
    # Ciclo PPO
    # ------------------------------------------------------------------

    def run_ppo_cycle(
        self,
        n_queries: int = 50,
        epochs: int = 5,
        exploration_noise: float = 0.05,
    ) -> dict:
        """
        Ciclo PPO con KL adaptativa.

        Requiere que el reward model esté entrenado previamente con
        train_pointwise_reward.py.

        Args:
            n_queries:         Queries para el ciclo
            epochs:            Épocas PPO
            exploration_noise: Ruido gaussiano de exploración
        Returns:
            dict con stats del ciclo
        """
        if not self.reward_trained:
            logger.error(
                "Entrena el reward primero: python train_pointwise_reward.py"
            )
            return {'error': 'reward not trained'}

        logger.info(f"\n{'='*60}")
        logger.info(f"CICLO PPO (KL adaptativa)")

        self.ppo_trainer.save_version(
            f"before_cycle{len(self.ppo_trainer.list_versions())}"
        )
        weight_before = self._get_policy_weight()

        # Queries desde preferencias recolectadas
        records = self.preference_collector.load_preferences(only_clear=False)
        queries = list({r['query'] for r in records})
        if not queries:
            return {'error': 'no queries disponibles'}

        random.shuffle(queries)
        queries = queries[:n_queries]

        ppo_rewards, ppo_kls = [], []

        for epoch in range(epochs):
            epoch_rewards, epoch_kls = [], []

            for query in queries:
                try:
                    result = self._ppo_step(query, exploration_noise)
                    if result:
                        epoch_rewards.append(result['reward'])
                        epoch_kls.append(result['kl'])
                except Exception as e:
                    logger.debug(f"PPO step error ({query}): {e}")

            avg_reward = float(np.mean(epoch_rewards)) if epoch_rewards else 0.0
            avg_kl     = float(np.mean(epoch_kls))     if epoch_kls     else 0.0
            ppo_rewards.append(avg_reward)
            ppo_kls.append(avg_kl)

            kl_status = self.ppo_trainer.get_kl_status()
            logger.info(
                f"  Epoch {epoch+1}/{epochs} | "
                f"reward={avg_reward:.4f} | kl={avg_kl:.4f} | "
                f"beta={kl_status['current_beta']:.3f} | "
                f"kl_status={kl_status['status']}"
            )

        # Diagnóstico
        weight_after = self._get_policy_weight()
        diag = self.ppo_trainer.check_policy_updated(weight_before, weight_after)
        if not diag['updated']:
            logger.warning(f"  [WARN] {diag.get('warning', 'Policy no se actualizó')}")

        # Guardar
        self.policy_trained = True
        torch.save(self.policy_model.state_dict(), POLICY_CKPT)
        self.ppo_trainer.save_version(
            f"after_cycle{len(self.ppo_trainer.list_versions())}"
        )
        self.training_stats['ppo_rewards'].extend(ppo_rewards)
        self.training_stats['ppo_kl'].extend(ppo_kls)
        self._save_stats()

        return {
            'epochs':         epochs,
            'n_queries':      len(queries),
            'final_reward':   ppo_rewards[-1] if ppo_rewards else 0.0,
            'avg_kl':         float(np.mean(ppo_kls)) if ppo_kls else 0.0,
            'policy_updated': diag['updated'],
            'ppo_summary':    self.ppo_trainer.get_training_summary(),
        }

    def _ppo_step(self, query: str, noise: float) -> Optional[dict]:
        """Un paso PPO para una query."""
        q_emb_np = self.emb_model.encode(query, normalize_embeddings=True)
        q_emb    = torch.tensor(q_emb_np, dtype=torch.float32, device=self.device)

        products = self.vector_store.search(q_emb_np, k=self.top_k * 2)
        if not products:
            return None

        prod_embs = self._products_to_embs(products[:self.top_k])
        if prod_embs is None:
            return None

        # Score del ranking baseline (orden FAISS)
        r_baseline = self._dcg_reward(q_emb, prod_embs)

        # Ranking de la policy con exploración
        n    = prod_embs.size(0)
        feats = torch.zeros(1, n, 8, dtype=torch.float32, device=self.device)

        self.policy_model.eval()
        with torch.no_grad():
            scores = self.policy_model(
                q_emb.unsqueeze(0),
                prod_embs.unsqueeze(0),
                feats,
            ).squeeze()
            scores       = scores + torch.randn_like(scores) * noise
            order        = torch.argsort(scores, descending=True)
            prod_embs_po = prod_embs[order]

        # Score del ranking generado por la policy
        r_policy  = self._dcg_reward(q_emb, prod_embs_po)
        advantage = torch.tensor(
            r_policy - r_baseline, dtype=torch.float32, device=self.device
        )

        result           = self.ppo_trainer.update(q_emb, prod_embs, prod_embs_po, advantage)
        result['reward'] = r_policy
        return result

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def rank_products(
        self,
        query: str,
        products: list,
        query_emb=None,
    ) -> list:
        """Reordena productos usando la PolicyModel entrenada."""
        if not self.policy_trained:
            return products

        if query_emb is None:
            query_emb = self.emb_model.encode(query, normalize_embeddings=True)

        q_emb = torch.tensor(query_emb, dtype=torch.float32, device=self.device)

        prod_embs = self._products_to_embs(products[:self.top_k])
        if prod_embs is None:
            return products

        n     = prod_embs.size(0)
        feats = torch.zeros(1, n, 8, dtype=torch.float32, device=self.device)

        self.policy_model.eval()
        with torch.no_grad():
            scores = self.policy_model(
                q_emb.unsqueeze(0),
                prod_embs.unsqueeze(0),
                feats,
            ).squeeze()

        order = torch.argsort(scores, descending=True).cpu().numpy()
        return [products[i] for i in order if i < len(products)]

    def retrieve_candidates(self, query: str, k: int = 20):
        q_emb    = self.emb_model.encode(query, normalize_embeddings=True)
        products = self.vector_store.search(q_emb, k=k)
        return products, q_emb, np.array([])

    # ------------------------------------------------------------------
    # Utilidades internas
    # ------------------------------------------------------------------

    def _products_to_embs(self, products: list) -> Optional[torch.Tensor]:
        embs = []
        for p in products[:self.top_k]:
            pid = getattr(p, 'id', None) or getattr(p, 'product_id', None)
            emb = self.product_index.get(pid, np.zeros(self.emb_dim, dtype=np.float32))
            embs.append(emb)
        if not embs:
            return None
        return torch.tensor(np.stack(embs), dtype=torch.float32, device=self.device)

    def _make_product_features(self, n_products: int) -> torch.Tensor:
        return torch.zeros(1, n_products, 8, dtype=torch.float32, device=self.device)

    def _get_policy_weight(self) -> float:
        for p in self.policy_model.parameters():
            return p.mean().item()
        return 0.0

    def _save_stats(self):
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Error guardando stats: {e}")

    # ------------------------------------------------------------------
    # Estado del pipeline (para CLI y diagnósticos)
    # ------------------------------------------------------------------

    def get_status(self) -> dict:
        return {
            'n_preferences':    self.preference_collector.count(),
            'reward_trained':   self.reward_trained,
            'policy_trained':   self.policy_trained,
            'device':           self.device,
            'checkpoints': {
                'reward':   REWARD_CKPT.exists(),
                'policy':   POLICY_CKPT.exists(),
                'versions': [p.name for p in self.ppo_trainer.list_versions()],
            },
            'collector_stats':  self.preference_collector.stats(),
            'ppo_status':       self.ppo_trainer.get_kl_status(),
        }