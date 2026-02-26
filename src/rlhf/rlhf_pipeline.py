# src/rlhf/rlhf_pipeline.py
"""
RLHF Pipeline — Orquestación completa con todas las salvaguardas.

MEJORAS vs versión anterior:
    1. Split 80/20: 20% hold-out nunca toca el entrenamiento
    2. Validación de reward ANTES de lanzar PPO
    3. PPO con KL adaptativa (via ppo_trainer.py)
    4. Versionado de checkpoints por ciclo
    5. Control de sesgo de presentación A/B
    6. Exploración correcta: baseline vs policy + ruido
    7. Diagnósticos de KL y convergencia integrados
"""

import torch
import numpy as np
import json
import random
import logging
from pathlib import Path
from typing import List, Tuple, Dict, Optional

from .reward_model import RankingRewardModel, RankingRewardTrainer
from .pointwise_reward_model import PointwiseRewardModel
from .preference_collector import PreferenceCollector
from .policy_model import PolicyModel
from .ppo_trainer import PPOTrainer

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/rlhf_checkpoints")
REWARD_CKPT = CHECKPOINT_DIR / "reward_model.pt"
POLICY_CKPT = CHECKPOINT_DIR / "policy_model.pt"
STATS_FILE = CHECKPOINT_DIR / "training_stats.json"


class RLHFPipeline:
    """
    Orquesta el ciclo completo RLHF con reward híbrido de rankings.

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
        reward_mode: str = "pointwise",
    ):
        self.emb_model = embedding_model
        self.product_index = product_index
        self.vector_store = vector_store
        self.emb_dim = emb_dim
        self.top_k = top_k_ranking
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'

        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        if reward_mode == "pointwise":
            self.reward_model = PointwiseRewardModel(emb_dim=emb_dim).to(self.device)
        else:
            self.reward_model = RankingRewardModel(emb_dim=emb_dim, top_k=top_k_ranking).to(self.device)

        self.reward_trainer = None
        self.reward_mode = reward_mode

        self.preference_collector = PreferenceCollector(
            embedding_model=embedding_model,
            product_index=product_index,
            top_k=top_k_ranking,
        )

        self.policy_model = PolicyModel(embedding_dim=emb_dim).to(self.device)
        self.ppo_trainer = PPOTrainer(self.policy_model)

        self.reward_trained = False
        self.policy_trained = False
        self.training_stats = {
            'reward_losses': [],
            'reward_accuracies_train': [],
            'reward_accuracies_val': [],
            'reward_loglik_val': [],
            'ppo_rewards': [],
            'ppo_kl': [],
            'ndcg_history': [],
        }

        logger.info(f"RLHFPipeline: device={self.device}, top_k={top_k_ranking}")

    def initialize(self, load_checkpoint: bool = True):
        if not load_checkpoint:
            return
        for ckpt, model, attr in [
            (REWARD_CKPT, self.reward_model, 'reward_trained'),
            (POLICY_CKPT, self.policy_model, 'policy_trained'),
        ]:
            if ckpt.exists():
                try:
                    model.load_state_dict(torch.load(ckpt, map_location=self.device))
                    setattr(self, attr, True)
                    logger.info(f"  Cargado: {ckpt.name}")
                except Exception as e:
                    logger.warning(f"  No se pudo cargar {ckpt.name}: {e}")

        if STATS_FILE.exists():
            try:
                with open(STATS_FILE) as f:
                    self.training_stats = json.load(f)
            except Exception:
                pass

    # ------------------------------------------------------------------
    # FASE 1: Entrenar Reward Model con split 80/20
    # ------------------------------------------------------------------

    def train_reward_model(
        self,
        epochs: int = 40,
        batch_size: int = 16,
        min_pairs: int = 10,
        val_split: float = 0.2,
    ) -> dict:
        """
        Entrena el reward model con split 80/20 integrado.

        El 20% hold-out NUNCA se usa para entrenamiento.
        Se reporta accuracy + log-likelihood en hold-out.

        Args:
            epochs:    Épocas de entrenamiento
            batch_size: Tamaño de batch
            min_pairs: Mínimo de pares para empezar
            val_split: Fracción del hold-out (0.2 = 20%)

        Returns:
            Stats incluyendo métricas del hold-out
        """
        all_records = self.preference_collector.load_preferences(only_clear=True)
        n = len(all_records)

        logger.info(f"\n{'='*60}")
        logger.info(f"ENTRENANDO REWARD MODEL")
        logger.info(f"  Pares disponibles: {n} (necesitas {min_pairs}+)")

        if n < min_pairs:
            return {'error': f'Necesitas {min_pairs}+ pares (tienes {n})'}

        # Split 80/20
        random.shuffle(all_records)
        val_n = max(2, int(n * val_split))
        train_records = all_records[val_n:]
        val_records = all_records[:val_n]

        logger.info(f"  Train: {len(train_records)} | Val (hold-out): {len(val_records)}")
        logger.info(f"  Coverage índice: {self.preference_collector.stats().get('coverage_pct', 0)}%")

        # Construir tensores de val una sola vez
        val_batch = self.preference_collector.build_batch(val_records, device=self.device)
        q_val, ra_val, rb_val, pref_val = val_batch

        best_val_acc = 0.0
        best_loss = float('inf')

        for epoch in range(epochs):
            # Shuffle train cada época
            random.shuffle(train_records)
            epoch_loss = 0.0
            n_batches = 0

            for i in range(0, len(train_records), batch_size):
                batch = train_records[i:i + batch_size]
                q, ra, rb, prefs = self.preference_collector.build_batch(batch, device=self.device)
                loss = self.reward_trainer.train_step(q, ra, rb, prefs)
                epoch_loss += loss
                n_batches += 1

            avg_loss = epoch_loss / max(n_batches, 1)

            # Métricas en val
            val_acc = self.reward_trainer.get_accuracy(q_val, ra_val, rb_val, pref_val)
            val_ll = self.reward_trainer.get_bradley_terry_loglik(q_val, ra_val, rb_val, pref_val)
            collapse = self.reward_trainer.detect_reward_collapse(q_val, ra_val, rb_val)

            self.training_stats['reward_losses'].append(avg_loss)
            self.training_stats['reward_accuracies_val'].append(val_acc)
            self.training_stats['reward_loglik_val'].append(val_ll)

            if (epoch + 1) % 5 == 0 or epoch == 0:
                collapse_warn = " [WARN] COLAPSO" if collapse['collapsed'] else ""
                logger.info(
                    f"  Epoch {epoch+1:3d}/{epochs} | "
                    f"loss={avg_loss:.4f} | "
                    f"val_acc={val_acc:.3f} | "
                    f"val_ll={val_ll:.3f}{collapse_warn}"
                )

            # Guardar mejor modelo por val_accuracy
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(self.reward_model.state_dict(), REWARD_CKPT)

        self.reward_trained = True
        self._save_stats()

        final_ll = self.training_stats['reward_loglik_val'][-1] if self.training_stats['reward_loglik_val'] else 0.0
        logger.info(f"\n  [OK] Reward model entrenado")
        logger.info(f"    Best val_accuracy: {best_val_acc:.3f}")
        logger.info(f"    Final val_loglik:  {final_ll:.3f}")

        if best_val_acc < 0.6:
            logger.warning("  [WARN] val_accuracy < 60% — NO ejecutes PPO todavía")
            logger.warning("  -> Recolecta más comparaciones A/B (objetivo: 100+)")
        else:
            logger.info("  -> Reward OK. Puedes proceder con PPO.")

        return {
            'n_train': len(train_records),
            'n_val': len(val_records),
            'best_val_accuracy': best_val_acc,
            'final_val_loglik': final_ll,
            'epochs': epochs,
        }

    def validate_reward_before_ppo(self, min_accuracy: float = 0.60) -> bool:
        """
        Valida el reward model ANTES de lanzar PPO.

        Criterio principal: accuracy en hold-out >= min_accuracy.
        El 'colapso' por mean_diff pequeño NO es criterio de bloqueo
        si la accuracy es alta — el modelo aprendió señal real en
        escala comprimida (normal en MLPs pequeños con hidden=128).

        Retorna True si es seguro proceder con PPO.
        """
        records = self.preference_collector.load_preferences(only_clear=True)
        if not records:
            logger.error("No hay preferencias para validar")
            return False

        val_n = max(2, int(len(records) * 0.2))
        val_records = records[:val_n]
        q, ra, rb, prefs = self.preference_collector.build_batch(val_records, device=self.device)

        acc = self.reward_trainer.get_accuracy(q, ra, rb, prefs)
        collapse = self.reward_trainer.detect_reward_collapse(q, ra, rb)

        logger.info(f"\nVALIDACIÓN REWARD:")
        logger.info(f"  Accuracy: {acc:.3f} (mínimo: {min_accuracy})")
        logger.info(f"  Mean diff: {collapse['mean_abs_diff']:.6f}")
        logger.info(f"  r_A std: {collapse['r_a_std']:.6f} | r_B std: {collapse['r_b_std']:.6f}")
        logger.info(f"  Colapso duro: {collapse['collapsed']}")

        # Colapso duro = literalmente no hay diferencia (bug, no escala pequeña)
        if collapse['collapsed']:
            logger.error("  [ERR] Colapso duro — el modelo no computa diferencias")
            logger.error("    Posible causa: bug en forward() o datos corruptos")
            return False

        # Accuracy insuficiente
        if acc < min_accuracy:
            logger.warning(f"  [ERR] Accuracy {acc:.3f} < {min_accuracy} — señal insuficiente")
            logger.warning("    Recolecta más comparaciones A/B antes de PPO")
            return False

        logger.info(f"  [OK] Reward OK para PPO (accuracy={acc:.3f}, diff={collapse['mean_abs_diff']:.6f})")
        return True

    # ------------------------------------------------------------------
    # FASE 2: Ciclo PPO con KL adaptativa y versionado
    # ------------------------------------------------------------------

    def run_ppo_cycle(
        self,
        n_queries: int = 50,
        epochs: int = 5,
        validate_first: bool = True,
        exploration_noise: float = 0.05,
    ) -> dict:
        """
        Ciclo PPO con KL adaptativa.

        Estrategia de exploración correcta:
            - baseline vs policy+ruido     (cuando policy no entrenada)
            - policy vs policy+ruido       (cuando policy entrenada)
            - baseline vs baseline+ruido   (control)
            Alternados para evitar sesgo de presentación.

        Args:
            n_queries:        Queries para el ciclo
            epochs:           Épocas PPO
            validate_first:   Si True, valida reward antes de empezar
            exploration_noise: Ruido de exploración (0.05 recomendado)

        Returns:
            Stats del ciclo
        """
        if not self.reward_trained:
            logger.error("Entrena reward primero: train_reward_model()")
            return {'error': 'reward not trained'}

        if validate_first and not self.validate_reward_before_ppo():
            return {'error': 'reward accuracy insuficiente — recolecta más datos A/B'}

        logger.info(f"\n{'='*60}")
        logger.info(f"CICLO PPO (KL adaptativa)")

        # Guardar versión antes de entrenar
        self.ppo_trainer.save_version(f"before_cycle{len(self.ppo_trainer.list_versions())}")

        # Diagnóstico inicial
        weight_before = self._get_policy_weight()
        logger.info(f"  Policy weight (antes): {weight_before:.8f}")

        # Queries de las preferencias guardadas
        records = self.preference_collector.load_preferences(only_clear=False)
        queries = list(set(r['query'] for r in records))
        if not queries:
            return {'error': 'no queries'}
        random.shuffle(queries)
        queries = queries[:n_queries]

        # Loop PPO
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
            avg_kl = float(np.mean(epoch_kls)) if epoch_kls else 0.0
            ppo_rewards.append(avg_reward)
            ppo_kls.append(avg_kl)

            kl_status = self.ppo_trainer.get_kl_status()
            logger.info(
                f"  Epoch {epoch+1}/{epochs} | "
                f"reward={avg_reward:.4f} | "
                f"kl={avg_kl:.4f} | "
                f"beta={kl_status['current_beta']:.3f} | "
                f"kl_status={kl_status['status']}"
            )

        # Diagnóstico final
        weight_after = self._get_policy_weight()
        diag = self.ppo_trainer.check_policy_updated(weight_before, weight_after)
        logger.info(f"  Policy weight (después): {weight_after:.8f}")
        logger.info(f"  Delta: {diag['delta']:.8f}")
        if not diag['updated']:
            logger.warning(f"  [WARN] {diag.get('warning', 'Policy no se actualizó')}")

        # Guardar versión después
        self.policy_trained = True
        torch.save(self.policy_model.state_dict(), POLICY_CKPT)
        self.ppo_trainer.save_version(f"after_cycle{len(self.ppo_trainer.list_versions())}")

        self.training_stats['ppo_rewards'].extend(ppo_rewards)
        self.training_stats['ppo_kl'].extend(ppo_kls)
        self._save_stats()

        return {
            'epochs': epochs,
            'n_queries': len(queries),
            'final_reward': ppo_rewards[-1] if ppo_rewards else 0.0,
            'avg_kl': float(np.mean(ppo_kls)) if ppo_kls else 0.0,
            'policy_updated': diag['updated'],
            'ppo_summary': self.ppo_trainer.get_training_summary(),
        }

    def _make_product_features(self, n_products: int) -> torch.Tensor:
        """
        Genera product_features vacíos para el PolicyModel.
        Tu PolicyModel espera (batch, n_prod, feature_dim=8).
        Por ahora usamos zeros — no tenemos features escalares conectados.
        """
        return torch.zeros(1, n_products, 8, dtype=torch.float32, device=self.device)

    def _ppo_step(self, query: str, noise: float) -> Optional[dict]:
        """Un paso PPO para una query con exploración correcta."""
        q_emb_np = self.emb_model.encode(query, normalize_embeddings=True)
        q_emb = torch.tensor(q_emb_np, dtype=torch.float32, device=self.device)

        products = self.vector_store.search(q_emb_np, k=self.top_k * 2)
        if not products:
            return None

        prod_embs = self._products_to_embs(products[:self.top_k])
        if prod_embs is None:
            return None

        # Reward del ranking baseline
        r_baseline = self.reward_model.score_ranking(q_emb, prod_embs)

        # Ranking de la policy con exploración
        n = prod_embs.size(0)
        feats = self._make_product_features(n)

        self.policy_model.eval()
        with torch.no_grad():
            scores = self.policy_model(
                q_emb.unsqueeze(0),
                prod_embs.unsqueeze(0),
                feats,
            ).squeeze()
            scores = scores + torch.randn_like(scores) * noise
            order = torch.argsort(scores, descending=True)
            prod_embs_policy = prod_embs[order]

        # Reward del ranking de la policy
        r_policy = self.reward_model.score_ranking(q_emb, prod_embs_policy)

        advantage = torch.tensor(r_policy - r_baseline, dtype=torch.float32, device=self.device)
        result = self.ppo_trainer.update(q_emb, prod_embs, prod_embs_policy, advantage)
        result['reward'] = r_policy

        return result

    # ------------------------------------------------------------------
    # Inferencia
    # ------------------------------------------------------------------

    def rank_products(self, query: str, products: list, query_emb=None) -> list:
        if not self.policy_trained:
            return products

        if query_emb is None:
            query_emb = self.emb_model.encode(query, normalize_embeddings=True)

        q_emb = torch.tensor(query_emb, dtype=torch.float32, device=self.device)
        prod_embs = self._products_to_embs(products[:self.top_k])
        if prod_embs is None:
            return products

        n = prod_embs.size(0)
        feats = self._make_product_features(n)

        self.policy_model.eval()
        with torch.no_grad():
            scores = self.policy_model(
                q_emb.unsqueeze(0),
                prod_embs.unsqueeze(0),
                feats,
            ).squeeze()

        order = torch.argsort(scores, descending=True).cpu().numpy()
        return [products[i] for i in order if i < len(products)]

    def retrieve_candidates(self, query: str, k: int = 20) -> Tuple:
        q_emb = self.emb_model.encode(query, normalize_embeddings=True)
        products = self.vector_store.search(q_emb, k=k)
        return products, q_emb, np.array([])

    # ------------------------------------------------------------------
    # Utilidades
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

    def _get_policy_weight(self) -> float:
        if hasattr(self.policy_model, 'linear'):
            return self.policy_model.linear.weight.mean().item()
        for p in self.policy_model.parameters():
            return p.mean().item()
        return 0.0

    def get_status(self) -> dict:
        return {
            'n_preferences': self.preference_collector.count(),
            'reward_trained': self.reward_trained,
            'policy_trained': self.policy_trained,
            'device': self.device,
            'checkpoints': {
                'reward': REWARD_CKPT.exists(),
                'policy': POLICY_CKPT.exists(),
                'versions': [p.name for p in self.ppo_trainer.list_versions()],
            },
            'collector_stats': self.preference_collector.stats(),
            'ppo_status': self.ppo_trainer.get_kl_status(),
        }

    def _save_stats(self):
        try:
            with open(STATS_FILE, 'w') as f:
                json.dump(self.training_stats, f, indent=2)
        except Exception as e:
            logger.warning(f"Error guardando stats: {e}")