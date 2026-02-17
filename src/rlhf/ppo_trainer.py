"""
PPO Trainer — Componente 5 del RLHF
Proximal Policy Optimization para el problema de ranking.

El ciclo de aprendizaje real:
    1. Policy genera ranking (acción)
    2. RewardModel evalúa el ranking (reward)
    3. PPO actualiza la policy para maximizar el reward
    4. KL divergence penaliza alejarse demasiado de la política de referencia

Adaptación de PPO para ranking:
    - "Estado":  query + candidatos (del retrieval FAISS)
    - "Acción":  ranking completo (permutación de los candidatos)
    - "Reward":  escalar del RewardModel
    - "Log-prob": bajo modelo Plackett-Luce

El clipping PPO evita actualizaciones demasiado grandes que destruirían
el conocimiento acumulado (estabilidad de entrenamiento).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List, Dict, Optional, Tuple
import numpy as np
import copy
import logging
from dataclasses import dataclass, field

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────────────────────────────────────
# Estructura de experiencia
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class RankingExperience:
    """Una experiencia completa: estado → acción (ranking) → reward."""
    query_embedding: torch.Tensor       # (emb_dim,)
    product_embeddings: torch.Tensor    # (n_prod, emb_dim)
    product_features: torch.Tensor      # (n_prod, feat_dim)
    ranking: torch.Tensor               # (n_prod,) — índices en orden
    log_prob: torch.Tensor              # escalar — log P(ranking | estado)
    reward: float                       # escalar del RewardModel
    query_text: str = ""               # para logging


# ─────────────────────────────────────────────────────────────────────────────
# Value Network (Critic / Baseline)
# ─────────────────────────────────────────────────────────────────────────────

class ValueNetwork(nn.Module):
    """
    Estima el valor esperado del estado (baseline para reducir varianza).
    Input:  representación comprimida del query + productos
    Output: escalar V(s)
    """

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, gain=0.5)
                nn.init.zeros_(m.bias)

    def forward(self, state_repr: torch.Tensor) -> torch.Tensor:
        return self.net(state_repr).squeeze(-1)


# ─────────────────────────────────────────────────────────────────────────────
# PPO Trainer
# ─────────────────────────────────────────────────────────────────────────────

class PPOTrainer:
    """
    PPO (Proximal Policy Optimization) para ranking de productos.

    Hiperparámetros clave:
        clip_epsilon:  clipping del ratio π_new/π_old (típico: 0.2)
        kl_target:     KL máxima permitida vs política de referencia
        entropy_coef:  coeficiente de entropía para exploración
        ppo_epochs:    veces que se pasa por el buffer por update
    """

    def __init__(
        self,
        policy_model,
        reward_model,
        learning_rate: float = 3e-5,
        clip_epsilon: float = 0.2,
        value_coef: float = 0.5,
        entropy_coef: float = 0.01,
        kl_coef: float = 0.1,
        kl_target: float = 0.01,
        max_grad_norm: float = 0.5,
        ppo_epochs: int = 4,
        min_buffer_size: int = 4,
        device: str = "cpu",
    ):
        self.policy = policy_model.to(device)
        self.reward_model = reward_model.to(device)
        self.device = device

        self.clip_epsilon = clip_epsilon
        self.value_coef = value_coef
        self.entropy_coef = entropy_coef
        self.kl_coef = kl_coef
        self.kl_target = kl_target
        self.max_grad_norm = max_grad_norm
        self.ppo_epochs = ppo_epochs
        self.min_buffer_size = min_buffer_size

        # Value network (critic / baseline)
        self.value_net = ValueNetwork(policy_model.hidden_dim).to(device)

        # Optimizadores
        self.policy_optimizer = torch.optim.Adam(
            list(policy_model.parameters()),
            lr=learning_rate,
            eps=1e-8,
        )
        self.value_optimizer = torch.optim.Adam(
            self.value_net.parameters(),
            lr=learning_rate * 3,  # Value network aprende más rápido
        )

        # Política de referencia (congelada) — penaliza desviarse demasiado
        self.reference_policy = copy.deepcopy(policy_model).to(device)
        for p in self.reference_policy.parameters():
            p.requires_grad_(False)

        # Buffer de experiencias
        self.buffer: List[RankingExperience] = []

        # Historial
        self.training_history: List[Dict] = []
        self.total_updates: int = 0

        logger.info(
            f"PPOTrainer inicializado — "
            f"lr={learning_rate}, ε={clip_epsilon}, "
            f"kl_target={kl_target}, device={device}"
        )

    # ─────────────────────────────────────────────────────────────────────
    # Generación de rankings (collect phase)
    # ─────────────────────────────────────────────────────────────────────

    @torch.no_grad()
    def generate_ranking(
        self,
        query_embedding: torch.Tensor,
        product_embeddings: torch.Tensor,
        product_features: torch.Tensor,
        temperature: float = 1.0,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Usa la política actual para generar un ranking.
        No actualiza gradientes (modo collect).
        """
        self.policy.eval()
        qe = query_embedding.unsqueeze(0).to(self.device)
        pe = product_embeddings.unsqueeze(0).to(self.device)
        pf = product_features.unsqueeze(0).to(self.device)

        ranking, log_prob = self.policy.get_ranking(
            qe, pe, pf, temperature=temperature, noise_scale=noise_scale
        )
        return ranking.squeeze(0), log_prob.squeeze(0)

    @torch.no_grad()
    def score_ranking(
        self,
        query_embedding: torch.Tensor,
        product_embeddings: torch.Tensor,
        product_features: torch.Tensor,
        ranking: torch.Tensor,
    ) -> float:
        """Usa el RewardModel para puntuar un ranking. Retorna reward escalar."""
        self.reward_model.eval()
        qe = query_embedding.unsqueeze(0).to(self.device)
        # Reordenar productos según el ranking
        pe_ordered = product_embeddings[ranking].unsqueeze(0).to(self.device)
        pf_ordered = product_features[ranking].unsqueeze(0).to(self.device)

        reward = self.reward_model(qe, pe_ordered, pf_ordered)
        return reward.item()

    # ─────────────────────────────────────────────────────────────────────
    # Buffer management
    # ─────────────────────────────────────────────────────────────────────

    def add_experience(self, exp: RankingExperience):
        self.buffer.append(exp)

    def collect_and_store(
        self,
        query_embedding: torch.Tensor,
        product_embeddings: torch.Tensor,
        product_features: torch.Tensor,
        query_text: str = "",
        temperature: float = 1.2,
        noise_scale: float = 0.1,
    ):
        """
        Genera un ranking, lo puntúa con el Reward Model, y lo guarda en el buffer.
        Este es el paso "collect" del ciclo RLHF.
        """
        ranking, log_prob = self.generate_ranking(
            query_embedding, product_embeddings, product_features,
            temperature=temperature, noise_scale=noise_scale
        )
        reward = self.score_ranking(
            query_embedding, product_embeddings, product_features, ranking
        )

        self.add_experience(RankingExperience(
            query_embedding=query_embedding.cpu(),
            product_embeddings=product_embeddings.cpu(),
            product_features=product_features.cpu(),
            ranking=ranking.cpu(),
            log_prob=log_prob.detach().cpu(),
            reward=reward,
            query_text=query_text,
        ))

        logger.debug(
            f"Experiencia guardada: reward={reward:.4f}, "
            f"query='{query_text[:30]}'"
        )

    # ─────────────────────────────────────────────────────────────────────
    # PPO Update
    # ─────────────────────────────────────────────────────────────────────

    def _get_state_repr(
        self,
        query_emb: torch.Tensor,
        prod_embs: torch.Tensor,
        prod_feats: torch.Tensor,
    ) -> torch.Tensor:
        """Extrae representación del estado para la Value Network."""
        # Usamos la proyección del query de la política
        with torch.no_grad():
            q_repr = self.policy.query_proj(query_emb)
        return q_repr

    def _compute_log_probs_new(
        self,
        queries: torch.Tensor,
        prod_embs: torch.Tensor,
        prod_feats: torch.Tensor,
        rankings: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula log-probabilidades NUEVAS bajo la política actual."""
        scores = self.policy(queries, prod_embs, prod_feats)
        log_probs = self.policy._plackett_luce_log_prob(scores, rankings)
        return log_probs

    @torch.no_grad()
    def _compute_log_probs_ref(
        self,
        queries: torch.Tensor,
        prod_embs: torch.Tensor,
        prod_feats: torch.Tensor,
        rankings: torch.Tensor,
    ) -> torch.Tensor:
        """Calcula log-probabilidades bajo la política de REFERENCIA (congelada)."""
        scores = self.reference_policy(queries, prod_embs, prod_feats)
        log_probs = self.reference_policy._plackett_luce_log_prob(scores, rankings)
        return log_probs

    def update(self) -> Optional[Dict]:
        """
        Ejecuta el update PPO sobre el buffer acumulado.

        Pasos:
          1. Normalizar advantages (reduce varianza)
          2. Para cada época PPO:
             a. Calcular ratio π_new/π_old
             b. Clipping PPO: min(ratio*A, clip(ratio)*A)
             c. Value loss (MSE entre V(s) y reward real)
             d. Entropía bonus (exploración)
             e. KL penalty vs política de referencia
          3. Early stop si KL > kl_target * 1.5
          4. Vaciar buffer
        """
        if len(self.buffer) < self.min_buffer_size:
            logger.warning(
                f"Buffer muy pequeño: {len(self.buffer)}/{self.min_buffer_size}. "
                f"Agrega más experiencias antes de update()."
            )
            return None

        self.policy.train()
        self.value_net.train()

        # Empaquetar buffer en tensores
        # (todos los batches tienen el mismo n_prod por padding en collect)
        n_prod = self.buffer[0].product_embeddings.shape[0]

        queries    = torch.stack([e.query_embedding for e in self.buffer]).to(self.device)
        prod_embs  = torch.stack([e.product_embeddings for e in self.buffer]).to(self.device)
        prod_feats = torch.stack([e.product_features for e in self.buffer]).to(self.device)
        rankings   = torch.stack([e.ranking for e in self.buffer]).to(self.device)
        old_lp     = torch.stack([e.log_prob for e in self.buffer]).to(self.device)
        rewards    = torch.tensor(
            [e.reward for e in self.buffer], dtype=torch.float32, device=self.device
        )

        # Normalizar rewards como advantages
        advantages = (rewards - rewards.mean()) / (rewards.std() + 1e-8)

        all_policy_losses, all_value_losses, all_kl_divs, all_entropies = [], [], [], []

        for epoch in range(self.ppo_epochs):
            idx = torch.randperm(len(self.buffer), device=self.device)

            new_lp = self._compute_log_probs_new(queries, prod_embs, prod_feats, rankings)
            ref_lp = self._compute_log_probs_ref(queries, prod_embs, prod_feats, rankings)

            # ── Ratio PPO ─────────────────────────────────────────────────
            ratio = torch.exp(new_lp - old_lp)

            # ── Objetivo PPO clipped ──────────────────────────────────────
            surr1 = ratio * advantages
            surr2 = torch.clamp(ratio, 1 - self.clip_epsilon, 1 + self.clip_epsilon) * advantages
            policy_loss = -torch.min(surr1, surr2).mean()

            # ── Value loss ────────────────────────────────────────────────
            state_repr = self._get_state_repr(queries, prod_embs, prod_feats)
            values = self.value_net(state_repr)
            value_loss = F.mse_loss(values, rewards)

            # ── Entropía (exploración) ────────────────────────────────────
            entropy = -new_lp.mean()   # Para Plackett-Luce, -E[log π] ≈ entropía

            # ── KL divergence vs referencia ───────────────────────────────
            # KL(π_new || π_ref) = E[log π_new - log π_ref]
            kl_div = (new_lp - ref_lp).mean()

            # ── Loss total ────────────────────────────────────────────────
            total_loss = (
                policy_loss
                + self.value_coef * value_loss
                - self.entropy_coef * entropy
                + self.kl_coef * kl_div
            )

            self.policy_optimizer.zero_grad()
            self.value_optimizer.zero_grad()
            total_loss.backward()

            nn.utils.clip_grad_norm_(self.policy.parameters(), self.max_grad_norm)
            nn.utils.clip_grad_norm_(self.value_net.parameters(), self.max_grad_norm)

            self.policy_optimizer.step()
            self.value_optimizer.step()

            all_policy_losses.append(policy_loss.item())
            all_value_losses.append(value_loss.item())
            all_kl_divs.append(kl_div.item())
            all_entropies.append(entropy.item())

            # ── Early stopping por KL ─────────────────────────────────────
            if abs(kl_div.item()) > self.kl_target * 1.5:
                logger.warning(
                    f"Early stop PPO: KL={kl_div.item():.4f} > "
                    f"{self.kl_target * 1.5:.4f} en época {epoch + 1}"
                )
                break

        self.total_updates += 1

        metrics = {
            "update": self.total_updates,
            "policy_loss": float(np.mean(all_policy_losses)),
            "value_loss": float(np.mean(all_value_losses)),
            "kl_divergence": float(np.mean(all_kl_divs)),
            "entropy": float(np.mean(all_entropies)),
            "mean_reward": rewards.mean().item(),
            "max_reward": rewards.max().item(),
            "buffer_size": len(self.buffer),
        }

        self.training_history.append(metrics)

        logger.info(
            f"PPO Update #{self.total_updates}: "
            f"policy_loss={metrics['policy_loss']:.4f}, "
            f"reward={metrics['mean_reward']:.4f}, "
            f"kl={metrics['kl_divergence']:.5f}"
        )

        # Vaciar buffer
        self.buffer.clear()

        # Actualizar política de referencia si entrenamos suficiente
        if self.total_updates % 10 == 0:
            self._update_reference_policy()

        return metrics

    def _update_reference_policy(self):
        """
        Actualiza la política de referencia con los pesos actuales.
        Esto permite que el KL penalty "siga" al modelo conforme aprende.
        Se hace de forma soft (EMA) para no perder la restricción de golpe.
        """
        alpha = 0.3  # Tasa de actualización
        for p_new, p_ref in zip(
            self.policy.parameters(), self.reference_policy.parameters()
        ):
            p_ref.data.copy_(alpha * p_new.data + (1 - alpha) * p_ref.data)

        logger.debug("Política de referencia actualizada (soft EMA)")

    # ─────────────────────────────────────────────────────────────────────
    # Persistencia
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "policy": self.policy.state_dict(),
            "value_net": self.value_net.state_dict(),
            "reference_policy": self.reference_policy.state_dict(),
            "policy_optimizer": self.policy_optimizer.state_dict(),
            "value_optimizer": self.value_optimizer.state_dict(),
            "training_history": self.training_history,
            "total_updates": self.total_updates,
        }, path)
        logger.info(f"PPOTrainer guardado: {path} (updates={self.total_updates})")

    def load(self, path: str):
        ck = torch.load(path, map_location=self.device)
        self.policy.load_state_dict(ck["policy"])
        self.value_net.load_state_dict(ck["value_net"])
        self.reference_policy.load_state_dict(ck["reference_policy"])
        self.policy_optimizer.load_state_dict(ck["policy_optimizer"])
        self.value_optimizer.load_state_dict(ck["value_optimizer"])
        self.training_history = ck.get("training_history", [])
        self.total_updates = ck.get("total_updates", 0)
        logger.info(f"PPOTrainer cargado: {path} (updates={self.total_updates})")

    def get_stats(self) -> Dict:
        if not self.training_history:
            return {"total_updates": 0, "trained": False}

        last = self.training_history[-1]
        return {
            "total_updates": self.total_updates,
            "trained": self.total_updates > 0,
            "last_reward": last.get("mean_reward", 0),
            "last_kl": last.get("kl_divergence", 0),
            "last_policy_loss": last.get("policy_loss", 0),
            "buffer_current": len(self.buffer),
        }