# src/rlhf/ppo_trainer.py
"""
PPO Trainer con KL Penalty Adaptativa.

PROBLEMA CON KL FIJO:
    Si beta (KL weight) es demasiado alto -> policy no aprende nada.
    Si beta es demasiado bajo -> policy diverge del baseline.
    KL fijo casi nunca funciona bien en la práctica.

SOLUCIÓN — KL Adaptativa (estilo Schulman et al.):
    if KL > target_kl:
        beta *= 1.5   # penalizar más si diverge demasiado
    elif KL < target_kl / 2:
        beta *= 0.8   # relajar si está muy conservador

VERSIONADO:
    Para paper IEEE necesitas reportar mejora incremental.
    Guardamos versiones: policy_v0.pt, policy_v1.pt, etc.
    Así puedes evaluar experimento_completo_4_metodos.py en cada versión
    y mostrar la curva de aprendizaje.

VERIFICACIÓN DE QUE PPO ACTUALIZA:
    Antes/después de cada ciclo se imprime:
        weight_before, weight_after, delta
    Si delta < 1e-6 -> PPO no está funcionando.
    Causas comunes:
        1. KL demasiado alto (beta muy grande)
        2. Reward plano (reward model no discrimina)
        3. Learning rate demasiado pequeño
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List, Dict

logger = logging.getLogger(__name__)

CHECKPOINT_DIR = Path("data/rlhf_checkpoints")


class PPOTrainer:
    """
    PPO con KL Penalty Adaptativa para PolicyModel.

    Args:
        policy_model:  El PolicyModel a entrenar
        lr:            Learning rate del optimizador
        target_kl:     KL objetivo (0.01 es conservador, 0.05 más agresivo)
        beta_init:     Peso inicial de la KL penalty
        beta_min:      Límite inferior de beta
        beta_max:      Límite superior de beta
        clip_eps:      Epsilon de clipping PPO (si usas ratio clipping)
    """

    def __init__(
        self,
        policy_model: nn.Module,
        lr: float = 3e-5,        # conservador — 1e-4 causó NaN con KL mal calculada
        target_kl: float = 0.02,
        beta_init: float = 0.1,
        beta_min: float = 0.001,
        beta_max: float = 10.0,
        clip_eps: float = 0.2,
    ):
        self.policy = policy_model
        self.target_kl = target_kl
        self.beta = beta_init
        self.beta_min = beta_min
        self.beta_max = beta_max
        self.clip_eps = clip_eps

        self.optimizer = torch.optim.Adam(
            policy_model.parameters(), lr=lr
        )

        # -- Copia frozen del policy inicial (referencia para KL real) --
        # Sin esto, KL(policy||policy) = 0 siempre -> gradiente sin control
        import copy
        self.policy_ref = copy.deepcopy(policy_model)
        for param in self.policy_ref.parameters():
            param.requires_grad = False
        self.policy_ref.eval()

        # Historial para diagnóstico
        self.kl_history: List[float] = []
        self.beta_history: List[float] = []
        self.loss_history: List[float] = []
        self.reward_history: List[float] = []

        self._version = 0
        CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

        logger.info(
            f"PPOTrainer: lr={lr}, target_kl={target_kl}, "
            f"beta_init={beta_init}, clip_eps={clip_eps}"
        )

    # ------------------------------------------------------------------
    # Update step principal
    # ------------------------------------------------------------------

    def update(
        self,
        query_emb: torch.Tensor,        # [emb_dim]
        prod_embs_baseline: torch.Tensor,  # [top_k, emb_dim]  — ranking de referencia
        prod_embs_policy: torch.Tensor,    # [top_k, emb_dim]  — ranking de la policy
        advantage: torch.Tensor,        # escalar — r_policy - r_baseline
    ) -> dict:
        """
        Un paso de actualización PPO.

        Objetivo:
            Maximizar advantage mientras la policy no se aleje
            demasiado del baseline (KL penalty adaptativa).

        Loss:
            L = -advantage * log_prob(ranking_policy | query)
                + beta * KL(policy || baseline)

        Returns:
            {'loss': float, 'kl': float, 'beta': float}
        """
        self.policy.train()

        q = query_emb.unsqueeze(0)                  # [1, emb_dim]
        pb = prod_embs_baseline.unsqueeze(0)        # [1, top_k, emb_dim]
        pp = prod_embs_policy.unsqueeze(0)          # [1, top_k, emb_dim]

        # product_features: zeros [1, top_k, 8] — PolicyModel los requiere
        n = pb.size(1)
        feats = torch.zeros(1, n, 8, dtype=torch.float32, device=query_emb.device)

        # -- Scores de la policy entrenada ------------------------------
        scores_policy_on_baseline = self.policy(q, pb, feats).squeeze()  # [top_k]
        scores_policy_on_policy   = self.policy(q, pp, feats).squeeze()  # [top_k]

        log_probs_current = F.log_softmax(scores_policy_on_policy, dim=-1)

        # -- Scores de la policy de referencia (frozen) -----------------
        # KL real = KL(policy_actual || policy_ref)
        # Sin policy_ref, KL(p||p)=0 siempre -> sin restricción -> NaN
        with torch.no_grad():
            scores_ref = self.policy_ref(q, pb, feats).squeeze()
            log_probs_ref = F.log_softmax(scores_ref, dim=-1)
            probs_ref = log_probs_ref.exp()

        log_probs_policy_on_baseline = F.log_softmax(scores_policy_on_baseline, dim=-1)

        # KL(policy_actual || policy_ref) — divergencia real del baseline
        kl = F.kl_div(
            log_probs_policy_on_baseline,
            probs_ref,
            reduction='sum',
            log_target=False,
        )
        kl_val = kl.item()

        # NaN guard — si KL explota, skip este step
        if not np.isfinite(kl_val):
            logger.warning(f"  KL={kl_val} — step ignorado (NaN/Inf)")
            return {'loss': 0.0, 'kl': 0.0, 'beta': self.beta, 'skipped': True}

        # -- Policy gradient loss ---------------------------------------
        # advantage > 0: policy_ranking > baseline_ranking según reward
        pg_loss = -advantage * log_probs_current.mean()

        # NaN en advantage
        if not torch.isfinite(advantage):
            return {'loss': 0.0, 'kl': kl_val, 'beta': self.beta, 'skipped': True}

        # -- Loss total con KL penalty ----------------------------------
        loss = pg_loss + self.beta * kl

        if not torch.isfinite(loss):
            return {'loss': 0.0, 'kl': kl_val, 'beta': self.beta, 'skipped': True}

        self.optimizer.zero_grad()
        loss.backward()
        # Clipping más conservador: 0.5 en lugar de 1.0
        torch.nn.utils.clip_grad_norm_(self.policy.parameters(), 0.5)
        self.optimizer.step()

        # Actualizar beta adaptativamente
        self._update_beta(kl_val)

        loss_val = loss.item()
        self.kl_history.append(kl_val)
        self.beta_history.append(self.beta)
        self.loss_history.append(loss_val)
        self.reward_history.append(advantage.item())

        return {'loss': loss_val, 'kl': kl_val, 'beta': self.beta}

    # ------------------------------------------------------------------
    # KL Adaptativa
    # ------------------------------------------------------------------

    def _update_beta(self, kl: float):
        """
        Ajusta beta basado en si el KL actual supera o es inferior al target.

        Lógica:
            kl > target_kl       -> la policy diverge demasiado -> aumentar penalización
            kl < target_kl / 2   -> la policy es demasiado conservadora -> reducir penalización
        """
        if kl > self.target_kl:
            self.beta = min(self.beta * 1.5, self.beta_max)
        elif kl < self.target_kl / 2:
            self.beta = max(self.beta * 0.8, self.beta_min)
        # Si está en la zona correcta (target_kl/2 <= kl <= target_kl), no cambiar

    # ------------------------------------------------------------------
    # Versionado de checkpoints
    # ------------------------------------------------------------------

    def save_version(self, label: str = None) -> Path:
        """
        Guarda una versión nombrada del modelo para comparación en paper.

        Uso:
            trainer.save_version("v0_baseline")   -> policy_v0_baseline.pt
            trainer.save_version("v1_reward")     -> policy_v1_reward.pt
            trainer.save_version("v2_ppo_c1")    -> policy_v2_ppo_c1.pt

        Args:
            label: Nombre descriptivo de la versión

        Returns:
            Path al archivo guardado
        """
        if label is None:
            label = f"cycle{self._version}"
        filename = f"policy_v{self._version}_{label}.pt"
        path = CHECKPOINT_DIR / filename
        torch.save(self.policy.state_dict(), path)
        logger.info(f"  [OK] Versión guardada: {filename}")
        self._version += 1
        return path

    def list_versions(self) -> List[Path]:
        """Lista todas las versiones guardadas."""
        return sorted(CHECKPOINT_DIR.glob("policy_v*.pt"))

    # ------------------------------------------------------------------
    # Diagnósticos
    # ------------------------------------------------------------------

    def check_policy_updated(
        self,
        weight_before: float,
        weight_after: float,
    ) -> dict:
        """
        Verifica si la policy realmente se actualizó.

        Uso:
            before = policy.linear.weight.mean().item()
            trainer.update(...)
            after = policy.linear.weight.mean().item()
            diag = trainer.check_policy_updated(before, after)

        Returns:
            {'delta': float, 'updated': bool, 'warning': str}
        """
        delta = abs(weight_after - weight_before)
        updated = delta > 1e-7

        result = {
            'weight_before': weight_before,
            'weight_after': weight_after,
            'delta': delta,
            'updated': updated,
        }

        if not updated:
            result['warning'] = (
                "Policy no cambió. Causas:\n"
                "  1. KL demasiado alto (beta muy grande)\n"
                "  2. Reward plano (reward model no discrimina)\n"
                "  3. Learning rate demasiado pequeño"
            )
        return result

    def get_kl_status(self) -> dict:
        """Estado actual del KL y beta."""
        recent_kl = self.kl_history[-10:] if self.kl_history else []
        avg_kl = np.mean(recent_kl) if recent_kl else 0.0

        status = 'ok'
        if avg_kl > self.target_kl * 2:
            status = 'too_high'
        elif avg_kl < self.target_kl / 4:
            status = 'too_low'

        return {
            'current_beta': self.beta,
            'target_kl': self.target_kl,
            'recent_avg_kl': avg_kl,
            'status': status,
            'n_updates': len(self.kl_history),
        }

    def get_training_summary(self) -> dict:
        """Resumen del entrenamiento para logging."""
        if not self.loss_history:
            return {'n_updates': 0}

        return {
            'n_updates': len(self.loss_history),
            'avg_loss_last10': float(np.mean(self.loss_history[-10:])),
            'avg_kl_last10': float(np.mean(self.kl_history[-10:])) if self.kl_history else 0,
            'avg_reward_last10': float(np.mean(self.reward_history[-10:])) if self.reward_history else 0,
            'current_beta': self.beta,
            'beta_range': f"[{self.beta_min}, {self.beta_max}]",
            'versions_saved': len(self.list_versions()),
        }