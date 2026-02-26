# src/rlhf/reward_model.py
"""
Reward Model Híbrido — estable con pocos datos.

PROBLEMA DEL DISEÑO PURO r(query, ranking):
    Con 30-50 comparaciones A/B, la varianza es muy alta.
    El reward aprende patrones globales difusos.
    PPO optimiza ruido.

SOLUCIÓN HÍBRIDA (crédito desagregado + sensibilidad a posición):

    r(query, ranking) = Σ_k  w_k * r_producto(query, producto_k)

    Donde:
        r_producto(query, p): MLP pequeño que puntúa query × producto
        w_k: pesos DCG  (posición 1 -> 1.0, posición 10 -> 0.29)

    Ventajas:
        [OK] Aprende señal más fina (por producto)
        [OK] Menos varianza con pocos datos
        [OK] Mantiene sensibilidad a posición (pesos DCG)
        [OK] Bradley-Terry opera sobre el score del ranking completo
        [OK] Convergencia mucho más rápida

    Desventajas vs puro r(ranking):
        [ERR] Asume independencia entre productos del ranking
        [ERR] No capta diversidad ni interacciones entre productos

    Decisión:
        Con <100 pares -> usa este Hybrid.
        Con >100 pares -> puedes probar RankingRewardModelPure.

ENTRENAMIENTO (Bradley-Terry idéntico):
    Cuando usuario prefirió ranking_A sobre ranking_B:
        r_A = Σ_k w_k * r_prod(query, prod_A_k)
        r_B = Σ_k w_k * r_prod(query, prod_B_k)
        loss = -log( σ(r_A - r_B) )
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List
import logging

logger = logging.getLogger(__name__)

EMB_DIM = 384
HIDDEN_DIM = 128   # intencional: pequeño evita sobreparametrización con pocos datos
TOP_K = 10


# ---------------------------------------------------------------------
# Módulo de score por producto
# ---------------------------------------------------------------------

class ProductScorer(nn.Module):
    """
    r_producto(query, producto) -> escalar.
    MLP pequeño: concat(q, p) [768] -> 128 -> 64 -> 1
    """

    def __init__(self, emb_dim: int = EMB_DIM, hidden_dim: int = HIDDEN_DIM):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(emb_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.15),
            nn.Linear(hidden_dim, 64),
            nn.GELU(),
            nn.Linear(64, 1),
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.1)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    def forward(self, query_emb: torch.Tensor, product_emb: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([query_emb, product_emb], dim=-1))


# ---------------------------------------------------------------------
# Reward model híbrido
# ---------------------------------------------------------------------

class RankingRewardModel(nn.Module):
    """
    r(query, ranking) = Σ_k w_k * r_producto(query, producto_k)

    Mantiene sensibilidad a posición (DCG) pero aprende
    señal a nivel de producto (menos varianza, más rápido).
    """

    def __init__(self, emb_dim: int = EMB_DIM, hidden_dim: int = HIDDEN_DIM, top_k: int = TOP_K):
        super().__init__()
        self.emb_dim = emb_dim
        self.top_k = top_k

        self.product_scorer = ProductScorer(emb_dim, hidden_dim)

        # Pesos DCG fijos (no entrenables)
        weights = torch.tensor(
            [1.0 / np.log2(k + 2) for k in range(top_k)], dtype=torch.float32
        )
        self.register_buffer('dcg_weights', weights / weights.sum())

        logger.info(
            f"RankingRewardModel(hybrid): emb={emb_dim}, hidden={hidden_dim}, "
            f"top_k={top_k}, params={sum(p.numel() for p in self.parameters()):,}"
        )

    def forward(self, query_emb: torch.Tensor, product_embs: torch.Tensor) -> torch.Tensor:
        """
        Args:
            query_emb:    [batch, emb_dim]
            product_embs: [batch, top_k, emb_dim]  — EN ORDEN del ranking
        Returns:
            reward: [batch, 1]
        """
        batch_size = query_emb.size(0)

        # Padding
        k = product_embs.size(1)
        if k < self.top_k:
            pad = torch.zeros(batch_size, self.top_k - k, self.emb_dim,
                              device=product_embs.device, dtype=product_embs.dtype)
            product_embs = torch.cat([product_embs, pad], dim=1)
        else:
            product_embs = product_embs[:, :self.top_k, :]

        # Score por producto: [batch*top_k, 1]
        q_exp = query_emb.unsqueeze(1).expand(-1, self.top_k, -1)
        q_flat = q_exp.reshape(batch_size * self.top_k, self.emb_dim)
        p_flat = product_embs.reshape(batch_size * self.top_k, self.emb_dim)
        scores = self.product_scorer(q_flat, p_flat).reshape(batch_size, self.top_k)

        # Suma ponderada DCG
        return (scores * self.dcg_weights.unsqueeze(0)).sum(dim=-1, keepdim=True)

    def score_ranking(self, query_emb: torch.Tensor, product_embs: torch.Tensor) -> float:
        """Inferencia de un solo ejemplo."""
        self.eval()
        with torch.no_grad():
            return self.forward(query_emb.unsqueeze(0), product_embs.unsqueeze(0)).item()


# ---------------------------------------------------------------------
# Entrenador con Bradley-Terry + split 80/20
# ---------------------------------------------------------------------

class RankingRewardTrainer:
    """
    Entrena RankingRewardModel con pares A/B.
    Incluye split 80/20, accuracy, log-likelihood, y detección de colapso.
    """

    def __init__(self, model: RankingRewardModel, lr: float = 2e-4, weight_decay: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=50, eta_min=1e-5
        )
        self.train_losses: List[float] = []

    def train_step(self, query_embs, ranking_a_embs, ranking_b_embs, preferences) -> float:
        self.model.train()
        clear = [i for i, p in enumerate(preferences) if p in ('A', 'B')]
        if not clear:
            return 0.0

        idx = torch.tensor(clear, device=query_embs.device)
        q, ra, rb = query_embs[idx], ranking_a_embs[idx], ranking_b_embs[idx]
        prefs = [preferences[i] for i in clear]

        r_a, r_b = self.model(q, ra), self.model(q, rb)
        preferred, rejected = [], []
        for j, pref in enumerate(prefs):
            preferred.append(r_a[j] if pref == 'A' else r_b[j])
            rejected.append(r_b[j] if pref == 'A' else r_a[j])

        loss = -F.logsigmoid(torch.stack(preferred) - torch.stack(rejected)).mean()
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
        self.optimizer.step()
        self.scheduler.step()

        self.train_losses.append(loss.item())
        return loss.item()

    def get_accuracy(self, query_embs, ranking_a_embs, ranking_b_embs, preferences) -> float:
        self.model.eval()
        correct = total = 0
        with torch.no_grad():
            for i, pref in enumerate(preferences):
                if pref not in ('A', 'B'):
                    continue
                r_a = self.model(query_embs[i:i+1], ranking_a_embs[i:i+1]).item()
                r_b = self.model(query_embs[i:i+1], ranking_b_embs[i:i+1]).item()
                correct += int(('A' if r_a > r_b else 'B') == pref)
                total += 1
        return correct / total if total > 0 else 0.0

    def get_bradley_terry_loglik(self, query_embs, ranking_a_embs, ranking_b_embs, preferences) -> float:
        """Log-likelihood en hold-out. Para reportar en el paper."""
        self.model.eval()
        lls, n = [], 0
        with torch.no_grad():
            for i, pref in enumerate(preferences):
                if pref not in ('A', 'B'):
                    continue
                r_a = self.model(query_embs[i:i+1], ranking_a_embs[i:i+1])
                r_b = self.model(query_embs[i:i+1], ranking_b_embs[i:i+1])
                ll = F.logsigmoid(r_a - r_b if pref == 'A' else r_b - r_a).item()
                lls.append(ll)
                n += 1
        return float(np.mean(lls)) if lls else float('-inf')

    def detect_reward_collapse(self, query_embs, ranking_a_embs, ranking_b_embs) -> dict:
        """
        Detecta si el reward ha colapsado (no discrimina A de B).

        Colapso real = el modelo no puede predecir preferencias mejor que random.
        Señal de eso: accuracy ~0.5, NO mean_diff < umbral_fijo.

        Un MLP pequeño (hidden=128) produce outputs naturalmente comprimidos
        en rango [-0.5, 0.5]. Con diff=0.002 pero accuracy=1.0, el modelo
        aprendió señal real — solo tiene outputs en escala pequeña.

        Criterio correcto:
            collapsed = mean_diff < 1e-6  (literalmente cero, saturación)
                     OR todos los scores exactamente iguales (std=0)
        """
        self.model.eval()
        with torch.no_grad():
            r_a = self.model(query_embs, ranking_a_embs)
            r_b = self.model(query_embs, ranking_b_embs)
        diff = (r_a - r_b).abs()
        r_a_std = r_a.std().item()
        r_b_std = r_b.std().item()

        # Colapso real: diferencias literalmente cero o varianza cero
        hard_collapse = diff.mean().item() < 1e-6 or (r_a_std < 1e-6 and r_b_std < 1e-6)

        return {
            'r_a_mean': r_a.mean().item(),
            'r_b_mean': r_b.mean().item(),
            'r_a_std': r_a_std,
            'r_b_std': r_b_std,
            'mean_abs_diff': diff.mean().item(),
            'collapsed': hard_collapse,
        }