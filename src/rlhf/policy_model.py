"""
Policy Model — Componente 1 del RLHF

El policy model es el modelo que será optimizado por PPO.
Toma (query_embedding, product_embeddings, product_features)
y genera scores de ranking para cada producto.

Arquitectura: Cross-attention transformer
- Los productos atienden al query (cross-attention)
- Self-attention entre productos
- Score head lineal por producto
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np
import logging

logger = logging.getLogger(__name__)


class PolicyModel(nn.Module):
    """
    Transformer de reranking entrenable para RLHF.

    Input:
        - query_embedding:    (batch, emb_dim)     — embedding del query
        - product_embeddings: (batch, n_prod, emb_dim) — embeddings de productos
        - product_features:   (batch, n_prod, feat_dim) — features escalares (rating, match, etc.)

    Output:
        - scores: (batch, n_prod) — score de ranking, mayor = mejor posición
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        feature_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim

        # ── Proyección del query ──────────────────────────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Proyección de productos (embedding + features escalares) ──────
        self.product_proj = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Self-attention entre productos ────────────────────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,           # Pre-LN: más estable para datasets pequeños
        )
        self.product_encoder = nn.TransformerEncoder(
            encoder_layer, num_layers=num_layers
        )

        # ── Cross-attention: productos ← query ───────────────────────────
        self.cross_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_dim)

        # ── Score head por producto ───────────────────────────────────────
        self.score_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),
        )

        self._init_weights()
        logger.info(
            f"PolicyModel inicializado — "
            f"emb={embedding_dim}, hidden={hidden_dim}, "
            f"heads={num_heads}, layers={num_layers}"
        )

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight, gain=0.5)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

    # ─────────────────────────────────────────────────────────────────────
    # Forward
    # ─────────────────────────────────────────────────────────────────────

    def forward(
        self,
        query_embedding: torch.Tensor,
        product_embeddings: torch.Tensor,
        product_features: torch.Tensor,
        src_key_padding_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """
        Returns scores: (batch, n_prod) — sin softmax, scores crudos para PPO.
        """
        # Proyecciones
        q = self.query_proj(query_embedding)                          # (B, H)
        p_in = torch.cat([product_embeddings, product_features], -1)  # (B, N, E+F)
        p = self.product_proj(p_in)                                    # (B, N, H)

        # Self-attention entre productos
        p = self.product_encoder(p, src_key_padding_mask=src_key_padding_mask)

        # Cross-attention: productos ← query
        q_exp = q.unsqueeze(1)                                         # (B, 1, H)
        attended, _ = self.cross_attn(
            query=p, key=q_exp, value=q_exp
        )
        p = self.cross_attn_norm(p + attended)

        # Concatenar representación propia + info del query
        q_broadcast = q.unsqueeze(1).expand_as(p)                      # (B, N, H)
        combined = torch.cat([p, q_broadcast], dim=-1)                 # (B, N, 2H)

        scores = self.score_head(combined).squeeze(-1)                 # (B, N)
        return scores

    # ─────────────────────────────────────────────────────────────────────
    # Métodos de ranking
    # ─────────────────────────────────────────────────────────────────────

    def get_ranking(
        self,
        query_embedding: torch.Tensor,
        product_embeddings: torch.Tensor,
        product_features: torch.Tensor,
        temperature: float = 1.0,
        noise_scale: float = 0.0,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Genera un ranking y sus log-probabilidades bajo el modelo Plackett-Luce.

        Args:
            temperature:  > 1 → más aleatorio, < 1 → más determinista
            noise_scale:  ruido gaussiano para exploración

        Returns:
            ranking:   (batch, n_prod) — índices de productos ordenados
            log_probs: (batch,)        — log P(ranking | query, products)
        """
        scores = self.forward(query_embedding, product_embeddings, product_features)

        if temperature != 1.0:
            scores = scores / max(temperature, 1e-6)

        if noise_scale > 0.0:
            scores = scores + torch.randn_like(scores) * noise_scale

        ranking = torch.argsort(scores, dim=-1, descending=True)
        log_probs = self._plackett_luce_log_prob(scores, ranking)

        return ranking, log_probs

    # ─────────────────────────────────────────────────────────────────────
    # Modelo Plackett-Luce
    # ─────────────────────────────────────────────────────────────────────

    def _plackett_luce_log_prob(
        self,
        scores: torch.Tensor,   # (B, N)
        ranking: torch.Tensor,  # (B, N) — índices en orden
    ) -> torch.Tensor:
        """
        log P(π) = Σ_i [ score[π_i] - log Σ_{j≥i} score[π_j] ]

        El modelo Plackett-Luce es la extensión natural de Bradley-Terry
        para rankings completos. Cada posición "elige" el siguiente ítem
        con probabilidad proporcional a exp(score).
        """
        # Scores en el orden del ranking
        ranked_scores = torch.gather(scores, 1, ranking)              # (B, N)

        n = ranked_scores.shape[1]
        log_probs = torch.zeros(ranked_scores.shape[0], device=scores.device)

        for i in range(n):
            score_i = ranked_scores[:, i]
            # Suma de scores restantes (desde posición i en adelante)
            log_denom = torch.logsumexp(ranked_scores[:, i:], dim=-1)
            log_probs = log_probs + score_i - log_denom

        return log_probs

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────

    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({"state_dict": self.state_dict(), "config": {
            "embedding_dim": self.embedding_dim,
            "feature_dim": self.feature_dim,
            "hidden_dim": self.hidden_dim,
        }}, path)
        logger.info(f"PolicyModel guardado: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "PolicyModel":
        data = torch.load(path, map_location=device)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        model.to(device)
        logger.info(f"PolicyModel cargado: {path}")
        return model