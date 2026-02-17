"""
Reward Model — Componente 3 del RLHF

Este modelo APRENDE a predecir qué ranking preferiría un humano.
No es una función hardcodeada — es una red neuronal entrenada con
datos de preferencias A-vs-B recolectados de usuarios reales.

Entrenamiento: Bradley-Terry loss sobre pares (preferido, rechazado)
    Loss = -log σ(r_preferido - r_rechazado)

Arquitectura: Transformer con position encoding sobre la lista ordenada
    - La posición importa: item en posición 1 ≠ item en posición 5
    - CLS token para agregar representación del ranking completo
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Dict
import numpy as np
import logging
import json
from pathlib import Path

logger = logging.getLogger(__name__)


class RewardModel(nn.Module):
    """
    Predice el reward escalar de un ranking dado un query.

    Input:
        - query_embedding:            (batch, emb_dim)
        - ranked_product_embeddings:  (batch, n_prod, emb_dim) — EN ORDEN DEL RANKING
        - ranked_product_features:    (batch, n_prod, feat_dim)

    Output:
        - reward: (batch,) — escalar, mayor = ranking mejor para humanos
    """

    def __init__(
        self,
        embedding_dim: int = 384,
        feature_dim: int = 8,
        hidden_dim: int = 128,
        num_heads: int = 4,
        num_layers: int = 2,
        max_products: int = 20,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.embedding_dim = embedding_dim
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.max_products = max_products

        # ── Proyecciones ─────────────────────────────────────────────────
        self.query_proj = nn.Sequential(
            nn.Linear(embedding_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        self.product_proj = nn.Sequential(
            nn.Linear(embedding_dim + feature_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
        )

        # ── Positional encoding (posición en el ranking importa) ─────────
        self.position_embedding = nn.Embedding(max_products + 1, hidden_dim)
        # +1: posición 0 reservada para CLS token (no tiene posición de ranking)

        # ── CLS token aprendible ─────────────────────────────────────────
        self.cls_token = nn.Parameter(torch.randn(1, 1, hidden_dim) * 0.02)

        # ── Transformer encoder sobre el ranking completo ────────────────
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=hidden_dim,
            nhead=num_heads,
            dim_feedforward=hidden_dim * 4,
            dropout=dropout,
            batch_first=True,
            norm_first=True,
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        # ── Reward head ───────────────────────────────────────────────────
        self.reward_head = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.GELU(),
            nn.Linear(hidden_dim // 2, 1),
        )

        self._init_weights()
        logger.info(
            f"RewardModel inicializado — "
            f"emb={embedding_dim}, hidden={hidden_dim}, "
            f"max_products={max_products}"
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
        ranked_product_embeddings: torch.Tensor,
        ranked_product_features: torch.Tensor,
    ) -> torch.Tensor:
        """Returns reward scalar (batch,)."""
        batch_size, n_prod, _ = ranked_product_embeddings.shape

        # Proyecciones
        q = self.query_proj(query_embedding)                                 # (B, H)
        p_in = torch.cat([ranked_product_embeddings, ranked_product_features], -1)
        p = self.product_proj(p_in)                                           # (B, N, H)

        # Positional encoding: posición 1..N en el ranking
        positions = torch.arange(1, n_prod + 1, device=p.device).unsqueeze(0)
        p = p + self.position_embedding(positions)                            # (B, N, H)

        # Prepend CLS token (no lleva posición de ranking)
        cls = self.cls_token.expand(batch_size, -1, -1)                      # (B, 1, H)
        sequence = torch.cat([cls, p], dim=1)                                 # (B, N+1, H)

        # Transformer
        encoded = self.transformer(sequence)                                  # (B, N+1, H)

        # CLS output + mean pool de productos
        cls_out = encoded[:, 0, :]                                            # (B, H)
        mean_pool = encoded[:, 1:, :].mean(dim=1)                            # (B, H)

        # Incluir query en la representación final
        q_gate = torch.sigmoid(q)  # gate suave con el query
        cls_out = cls_out * q_gate

        combined = torch.cat([cls_out, mean_pool], dim=-1)                   # (B, 2H)
        reward = self.reward_head(combined).squeeze(-1)                       # (B,)

        return reward

    # ─────────────────────────────────────────────────────────────────────
    # Training
    # ─────────────────────────────────────────────────────────────────────

    def compute_preference_loss(
        self,
        query_embedding: torch.Tensor,
        preferred_embeddings: torch.Tensor,
        preferred_features: torch.Tensor,
        rejected_embeddings: torch.Tensor,
        rejected_features: torch.Tensor,
    ) -> Tuple[torch.Tensor, Dict]:
        """
        Bradley-Terry pairwise preference loss:
            L = -log σ(r_preferido - r_rechazado)

        Si el modelo asigna mayor reward al ranking preferido,
        la loss es baja. Si los confunde, la loss sube.
        """
        r_pref = self.forward(query_embedding, preferred_embeddings, preferred_features)
        r_rej = self.forward(query_embedding, rejected_embeddings, rejected_features)

        # Bradley-Terry loss
        loss = -F.logsigmoid(r_pref - r_rej).mean()

        # Accuracy: ¿cuántas veces el modelo elige correctamente?
        accuracy = (r_pref > r_rej).float().mean()

        metrics = {
            "loss": loss.item(),
            "accuracy": accuracy.item(),
            "reward_preferred": r_pref.mean().item(),
            "reward_rejected": r_rej.mean().item(),
            "margin": (r_pref - r_rej).mean().item(),
        }

        return loss, metrics

    def train_on_preferences(
        self,
        preference_dataset: list,
        epochs: int = 20,
        lr: float = 1e-4,
        batch_size: int = 8,
        device: str = "cpu",
    ) -> Dict:
        """
        Entrena el Reward Model sobre un dataset de preferencias.

        preference_dataset: lista de dicts con keys:
            - query_emb:        torch.Tensor (emb_dim,)
            - preferred_embs:   torch.Tensor (n_prod, emb_dim)
            - preferred_feats:  torch.Tensor (n_prod, feat_dim)
            - rejected_embs:    torch.Tensor (n_prod, emb_dim)
            - rejected_feats:   torch.Tensor (n_prod, feat_dim)
        """
        self.to(device)
        self.train()

        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=epochs, eta_min=lr / 10
        )

        history = []
        best_accuracy = 0.0
        best_state = None

        logger.info(
            f"Entrenando Reward Model: {len(preference_dataset)} prefs, "
            f"{epochs} épocas, lr={lr}"
        )

        for epoch in range(epochs):
            # Shuffle
            indices = np.random.permutation(len(preference_dataset))
            epoch_losses, epoch_accs = [], []

            for start in range(0, len(indices), batch_size):
                batch_idx = indices[start : start + batch_size]
                batch = [preference_dataset[i] for i in batch_idx]

                # Padding al mismo n_prod si varía
                n_prod = max(b["preferred_embs"].shape[0] for b in batch)
                emb_dim = batch[0]["preferred_embs"].shape[1]
                feat_dim = batch[0]["preferred_feats"].shape[1]

                def pad(t, n):
                    if t.shape[0] >= n:
                        return t[:n]
                    pad_size = n - t.shape[0]
                    return torch.cat([t, torch.zeros(pad_size, t.shape[1])], 0)

                queries = torch.stack([b["query_emb"] for b in batch]).to(device)
                pref_embs = torch.stack([pad(b["preferred_embs"], n_prod) for b in batch]).to(device)
                pref_feats = torch.stack([pad(b["preferred_feats"], n_prod) for b in batch]).to(device)
                rej_embs = torch.stack([pad(b["rejected_embs"], n_prod) for b in batch]).to(device)
                rej_feats = torch.stack([pad(b["rejected_feats"], n_prod) for b in batch]).to(device)

                optimizer.zero_grad()
                loss, metrics = self.compute_preference_loss(
                    queries, pref_embs, pref_feats, rej_embs, rej_feats
                )
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.parameters(), 1.0)
                optimizer.step()

                epoch_losses.append(metrics["loss"])
                epoch_accs.append(metrics["accuracy"])

            scheduler.step()

            mean_loss = np.mean(epoch_losses)
            mean_acc = np.mean(epoch_accs)
            history.append({"epoch": epoch + 1, "loss": mean_loss, "accuracy": mean_acc})

            if mean_acc > best_accuracy:
                best_accuracy = mean_acc
                best_state = {k: v.clone() for k, v in self.state_dict().items()}

            if (epoch + 1) % 5 == 0 or epoch == 0:
                logger.info(
                    f"  Época {epoch+1}/{epochs}: "
                    f"loss={mean_loss:.4f}, accuracy={mean_acc:.4f}"
                )

        # Restaurar mejor estado
        if best_state:
            self.load_state_dict(best_state)
            logger.info(f"Reward Model — mejor accuracy: {best_accuracy:.4f}")

        return {"history": history, "best_accuracy": best_accuracy}

    # ─────────────────────────────────────────────────────────────────────
    # Utilidades
    # ─────────────────────────────────────────────────────────────────────

    def save(self, path: str):
        import os
        os.makedirs(os.path.dirname(path) if os.path.dirname(path) else ".", exist_ok=True)
        torch.save({
            "state_dict": self.state_dict(),
            "config": {
                "embedding_dim": self.embedding_dim,
                "feature_dim": self.feature_dim,
                "hidden_dim": self.hidden_dim,
                "max_products": self.max_products,
            },
        }, path)
        logger.info(f"RewardModel guardado: {path}")

    @classmethod
    def load(cls, path: str, device: str = "cpu") -> "RewardModel":
        data = torch.load(path, map_location=device)
        model = cls(**data["config"])
        model.load_state_dict(data["state_dict"])
        model.to(device)
        logger.info(f"RewardModel cargado: {path}")
        return model