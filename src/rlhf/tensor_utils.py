"""
Tensor Utils — Convierte productos canónicos a tensores PyTorch para RLHF.

Los modelos de RLHF trabajan con tensores, pero el sistema existente
usa objetos Python (CanonicalProduct). Este módulo hace el puente.

Features escalares extraídas por producto (feature_dim=8):
    0: rating normalizado (0-1)
    1: tiene rating (binary)
    2: word match ratio query-título (0-1)
    3: match exacto ≥ 0.7 (binary)
    4: category match (binary)
    5: tiene imagen (binary)
    6: cosine similarity con query (0-1, si está disponible)
    7: longitud título normalizada (0-1)
"""

import torch
import numpy as np
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)

FEATURE_DIM = 8
FEATURE_NAMES = [
    "rating_norm",
    "has_rating",
    "word_match_ratio",
    "good_match",
    "category_match",
    "has_image",
    "cosine_sim",
    "title_length_norm",
]


class ProductTensorizer:
    """
    Convierte listas de productos canónicos a tensores para los modelos RLHF.
    """

    def __init__(self, embedding_dim: int = 384, max_products: int = 20):
        self.embedding_dim = embedding_dim
        self.max_products = max_products
        self.feature_dim = FEATURE_DIM

    # ─────────────────────────────────────────────────────────────────────
    # Features escalares de un producto
    # ─────────────────────────────────────────────────────────────────────

    def extract_product_features(
        self,
        product,
        query: str = "",
        cosine_sim: float = 0.0,
    ) -> torch.Tensor:
        """
        Extrae el vector de features escalares de un producto.
        Retorna tensor (feature_dim,).
        """
        feats = [0.0] * FEATURE_DIM

        # 0: rating normalizado
        try:
            rating = float(product.rating) if hasattr(product, "rating") and product.rating else 0.0
            feats[0] = rating / 5.0
        except (ValueError, TypeError):
            feats[0] = 0.0

        # 1: tiene rating
        feats[1] = 1.0 if feats[0] > 0 else 0.0

        # 2-3: word match ratio query-título
        if query and hasattr(product, "title") and product.title:
            q_words = set(query.lower().split())
            t_words = set(product.title.lower().split())
            if q_words:
                ratio = len(q_words & t_words) / len(q_words)
                feats[2] = min(ratio, 1.0)
                feats[3] = 1.0 if ratio >= 0.7 else 0.0

        # 4: category match
        if query and hasattr(product, "category") and product.category:
            cat = str(product.category).lower()
            feats[4] = 1.0 if any(w in cat for w in query.lower().split()) else 0.0

        # 5: tiene imagen
        feats[5] = 1.0 if (hasattr(product, "image_url") and product.image_url) else 0.0

        # 6: cosine similarity (pasada externamente desde FAISS/vector store)
        feats[6] = float(np.clip(cosine_sim, 0.0, 1.0))

        # 7: longitud normalizada del título
        if hasattr(product, "title") and product.title:
            feats[7] = min(len(product.title) / 200.0, 1.0)

        return torch.tensor(feats, dtype=torch.float32)

    # ─────────────────────────────────────────────────────────────────────
    # Embedding de un producto
    # ─────────────────────────────────────────────────────────────────────

    def get_product_embedding(self, product) -> Optional[torch.Tensor]:
        """
        Retorna el embedding del producto si está disponible.
        El sistema existente guarda 'content_embedding' en cada producto.
        """
        emb = None

        if hasattr(product, "content_embedding") and product.content_embedding is not None:
            emb = product.content_embedding
        elif hasattr(product, "embedding") and product.embedding is not None:
            emb = product.embedding

        if emb is None:
            return torch.zeros(self.embedding_dim, dtype=torch.float32)

        if isinstance(emb, np.ndarray):
            emb = torch.from_numpy(emb).float()
        elif not isinstance(emb, torch.Tensor):
            emb = torch.tensor(emb, dtype=torch.float32)

        # Normalizar L2
        norm = emb.norm()
        if norm > 0:
            emb = emb / norm

        return emb.float()

    # ─────────────────────────────────────────────────────────────────────
    # Conversión de lista de productos
    # ─────────────────────────────────────────────────────────────────────

    def products_to_tensors(
        self,
        products: list,
        query: str = "",
        query_embedding: Optional[np.ndarray] = None,
        max_products: Optional[int] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Convierte lista de productos a tensores.

        Returns:
            product_embeddings: (n_prod, emb_dim)
            product_features:   (n_prod, feat_dim)
        """
        n = min(len(products), max_products or self.max_products)
        products = products[:n]

        embeddings = []
        features = []

        for i, product in enumerate(products):
            # Cosine sim con query si disponemos del embedding del query
            cosine_sim = 0.0
            if query_embedding is not None:
                prod_emb_np = None
                if hasattr(product, "content_embedding") and product.content_embedding is not None:
                    prod_emb_np = product.content_embedding
                if prod_emb_np is not None:
                    q_norm = query_embedding / (np.linalg.norm(query_embedding) + 1e-8)
                    p_norm = prod_emb_np / (np.linalg.norm(prod_emb_np) + 1e-8)
                    cosine_sim = float(np.dot(q_norm, p_norm))

            emb = self.get_product_embedding(product)
            feat = self.extract_product_features(product, query, cosine_sim)

            embeddings.append(emb)
            features.append(feat)

        emb_tensor = torch.stack(embeddings)    # (n, emb_dim)
        feat_tensor = torch.stack(features)     # (n, feat_dim)

        return emb_tensor, feat_tensor

    def query_to_tensor(self, query_embedding: np.ndarray) -> torch.Tensor:
        """Convierte el embedding del query (numpy) a tensor normalizado."""
        t = torch.from_numpy(query_embedding).float()
        norm = t.norm()
        if norm > 0:
            t = t / norm
        return t

    # ─────────────────────────────────────────────────────────────────────
    # Ranking a lista de dicts para la UI
    # ─────────────────────────────────────────────────────────────────────

    def ranking_to_display(
        self,
        products: list,
        ranking_indices: torch.Tensor,
    ) -> List[dict]:
        """
        Convierte un tensor de índices de ranking a una lista de dicts
        legibles para el PreferenceCollector UI.
        """
        indices = ranking_indices.tolist()
        result = []

        for idx in indices:
            if idx >= len(products):
                continue
            p = products[idx]
            result.append({
                "id": getattr(p, "id", f"unk_{idx}"),
                "title": getattr(p, "title", "Sin título"),
                "category": str(getattr(p, "category", "")),
                "rating": float(p.rating) if hasattr(p, "rating") and p.rating else None,
                "price": float(p.price) if hasattr(p, "price") and p.price else None,
                "image_url": getattr(p, "image_url", None),
                "original_index": idx,
            })

        return result

    # ─────────────────────────────────────────────────────────────────────
    # Preferencia a tensores de entrenamiento
    # ─────────────────────────────────────────────────────────────────────

    def preference_to_training_sample(
        self,
        query_emb_tensor: torch.Tensor,
        all_products: list,
        preferred_ids: List[str],
        rejected_ids: List[str],
        query: str = "",
        query_embedding_np: Optional[np.ndarray] = None,
    ) -> Optional[dict]:
        """
        Convierte una preferencia guardada (IDs) a tensores de entrenamiento
        para el RewardModel.

        Returns dict con:
            query_emb, preferred_embs, preferred_feats,
            rejected_embs, rejected_feats
        """
        # Construir mapa id → producto
        id_map = {
            getattr(p, "id", ""): p
            for p in all_products
            if hasattr(p, "id")
        }

        def ids_to_tensors(ids):
            prods = [id_map[pid] for pid in ids if pid in id_map]
            if not prods:
                return None, None
            return self.products_to_tensors(prods, query, query_embedding_np)

        pref_embs, pref_feats = ids_to_tensors(preferred_ids)
        rej_embs, rej_feats = ids_to_tensors(rejected_ids)

        if pref_embs is None or rej_embs is None:
            return None

        # Igualar longitud (padding con ceros si es necesario)
        n = max(pref_embs.shape[0], rej_embs.shape[0])
        emb_dim = pref_embs.shape[1]
        feat_dim = pref_feats.shape[1]

        def pad_to(t, target_n):
            if t.shape[0] >= target_n:
                return t[:target_n]
            pad = torch.zeros(target_n - t.shape[0], t.shape[1])
            return torch.cat([t, pad], 0)

        return {
            "query_emb": query_emb_tensor,
            "preferred_embs": pad_to(pref_embs, n),
            "preferred_feats": pad_to(pref_feats, n),
            "rejected_embs": pad_to(rej_embs, n),
            "rejected_feats": pad_to(rej_feats, n),
        }