# src/rlhf/preference_collector.py
"""
Preference Collector — Recolecta y convierte preferencias A/B en tensores.

RESPONSABILIDAD:
    1. Leer preferencias guardadas en data/preferences/preferences.jsonl
    2. Para cada par (query, ranking_A_ids, ranking_B_ids, preference):
       - Obtener embeddings de la query
       - Obtener embeddings de los productos en el orden del ranking
    3. Construir batches (query_emb, ranking_a_embs, ranking_b_embs, prefs)
       que el RankingRewardTrainer pueda consumir directamente.

FORMATO del JSONL (guardado por sistema_interactivo.py):
    {
        "timestamp": "2026-02-22T...",
        "session_id": "ab_...",
        "query": "car parts",
        "ranking_a_ids": ["B001", "B002", ...],
        "ranking_b_ids": ["B010", "B011", ...],
        "ranking_a_titles": [...],    # solo para debug
        "ranking_b_titles": [...],    # solo para debug
        "preference": "A"             # 'A', 'B', o 'equal'
    }

POR QUÉ PRESERVAR EL ORDEN:
    El reward model usa pesos de posición (DCG-style).
    El producto en posición 1 tiene más peso que el de posición 10.
    Si reordenamos los embeddings, destruimos la señal de posición.
"""

import json
import numpy as np
import torch
from pathlib import Path
from typing import List, Tuple, Optional, Dict
import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------
# Constantes
# ---------------------------------------------------------------------

PREFERENCES_FILE = Path("data/preferences/preferences.jsonl")
TOP_K = 10           # número de productos por ranking
EMB_DIM = 384        # sentence-transformer


# ---------------------------------------------------------------------
# Collector
# ---------------------------------------------------------------------

class PreferenceCollector:
    """
    Convierte preferencias guardadas en batches para entrenar el RewardModel.

    Args:
        preferences_file: Ruta al JSONL de preferencias
        embedding_model:  Modelo de sentence-transformers (ya cargado)
        product_index:    Dict[product_id -> np.ndarray (emb_dim,)]
                         Mapa de ID -> embedding precomputado
        top_k:            Cuántos productos del ranking usar
    """

    def __init__(
        self,
        embedding_model,
        product_index: Dict[str, np.ndarray],
        preferences_file: Path = PREFERENCES_FILE,
        top_k: int = TOP_K,
    ):
        self.emb_model = embedding_model
        self.product_index = product_index
        self.preferences_file = preferences_file
        self.top_k = top_k

    # ------------------------------------------------------------------
    # Leer preferencias del disco
    # ------------------------------------------------------------------

    def load_preferences(self, only_clear: bool = True) -> List[dict]:
        """
        Carga las preferencias desde el JSONL.

        Args:
            only_clear: Si True, filtra los 'equal' (no aportan señal clara).

        Returns:
            Lista de dicts con las preferencias.
        """
        if not self.preferences_file.exists():
            logger.warning(f"No hay preferencias en {self.preferences_file}")
            return []

        records = []
        with open(self.preferences_file, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    r = json.loads(line)
                    if only_clear and r.get('preference') == 'equal':
                        continue
                    records.append(r)
                except json.JSONDecodeError:
                    continue

        logger.info(f"Cargadas {len(records)} preferencias (only_clear={only_clear})")
        return records

    def count(self) -> int:
        return len(self.load_preferences(only_clear=False))

    # ------------------------------------------------------------------
    # Construir tensores
    # ------------------------------------------------------------------

    def _get_query_emb(self, query: str) -> np.ndarray:
        """Embedding de la query."""
        return self.emb_model.encode(query, normalize_embeddings=True)

    def _get_ranking_embs(self, product_ids: List[str]) -> np.ndarray:
        """
        Embeddings del ranking PRESERVANDO EL ORDEN.

        Si un producto no está en el índice, se rellena con ceros
        (el reward model maneja padding explícitamente).

        Returns:
            [top_k, emb_dim]
        """
        embs = []
        for pid in product_ids[:self.top_k]:
            if pid in self.product_index:
                embs.append(self.product_index[pid])
            else:
                embs.append(np.zeros(EMB_DIM, dtype=np.float32))

        # Padding si hay menos de top_k productos
        while len(embs) < self.top_k:
            embs.append(np.zeros(EMB_DIM, dtype=np.float32))

        return np.stack(embs[:self.top_k])  # [top_k, emb_dim]

    def build_batch(
        self,
        records: List[dict],
        device: str = 'cpu',
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        Construye un batch listo para RankingRewardTrainer.

        Returns:
            query_embs:    [N, emb_dim]
            ranking_a_embs:[N, top_k, emb_dim]
            ranking_b_embs:[N, top_k, emb_dim]
            preferences:   List['A' | 'B' | 'equal'], len=N
        """
        query_embs = []
        ra_embs = []
        rb_embs = []
        prefs = []

        for r in records:
            q_emb = self._get_query_emb(r['query'])
            ra = self._get_ranking_embs(r['ranking_a_ids'])
            rb = self._get_ranking_embs(r['ranking_b_ids'])

            query_embs.append(q_emb)
            ra_embs.append(ra)
            rb_embs.append(rb)
            prefs.append(r['preference'])

        q_t = torch.tensor(np.stack(query_embs), dtype=torch.float32, device=device)
        ra_t = torch.tensor(np.stack(ra_embs), dtype=torch.float32, device=device)
        rb_t = torch.tensor(np.stack(rb_embs), dtype=torch.float32, device=device)

        return q_t, ra_t, rb_t, prefs

    def get_training_batches(
        self,
        batch_size: int = 16,
        device: str = 'cpu',
        shuffle: bool = True,
    ) -> List[Tuple]:
        """
        Carga todas las preferencias y las divide en batches.

        Returns:
            Lista de tuplas (query_embs, ra_embs, rb_embs, prefs)
        """
        records = self.load_preferences(only_clear=True)
        if not records:
            return []

        if shuffle:
            import random
            random.shuffle(records)

        batches = []
        for i in range(0, len(records), batch_size):
            batch_records = records[i:i + batch_size]
            batch = self.build_batch(batch_records, device=device)
            batches.append(batch)

        logger.info(f"  {len(records)} preferencias -> {len(batches)} batches (bs={batch_size})")
        return batches

    # ------------------------------------------------------------------
    # Diagnóstico
    # ------------------------------------------------------------------

    def stats(self) -> dict:
        records = self.load_preferences(only_clear=False)
        if not records:
            return {'total': 0}

        prefs = [r.get('preference') for r in records]
        queries = [r.get('query') for r in records]

        # Cobertura del índice
        all_ids = set()
        for r in records:
            all_ids.update(r.get('ranking_a_ids', []))
            all_ids.update(r.get('ranking_b_ids', []))
        covered = sum(1 for pid in all_ids if pid in self.product_index)

        return {
            'total': len(records),
            'prefer_A': prefs.count('A'),
            'prefer_B': prefs.count('B'),
            'equal': prefs.count('equal'),
            'unique_queries': len(set(queries)),
            'product_ids_referenced': len(all_ids),
            'product_ids_in_index': covered,
            'coverage_pct': round(covered / len(all_ids) * 100, 1) if all_ids else 0,
        }