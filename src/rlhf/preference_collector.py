"""
Preference Collector — Componente 2 del RLHF (Dataset de feedback humano)

Recolecta preferencias EXPLÍCITAS entre rankings:
    "¿Cuál lista de resultados es mejor para tu búsqueda, A o B?"

Esto produce comparaciones pareadas (A ≻ B) que son el input real del
Reward Model en RLHF. Es fundamentalmente diferente a registrar un click:
- Click: feedback implícito, solo positivo
- A vs B: feedback explícito, comparativo, el humano evalúa AMBAS opciones

Guardado: data/preferences/preferences.jsonl
Formato por línea:
    {
        "query": "car parts",
        "ranking_a_ids": ["B001", "B002", ...],
        "ranking_b_ids": ["B003", "B001", ...],
        "preference": "A",       # o "B" o "equal"
        "timestamp": "...",
        "session_id": "..."
    }
"""

import json
import os
import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PreferenceCollector:
    """
    Interfaz CLI para recolectar comparaciones A-vs-B entre rankings.

    El flujo por query:
      1. Genera RANKING_A (política greedy / temperatura baja)
      2. Genera RANKING_B (política exploratoria / temperatura alta + ruido)
      3. Muestra ambos rankings al usuario en paralelo
      4. Usuario escoge A, B, o igual (=)
      5. Guarda la preferencia en JSONL
    """

    def __init__(
        self,
        output_file: str = "data/preferences/preferences.jsonl",
        top_k_display: int = 10,
    ):
        self.output_file = Path(output_file)
        self.output_file.parent.mkdir(parents=True, exist_ok=True)

        self.top_k_display = top_k_display
        self.session_id = f"pref_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        self.session_count = 0
        self.total_preferences = self._count_existing_preferences()

        logger.info(
            f"PreferenceCollector iniciado — "
            f"archivo: {self.output_file}, "
            f"preferencias existentes: {self.total_preferences}"
        )

    def _count_existing_preferences(self) -> int:
        if not self.output_file.exists():
            return 0
        try:
            with open(self.output_file, "r", encoding="utf-8") as f:
                return sum(1 for line in f if line.strip())
        except Exception:
            return 0

    # ─────────────────────────────────────────────────────────────────────
    # Mostrar comparación
    # ─────────────────────────────────────────────────────────────────────

    def display_comparison(
        self,
        query: str,
        ranking_a: List[Dict],
        ranking_b: List[Dict],
    ):
        """
        Muestra los dos rankings en paralelo con formato legible.

        ranking_a / ranking_b: lista de dicts con keys:
            - title, id, category, rating, price
        """
        k = min(self.top_k_display, len(ranking_a), len(ranking_b))

        width = 58
        sep = "│"

        print("\n" + "═" * (width * 2 + 3))
        print(f"  QUERY: \"{query}\"")
        print("═" * (width * 2 + 3))
        print(f"  {'RANKING  A':<{width}} {sep} {'RANKING  B':<{width}}")
        print("─" * (width * 2 + 3))

        for i in range(k):
            prod_a = ranking_a[i] if i < len(ranking_a) else {}
            prod_b = ranking_b[i] if i < len(ranking_b) else {}

            # Línea 1: número + título truncado
            title_a = (prod_a.get("title", "")[:54] + "…") if len(prod_a.get("title", "")) > 55 else prod_a.get("title", "—")
            title_b = (prod_b.get("title", "")[:54] + "…") if len(prod_b.get("title", "")) > 55 else prod_b.get("title", "—")
            print(f"  {i+1:>2}. {title_a:<{width-5}} {sep}  {i+1:>2}. {title_b}")

            # Línea 2: ID + rating
            id_a = prod_a.get("id", "")
            id_b = prod_b.get("id", "")
            rat_a = f"⭐ {prod_a['rating']:.1f}" if prod_a.get("rating") else "     "
            rat_b = f"⭐ {prod_b['rating']:.1f}" if prod_b.get("rating") else "     "
            cat_a = str(prod_a.get("category", ""))[:20]
            cat_b = str(prod_b.get("category", ""))[:20]
            print(f"      ID:{id_a:<12} {rat_a}  {cat_a:<20} {sep}      ID:{id_b:<12} {rat_b}  {cat_b}")
            print(f"  {'─' * width} {sep} {'─' * width}")

        print("═" * (width * 2 + 3))

    # ─────────────────────────────────────────────────────────────────────
    # Recolección de preferencia
    # ─────────────────────────────────────────────────────────────────────

    def collect_preference(
        self,
        query: str,
        ranking_a: List[Dict],
        ranking_b: List[Dict],
    ) -> Optional[str]:
        """
        Muestra la comparación y pide al usuario su preferencia.

        Retorna: "A", "B", "equal", o None si el usuario omite.
        """
        self.display_comparison(query, ranking_a, ranking_b)

        print("\n  ¿Cuál ranking es MÁS RELEVANTE para la búsqueda?")
        print("  [ A ]  Prefiero el RANKING A (izquierda)")
        print("  [ B ]  Prefiero el RANKING B (derecha)")
        print("  [ = ]  Son igual de buenos / no puedo decidir")
        print("  [ s ]  Saltar esta query")

        while True:
            try:
                choice = input("\n  Tu elección (A/B/=/s): ").strip().upper()
                if choice in ("A", "B"):
                    return choice
                elif choice in ("=", "E", "EQUAL", "IGUAL"):
                    return "equal"
                elif choice in ("S", "SKIP", ""):
                    print("  (query omitida)")
                    return None
                else:
                    print("  Por favor escribe A, B, = o s")
            except (EOFError, KeyboardInterrupt):
                return None

    # ─────────────────────────────────────────────────────────────────────
    # Guardar preferencia
    # ─────────────────────────────────────────────────────────────────────

    def save_preference(
        self,
        query: str,
        ranking_a: List[Dict],
        ranking_b: List[Dict],
        preference: str,
        metadata: Optional[Dict] = None,
    ) -> bool:
        """
        Guarda una preferencia en el archivo JSONL.
        Devuelve True si se guardó correctamente.
        """
        record = {
            "timestamp": datetime.now().isoformat(),
            "session_id": self.session_id,
            "query": query,
            "ranking_a_ids": [p.get("id", "") for p in ranking_a],
            "ranking_b_ids": [p.get("id", "") for p in ranking_b],
            "ranking_a_titles": [p.get("title", "")[:60] for p in ranking_a],
            "ranking_b_titles": [p.get("title", "")[:60] for p in ranking_b],
            "preference": preference,  # "A", "B", o "equal"
            "metadata": metadata or {},
        }

        try:
            with open(self.output_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

            self.total_preferences += 1
            self.session_count += 1
            logger.info(
                f"Preferencia guardada: {preference} para '{query[:30]}' "
                f"(total: {self.total_preferences})"
            )
            return True

        except Exception as e:
            logger.error(f"Error guardando preferencia: {e}")
            return False

    # ─────────────────────────────────────────────────────────────────────
    # Sesión interactiva completa
    # ─────────────────────────────────────────────────────────────────────

    def run_comparison_session(
        self,
        queries_and_rankings: List[Tuple[str, List[Dict], List[Dict]]],
    ) -> Dict:
        """
        Corre una sesión completa de comparaciones A-vs-B.

        Args:
            queries_and_rankings: lista de (query, ranking_a, ranking_b)

        Returns:
            dict con estadísticas de la sesión
        """
        print("\n" + "═" * 120)
        print("  SESIÓN DE PREFERENCIAS — RLHF")
        print("  Comparas dos listas de resultados y eliges cuál te parece mejor.")
        print("  Tus elecciones entrenan el sistema para mejorar.")
        print("═" * 120)
        print(f"\n  Comparaciones a evaluar: {len(queries_and_rankings)}")
        print(f"  Preferencias previas:    {self.total_preferences}")
        print(f"  Objetivo recomendado:    30+ preferencias\n")

        stats = {"A": 0, "B": 0, "equal": 0, "skip": 0}

        for i, (query, rank_a, rank_b) in enumerate(queries_and_rankings, 1):
            print(f"\n  [{i}/{len(queries_and_rankings)}] Comparación")

            preference = self.collect_preference(query, rank_a, rank_b)

            if preference is None:
                stats["skip"] += 1
                continue

            self.save_preference(query, rank_a, rank_b, preference)
            stats[preference] += 1

            print(f"\n  ✓ Guardado: preferiste {preference}")
            print(f"  Total en esta sesión: {self.session_count}")

            if self.total_preferences >= 30 and self.total_preferences % 10 == 0:
                print(f"\n  🎉 {self.total_preferences} preferencias — suficiente para entrenar RLHF completo")

        # Resumen de sesión
        print("\n" + "═" * 120)
        print("  RESUMEN DE LA SESIÓN")
        print(f"  Preferencias recolectadas: {self.session_count}")
        print(f"    → Prefirió A: {stats['A']}")
        print(f"    → Prefirió B: {stats['B']}")
        print(f"    → Empate:     {stats['equal']}")
        print(f"    → Omitidas:   {stats['skip']}")
        print(f"  Total acumulado: {self.total_preferences}")
        print("═" * 120)

        return stats

    # ─────────────────────────────────────────────────────────────────────
    # Cargar preferencias guardadas
    # ─────────────────────────────────────────────────────────────────────

    def load_preferences(self) -> List[Dict]:
        """Carga todas las preferencias guardadas en el archivo JSONL."""
        if not self.output_file.exists():
            return []

        preferences = []
        with open(self.output_file, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    preferences.append(json.loads(line))
                except json.JSONDecodeError:
                    continue

        logger.info(f"Cargadas {len(preferences)} preferencias desde {self.output_file}")
        return preferences

    def get_stats(self) -> Dict:
        prefs = self.load_preferences()
        if not prefs:
            return {"total": 0, "ready_for_training": False}

        choices = [p.get("preference", "") for p in prefs]
        return {
            "total": len(prefs),
            "prefer_a": choices.count("A"),
            "prefer_b": choices.count("B"),
            "equal": choices.count("equal"),
            "unique_queries": len(set(p["query"] for p in prefs)),
            "ready_for_training": len(prefs) >= 10,
        }