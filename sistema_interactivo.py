# sistema_interactivo.py — Versión A/B ranking 
"""
Sistema interactivo con comparación A/B explícita.

Flujo correcto:
    1. Usuario escribe query
    2. Sistema genera DOS rankings distintos (A y B)
       A: baseline FAISS puro
       B: política actual con temperatura alta (exploración)
    3. Usuario compara ambos y elige cuál es mejor
    4. Se guarda el par (query, ranking_A, ranking_B, preferencia)
    5. Ese par alimenta el Reward Model -> PPO

Esto sí es RLHF: preferencia humana explícita entre dos alternativas.
"""
import json
import random
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Any, Optional
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class SistemaInteractivoAB:
    """
    Interfaz de comparación A/B para recolección de preferencias RLHF.
    """

    def __init__(self):
        print("\n" + "="*80)
        print("  SISTEMA INTERACTIVO — COMPARACIÓN A/B")
        print("="*80)

        self.preferences_file = Path("data/preferences/preferences.jsonl")
        self.preferences_file.parent.mkdir(parents=True, exist_ok=True)

        self.system = None
        self.rlhf_pipeline = None
        self.session_id = f"ab_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        self.session_count = 0
        self.total_prefs = self._count_existing()

        self._cargar_sistema()

        print(f"\n  Preferencias previas: {self.total_prefs}")
        print(f"  Objetivo mínimo:      30 comparaciones A/B")
        print("\n  COMANDOS:")
        print("    query [texto]  — hacer una comparación A/B")
        print("    stats          — ver estadísticas")
        print("    help           — ayuda")
        print("    exit           — guardar y salir")

    # ------------------------------------------------------------------
    # Carga del sistema
    # ------------------------------------------------------------------

    def _count_existing(self) -> int:
        if not self.preferences_file.exists():
            return 0
        try:
            return sum(1 for line in open(self.preferences_file) if line.strip())
        except Exception:
            return 0

    def _cargar_sistema(self):
        print("\nCargando sistema...")
        system_cache = Path("data/cache/unified_system_v2.pkl")
        if not system_cache.exists():
            print("  Sistema no inicializado. Ejecuta: python main.py init")
            return

        try:
            from src.unified_system_v2 import UnifiedSystemV2
            self.system = UnifiedSystemV2.load_from_cache()
            if not self.system:
                print("  Error cargando sistema")
                return

            print(f"  Sistema: {len(self.system.canonical_products):,} productos")

            # Intentar cargar pipeline RLHF
            try:
                from src.rlhf_integration import add_rlhf_to_system
                self.rlhf_pipeline = add_rlhf_to_system(self.system)
                self.rlhf_pipeline.initialize(load_checkpoint=True)
                status = "entrenado" if self.rlhf_pipeline.policy_trained else "sin entrenar"
                print(f"  RLHFPipeline: {status}")
            except Exception as e:
                logger.warning(f"  RLHFPipeline no disponible: {e}")
                self.rlhf_pipeline = None

        except Exception as e:
            print(f"  Error: {e}")

    # ------------------------------------------------------------------
    # Generación de rankings A y B
    # ------------------------------------------------------------------

    def _get_query_embedding(self, query: str) -> Optional[np.ndarray]:
        if self.system and hasattr(self.system, 'canonicalizer'):
            return self.system.canonicalizer.embedding_model.encode(
                query, normalize_embeddings=True
            )
        return None

    def _get_faiss_scores(self, query_emb: np.ndarray, products: list) -> np.ndarray:
        """
        Recupera los scores coseno reales del índice FAISS para los productos dados.

        Si el vector_store expone los scores directamente los usa.
        Si no, los estima por posición (monotónico decreciente entre 0.95 y 0.70).
        Los scores reales son mejores porque capturan la distancia real al embedding.
        """
        vs = self.system.vector_store if self.system else None
        if vs is None:
            n = len(products)
            return np.linspace(0.95, 0.70, n)

        # Intentar obtener scores reales del vector store
        # (depende de cómo esté implementado tu VectorStore)
        try:
            _, scores = vs.search_with_scores(query_emb, k=len(products))
            if scores is not None and len(scores) == len(products):
                return np.array(scores, dtype=np.float32)
        except AttributeError:
            pass

        # Fallback: scores aproximados por posición
        n = len(products)
        return np.linspace(0.95, 0.70, n).astype(np.float32)

    def _baseline_ranking(self, query: str, k: int = 10) -> List[Dict]:
        """
        Ranking A: FAISS top-k, orden por similitud coseno.
        Determinista. Siempre los mismos resultados para la misma query.
        """
        if not self.system or not self.system.vector_store:
            return []
        emb = self._get_query_embedding(query)
        if emb is None:
            return []
        products = self.system.vector_store.search(emb, k=k)
        return self._to_display(products)

    def _alternative_ranking(self, query: str, k: int = 10) -> List[Dict]:
        """
        Ranking B: dos estrategias según el estado del sistema.

        FASE 1 — Policy RLHF entrenada:
            FAISS recupera top-20 -> PolicyModel reordena.
            Esto es RLHF real. La señal de preferencia retroalimenta PPO.

        FASE 2 — Sin policy (recolección inicial):
            FAISS recupera top-20 (mismo pool relevante).
            Scores coseno reales + ruido N(0, 0.01) -> reordenar -> top-10.

            Propiedades:
                - Pool idéntico al baseline (todos relevantes)
                - El ruido cambia el orden marginalmente
                - El usuario elige entre dos ordenamientos plausibles
                - Señal fina que el reward model puede aprender
                - sigma=0.01 cambia orden sin destruir relevancia base

            Por qué sigma=0.01 y no más grande:
                Si sigma >> (score_i - score_{i+1}), el ruido destruye
                la señal de FAISS y genera basura. Con scores en [0.70, 0.95],
                la diferencia entre posiciones contiguas es ~0.003-0.010,
                así que sigma=0.01 produce mezcla visible pero controlada.
        """
        # -- FASE 1: Policy RLHF entrenada ----------------------------
        if self.rlhf_pipeline and self.rlhf_pipeline.policy_trained:
            try:
                emb = self._get_query_embedding(query)
                products, _, _ = self.rlhf_pipeline.retrieve_candidates(query, k=k * 2)
                if products:
                    ranked = self.rlhf_pipeline.rank_products(query, products, emb)
                    return self._to_display(ranked[:k])
            except Exception as e:
                logger.error(f"Error en policy ranking: {e}")

        # -- FASE 2: Score perturbado sobre pool relevante -------------
        if not self.system or not self.system.vector_store:
            return []

        emb = self._get_query_embedding(query)
        if emb is None:
            return []

        # Pool = top-20 de FAISS (todos relevantes, misma distribución semántica)
        pool = self.system.vector_store.search(emb, k=k * 2)
        if not pool:
            return []

        # Scores reales o aproximados por posición
        scores = self._get_faiss_scores(emb, pool)

        # Ruido pequeño: N(0, sigma=0.01)
        # Cambia el orden marginalmente sin destruir la relevancia base
        sigma = np.std(scores) * 0.5
        noise = np.random.normal(0, sigma, size=len(scores))
        perturbed = scores + noise

        order = np.argsort(perturbed)[::-1]
        reordered = [pool[i] for i in order]

        logger.debug(f"  Alternativo: score perturbado sigma=0.01 sobre pool de {len(pool)}")
        return self._to_display(reordered[:k])

    def _to_display(self, products: list) -> List[Dict]:
        result = []
        for p in products:
            pid = getattr(p, 'id', '')
            result.append({
                'id': pid,
                'title': getattr(p, 'title', 'Sin título'),
                'category': str(getattr(p, 'category', '')),
                'rating': float(p.rating) if getattr(p, 'rating', None) else None,
                'price': float(p.price) if getattr(p, 'price', None) else None,
            })
        return result

    # ------------------------------------------------------------------
    # Mostrar comparación A/B
    # ------------------------------------------------------------------

    def _display_ab(self, query: str, rank_a: List[Dict], rank_b: List[Dict]):
        k = min(10, len(rank_a), len(rank_b))
        w = 55
        sep = "|"
        print("\n" + "=" * (w * 2 + 5))
        print(f"  QUERY: \"{query}\"")
        print("=" * (w * 2 + 5))
        print(f"  {'RANKING  A  (baseline)':<{w}} {sep} {'RANKING  B  (alternativo)':<{w}}")
        print("-" * (w * 2 + 5))

        for i in range(k):
            pa = rank_a[i] if i < len(rank_a) else {}
            pb = rank_b[i] if i < len(rank_b) else {}

            ta = pa.get('title', '')
            tb = pb.get('title', '')
            ta = (ta[:w - 8] + "…") if len(ta) > w - 7 else ta
            tb = (tb[:w - 8] + "…") if len(tb) > w - 7 else tb

            print(f"  {i+1:2}. {ta:<{w-5}} {sep}  {i+1:2}. {tb}")

            ra = f"*{pa['rating']:.1f}" if pa.get('rating') else "     "
            rb = f"*{pb['rating']:.1f}" if pb.get('rating') else "     "
            cat_a = str(pa.get('category', ''))[:18]
            cat_b = str(pb.get('category', ''))[:18]
            print(f"      {ra}  {cat_a:<20} {sep}      {rb}  {cat_b}")
            print(f"  {'-'*w} {sep} {'-'*w}")

        print("=" * (w * 2 + 5))

    # ------------------------------------------------------------------
    # Recolección de preferencia
    # ------------------------------------------------------------------

    def _ask_preference(self) -> Optional[str]:
        print("\n  ¿Cuál ranking es MÁS RELEVANTE para la búsqueda?")
        print("  [ A ]  Prefiero el RANKING A")
        print("  [ B ]  Prefiero el RANKING B")
        print("  [ = ]  Son equivalentes")
        print("  [ s ]  Saltar esta query")
        while True:
            try:
                choice = input("\n  Tu elección (A/B/=/s): ").strip().upper()
                if choice in ("A", "B"):
                    return choice
                elif choice in ("=", "E", "EQUAL"):
                    return "equal"
                elif choice in ("S", "SKIP", ""):
                    return None
                else:
                    print("  Escribe A, B, = o s")
            except (EOFError, KeyboardInterrupt):
                return None

    def _save_preference(self, query: str, rank_a: List[Dict],
                         rank_b: List[Dict], preference: str):
        record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'ranking_a_ids': [p['id'] for p in rank_a],
            'ranking_b_ids': [p['id'] for p in rank_b],
            'ranking_a_titles': [p['title'][:60] for p in rank_a],
            'ranking_b_titles': [p['title'][:60] for p in rank_b],
            'preference': preference,
        }
        with open(self.preferences_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.total_prefs += 1
        self.session_count += 1

    # ------------------------------------------------------------------
    # Comparación completa para un query
    # ------------------------------------------------------------------

    def hacer_comparacion(self, query: str):
        """
        Comparación A/B con control de sesgo de presentación.

        El sesgo de presentación ocurre cuando el usuario tiende a elegir
        siempre el lado izquierdo (A) por defecto. Para evitarlo:
            - Se aleatoriza qué ranking aparece como A y cuál como B
            - Se guarda cuál era baseline y cuál policy en 'ranking_a_type'
            - El JSONL guarda ranking_a_ids / ranking_b_ids según lo que VIO el usuario

        Así el reward model aprende de la preferencia real, no de la posición.
        """
        if not self.system:
            print("  Sistema no cargado")
            return

        print(f"\n  Generando rankings para: '{query}'...")
        baseline = self._baseline_ranking(query, k=10)
        policy = self._alternative_ranking(query, k=10)

        if not baseline or not policy:
            print("  Sin resultados suficientes para comparar")
            return

        # Verificar que son distintos
        ids_bl = {p['id'] for p in baseline}
        ids_po = {p['id'] for p in policy}
        overlap = len(ids_bl & ids_po) / max(len(ids_bl), len(ids_po), 1)
        if overlap > 0.8:
            print("  Advertencia: los rankings son muy similares (normal al inicio)")

        # -- CONTROL DE SESGO: aleatorizar qué lado es A y cuál es B --
        flip = random.random() < 0.5
        if flip:
            rank_a, rank_b = policy, baseline
            type_a, type_b = 'policy', 'baseline'
        else:
            rank_a, rank_b = baseline, policy
            type_a, type_b = 'baseline', 'policy'

        self._display_ab(query, rank_a, rank_b)
        preference = self._ask_preference()

        if preference is None:
            print("  (query omitida)")
            return

        # Guardar preferencia con metadato de tipo
        record = {
            'timestamp': datetime.now().isoformat(),
            'session_id': self.session_id,
            'query': query,
            'ranking_a_ids': [p['id'] for p in rank_a],
            'ranking_b_ids': [p['id'] for p in rank_b],
            'ranking_a_titles': [p['title'][:60] for p in rank_a],
            'ranking_b_titles': [p['title'][:60] for p in rank_b],
            'ranking_a_type': type_a,   # 'baseline' o 'policy'
            'ranking_b_type': type_b,
            'preference': preference,
            # Cuál ganó en términos de baseline/policy (para análisis)
            'preferred_type': type_a if preference == 'A' else (type_b if preference == 'B' else 'equal'),
        }
        with open(self.preferences_file, 'a', encoding='utf-8') as f:
            f.write(json.dumps(record, ensure_ascii=False) + '\n')
        self.total_prefs += 1
        self.session_count += 1

        pref_type = record['preferred_type']
        print(f"\n  [OK] Guardado: preferiste {preference} ({pref_type})")
        print(f"  Total en esta sesión: {self.session_count}")
        print(f"  Total acumulado:      {self.total_prefs}")

        if self.total_prefs >= 30 and self.total_prefs % 10 == 0:
            print(f"\n  🎉 {self.total_prefs} preferencias — puedes entrenar el Reward Model:")
            print("     python main.py rlhf --train-reward")

    # ------------------------------------------------------------------
    # Stats
    # ------------------------------------------------------------------

    def mostrar_estadisticas(self):
        print("\n  ESTADÍSTICAS")
        print(f"  Sesión:              {self.session_count} comparaciones")
        print(f"  Total acumulado:     {self.total_prefs} preferencias")

        if self.preferences_file.exists():
            prefs = []
            with open(self.preferences_file) as f:
                for line in f:
                    try:
                        prefs.append(json.loads(line))
                    except Exception:
                        pass

            choices = [p.get('preference') for p in prefs]
            print(f"    -> Prefirió A:  {choices.count('A')}")
            print(f"    -> Prefirió B:  {choices.count('B')}")
            print(f"    -> Empate:      {choices.count('equal')}")
            print(f"    -> Queries únicas: {len(set(p['query'] for p in prefs))}")

            # Análisis de sesgo de presentación
            preferred_types = [p.get('preferred_type') for p in prefs if p.get('preferred_type')]
            if preferred_types:
                print(f"\n  Análisis de preferencia por tipo:")
                print(f"    -> Prefirió baseline: {preferred_types.count('baseline')}")
                print(f"    -> Prefirió policy:   {preferred_types.count('policy')}")
                print(f"    -> Empate:            {preferred_types.count('equal')}")

                # Si policy no está entrenada, el usuario siempre preferirá baseline
                # Si policy mejora, la proporción debería cambiar
                n_clear = preferred_types.count('baseline') + preferred_types.count('policy')
                if n_clear > 5:
                    policy_win_rate = preferred_types.count('policy') / n_clear
                    print(f"    -> Win rate de policy: {policy_win_rate:.1%}")
                    if policy_win_rate < 0.3:
                        print("      (Policy aún no supera baseline — normal al inicio)")
                    elif policy_win_rate > 0.55:
                        print("      [OK] Policy empieza a superar baseline")

            # Detectar sesgo de presentación posicional
            a_choices = choices.count('A')
            b_choices = choices.count('B')
            total_clear = a_choices + b_choices
            if total_clear > 10:
                a_rate = a_choices / total_clear
                if a_rate > 0.75:
                    print(f"\n  [WARN] Sesgo posicional: eliges A el {a_rate:.0%} de las veces")
                    print("    (El orden A/B ya está aleatorizado — intenta ser más neutral)")
                elif a_rate < 0.25:
                    print(f"\n  [WARN] Sesgo posicional: eliges B el {1-a_rate:.0%} de las veces")

        ready = self.total_prefs >= 10
        print(f"\n  Estado: {'[OK] Listo para entrenar' if ready else '[ERR] Necesitas 10+ preferencias'}")
        if not ready:
            print(f"  Faltan: {max(0, 10 - self.total_prefs)} comparaciones")
        if self.rlhf_pipeline:
            print(f"\n  Policy RLHF entrenada: {'[OK]' if self.rlhf_pipeline.policy_trained else '[ERR]'}")

    # ------------------------------------------------------------------
    # Loop principal
    # ------------------------------------------------------------------

    def ejecutar(self):
        print("\n  ¡Empieza a comparar rankings!")
        print("  Ejemplo: query car parts\n")

        while True:
            try:
                cmd = input("sistema> ").strip()
                if not cmd:
                    continue
                elif cmd.lower() == "exit":
                    print(f"\n  Sesión finalizada: {self.session_count} comparaciones")
                    print(f"  Total acumulado:    {self.total_prefs} preferencias")
                    if self.total_prefs >= 10:
                        print("\n  Próximos pasos:")
                        print("    python main.py rlhf --train-reward")
                        print("    python main.py rlhf --ppo")
                    break
                elif cmd.lower() == "stats":
                    self.mostrar_estadisticas()
                elif cmd.lower() == "help":
                    print("\n  COMANDOS:")
                    print("    query [texto]  — comparación A/B")
                    print("    stats          — estadísticas")
                    print("    exit           — salir")
                    print("\n  FLUJO RLHF:")
                    print("    1. Haz 30+ comparaciones (query ...)")
                    print("    2. python main.py rlhf --train-reward")
                    print("    3. python main.py rlhf --ppo")
                    print("    4. python main.py experimento")
                elif cmd.lower().startswith("query "):
                    query_text = cmd[6:].strip()
                    if query_text:
                        self.hacer_comparacion(query_text)
                    else:
                        print("  Falta el texto. Ejemplo: query car parts")
                else:
                    # Tratar input directo como query
                    self.hacer_comparacion(cmd)

            except KeyboardInterrupt:
                print("\n  Interrumpido")
                break
            except Exception as e:
                print(f"  Error: {e}")


def main():
    try:
        sistema = SistemaInteractivoAB()
        if sistema.system:
            sistema.ejecutar()
        else:
            print("\n  Ejecuta primero: python main.py init")
    except Exception as e:
        print(f"\nError crítico: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()