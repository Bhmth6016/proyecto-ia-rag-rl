# src/ranking/ner_enhanced_ranker.py
"""
src/ranking/ner_enhanced_ranker.py
====================================
Ranker NER real: usa DeBERTa zero-shot NLI para clasificar el intent
de la query, luego aplica bonus a productos cuyas entidades NER coincidan.

Cambios respecto a versión anterior:
  - _extract_query_intent_aggressive() eliminado (era keyword puro)
  - Toda detección de intent pasa por OptimizedNERExtractor.extract_query_intent()
  - ner_weight reducido a 0.10 para no volcar rankings donde FAISS ya es bueno
  - Bonus scoring con caps por componente para mejor discriminación
  - Full Hybrid: NER como bonus ligero sobre score final, no como reranker previo
"""

from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger(__name__)


class NEREnhancedRanker:

    def __init__(
        self,
        ner_weight: float = 0.10,
        ner_extractor=None,
    ):
        """
        Args:
            ner_weight:     Peso del bonus NER sobre el score FAISS (0.10 recomendado).
            ner_extractor:  Instancia de OptimizedNERExtractor ya inicializada.
                            Si es None, usa keyword fallback interno.
        """
        self.ner_weight = ner_weight
        self.ner_extractor = ner_extractor

        # Cache de intents por query para no llamar al modelo dos veces
        self._intent_cache: Dict[str, Dict[str, List[str]]] = {}

        # Para logging deduplicado
        self._logged_queries: set = set()
        self._logged_bonus: set = set()

        logger.info(
            f"NEREnhancedRanker | weight={ner_weight} | "
            f"extractor={'DeBERTa' if ner_extractor and ner_extractor.use_zero_shot else 'keyword-fallback'}"
        )

    # ── API pública ────────────────────────────────────────────────────────────

    def rank_with_ner(self, products, query, baseline_scores, ner_weight_override=None):
        weight = ner_weight_override if ner_weight_override is not None else self.ner_weight
        query_clean = self._clean_query(query)
        query_intent = self._get_intent(query_clean)
        if not query_intent:
            logger.info(f"   Query: '{query_clean}' → Intent vacío, ranking sin cambios")
            return products
        # Log de intent una sola vez por query
        if query_clean not in self._logged_queries:
            intent_summary = {k: v for k, v in query_intent.items()}
            logger.info(f"   Query: '{query_clean}' → Intent NER: {intent_summary}")
            self._logged_queries.add(query_clean)

        scored = []
        bonus_applied = 0
        MIN_BONUS_TO_RERANK = 0.15
        for i, product in enumerate(products):
            base = baseline_scores[i] if i < len(baseline_scores) else 0.0
            bonus = self._compute_bonus(product, query_clean, query_intent)

            # Solo aplicar si el bonus supera el umbral
            if bonus >= MIN_BONUS_TO_RERANK:
                bonus_applied += 1
                ner_weight = 0.10
                final = (1 - ner_weight) * base + ner_weight * bonus
            else:
                final = base  # mantener score FAISS/PPO sin modificar

            final = max(0.0, min(1.0, final))
            scored.append((product, final, bonus))

        scored.sort(key=lambda x: x[1], reverse=True)

        if query_clean not in self._logged_bonus:
            logger.info(
                f"    NER bonus aplicado a {bonus_applied}/{len(products)} productos"
            )
            self._logged_bonus.add(query_clean)

        return [p for p, _, _ in scored]

    def rank_products(
        self,
        products: List,
        query: str,
        baseline_scores: List[float],
    ) -> List:
        return self.rank_with_ner(products, query, baseline_scores)

    def rank_with_human_preferences(
        self,
        products: List,
        query: str,
        baseline_scores: List[float],
    ) -> List:
        return self.rank_with_ner(products, query, baseline_scores)

    # ── Intent de query ────────────────────────────────────────────────────────

    def _get_intent(self, query: str) -> Dict[str, List[str]]:
        """
        Obtiene el intent NER de la query.
        Usa DeBERTa si el extractor está disponible, keyword fallback si no.
        Cachea el resultado para no repetir llamadas al modelo.
        """
        if query in self._intent_cache:
            return self._intent_cache[query]

        if self.ner_extractor is not None:
            intent = self.ner_extractor.extract_query_intent(query)
        else:
            intent = self._keyword_intent_fallback(query)

        self._intent_cache[query] = intent
        return intent

    def _keyword_intent_fallback(self, query: str) -> Dict[str, List[str]]:
        """
        Fallback puro de keywords cuando no hay extractor NER disponible.
        Cubre el dominio de videojuegos con los términos más frecuentes.
        """
        ql = query.lower()
        intent: Dict[str, List[str]] = {}

        platform_map = {
            "playstation": ["playstation", "ps4", "ps5", "ps3", "ps "],
            "xbox": ["xbox", "xbox 360", "xbox one"],
            "nintendo switch": ["nintendo switch", "switch"],
            "nintendo": ["nintendo", "wii", "gameboy", "3ds"],
            "pc": ["pc gaming", "steam", "windows game"],
        }
        found_platforms = [p for p, kws in platform_map.items() if any(k in ql for k in kws)]
        if found_platforms:
            intent["platform"] = found_platforms

        genre_map = {
            "racing": ["racing", "race"],
            "survival": ["survival", "survive"],
            "horror": ["horror", "survival horror"],
            "first-person shooter": ["fps", "first person shooter"],
            "role-playing game": ["rpg", "role playing"],
            "massively multiplayer online": ["mmorpg", "mmo"],
            "strategy": ["strategy", "tactical", "tower defense"],
            "sports": ["sport", "fifa", "nba", "wwe", "football"],
            "fighting": ["fighting", "smash"],
            "platformer": ["platformer", "platform game"],
            "sandbox": ["sandbox", "open world"],
            "action": ["action"],
            "adventure": ["adventure"],
        }
        found_genres = [g for g, kws in genre_map.items() if any(k in ql for k in kws)]
        if found_genres:
            intent["genre"] = found_genres

        franchise_map = {
            "Mario": ["mario"],
            "Zelda": ["zelda"],
            "Pokemon": ["pokemon", "pokémon"],
            "Minecraft": ["minecraft"],
            "Call of Duty": ["call of duty", "cod"],
            "Halo": ["halo"],
            "God of War": ["god of war"],
            "The Last of Us": ["last of us"],
            "Resident Evil": ["resident evil", "residen evil"],
            "Monster Hunter": ["monster hunter"],
            "Elden Ring": ["elden ring"],
            "Skyrim": ["skyrim"],
            "Red Dead Redemption": ["red dead"],
            "Grand Theft Auto": ["gta", "grand theft auto"],
            "Need for Speed": ["need for speed", "nfs"],
            "Smash Bros": ["smash bros"],
            "Cyberpunk": ["cyberpunk"],
            "The Witcher": ["witcher"],
            "Hollow Knight": ["hollow knight"],
            "Lego": ["lego"],
            "FIFA": ["fifa"],
            "WWE": ["wwe"],
            "Sonic": ["sonic"],
        }
        found_franchises = [f for f, kws in franchise_map.items() if any(k in ql for k in kws)]
        if found_franchises:
            intent["franchise"] = found_franchises

        return intent

    # ── Cálculo de bonus ───────────────────────────────────────────────────────

    def _compute_bonus(
        self,
        product,
        query: str,
        query_intent: Dict[str, List[str]],
    ) -> float:
        """
        Calcula el bonus NER para un producto dado el intent de la query.

        Componentes con cap individual para evitar saturación:
          - intent_score:      match entre entidades de la query y atributos del producto
          - keyword_score:     valores de atributos del producto que aparecen en la query
          - specificity_score: penalización a productos sin atributos NER (menos información)

        Retorna un valor en [0, 1].
        """
        product_attrs: Dict[str, List[str]] = getattr(product, "ner_attributes", {})

        if not product_attrs:
            logger.debug(f"Producto {getattr(product, 'id', '?')[:12]} sin ner_attributes")
            return 0.0

        # ── Intent score: entidades de la query vs atributos del producto ──────
        intent_score = 0.0
        for intent_key, intent_values in query_intent.items():
            if intent_key in product_attrs:
                product_values = product_attrs[intent_key]
                for iv in intent_values:
                    for pv in product_values:
                        if self._fuzzy_match(str(iv), str(pv)):
                            intent_score += 0.20
                            logger.debug(f"Intent match: {iv} ≈ {pv}")
        intent_score = min(0.50, intent_score)  # cap: 0.50

        # ── Keyword score: valores de atributos del producto en la query ───────
        ql = query.lower()
        keyword_score = 0.0
        for attr_values in product_attrs.values():
            for val in attr_values:
                val_lower = str(val).lower()
                if len(val_lower) > 2 and val_lower in ql:
                    keyword_score += 0.15
                    logger.debug(f"Keyword match: '{val_lower}' en query")
        keyword_score = min(0.30, keyword_score)  # cap: 0.30

        # ── Specificity: productos con más atributos NER son más informativos ──
        specificity_score = min(0.10, len(product_attrs) * 0.025)

        total = intent_score + keyword_score + specificity_score

        logger.debug(
            f"Bonus breakdown — intent={intent_score:.2f} "
            f"keyword={keyword_score:.2f} specificity={specificity_score:.2f} "
            f"total={total:.2f}"
        )

        return min(1.0, total)

    # ── Utilidades ─────────────────────────────────────────────────────────────

    def _clean_query(self, query: str) -> str:
        cleaned = query.replace('"', "").replace("'", "").replace("...", "")
        return " ".join(cleaned.split()).strip().lower()

    def _fuzzy_match(self, s1: str, s2: str) -> bool:
        a = s1.lower().strip()
        b = s2.lower().strip()
        if a == b:
            return True
        if a in b or b in a:
            return True
        wa = set(a.split())
        wb = set(b.split())
        if not wa or not wb:
            return False
        return len(wa & wb) / min(len(wa), len(wb)) >= 0.5

    def get_stats(self) -> Dict[str, Any]:
        return {
            "ranker_type": "NER_DeBERTa_v3",
            "ner_weight": self.ner_weight,
            "intent_cache_size": len(self._intent_cache),
            "extractor": (
                "DeBERTa"
                if self.ner_extractor and self.ner_extractor.use_zero_shot
                else "keyword-fallback"
            ),
        }