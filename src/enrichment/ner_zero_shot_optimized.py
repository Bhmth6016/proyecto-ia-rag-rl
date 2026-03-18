# src/enrichment/ner_zero_shot_optimized.py
"""
src/enrichment/ner_zero_shot_optimized.py
==========================================
NER real usando DeBERTa zero-shot NLI para:
  1. Extracción de atributos de productos (título → entidades)
  2. Clasificación de intent de queries (query → tipos de entidad + valores)

El modelo cross-encoder/nli-deberta-v3-small actúa como clasificador NLI:
  "Este texto menciona una {entidad}" → score de confianza
Eso es NER via inferencia de lenguaje natural, no keyword matching.
"""

import pickle
import hashlib
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable, cast

from tqdm import tqdm

logger = logging.getLogger(__name__)

# ── Dominio de videojuegos ─────────────────────────────────────────────────────

# Keywords para filtrar productos que no son de videojuegos
VIDEOGAME_KEYWORDS = [
    'game', 'games', 'gaming', 'playstation', 'xbox', 'nintendo',
    'switch', 'ps4', 'ps5', 'ps3', 'wii', 'gameboy', 'game boy',
    'pc game', 'video game', 'videogame', 'console', 'controller',
    'gameplay', 'rpg', 'fps', 'mmorpg', 'dlc', 'expansion pack',
    'mario', 'zelda', 'pokemon', 'sonic', 'halo', 'minecraft',
    'call of duty', 'cod', 'gta', 'grand theft auto',
    'playthrough', 'walkthrough', 'achievement', 'quest',
    'level', 'dungeon', 'raid', 'multiplayer', 'co-op',
    'steam', 'origin', 'ubisoft', 'activision', 'capcom',
    'bandai', 'namco', 'square enix', 'bethesda', 'rockstar'
]

PLATFORMS = [
    "PlayStation 4", "PlayStation 5", "PlayStation 3",
    "Xbox One", "Xbox 360", "Xbox Series X",
    "Nintendo Switch", "Nintendo Wii", "Nintendo 3DS", "Game Boy",
    "PC", "Steam",
]

GENRES = [
    "action", "adventure", "role-playing game", "strategy", "sports",
    "racing", "survival", "horror", "first-person shooter", "platformer",
    "fighting", "sandbox", "simulation", "puzzle", "tower defense",
    "massively multiplayer online", "roguelike", "metroidvania",
    "tactical", "stealth", "rhythm",
]

FRANCHISES = [
    "Mario", "Zelda", "Pokemon", "Sonic", "Minecraft", "Fortnite",
    "Call of Duty", "Halo", "God of War", "The Last of Us",
    "Resident Evil", "Monster Hunter", "Elden Ring", "Skyrim",
    "Red Dead Redemption", "Grand Theft Auto", "Need for Speed",
    "Smash Bros", "Cyberpunk", "The Witcher", "FIFA", "WWE",
    "Lego", "Hollow Knight", "Metroid", "Donkey Kong", "Kirby",
    "Final Fantasy", "Dragon Quest", "Dark Souls", "Bloodborne",
    "Assassin's Creed", "Far Cry", "Tom Clancy",
]

FEATURES = [
    "multiplayer", "single-player", "co-op", "online multiplayer",
    "local multiplayer", "open world", "story-driven",
]

# Abreviaciones de dominio → texto normalizado para el modelo
DOMAIN_ALIASES: Dict[str, str] = {
    "ps4": "PlayStation 4",
    "ps5": "PlayStation 5",
    "ps3": "PlayStation 3",
    "ps ": "PlayStation ",
    "xbox 360": "Xbox 360",
    "xbox one": "Xbox One",
    "cod": "Call of Duty",
    "gta": "Grand Theft Auto",
    "nfs": "Need for Speed",
    "fps": "first-person shooter",
    "rpg": "role-playing game",
    "mmorpg": "massively multiplayer online role-playing game",
    "mmo": "massively multiplayer online",
    "rdr": "Red Dead Redemption",
    "nds": "Nintendo DS",
    "3ds": "Nintendo 3DS",
    "wii u": "Nintendo Wii U",
    "smash bros": "Nintendo Switch fighting game",
    "smash brothers": "Nintendo Switch fighting game", 
    "need for speed": "Need for Speed racing game",
    "the last of us": "The Last of Us PlayStation game",
    "god of war": "God of War PlayStation game",
}


class OptimizedNERExtractor:
    """
    Extractor NER basado en DeBERTa zero-shot NLI.

    Para productos: detecta plataforma, género, franquicia y features
                    en el título del producto.
    Para queries:   clasifica qué tipo de entidad menciona la query
                    y extrae los valores concretos.
    """

    def __init__(
        self,
        use_zero_shot: bool = True,
        model_name: str = "cross-encoder/nli-deberta-v3-small",
    ):
        self.use_zero_shot = use_zero_shot
        self.model_name = model_name
        self.classifier: Optional[Callable[..., Any]] = None
        self._init_error: Optional[str] = None

        if use_zero_shot:
            self._initialize_model(model_name)

        logger.info(
            f"NERExtractor listo | zero-shot={self.use_zero_shot} | "
            f"modelo={model_name if self.use_zero_shot else 'keyword-fallback'}"
        )

    # ── Inicialización ─────────────────────────────────────────────────────────

    def _initialize_model(self, model_name: str) -> None:
        try:
            import torch
            from transformers import pipeline

            device = 0 if torch.cuda.is_available() else -1
            if device == -1:
                logger.warning("Sin GPU: NER zero-shot será lento")

            self.classifier = pipeline(
                "zero-shot-classification",
                model=model_name,
                device=device,
                batch_size=4 if device == 0 else 2,
                framework="pt",
            )
            logger.info(f"DeBERTa cargado en {'GPU' if device == 0 else 'CPU'}")

        except Exception as e:
            self._init_error = str(e)
            logger.warning(f"Error cargando DeBERTa: {e} → usando keyword fallback")
            self.use_zero_shot = False

    # ── Filtro de dominio ─────────────────────────────────────────────────────

    def _is_videogame_product(self, title: str) -> bool:
        """Verifica si el título corresponde a un producto de videojuegos."""
        tl = title.lower()
        return any(kw in tl for kw in VIDEOGAME_KEYWORDS)

    # ── API pública ────────────────────────────────────────────────────────────

    def extract_attributes(self, title: str, category: str = "") -> Dict[str, List[str]]:
        """Extrae entidades de un título de producto."""
        if not title or len(title.strip()) < 3:
            return {}

        # Filtro de dominio: solo procesar productos de videojuegos
        if not self._is_videogame_product(title):
            return {}

        normalized = self._normalize(title)

        if self.use_zero_shot and self.classifier:
            try:
                return self._extract_product_entities_deberta(normalized)
            except Exception as e:
                logger.debug(f"DeBERTa falló en producto: {e}")

        return self._extract_product_entities_keywords(normalized)

    def extract_query_intent(self, query: str) -> Dict[str, List[str]]:
        """
        Clasifica el intent de una query usando DeBERTa NLI.
        Retorna un dict con las entidades detectadas y sus valores.
        """
        if not query or len(query.strip()) < 2:
            return {}

        normalized = self._normalize(query)

        if self.use_zero_shot and self.classifier:
            try:
                return self._extract_query_entities_deberta(normalized)
            except Exception as e:
                logger.debug(f"DeBERTa falló en query: {e}")

        return self._extract_query_entities_keywords(normalized)

    # ── Extracción de productos con DeBERTa ───────────────────────────────────

    def _extract_product_entities_deberta(self, title: str) -> Dict[str, List[str]]:
        assert self.classifier is not None
        result: Dict[str, List[str]] = {}

        # 1. Plataforma
        platforms = self._classify(
            title, PLATFORMS,
            template="This product is for {}.",
            threshold=0.65, top_k=2,
        )
        if platforms:
            result["platform"] = platforms

        # 2. Género
        genres = self._classify(
            title, GENRES,
            template="This is a {} game.",
            threshold=0.60, top_k=2,
        )
        if genres:
            result["genre"] = genres

        # 3. Franquicia
        franchises = self._classify(
            title, FRANCHISES,
            template="This product belongs to the {} franchise.",
            threshold=0.70, top_k=1,
        )
        if franchises:
            result["franchise"] = franchises

        # 4. Features
        features = self._classify(
            title, FEATURES,
            template="This game supports {}.",
            threshold=0.65, top_k=2,
        )
        if features:
            result["features"] = features

        return result

    # ── Extracción de query intent con DeBERTa ────────────────────────────────

    def _extract_query_entities_deberta(self, query: str) -> Dict[str, List[str]]:
        assert self.classifier is not None
        result: Dict[str, List[str]] = {}

        # Paso 1: detectar qué TIPOS de entidad menciona la query
        entity_types = self._classify(
            query,
            candidate_labels=[
                "specific game title",
                "gaming platform or console",
                "game genre or category",
                "game franchise or series",
                "game feature or mode",
                "game developer or publisher",
            ],
            template="This search query is looking for a {}.",
            threshold=0.50,
            top_k=4,
            multi_label=True,
        )

        logger.debug(f"Query '{query}' → tipos detectados: {entity_types}")

        # Paso 2: para cada tipo, extraer el valor concreto

        if "gaming platform or console" in entity_types:
            platforms = self._classify(
                query, PLATFORMS,
                template="This query mentions {}.",
                threshold=0.70, top_k=1,
            )
            if platforms:
                result["platform"] = platforms

        if "game genre or category" in entity_types:
            genres = self._classify(
                query, GENRES,
                template="This query is about {} games.",
                threshold=0.55, top_k=3,
            )
            if genres:
                result["genre"] = genres

        if "game franchise or series" in entity_types:
            franchises = self._classify(
                query,          
                FRANCHISES,
                template="This query is about the {} franchise.",
                threshold=0.75,
                top_k=1,
            )
            if franchises:
                result["franchise"] = franchises

        if "game feature or mode" in entity_types:
            features = self._classify(
                query, FEATURES,
                template="This query mentions {} gameplay.",
                threshold=0.60, top_k=2,
            )
            if features:
                result["features"] = features

        if "specific game title" in entity_types and not result:
            # Query es probablemente un título específico
            result["title_search"] = [query]

        return result

    # ── Clasificador base ─────────────────────────────────────────────────────

    def _classify(
        self,
        text: str,
        candidate_labels: List[str],
        template: str,
        threshold: float,
        top_k: int,
        multi_label: bool = True,
    ) -> List[str]:
        """Llama al pipeline NLI y filtra por threshold."""
        assert self.classifier is not None

        # DeBERTa tiene límite de candidatos por llamada
        all_selected: List[str] = []

        for chunk_start in range(0, len(candidate_labels), 20):
            chunk = candidate_labels[chunk_start: chunk_start + 20]
            try:
                out = cast(
                    Dict[str, Any],
                    self.classifier(
                        text,
                        candidate_labels=chunk,
                        multi_label=multi_label,
                        hypothesis_template=template,
                    ),
                )
                selected = [
                    label
                    for label, score in zip(out["labels"], out["scores"])
                    if score >= threshold
                ]
                all_selected.extend(selected)
            except Exception as e:
                logger.debug(f"Chunk NLI falló: {e}")
                continue

        return all_selected[:top_k]

    # ── Keyword fallbacks ─────────────────────────────────────────────────────

    def _extract_product_entities_keywords(self, title: str) -> Dict[str, List[str]]:
        """Fallback keyword para productos cuando DeBERTa no está disponible."""
        tl = title.lower()
        result: Dict[str, List[str]] = {}

        platforms = [p for p in PLATFORMS if p.lower() in tl]
        if platforms:
            result["platform"] = platforms[:2]

        genres = [g for g in GENRES if g.lower() in tl]
        if genres:
            result["genre"] = genres[:2]

        franchises = [f for f in FRANCHISES if f.lower() in tl]
        if franchises:
            result["franchise"] = franchises[:1]

        features = [f for f in FEATURES if f.lower() in tl]
        if features:
            result["features"] = features[:2]

        return result

    def _extract_query_entities_keywords(self, query: str) -> Dict[str, List[str]]:
        """Fallback keyword para queries cuando DeBERTa no está disponible."""
        ql = query.lower()
        result: Dict[str, List[str]] = {}

        platforms = [p for p in PLATFORMS if p.lower() in ql]
        if platforms:
            result["platform"] = platforms[:2]

        genres = [g for g in GENRES if g.lower() in ql]
        if genres:
            result["genre"] = genres[:2]

        franchises = [f for f in FRANCHISES if f.lower() in ql]
        if franchises:
            result["franchise"] = franchises[:2]

        return result

    # ── Normalización ─────────────────────────────────────────────────────────

    def _normalize(self, text: str) -> str:
        """Expande abreviaciones de dominio antes de pasar al modelo."""
        normalized = text.strip()
        normalized_lower = normalized.lower()
        for alias, full in DOMAIN_ALIASES.items():
            if alias in normalized_lower:
                normalized_lower = normalized_lower.replace(alias, full.lower())
        return normalized_lower

    # ── Enriquecimiento batch de productos ────────────────────────────────────

    def enrich_products_batch(
        self,
        products: List,
        batch_size: int = 500,
        cache_path: Optional[str] = None,
    ) -> List:
        if cache_path and Path(cache_path).exists():
            logger.info(f"Cargando NER cache: {cache_path}")
            return self._load_from_cache(products, cache_path)

        total = len(products)
        logger.info(f"Enriqueciendo {total:,} productos con NER...")

        enriched = []
        stats = {"deberta": 0, "keywords": 0, "empty": 0, "errors": 0, "filtered": 0}

        for i in tqdm(range(0, total, batch_size), desc="NER"):
            batch = products[i: i + batch_size]
            for product in batch:
                try:
                    title = getattr(product, "title", "") or ""
                    category = getattr(product, "category", "") or ""

                    if title:
                        # Verificar si es producto de videojuegos
                        if not self._is_videogame_product(title):
                            stats["filtered"] += 1
                            product.ner_attributes = {}
                            product.enriched_text = title
                            enriched.append(product)
                            continue

                        attrs = self.extract_attributes(title, category)
                        if attrs:
                            if self.use_zero_shot:
                                stats["deberta"] += 1
                            else:
                                stats["keywords"] += 1
                        else:
                            stats["empty"] += 1

                        product.ner_attributes = attrs
                        product.enriched_text = self._build_enriched_text(title, attrs)
                    else:
                        product.ner_attributes = {}
                        product.enriched_text = ""
                        stats["empty"] += 1

                    enriched.append(product)

                except Exception as e:
                    stats["errors"] += 1
                    product.ner_attributes = {}
                    product.enriched_text = getattr(product, "title", "")
                    enriched.append(product)

        logger.info(
            f"NER completado: deberta={stats['deberta']:,} | "
            f"keywords={stats['keywords']:,} | "
            f"empty={stats['empty']:,} | filtered={stats['filtered']:,} | "
            f"errors={stats['errors']:,}"
        )

        if cache_path:
            self._save_to_cache(enriched, cache_path)

        return enriched

    def _build_enriched_text(self, title: str, attributes: Dict) -> str:
        if not attributes:
            return title
        parts = [title]
        for attr_type, values in attributes.items():
            if values:
                parts.append(f"{attr_type}:{','.join(str(v) for v in values[:2])}")
        return " | ".join(parts)

    # ── Cache ─────────────────────────────────────────────────────────────────

    def _save_to_cache(self, products: List, cache_path: str) -> None:
        try:
            data = [
                {
                    "id": getattr(p, "id", ""),
                    "ner_attributes": getattr(p, "ner_attributes", {}),
                    "enriched_text": getattr(p, "enriched_text", ""),
                    "title_hash": hashlib.md5(
                        str(getattr(p, "title", "")).encode()
                    ).hexdigest()[:8],
                }
                for p in products
            ]
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, "wb") as f:
                pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
            logger.info(f"Cache guardado: {len(data):,} productos → {cache_path}")
        except Exception as e:
            logger.error(f"Error guardando cache: {e}")

    def _load_from_cache(self, products: List, cache_path: str) -> List:
        try:
            with open(cache_path, "rb") as f:
                cache_data = pickle.load(f)
            cache_dict = {item["id"]: item for item in cache_data}
            loaded = 0
            for product in products:
                pid = getattr(product, "id", "")
                if pid in cache_dict:
                    entry = cache_dict[pid]
                    current_hash = hashlib.md5(
                        str(getattr(product, "title", "")).encode()
                    ).hexdigest()[:8]
                    if entry.get("title_hash") == current_hash:
                        product.ner_attributes = entry.get("ner_attributes", {})
                        product.enriched_text = entry.get(
                            "enriched_text", getattr(product, "title", "")
                        )
                        loaded += 1
                    else:
                        product.ner_attributes = {}
                        product.enriched_text = getattr(product, "title", "")
            logger.info(f"Cache cargado: {loaded:,}/{len(products):,}")
            return products
        except Exception as e:
            logger.error(f"Error cargando cache: {e}")
            return products


# ── Función de conveniencia ────────────────────────────────────────────────────

def enrich_dataset_with_ner(
    products: List,
    use_zero_shot: bool = True,
    cache_file: str = "data/cache/ner_attributes.pkl",
) -> List:
    extractor = OptimizedNERExtractor(use_zero_shot=use_zero_shot)
    return extractor.enrich_products_batch(
        products,
        batch_size=500,
        cache_path=cache_file,
    )