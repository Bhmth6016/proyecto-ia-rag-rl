# src/unified_system_v2.py
import pickle
import json
import os
import numpy as np
import random  # <-- AÑADIDO: importar random
from pathlib import Path
from typing import Dict, List, Any, TYPE_CHECKING, Optional
import logging
import time
import sys

current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)

if TYPE_CHECKING:
    from unified_system import UnifiedRAGRLSystem
else:
    from unified_system import UnifiedRAGRLSystem

if TYPE_CHECKING:
    NER_available: bool = True
    from enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
    from ranking.ner_enhanced_ranker import NEREnhancedRanker
else:
    NER_available = False

    try:
        from enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
        from ranking.ner_enhanced_ranker import NEREnhancedRanker
        NER_available = True
    except ImportError as e:
        logger.error(f"No se pudieron importar módulos NER: {e}")
        OptimizedNERExtractor = None
        NEREnhancedRanker = None


class UnifiedSystemV2(UnifiedRAGRLSystem):

    def __init__(self, config_path: str = "config/config.yaml"):
        super().__init__(config_path)

        self.ner_extractor = None
        self.ner_ranker = None

        # rl_ranker eliminado — era re-ranking heurístico, no RLHF real
        # El RLHF real vive en self.rlhf_pipeline (inyectado por rlhf_integration.py)

        self.ner_cache_path = Path("data/cache/ner_attributes_v2.pkl")
        self.system_cache_path = Path("data/cache/unified_system_v2.pkl")

        self.available_methods = ['baseline', 'ner_enhanced', 'reward_only', 'rlhf', 'full_hybrid']

        logger.info(" UnifiedSystemV2 inicializado")

    def initialize_with_ner(
            self,
            limit: Optional[int] = None,
            use_cache: bool = True,
            use_zero_shot: bool = False,
            ner_sample_size: int = 10000  # <-- AÑADIDO: tamaño de muestra para NER
        ) -> bool:
        logger.info(" Inicializando sistema V2 con NER...")

        if not self.canonical_products:
            logger.info(" Cargando sistema base...")

            base_cache = Path("data/cache/unified_system.pkl")
            if base_cache.exists():
                try:
                    with open(base_cache, 'rb') as f:
                        base_system = pickle.load(f)
                    self.canonical_products = base_system.canonical_products
                    self.canonicalizer = base_system.canonicalizer
                    self.vector_store = base_system.vector_store
                    logger.info(f"   Cargados {len(self.canonical_products):,} productos")
                except Exception as e:
                    logger.error(f"Error cargando caché base: {e}")
                    return False
            else:
                if not self.initialize_from_raw_all_files(limit=limit or 5000):
                    logger.error(" Error inicializando sistema base")
                    return False

        if NER_available and OptimizedNERExtractor is not None:
            logger.info(" Configurando NER...")
            self.ner_extractor = OptimizedNERExtractor(use_zero_shot=use_zero_shot)

            if use_cache and self.ner_cache_path.exists():
                try:
                    with open(self.ner_cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # cache_data es una LISTA de dicts, hay que convertirla a dict keyed por id
                    cache_dict = {item["id"]: item for item in cache_data}
                    
                    loaded = 0
                    for product in self.canonical_products:
                        pid = getattr(product, 'id', '')
                        if pid and pid in cache_dict:
                            entry = cache_dict[pid]
                            product.ner_attributes = entry.get('ner_attributes', {})
                            product.enriched_text = entry.get('enriched_text', '')
                            loaded += 1
                    
                    logger.info(f"   Cache NER aplicado: {loaded:,}/{len(self.canonical_products):,} productos")
                except Exception as e:
                    logger.error(f"Error cargando cache NER: {e}")
            else:
                try:
                    # --- MODIFICACIÓN: usar random.sample para seleccionar subconjunto ---
                    num_products = len(self.canonical_products)
                    sample_size = min(ner_sample_size, num_products)
                    
                    logger.info(f" Seleccionando muestra aleatoria de {sample_size} productos para NER...")
                    subset = random.sample(self.canonical_products, sample_size)
                    
                    # Aplicar NER solo al subconjunto
                    logger.info(f" Aplicando NER a {len(subset)} productos...")
                    enriched_subset = self.ner_extractor.enrich_products_batch(
                        subset,
                        batch_size=100,
                        cache_path=str(self.ner_cache_path)
                    )
                    
                    # Actualizar los productos originales con los resultados NER
                    enriched_dict = {p.id: p for p in enriched_subset}
                    for i, product in enumerate(self.canonical_products):
                        if hasattr(product, 'id') and product.id in enriched_dict:
                            enriched_product = enriched_dict[product.id]
                            product.ner_attributes = enriched_product.ner_attributes
                            product.enriched_text = enriched_product.enriched_text
                    
                    logger.info(f" NER completado para {len(enriched_subset)} productos")
                    
                except Exception as e:
                    logger.error(f"Error procesando NER: {e}")

        if NER_available and NEREnhancedRanker is not None:
            self.ner_ranker = NEREnhancedRanker(
                ner_weight=0.10,
                ner_extractor=self.ner_extractor
            )
        else:
            self.ner_ranker = None

        logger.info(" Sistema V2 inicializado")
        logger.info(f"   • Productos totales: {len(self.canonical_products):,}")
        logger.info(f"   • Productos con NER: {ner_sample_size if NER_available else 0}")
        logger.info(f"   • Métodos: {', '.join(self.available_methods)}")

        return True

    def query_four_methods(self, query_text: str, k: int = 10) -> Dict[str, Any]:
        results = {
            'query': query_text,
            'methods': {},
            'scores': {},
            'timing': {},
            'metadata': {}
        }

        try:
            start_time = time.time()
            baseline_results = self._process_query_baseline(query_text, k)
            results['methods']['baseline'] = baseline_results
            results['timing']['baseline'] = time.time() - start_time
            results['scores']['baseline'] = self._calculate_method_score(baseline_results, query_text)

            start_time = time.time()
            ner_results = self._method_ner_enhanced(query_text, k)
            results['methods']['ner_enhanced'] = ner_results
            results['timing']['ner_enhanced'] = time.time() - start_time
            results['scores']['ner_enhanced'] = self._calculate_method_score(ner_results, query_text)

            start_time = time.time()
            reward_only_results = self._method_reward_only(query_text, k)
            results['methods']['reward_only'] = reward_only_results
            results['timing']['reward_only'] = time.time() - start_time
            results['scores']['reward_only'] = self._calculate_method_score(reward_only_results, query_text)

            start_time = time.time()
            rlhf_results = self._method_rlhf(query_text, k)
            results['methods']['rlhf'] = rlhf_results
            results['timing']['rlhf'] = time.time() - start_time
            results['scores']['rlhf'] = self._calculate_method_score(rlhf_results, query_text)

            start_time = time.time()
            hybrid_results = self._method_full_hybrid(query_text, k)
            results['methods']['full_hybrid'] = hybrid_results
            results['timing']['full_hybrid'] = time.time() - start_time
            results['scores']['full_hybrid'] = self._calculate_method_score(hybrid_results, query_text)

            results['metadata'] = {
                'k': k,
                'timestamp': time.time(),
                'has_ner': self.ner_ranker is not None,
                'has_rlhf': hasattr(self, 'rlhf_pipeline') and getattr(
                    self.rlhf_pipeline, 'policy_trained', False
                )
            }

        except Exception as e:
            logger.error(f" Error en query_four_methods: {e}")
            baseline_results = self._process_query_baseline(query_text, k)
            for method in self.available_methods:
                results['methods'][method] = baseline_results
                results['scores'][method] = 0.5
                results['timing'][method] = 0.0

        return results

    def _process_query_baseline(self, query_text: str, k: int) -> List:
        try:
            if self.vector_store and self.canonicalizer:
                query_embedding = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                return self.vector_store.search(query_embedding, k=k)
            return []
        except Exception as e:
            logger.error(f"Error en baseline: {e}")
            return []

    def _method_ner_enhanced(self, query_text: str, k: int) -> List:
        if not self.ner_ranker:
            return self._process_query_baseline(query_text, k)

        try:
            baseline_results = self._process_query_baseline(query_text, k * 2)
            if not baseline_results:
                return []

            query_embedding = self.canonicalizer.embedding_model.encode(
                query_text, normalize_embeddings=True
            )
            baseline_scores = []
            for product in baseline_results:
                if hasattr(product, 'content_embedding'):
                    p = product.content_embedding / np.linalg.norm(product.content_embedding)
                    q = query_embedding / np.linalg.norm(query_embedding)
                    baseline_scores.append(float(np.dot(q, p)))
                else:
                    baseline_scores.append(0.0)

            ranked = self.ner_ranker.rank_with_ner(baseline_results, query_text, baseline_scores)
            return ranked[:k]

        except Exception as e:
            logger.error(f"Error en NER-enhanced: {e}")
            return self._process_query_baseline(query_text, k)

    def _method_reward_only(self, query_text: str, k: int) -> list:
            """
            Reranking con PointwiseRewardModel, sin PPO.

            Retrieval:  FAISS top-k*2
            Reranking:  reward model puntúa cada candidato individualmente
                        y reordena. Si el reward no está entrenado → baseline.
            """
            import torch
            import numpy as np

            if not hasattr(self, 'rlhf_pipeline') or self.rlhf_pipeline is None:
                return self._process_query_baseline(query_text, k)

            pipeline = self.rlhf_pipeline
            if not pipeline.reward_trained:
                logger.debug("reward_only: reward no entrenado — usando baseline")
                return self._process_query_baseline(query_text, k)

            try:
                q_emb = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                candidates = self.vector_store.search(q_emb, k=k * 2)
                if not candidates:
                    return []

                top_cands = candidates[:pipeline.top_k]
                prod_embs = pipeline._products_to_embs(top_cands)
                if prod_embs is None:
                    return candidates[:k]

                # q_t: [emb_dim]  →  expandir a [1, emb_dim] para el modelo pointwise
                q_t = torch.tensor(q_emb, dtype=torch.float32, device=pipeline.device)

                pipeline.reward_model.eval()
                scores = []
                with torch.no_grad():
                    for i in range(prod_embs.size(0)):
                        # p_emb: [1, emb_dim]  (2D — lo que espera PointwiseRewardModel)
                        p_emb = prod_embs[i:i+1]
                        r     = pipeline.reward_model(q_t.unsqueeze(0), p_emb)
                        scores.append(r.item())

                order = np.argsort(scores)[::-1]
                return [top_cands[j] for j in order if j < len(top_cands)][:k]

            except Exception as e:
                logger.error(f"Error en reward_only: {e}")
                return self._process_query_baseline(query_text, k)

    def _method_rlhf(self, query_text: str, k: int) -> List:
        """
        RLHF real: usa RLHFPipeline (PolicyModel + PPO).
        Si no está entrenado, devuelve baseline — sin fallback al ranker falso.
        """
        if hasattr(self, 'rlhf_pipeline') and self.rlhf_pipeline is not None:
            pipeline = self.rlhf_pipeline
            if pipeline.policy_trained:
                products, query_emb_np, _ = pipeline.retrieve_candidates(query_text, k=k * 2)
                if products:
                    return pipeline.rank_products(query_text, products, query_emb_np)[:k]

        # Si policy no está entrenada: baseline puro, sin heurística
        logger.debug("Policy RLHF no entrenada — usando baseline")
        return self._process_query_baseline(query_text, k)

    def _method_full_hybrid(self, query: str, top_k: int = 10):

        # 1️⃣ Retrieval base FAISS
        candidates, faiss_scores = self._get_candidates_with_scores(
            query,
            top_k=40
        )

        # 2️⃣ PPO reranking sobre distribución original
        if self.rlhf_pipeline and self.rlhf_pipeline.policy_trained:
            candidates = self.rlhf_pipeline.rerank_with_policy(
                candidates,
                query
            )

            policy_scores = [
                1.0 - (i / len(candidates))
                for i in range(len(candidates))
            ]
        else:
            policy_scores = faiss_scores

        # 3️⃣ Boost NER ligero (NO rerank agresivo)
        reranked = self.ner_ranker.rank_with_ner(
            candidates, 
            query, 
            policy_scores,
            ner_weight_override=0.05  # ← override solo para este método
        )

        return reranked[:top_k]

    def _calculate_method_score(self, products: List, query: str) -> float:
        if not products:
            return 0.0
        rating_total = 0.0
        rating_count = 0
        for product in products[:10]:
            if hasattr(product, 'rating') and product.rating:
                try:
                    rating_total += float(product.rating) / 5.0
                    rating_count += 1
                except (ValueError, TypeError):
                    continue
        score = (rating_total / rating_count * 0.7) if rating_count > 0 else 0.0
        titles = [getattr(p, 'title', '') for p in products[:10]]
        if titles:
            score += (len(set(titles)) / len(titles)) * 0.3
        return min(1.0, score)

    def save_to_cache(self, cache_path: Optional[str] = None) -> bool:
        if cache_path is None:
            cache_path = str(self.system_cache_path)
        try:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(cache_file, 'wb') as f:
                pickle.dump(self, f)
            logger.info(f" Sistema V2 guardado: {cache_file}")
            return True
        except Exception as e:
            logger.error(f" Error guardando caché: {e}")
            return False

    def get_system_stats(self):
        stats = super().get_system_stats()
        stats.update({
            'has_ner_ranker': self.ner_ranker is not None,
            'has_ner_extractor': self.ner_extractor is not None,
            'ner_enriched_count': sum(
                1 for p in self.canonical_products
                if hasattr(p, 'ner_attributes') and p.ner_attributes
            ),
            'available_methods': self.available_methods,
            'has_rlhf_pipeline': hasattr(self, 'rlhf_pipeline') and self.rlhf_pipeline is not None,
            'rlhf_policy_trained': getattr(
                getattr(self, 'rlhf_pipeline', None), 'policy_trained', False
            )
        })
        return stats

    @classmethod
    def load_from_cache(cls, cache_path: str = "data/cache/unified_system_v2.pkl"):
        if not os.path.exists(cache_path):
            logger.warning(f"  Cache no encontrado: {cache_path}")
            return None
        try:
            with open(cache_path, 'rb') as f:
                system = pickle.load(f)
            logger.info(f" Sistema V2 cargado: {len(system.canonical_products):,} productos")
            return system
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f" Error cargando cache: {e}")
            try:
                os.remove(cache_path)
            except OSError:
                pass
            return None