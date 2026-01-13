# src/unified_system_v2.py
"""
Sistema Unificado V2 - Extensión para 4 métodos
VERSIÓN CORREGIDA - Fix de error NEREnhancedRanker.rank_with_ner(k=...)
"""
import pickle
import json
import os
import numpy as np
from pathlib import Path
from typing import Dict, List, Any, Optional
import logging
import time
import sys

# Añadir el directorio src al path para importaciones
current_dir = Path(__file__).parent
project_root = current_dir.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(current_dir))

logger = logging.getLogger(__name__)

# Intentar importar sistema base
try:
    from unified_system import UnifiedRAGRLSystem
except ImportError as e:
    logger.warning(f"No se pudo importar UnifiedRAGRLSystem: {e}")
    # Crear clase base simple
    class UnifiedRAGRLSystem:
        def __init__(self, config_path: str = "config/config.yaml"):
            self.canonical_products = []
            self.canonicalizer = None
            self.vector_store = None
            self.rl_ranker = None
            self.config = {}

# Intentar importar componentes
try:
    from ranking.rl_ranker_fixed import RLHFRankerFixed
    RLHFRankerFixed_available = True
except ImportError:
    logger.warning("RLHFRankerFixed no disponible")
    RLHFRankerFixed_available = False

try:
    from enrichment.ner_zero_shot_optimized import OptimizedNERExtractor
    from ranking.ner_enhanced_ranker import NEREnhancedRanker
    NER_available = True
except ImportError as e:
    logger.error(f"No se pudieron importar módulos NER: {e}")
    NER_available = False

class UnifiedSystemV2(UnifiedRAGRLSystem):
    """
    Extensión V2 con 4 métodos de ranking
    """
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """Inicializa heredando del sistema base"""
        super().__init__(config_path)
        
        # Componentes adicionales para V2
        self.ner_extractor = None
        self.ner_ranker = None
        
        # Cache paths
        self.ner_cache_path = Path("data/cache/ner_attributes_v2.pkl")
        self.system_cache_path = Path("data/cache/unified_system_v2.pkl")
        
        # Métodos disponibles
        self.available_methods = ['baseline', 'ner_enhanced', 'rlhf', 'full_hybrid']
        
        logger.info("✅ UnifiedSystemV2 inicializado")
    
    def initialize_with_ner(self, limit: int = None, 
                          use_cache: bool = True,
                          use_zero_shot: bool = False) -> bool:
        """
        Inicializa sistema completo con NER
        """
        logger.info("🚀 Inicializando sistema V2 con NER...")
        
        # 1. Verificar si ya tenemos sistema base
        if not self.canonical_products:
            logger.info("📥 Cargando sistema base...")
            
            # Intentar cargar desde caché base primero
            base_cache = Path("data/cache/unified_system.pkl")
            if base_cache.exists():
                logger.info("   Desde caché base...")
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
                logger.info("   Creando nuevo sistema base...")
                if not self.initialize_from_raw_all_files(limit=limit or 10000):
                    logger.error("❌ Error inicializando sistema base")
                    return False
        
        logger.info(f"   • Productos base: {len(self.canonical_products):,}")
        
        # 2. Inicializar extractor NER si está disponible
        if NER_available:
            logger.info("🎯 Configurando NER...")
            self.ner_extractor = OptimizedNERExtractor(use_zero_shot=use_zero_shot)
            
            # 3. Enriquecer productos con NER
            if use_cache and self.ner_cache_path.exists():
                logger.info("📂 Cargando atributos NER desde cache...")
                try:
                    with open(self.ner_cache_path, 'rb') as f:
                        cache_data = pickle.load(f)
                    
                    # Aplicar cache a productos
                    for product in self.canonical_products:
                        if hasattr(product, 'id'):
                            product_id = product.id
                            if product_id in cache_data:
                                product.ner_attributes = cache_data[product_id].get('ner_attributes', {})
                                product.enriched_text = cache_data[product_id].get('enriched_text', '')
                    logger.info("   Cache NER aplicado")
                except Exception as e:
                    logger.error(f"Error cargando cache NER: {e}")
                    self.ner_extractor = None
            else:
                logger.info("🔨 Procesando NER (primera vez, puede tardar)...")
                try:
                    # Procesar solo primeros 1000 productos para prueba
                    sample_size = min(1000, len(self.canonical_products))
                    sample_products = self.canonical_products[:sample_size]
                    
                    enriched = self.ner_extractor.enrich_products_batch(
                        sample_products,
                        batch_size=100,
                        cache_path=str(self.ner_cache_path)
                    )
                    
                    # Actualizar productos
                    for i, product in enumerate(enriched):
                        if i < len(self.canonical_products):
                            self.canonical_products[i] = product
                    
                    logger.info(f"   NER aplicado a {sample_size} productos")
                except Exception as e:
                    logger.error(f"Error procesando NER: {e}")
                    self.ner_extractor = None
        
        # 4. Inicializar rankers
        logger.info("⚙️  Inicializando rankers...")
        
        if NER_available:
            self.ner_ranker = NEREnhancedRanker(ner_weight=0.15)
        else:
            self.ner_ranker = None
            logger.warning("NER ranker no disponible")
        
        # RLHF se inicializará después con entrenamiento
        self.rl_ranker = None
        
        # 5. Verificar integridad
        ner_enriched = sum(1 for p in self.canonical_products 
                         if hasattr(p, 'ner_attributes') and p.ner_attributes) if self.ner_extractor else 0
        
        logger.info(f"✅ Sistema V2 inicializado exitosamente")
        logger.info(f"   • Productos totales: {len(self.canonical_products):,}")
        if self.ner_extractor:
            logger.info(f"   • Con atributos NER: {ner_enriched:,} ({ner_enriched/len(self.canonical_products)*100:.1f}%)")
        logger.info(f"   • Métodos disponibles: {', '.join(self.available_methods)}")
        
        return True
    
    def query_four_methods(self, query_text: str, k: int = 10) -> Dict[str, Any]:
        """
        Ejecuta los 4 métodos y devuelve resultados comparativos
        """
        results = {
            'query': query_text,
            'methods': {},
            'scores': {},
            'timing': {},
            'metadata': {}
        }
        
        try:
            # 1. Baseline
            start_time = time.time()
            baseline_results = self._process_query_baseline(query_text, k)
            results['methods']['baseline'] = baseline_results
            results['timing']['baseline'] = time.time() - start_time
            results['scores']['baseline'] = self._calculate_method_score(baseline_results, query_text)
            
            # 2. NER-Enhanced (si está disponible)
            start_time = time.time()
            ner_results = self._method_ner_enhanced(query_text, k)
            results['methods']['ner_enhanced'] = ner_results
            results['timing']['ner_enhanced'] = time.time() - start_time
            results['scores']['ner_enhanced'] = self._calculate_method_score(ner_results, query_text)
            
            # 3. RLHF (si está disponible)
            start_time = time.time()
            rlhf_results = self._method_rlhf(query_text, k)
            results['methods']['rlhf'] = rlhf_results
            results['timing']['rlhf'] = time.time() - start_time
            results['scores']['rlhf'] = self._calculate_method_score(rlhf_results, query_text)
            
            # 4. Full Hybrid
            start_time = time.time()
            hybrid_results = self._method_full_hybrid(query_text, k)
            results['methods']['full_hybrid'] = hybrid_results
            results['timing']['full_hybrid'] = time.time() - start_time
            results['scores']['full_hybrid'] = self._calculate_method_score(hybrid_results, query_text)
            
            # Metadata
            results['metadata'] = {
                'k': k,
                'timestamp': time.time(),
                'has_ner': self.ner_ranker is not None,
                'has_rlhf': self.rl_ranker is not None
            }
            
            logger.debug(f"✅ Query procesada: '{query_text[:30]}...'")
            
        except Exception as e:
            logger.error(f"❌ Error en query_four_methods: {e}")
            # Fallback a baseline
            baseline_results = self._process_query_baseline(query_text, k)
            for method in self.available_methods:
                results['methods'][method] = baseline_results
                results['scores'][method] = 0.5
                results['timing'][method] = 0.0
        
        return results
    
    def _process_query_baseline(self, query_text: str, k: int) -> List:
        """Método 1: Baseline puro (FAISS)"""
        try:
            if self.vector_store and self.canonicalizer:
                query_embedding = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                return self.vector_store.search(query_embedding, k=k)
            else:
                logger.warning("Vector store no disponible")
                return []
        except Exception as e:
            logger.error(f"Error en baseline: {e}")
            return []
    
    def _method_ner_enhanced(self, query_text: str, k: int) -> List:
        """Método 2: NER-Enhanced - VERSIÓN CORREGIDA"""
        if not self.ner_ranker:
            logger.warning("NER ranker no disponible, usando baseline")
            return self._process_query_baseline(query_text, k)
        
        try:
            # Obtener más resultados base para tener pool de re-ranking
            baseline_results = self._process_query_baseline(query_text, k*2)
            
            if not baseline_results:
                return []
            
            # Calcular scores baseline
            if self.canonicalizer:
                query_embedding = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                
                baseline_scores = []
                for product in baseline_results:
                    if hasattr(product, 'content_embedding'):
                        prod_emb = product.content_embedding
                        prod_norm = prod_emb / np.linalg.norm(prod_emb)
                        query_norm = query_embedding / np.linalg.norm(query_embedding)
                        score = float(np.dot(query_norm, prod_norm))
                        baseline_scores.append(score)
                    else:
                        baseline_scores.append(0.0)
            else:
                baseline_scores = [0.5] * len(baseline_results)
            
            # ✅ FIX: rank_with_ner() NO acepta parámetro k
            # Solo pasa: products, query, baseline_scores
            ranked = self.ner_ranker.rank_with_ner(
                baseline_results, 
                query_text, 
                baseline_scores
            )
            
            # Retornar solo los top k
            return ranked[:k]
            
        except Exception as e:
            logger.error(f"Error en NER-enhanced: {e}")
            return self._process_query_baseline(query_text, k)
    
    def _method_rlhf(self, query_text: str, k: int) -> List:
        """Método 3: RLHF"""
        if not self.rl_ranker or not self.rl_ranker.has_learned:
            logger.warning("RLHF ranker no disponible o no entrenado, usando baseline")
            return self._process_query_baseline(query_text, k)
        
        try:
            # Obtener resultados baseline (no NER, para ser justo)
            baseline_results = self._process_query_baseline(query_text, k*2)
            
            if not baseline_results:
                return []
            
            # Calcular scores para RLHF
            if self.canonicalizer:
                query_embedding = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                
                baseline_scores = []
                for product in baseline_results:
                    if hasattr(product, 'content_embedding'):
                        prod_emb = product.content_embedding
                        prod_norm = prod_emb / np.linalg.norm(prod_emb)
                        query_norm = query_embedding / np.linalg.norm(query_embedding)
                        score = float(np.dot(query_norm, prod_norm))
                        baseline_scores.append(score)
                    else:
                        baseline_scores.append(0.0)
            else:
                baseline_scores = [0.5] * len(baseline_results)
            
            # Aplicar RLHF
            rlhf_results = self.rl_ranker.rank_with_human_preferences(
                baseline_results, query_text, baseline_scores
            )
            
            return rlhf_results[:k]
            
        except Exception as e:
            logger.error(f"Error en RLHF: {e}")
            return self._process_query_baseline(query_text, k)
    
    def _method_full_hybrid(self, query_text: str, k: int) -> List:
        """
        Método 4: Full Hybrid - Combina Baseline + NER + RLHF
        VERSIÓN CORREGIDA
        """
        if not self.ner_ranker or not self.rl_ranker or not self.rl_ranker.has_learned:
            logger.warning("Full Hybrid requiere NER y RLHF, usando NER solo")
            return self._method_ner_enhanced(query_text, k)
        
        try:
            # 1. Obtener candidatos base
            baseline_products = self._process_query_baseline(query_text, k*2)
            
            if not baseline_products:
                return []
            
            # 2. Calcular baseline scores
            if self.canonicalizer:
                query_embedding = self.canonicalizer.embedding_model.encode(
                    query_text, normalize_embeddings=True
                )
                
                baseline_scores = []
                for product in baseline_products:
                    if hasattr(product, 'content_embedding'):
                        prod_emb = product.content_embedding
                        prod_norm = prod_emb / np.linalg.norm(prod_emb)
                        query_norm = query_embedding / np.linalg.norm(query_embedding)
                        score = float(np.dot(query_norm, prod_norm))
                        baseline_scores.append(score)
                    else:
                        baseline_scores.append(0.0)
            else:
                baseline_scores = [0.5] * len(baseline_products)
            
            # 3. Aplicar NER primero (✅ SIN parámetro k)
            ner_ranked = self.ner_ranker.rank_with_ner(
                baseline_products,
                query_text,
                baseline_scores
            )
            
            # 4. Recalcular scores después de NER para RLHF
            ner_scores = []
            for i, product in enumerate(ner_ranked):
                # Score basado en posición en NER ranking
                ner_score = 1.0 - (i / len(ner_ranked))
                ner_scores.append(ner_score)
            
            # 5. Aplicar RLHF sobre NER
            final_ranked = self.rl_ranker.rank_with_human_preferences(
                ner_ranked,
                query_text,
                ner_scores
            )
            
            return final_ranked[:k]
            
        except Exception as e:
            logger.error(f"Error en Full Hybrid: {e}")
            return self._method_ner_enhanced(query_text, k)
    
    def _calculate_method_score(self, products: List, query: str) -> float:
        """Calcula score de calidad para un método"""
        if not products:
            return 0.0
        
        score = 0.0
        
        # Score por rating
        rating_total = 0.0
        rating_count = 0
        
        for product in products[:10]:
            if hasattr(product, 'rating') and product.rating:
                try:
                    rating_val = float(product.rating)
                    rating_total += rating_val / 5.0  # Normalizar a 0-1
                    rating_count += 1
                except:
                    continue
        
        if rating_count > 0:
            score += (rating_total / rating_count) * 0.7
        
        # Score por diversidad (títulos únicos)
        titles = [getattr(p, 'title', '') for p in products[:10]]
        unique_titles = len(set(titles))
        
        if len(titles) > 0:
            score += (unique_titles / len(titles)) * 0.3
        
        return min(1.0, score)
    
    def train_rlhf_with_queries(self, train_queries: List[str], 
                              interactions_file: Path) -> bool:
        """
        Entrena RLHF solo con queries específicas
        """
        if not RLHFRankerFixed_available:
            logger.error("RLHFRankerFixed no disponible")
            return False
        
        logger.info(f"🎯 Entrenando RLHF con {len(train_queries)} queries...")
        
        # Filtrar interacciones
        train_interactions = []
        try:
            with open(interactions_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        data = json.loads(line)
                        if data.get('interaction_type') == 'click':
                            context = data.get('context', {})
                            query = context.get('query', '')
                            if query in train_queries:
                                train_interactions.append(data)
                    except:
                        continue
        except FileNotFoundError:
            logger.error(f"Archivo de interacciones no encontrado: {interactions_file}")
            return False
        
        logger.info(f"   • Interacciones filtradas: {len(train_interactions)}")
        
        if len(train_interactions) < 5:
            logger.warning(f"⚠️  Muy pocas interacciones ({len(train_interactions)}) para entrenar RLHF")
            return False
        
        # Crear RL ranker
        self.rl_ranker = RLHFRankerFixed(
            learning_rate=0.5,
            match_rating_balance=1.5
        )
        
        # Mapa de productos
        products_by_id = {p.id: p for p in self.canonical_products if hasattr(p, 'id')}
        
        # Entrenar
        trained = 0
        failed = 0
        
        for interaction in train_interactions:
            try:
                context = interaction.get('context', {})
                query = context.get('query', '').strip()
                product_id = context.get('product_id')
                position = context.get('position', 1)
                
                if not query or not product_id:
                    continue
                
                # Buscar producto
                product = products_by_id.get(product_id)
                if not product:
                    # Intentar búsqueda parcial
                    for pid, prod in products_by_id.items():
                        if product_id in pid or pid in product_id:
                            product = prod
                            break
                
                if not product:
                    failed += 1
                    continue
                
                # Calcular reward adaptativo
                if position == 1:
                    reward = 0.3  # Bajo para clicks obvios
                elif position <= 3:
                    reward = 0.7
                elif position <= 10:
                    reward = 1.2  # Bueno para descubrimiento
                else:
                    reward = 1.5  # Excelente para clicks profundos
                
                # Aplicar aprendizaje
                self.rl_ranker.learn_from_human_feedback(product, query, position, reward)
                trained += 1
                
                # Log progreso
                if trained % 10 == 0:
                    logger.info(f"   • Entrenados: {trained}/{len(train_interactions)}")
                
            except Exception as e:
                failed += 1
                logger.debug(f"Error en interacción: {e}")
                continue
        
        logger.info(f"✅ RLHF entrenado: {trained} exitosas, {failed} fallidas")
        
        if trained > 0:
            stats = self.rl_ranker.get_stats()
            logger.info(f"   • Features aprendidas: {stats.get('weights_count', 0)}")
            logger.info(f"   • Feedback procesado: {stats.get('feedback_count', 0)}")
            
            # Guardar sistema actualizado
            self.save_to_cache(str(self.system_cache_path))
            logger.info(f"💾 Sistema guardado: {self.system_cache_path}")
            
            return True
        else:
            logger.error("❌ RLHF no pudo ser entrenado (0 interacciones exitosas)")
            self.rl_ranker = None
            return False
    
    def save_to_cache(self, cache_path: str = None) -> bool:
        """Guarda sistema en caché"""
        if cache_path is None:
            cache_path = str(self.system_cache_path)
        
        try:
            cache_file = Path(cache_path)
            cache_file.parent.mkdir(parents=True, exist_ok=True)
            
            with open(cache_file, 'wb') as f:
                pickle.dump(self, f)
            
            logger.info(f"💾 Sistema V2 guardado: {cache_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error guardando caché: {e}")
            return False
    
    def get_system_stats(self):
        """Obtiene estadísticas completas del sistema V2"""
        stats = super().get_system_stats() if hasattr(super(), 'get_system_stats') else {}
        
        # Añadir estadísticas específicas de V2
        stats.update({
            'has_ner_ranker': self.ner_ranker is not None,
            'has_ner_extractor': self.ner_extractor is not None,
            'ner_enriched_count': sum(1 for p in self.canonical_products 
                                    if hasattr(p, 'ner_attributes') and p.ner_attributes),
            'available_methods': self.available_methods,
            'has_learned_rlhf': self.rl_ranker is not None and 
                            hasattr(self.rl_ranker, 'has_learned') and 
                            self.rl_ranker.has_learned
        })
        
        # RLHF stats
        if self.rl_ranker and hasattr(self.rl_ranker, 'get_stats'):
            rl_stats = self.rl_ranker.get_stats()
            stats['rl_stats'] = {
                'feedback_count': rl_stats.get('feedback_count', 0),
                'has_learned': rl_stats.get('has_learned', False),
                'weights_count': rl_stats.get('weights_count', 0)
            }
        
        return stats
    
    @classmethod
    def load_from_cache(cls, cache_path: str = "data/cache/unified_system_v2.pkl"):
        """Carga sistema desde cache"""
        if not os.path.exists(cache_path):
            logger.warning(f"⚠️  Cache no encontrado: {cache_path}")
            return None
        
        try:
            with open(cache_path, 'rb') as f:
                system = pickle.load(f)
            logger.info(f"✅ Sistema V2 cargado: {len(system.canonical_products):,} productos")
            return system
        
        except (EOFError, pickle.UnpicklingError) as e:
            logger.error(f"❌ Error cargando cache: {e}")
            logger.info("🔄 Eliminando cache corrupto y reconstruyendo...")
            
            # Eliminar cache corrupto
            try:
                os.remove(cache_path)
            except:
                pass
            
            # Retornar None para forzar reconstrucción
            return None