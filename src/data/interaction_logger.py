# src/data/interaction_logger.py
# src/data/interaction_logger.py
"""
Logger de interacciones REALES - Single Source of Truth
Registra TODAS las interacciones para métricas REALES
"""
import json
from pathlib import Path
from datetime import datetime, date
from typing import Dict, List, Any, Optional
import logging
import hashlib
from collections import defaultdict

logger = logging.getLogger(__name__)

class RealInteractionLogger:
    """Registra TODAS las interacciones del usuario para métricas REALES"""
    
    def __init__(self, log_dir: str = "data/interactions"):
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # Archivo de log del día actual
        self.current_log_file = None
        self._update_log_file()
        
        # Cache para análisis rápido
        self.interactions_cache = []
        
        logger.info(f"📝 Logger de interacciones REALES inicializado en: {self.log_dir}")
    
    def _update_log_file(self):
        """Actualiza el archivo de log según la fecha actual"""
        today = date.today().strftime("%Y%m%d")
        self.current_log_file = self.log_dir / f"interactions_{today}.jsonl"
        
        # Crear archivo si no existe
        if not self.current_log_file.exists():
            with open(self.current_log_file, 'w', encoding='utf-8') as f:
                f.write(f"# Interaction log started at {datetime.now().isoformat()}\n")
    
    def log_interaction(
        self,
        session_id: str,
        mode: str,  # "baseline", "with_features", "with_rlhf"
        query: str,
        results: List[Dict],  # [{id, title, rank, score}]
        clicked_product_id: Optional[str] = None,
        clicked_rank: Optional[int] = None,
        dwell_time: float = 0.0,
        feedback_type: str = "shown",  # "click", "select", "purchase", "skip"
        additional_context: Dict[str, Any] = None
    ) -> Dict[str, Any]:
        """
        Registra una interacción COMPLETA del usuario
        
        Returns:
            Dict con la interacción registrada
        """
        
        # Actualizar archivo si cambió el día
        self._update_log_file()
        
        # Crear objeto de interacción
        interaction = {
            "session_id": session_id,
            "timestamp": datetime.now().isoformat(),
            "mode": mode,
            "query": query,
            "query_hash": hashlib.md5(query.encode()).hexdigest()[:12],
            "results_shown": [
                {
                    "product_id": r.get("id", ""),
                    "rank": idx + 1,
                    "title": r.get("title", "")[:100] if r.get("title") else "",
                    "score": float(r.get("score", 0.0)) if r.get("score") else None
                }
                for idx, r in enumerate(results[:10])  # Solo primeros 10
            ],
            "interaction": {
                "clicked_product_id": clicked_product_id,
                "clicked_rank": clicked_rank,
                "dwell_time": dwell_time,
                "feedback_type": feedback_type,
                "success": clicked_product_id is not None  # ¿Hubo click?
            },
            "metadata": {
                "results_count": len(results),
                "source": "real_user_interaction",
                "system_version": "1.0"
            }
        }
        
        # Añadir contexto adicional si existe
        if additional_context:
            interaction["context"] = additional_context
        
        # Guardar en archivo JSONL
        try:
            with open(self.current_log_file, "a", encoding="utf-8") as f:
                f.write(json.dumps(interaction, ensure_ascii=False) + "\n")
        except Exception as e:
            logger.error(f"Error guardando interacción: {e}")
        
        # Agregar al cache
        self.interactions_cache.append(interaction)
        
        # Limitar cache a últimos 1000 interacciones
        if len(self.interactions_cache) > 1000:
            self.interactions_cache = self.interactions_cache[-1000:]
        
        logger.debug(f"📝 Interacción registrada: {query[:30]}... -> {clicked_product_id or 'no click'}")
        
        return interaction
    
    def calculate_real_metrics(self, mode: Optional[str] = None) -> Dict[str, Any]:
        """Calcula métricas REALES a partir de interacciones"""
        
        # Cargar interacciones del día
        interactions = self._load_today_interactions()
        
        if mode:
            interactions = [i for i in interactions if i.get("mode") == mode]
        
        if not interactions:
            return {
                "error": "No hay interacciones registradas",
                "timestamp": datetime.now().isoformat()
            }
        
        # Métricas básicas
        total_queries = len(interactions)
        successful_queries = sum(1 for i in interactions if i["interaction"]["success"])
        
        # CTR (Click-Through Rate) REAL
        ctr = successful_queries / total_queries if total_queries > 0 else 0.0
        
        # Posición promedio del click
        clicked_positions = [
            i["interaction"]["clicked_rank"]
            for i in interactions
            if i["interaction"]["clicked_rank"] is not None
        ]
        
        avg_click_position = sum(clicked_positions) / len(clicked_positions) if clicked_positions else None
        
        # Análisis por query
        query_stats = defaultdict(lambda: {"total": 0, "clicks": 0})
        for interaction in interactions:
            query = interaction["query"]
            query_stats[query]["total"] += 1
            if interaction["interaction"]["success"]:
                query_stats[query]["clicks"] += 1
        
        # Top queries por CTR
        top_queries = []
        for query, stats in query_stats.items():
            if stats["total"] > 0:
                query_ctr = stats["clicks"] / stats["total"]
                top_queries.append((query, query_ctr, stats["clicks"], stats["total"]))
        
        top_queries.sort(key=lambda x: x[1], reverse=True)
        
        return {
            "timestamp": datetime.now().isoformat(),
            "total_interactions": total_queries,
            "successful_interactions": successful_queries,
            "ctr": ctr,
            "avg_click_position": avg_click_position,
            "top_queries_by_ctr": [
                {
                    "query": q[0],
                    "ctr": q[1],
                    "clicks": q[2],
                    "total": q[3]
                }
                for q in top_queries[:5]
            ],
            "modes_used": list(set(i.get("mode", "unknown") for i in interactions)),
            "is_real_data": True,
            "data_source": "real_user_interactions"
        }
    
    def get_relevance_labels(self) -> Dict[str, List[str]]:
        """Extrae ground truth REAL: query → productos relevantes (clickeados)"""
        
        interactions = self._load_today_interactions()
        relevance = {}
        
        for interaction in interactions:
            query = interaction["query"]
            clicked_id = interaction["interaction"]["clicked_product_id"]
            
            if clicked_id:
                if query not in relevance:
                    relevance[query] = []
                
                if clicked_id not in relevance[query]:
                    relevance[query].append(clicked_id)
        
        logger.info(f"📊 Ground truth REAL extraído: {len(relevance)} queries con productos relevantes")
        return relevance
    
    def _load_today_interactions(self) -> List[Dict]:
        """Carga todas las interacciones de hoy"""
        if not self.current_log_file or not self.current_log_file.exists():
            return []
        
        interactions = []
        try:
            with open(self.current_log_file, "r", encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if line and not line.startswith("#"):  # Ignorar comentarios
                        try:
                            interactions.append(json.loads(line))
                        except json.JSONDecodeError:
                            continue
        except Exception as e:
            logger.error(f"Error cargando interacciones: {e}")
        
        return interactions
    
    def clear_cache(self):
        """Limpia el cache de interacciones"""
        self.interactions_cache = []