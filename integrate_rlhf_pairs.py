# integrate_rlhf_pairs.py
#!/usr/bin/env python3
"""
Integrador de Pares RLHF con Sistema Existente
==============================================

Convierte los pares (chosen, rejected) generados desde reviews
al formato de interacciones que usa tu sistema actual.

Compatible con:
- src.ranking.rl_ranker_fixed.RLHFRankerFixed
- experimento_completo_4_metodos.py
"""

import json
from pathlib import Path
from typing import List, Dict, Optional
from datetime import datetime
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLHFPairsIntegrator:
    """
    Integra pares RLHF en el formato esperado por el sistema.
    
    Formato de salida (compatible con real_interactions.jsonl):
    {
        "timestamp": "2025-01-18T10:30:00",
        "session_id": "rlhf_training",
        "interaction_type": "click",
        "context": {
            "query": "beauty products",
            "product_id": "B08ZRWZ2DJ",
            "position": 1,
            "product_title": "...",
            "is_relevant": true,
            "feedback_type": "implicit_preference",
            "reward_score": 0.85,
            "source": "review_aggregation"
        }
    }
    """
    
    def __init__(
        self,
        pairs_file: Path,
        output_file: Path = Path("data/interactions/rlhf_interactions_from_reviews.jsonl"),
        ground_truth_file: Path = Path("data/interactions/ground_truth_from_reviews.json")
    ):
        self.pairs_file = pairs_file
        self.output_file = output_file
        self.ground_truth_file = ground_truth_file
        
        logger.info("🔧 RLHF Integrator inicializado")
        logger.info(f"   Input:  {pairs_file}")
        logger.info(f"   Output: {output_file}")
    
    def load_pairs(self) -> List[Dict]:
        """Carga pares RLHF"""
        logger.info(f"📂 Cargando pares desde {self.pairs_file}...")
        
        pairs = []
        with open(self.pairs_file, 'r', encoding='utf-8') as f:
            for line in f:
                try:
                    pair = json.loads(line)
                    pairs.append(pair)
                except json.JSONDecodeError:
                    continue
        
        logger.info(f"✅ Pares cargados: {len(pairs):,}")
        return pairs
    
    def convert_to_interactions(self, pairs: List[Dict]) -> List[Dict]:
        """
        Convierte pares (chosen, rejected) a interacciones de click.
        
        Estrategia:
        - chosen → click en posición baja (alta recompensa)
        - rejected → no incluir (o click en posición alta con baja recompensa)
        """
        logger.info("🔄 Convirtiendo a formato de interacciones...")
        
        interactions = []
        session_id = f"rlhf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        for i, pair in enumerate(pairs):
            query = pair['query']
            chosen = pair['chosen']
            rejected = pair['rejected']
            margin = pair.get('margin', 0.0)
            
            # Interacción para CHOSEN (producto preferido)
            # Posición baja simula "usuario scrolleó pero encontró esto valioso"
            chosen_position = 3  # Posición media-alta
            chosen_reward = self._calculate_reward_from_position(
                chosen_position,
                chosen['reward_score']
            )
            
            interaction_chosen = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'interaction_type': 'click',
                'context': {
                    'query': query,
                    'product_id': chosen['parent_asin'],
                    'position': chosen_position,
                    'product_title': chosen['title'],
                    'is_relevant': True,
                    'feedback_type': 'implicit_preference',
                    'reward_score': chosen['reward_score'],
                    'avg_rating': chosen.get('avg_rating', 0.0),
                    'num_reviews': chosen.get('num_reviews', 0),
                    'source': 'review_aggregation',
                    'pair_id': i,
                    'pair_margin': margin
                }
            }
            
            interactions.append(interaction_chosen)
            
            # OPCIONAL: También podemos generar interacción negativa para rejected
            # (útil para contrastive learning)
            if margin > 0.3:  # Solo si hay margen significativo
                rejected_position = 10  # Posición baja
                
                interaction_rejected = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'interaction_type': 'view',  # View sin click
                    'context': {
                        'query': query,
                        'product_id': rejected['parent_asin'],
                        'position': rejected_position,
                        'product_title': rejected['title'],
                        'is_relevant': False,
                        'feedback_type': 'implicit_rejection',
                        'reward_score': rejected['reward_score'],
                        'avg_rating': rejected.get('avg_rating', 0.0),
                        'num_reviews': rejected.get('num_reviews', 0),
                        'source': 'review_aggregation',
                        'pair_id': i,
                        'pair_margin': -margin  # Negativo para rejected
                    }
                }
                
                interactions.append(interaction_rejected)
        
        logger.info(f"✅ Interacciones generadas: {len(interactions):,}")
        
        return interactions
    
    def _calculate_reward_from_position(self, position: int, base_reward: float) -> float:
        """
        Ajusta reward según posición (clicks profundos = más valioso).
        
        Similar a RLHFRankerFixed.learn_from_human_feedback()
        """
        if position == 1:
            position_factor = 0.3  # Bajo para clicks obvios
        elif position <= 3:
            position_factor = 0.7
        elif position <= 10:
            position_factor = 1.2  # Bueno para descubrimiento
        else:
            position_factor = 1.5  # Excelente para clicks profundos
        
        return base_reward * position_factor
    
    def create_ground_truth(self, pairs: List[Dict]) -> Dict[str, List[str]]:
        """
        Crea ground truth: {query: [producto_relevante_1, ...]}
        """
        logger.info("📝 Creando ground truth...")
        
        ground_truth = {}
        
        for pair in pairs:
            query = pair['query']
            chosen_id = pair['chosen']['parent_asin']
            
            if query not in ground_truth:
                ground_truth[query] = []
            
            # Solo agregar si no está (evitar duplicados)
            if chosen_id not in ground_truth[query]:
                ground_truth[query].append(chosen_id)
        
        logger.info(f"✅ Ground truth: {len(ground_truth)} queries")
        
        total_relevant = sum(len(ids) for ids in ground_truth.values())
        logger.info(f"   • Productos relevantes totales: {total_relevant}")
        
        return ground_truth
    
    def save_interactions(self, interactions: List[Dict]):
        """Guarda interacciones en formato JSONL"""
        logger.info(f"💾 Guardando interacciones en {self.output_file}...")
        
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for interaction in interactions:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Guardadas: {len(interactions):,} interacciones")
    
    def save_ground_truth(self, ground_truth: Dict[str, List[str]]):
        """Guarda ground truth en JSON"""
        logger.info(f"💾 Guardando ground truth en {self.ground_truth_file}...")
        
        self.ground_truth_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Ground truth guardado")
    
    def run(self):
        """Pipeline completo"""
        logger.info("="*60)
        logger.info("🚀 INTEGRANDO PARES RLHF CON SISTEMA")
        logger.info("="*60)
        
        # 1. Cargar pares
        pairs = self.load_pairs()
        
        if not pairs:
            logger.error("❌ No hay pares para procesar")
            return
        
        # 2. Convertir a interacciones
        interactions = self.convert_to_interactions(pairs)
        
        # 3. Crear ground truth
        ground_truth = self.create_ground_truth(pairs)
        
        # 4. Guardar
        self.save_interactions(interactions)
        self.save_ground_truth(ground_truth)
        
        # Estadísticas
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN")
        logger.info("="*60)
        logger.info(f"Pares procesados:        {len(pairs):,}")
        logger.info(f"Interacciones generadas: {len(interactions):,}")
        logger.info(f"Queries únicas:          {len(ground_truth):,}")
        
        clicks = sum(1 for i in interactions if i['interaction_type'] == 'click')
        views = sum(1 for i in interactions if i['interaction_type'] == 'view')
        
        logger.info(f"\nPor tipo:")
        logger.info(f"  • Clicks (chosen):     {clicks:,}")
        logger.info(f"  • Views (rejected):    {views:,}")
        
        logger.info("\n✅ Listo para entrenar RLHF!")
        logger.info(f"   python main.py experimento")
        logger.info("="*60)


def main():
    """Ejemplo de uso"""
    
    # Archivos
    pairs_file = Path("data/rlhf_pairs/rlhf_pairs_beauty.jsonl")
    output_file = Path("data/interactions/rlhf_interactions_from_reviews.jsonl")
    ground_truth_file = Path("data/interactions/ground_truth_from_reviews.json")
    
    if not pairs_file.exists():
        logger.error(f"❌ Archivo de pares no existe: {pairs_file}")
        logger.info("   Ejecuta primero: python generate_rlhf_pairs_from_reviews.py")
        return
    
    # Integrador
    integrator = RLHFPairsIntegrator(
        pairs_file=pairs_file,
        output_file=output_file,
        ground_truth_file=ground_truth_file
    )
    
    # Ejecutar
    integrator.run()


if __name__ == "__main__":
    main()