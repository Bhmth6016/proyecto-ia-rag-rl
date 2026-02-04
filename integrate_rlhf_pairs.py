# integrate_rlhf_pairs.py
#!/usr/bin/env python3
"""
Integrador de Pares RLHF - VERSIÓN MEJORADA
===========================================

Mejoras:
1. Combina pares de TODAS las categorías automáticamente
2. Mejor validación de datos
3. Estadísticas detalladas por categoría
"""

import json
from pathlib import Path
from typing import List, Dict
from datetime import datetime
from collections import defaultdict
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RLHFPairsIntegrator:
    """Integra pares RLHF de múltiples categorías"""
    
    def __init__(
        self,
        pairs_dir: Path = Path("data/rlhf_pairs"),
        output_file: Path = Path("data/interactions/rlhf_interactions_from_reviews.jsonl"),
        ground_truth_file: Path = Path("data/interactions/ground_truth_from_reviews.json")
    ):
        self.pairs_dir = pairs_dir
        self.output_file = output_file
        self.ground_truth_file = ground_truth_file
        
        logger.info("🔧 RLHF Integrator inicializado")
        logger.info(f"   Pairs dir: {pairs_dir}")
        logger.info(f"   Output: {output_file}")
    
    def find_pair_files(self) -> List[Path]:
        """✅ NUEVO: Encuentra TODOS los archivos de pares"""
        if not self.pairs_dir.exists():
            logger.warning(f"⚠️ Directorio no existe: {self.pairs_dir}")
            return []
        
        pair_files = list(self.pairs_dir.glob("rlhf_pairs_*.jsonl"))
        
        logger.info(f"\n📂 Archivos de pares encontrados: {len(pair_files)}")
        for pf in pair_files:
            logger.info(f"   • {pf.name}")
        
        return pair_files
    
    def load_all_pairs(self) -> List[Dict]:
        """✅ MEJORADO: Carga pares de TODAS las categorías"""
        pair_files = self.find_pair_files()
        
        if not pair_files:
            logger.error("❌ No se encontraron archivos de pares")
            logger.info("   Ejecuta primero: python generate_rlhf_pairs_from_reviews_FIXED.py")
            return []
        
        all_pairs = []
        stats_by_category = defaultdict(int)
        
        for pair_file in pair_files:
            logger.info(f"\n📂 Cargando: {pair_file.name}")
            
            count = 0
            with open(pair_file, 'r', encoding='utf-8') as f:
                for line in f:
                    try:
                        pair = json.loads(line)
                        all_pairs.append(pair)
                        
                        # Extraer categoría del par - CON VALIDACIÓN
                        category = pair.get('category', 'Unknown')
                        # Si category es None, usar "Unknown"
                        if category is None:
                            category = "Unknown"
                        
                        stats_by_category[category] += 1
                        count += 1
                    except json.JSONDecodeError:
                        continue
            
            logger.info(f"   ✓ Pares cargados: {count:,}")
        
        logger.info(f"\n✅ TOTAL PARES: {len(all_pairs):,}")
        logger.info(f"\n📊 Por categoría:")
        
        # CORRECCIÓN: Ordenar por clave (categoría) en lugar de por valor
        # Usamos una función de ordenamiento segura que maneja None
        for category in sorted(stats_by_category.keys(), 
                              key=lambda x: (x is None, x)):
            count = stats_by_category[category]
            # Mostrar "None" como "Unknown" para mejor claridad
            display_category = "Unknown" if category is None else category
            logger.info(f"   • {display_category:40} {count:>6,}")
        
        return all_pairs
    
    def convert_to_interactions(self, pairs: List[Dict]) -> List[Dict]:
        """Convierte pares a interacciones"""
        logger.info("\n🔄 Convirtiendo a interacciones...")
        
        interactions = []
        session_id = f"rlhf_training_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        stats = {'clicks': 0, 'views': 0}
        
        for i, pair in enumerate(pairs):
            query = pair['query']
            chosen = pair['chosen']
            rejected = pair['rejected']
            margin = pair.get('margin', 0.0)
            
            # Obtener categoría con validación
            category = pair.get('category', 'Unknown')
            if category is None:
                category = 'Unknown'
            
            # Interacción CHOSEN (click)
            chosen_position = 3
            
            interaction_chosen = {
                'timestamp': datetime.now().isoformat(),
                'session_id': session_id,
                'interaction_type': 'click',
                'context': {
                    'query': query,
                    'product_id': chosen['parent_asin'],  # Usar parent_asin como ID
                    'position': chosen_position,
                    'product_title': chosen['title'],
                    'product_image': chosen.get('image_url'),  # NUEVO
                    'is_relevant': True,
                    'feedback_type': 'implicit_preference',
                    'reward_score': chosen['reward_score'],
                    'avg_rating': chosen.get('avg_rating', 0.0),
                    'num_reviews': chosen.get('num_reviews', 0),
                    'source': 'review_aggregation',
                    'pair_id': i,
                    'pair_margin': margin,
                    'category': category
                }
            }
            
            interactions.append(interaction_chosen)
            stats['clicks'] += 1
            
            # Interacción REJECTED (view)
            if margin > 0.3:
                rejected_position = 10
                
                interaction_rejected = {
                    'timestamp': datetime.now().isoformat(),
                    'session_id': session_id,
                    'interaction_type': 'view',
                    'context': {
                        'query': query,
                        'product_id': rejected['parent_asin'],  # Usar parent_asin como ID
                        'position': rejected_position,
                        'product_title': rejected['title'],
                        'product_image': rejected.get('image_url'),  # NUEVO
                        'is_relevant': False,
                        'feedback_type': 'implicit_rejection',
                        'reward_score': rejected['reward_score'],
                        'avg_rating': rejected.get('avg_rating', 0.0),
                        'num_reviews': rejected.get('num_reviews', 0),
                        'source': 'review_aggregation',
                        'pair_id': i,
                        'pair_margin': -margin,
                        'category': category
                    }
                }
                
                interactions.append(interaction_rejected)
                stats['views'] += 1
        
        logger.info(f"✅ Interacciones generadas: {len(interactions):,}")
        logger.info(f"   • Clicks (chosen):  {stats['clicks']:,}")
        logger.info(f"   • Views (rejected): {stats['views']:,}")
        
        return interactions
    
    def create_ground_truth(self, pairs: List[Dict]) -> Dict[str, List[str]]:
        """Crea ground truth desde pares"""
        logger.info("\n📝 Creando ground truth...")
        
        ground_truth = {}
        
        for pair in pairs:
            query = pair['query']
            chosen_id = pair['chosen']['parent_asin']
            
            if query not in ground_truth:
                ground_truth[query] = []
            
            if chosen_id not in ground_truth[query]:
                ground_truth[query].append(chosen_id)
        
        logger.info(f"✅ Ground truth: {len(ground_truth)} queries únicas")
        
        total_relevant = sum(len(ids) for ids in ground_truth.values())
        logger.info(f"   • Productos relevantes: {total_relevant:,}")
        
        # Estadísticas
        query_lengths = [len(ids) for ids in ground_truth.values()]
        logger.info(f"   • Productos/query (promedio): {sum(query_lengths)/len(query_lengths):.1f}")
        
        return ground_truth
    
    def save_interactions(self, interactions: List[Dict]):
        """Guarda interacciones"""
        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"\n💾 Guardando interacciones: {self.output_file}")
        
        with open(self.output_file, 'w', encoding='utf-8') as f:
            for interaction in interactions:
                f.write(json.dumps(interaction, ensure_ascii=False) + '\n')
        
        logger.info(f"✅ Guardadas: {len(interactions):,} interacciones")
    
    def save_ground_truth(self, ground_truth: Dict[str, List[str]]):
        """Guarda ground truth"""
        self.ground_truth_file.parent.mkdir(parents=True, exist_ok=True)
        
        logger.info(f"💾 Guardando ground truth: {self.ground_truth_file}")
        
        with open(self.ground_truth_file, 'w', encoding='utf-8') as f:
            json.dump(ground_truth, f, indent=2, ensure_ascii=False)
        
        logger.info(f"✅ Ground truth guardado")
    
    def run(self):
        """Pipeline completo"""
        logger.info("\n" + "="*60)
        logger.info("🚀 INTEGRANDO PARES RLHF CON SISTEMA")
        logger.info("="*60)
        
        # 1. Cargar TODOS los pares
        pairs = self.load_all_pairs()
        
        if not pairs:
            logger.error("❌ No hay pares para procesar")
            logger.info("\n💡 SOLUCIÓN:")
            logger.info("   python generate_rlhf_pairs_from_reviews_FIXED.py")
            return False
        
        # 2. Convertir a interacciones
        interactions = self.convert_to_interactions(pairs)
        
        # 3. Crear ground truth
        ground_truth = self.create_ground_truth(pairs)
        
        # 4. Guardar
        self.save_interactions(interactions)
        self.save_ground_truth(ground_truth)
        
        # Resumen
        logger.info("\n" + "="*60)
        logger.info("📊 RESUMEN FINAL")
        logger.info("="*60)
        logger.info(f"Pares procesados:        {len(pairs):,}")
        logger.info(f"Interacciones generadas: {len(interactions):,}")
        logger.info(f"Queries únicas:          {len(ground_truth):,}")
        logger.info(f"\nArchivos generados:")
        logger.info(f"   • {self.output_file}")
        logger.info(f"   • {self.ground_truth_file}")
        logger.info("\n✅ Listo para entrenar RLHF!")
        logger.info("   python main.py experimento")
        logger.info("="*60)
        
        return True


def main():
    """Ejemplo de uso"""
    
    # Integrador automático
    integrator = RLHFPairsIntegrator(
        pairs_dir=Path("data/rlhf_pairs"),
        output_file=Path("data/interactions/rlhf_interactions_from_reviews.jsonl"),
        ground_truth_file=Path("data/interactions/ground_truth_from_reviews.json")
    )
    
    # Ejecutar
    success = integrator.run()
    
    if not success:
        logger.error("\n❌ Integración fallida")
        return 1
    
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())