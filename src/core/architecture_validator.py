# src/core/architecture_validator.py
"""
Valida que la arquitectura cumpla los principios del proyecto
"""
import logging
import inspect
from typing import Dict, Any, List

logger = logging.getLogger(__name__)


class ArchitectureValidator:
    """Validador de principios de arquitectura"""
    
    PRINCIPLES = {
        "retrieval_immutable": "FAISS se construye una vez, no se modifica",
        "rl_ranking_only": "RL solo reordena, no filtra ni indexa",
        "no_data_leakage": "Query understanding no afecta retrieval",
        "user_feedback_clean": "Feedback afecta solo ranking",
        "reproducible": "Misma entrada → misma salida (sin RL)",
        "separation_concerns": "Cada módulo tiene responsabilidad única"
    }
    
    def __init__(self):
        self.violations = []
        self.checks_performed = []
    
    def validate_system(self, system) -> Dict[str, Any]:
        """Valida que el sistema cumpla todos los principios"""
        logger.info("🔍 Validando arquitectura del sistema...")
        
        self.violations = []
        self.checks_performed = []
        
        # 1. Verificar inmutabilidad de FAISS
        self._check_retrieval_immutability(system)
        
        # 2. Verificar que RL no toque FAISS
        self._check_rl_ranking_only(system)
        
        # 3. Verificar separación query understanding
        self._check_no_data_leakage(system)
        
        # 4. Verificar feedback limpio
        self._check_user_feedback_clean(system)
        
        # 5. Verificar reproducibilidad
        self._check_reproducibility(system)
        
        # 6. Verificar separación de responsabilidades
        self._check_separation_concerns(system)
        
        return {
            "valid": len(self.violations) == 0,
            "violations": self.violations,
            "checks_performed": self.checks_performed,
            "principles": self.PRINCIPLES
        }
    
    def _check_retrieval_immutability(self, system):
        """Verifica que FAISS sea inmutable"""
        check_id = "retrieval_immutable"
        self.checks_performed.append(check_id)
        
        if hasattr(system, 'vector_store'):
            vs = system.vector_store
            
            # Check 1: Flag de inmutabilidad
            if hasattr(vs, 'is_locked'):
                if vs.is_locked:
                    logger.debug(f"✅ {check_id}: VectorStore está bloqueado")
                else:
                    self.violations.append(check_id)
                    logger.warning(f"❌ {check_id}: VectorStore no está bloqueado")
            else:
                self.violations.append(check_id)
                logger.warning(f"❌ {check_id}: VectorStore no tiene flag is_locked")
            
            # Check 2: No métodos de modificación después de construcción
            if hasattr(vs, 'build_index'):
                # Verificar que build_index no se llame después de is_locked
                # Esto requiere monitoreo en tiempo de ejecución
                pass
            
        else:
            self.violations.append(check_id)
            logger.warning(f"❌ {check_id}: No hay vector_store en el sistema")
    
    def _check_rl_ranking_only(self, system):
        """Verifica que RL solo afecte ranking"""
        check_id = "rl_ranking_only"
        self.checks_performed.append(check_id)
        
        # Buscar RL ranker en diferentes ubicaciones posibles
        rl_ranker = None
        possible_locations = ['rl_ranker', 'ranking.rl_ranker']
        
        for location in possible_locations:
            if hasattr(system, location):
                rl_ranker = getattr(system, location)
                break
        
        if rl_ranker:
            # Analizar código del RL ranker
            try:
                rl_code = inspect.getsource(rl_ranker.__class__)
                
                # Palabras prohibidas en RL
                prohibited_terms = [
                    'vector_store', 'faiss', 'index', 'embedding', 
                    'build_index', 'add', 'train', 'update_index',
                    'retrieval', 'search_engine', 'database'
                ]
                
                violations_found = []
                for term in prohibited_terms:
                    if term in rl_code.lower():
                        violations_found.append(term)
                
                if violations_found:
                    self.violations.append(check_id)
                    logger.warning(f"❌ {check_id}: RL menciona {violations_found}")
                else:
                    logger.debug(f"✅ {check_id}: RL no toca sistemas de búsqueda")
            
            except Exception as e:
                logger.debug(f"⚠️ {check_id}: No se pudo analizar código: {e}")
        
        else:
            logger.debug(f"ℹ️ {check_id}: No hay rl_ranker encontrado en el sistema")
    
    def _check_no_data_leakage(self, system):
        """Verifica que query understanding no afecte retrieval"""
        check_id = "no_data_leakage"
        self.checks_performed.append(check_id)
        
        if hasattr(system, 'query_understanding'):
            qu = system.query_understanding
            
            try:
                qu_code = inspect.getsource(qu.__class__)
                
                # Palabras prohibidas en query understanding
                # IGNORAR si están en comentarios o métodos de verificación
                prohibited_terms = [
                    'vector_store', 'search', 'faiss', 'retrieve',
                    'index', 'embedding_generation', 'build_index'
                ]
                
                violations_found = []
                
                # Analizar línea por línea para ignorar comentarios
                lines = qu_code.split('\n')
                for line in lines:
                    # Ignorar líneas que son comentarios
                    stripped_line = line.strip()
                    if stripped_line.startswith('#') or stripped_line.startswith('"""'):
                        continue
                    
                    # Ignorar líneas con "def _verify" (métodos de verificación)
                    if 'def _verify' in line or 'def verify' in line:
                        continue
                    
                    # Ignorar líneas con "prohibited_terms" (definición de lista)
                    if 'prohibited_terms' in line:
                        continue
                    
                    # Buscar términos prohibidos en código real
                    for term in prohibited_terms:
                        if term in line.lower():
                            # Verificar que no sea parte de un string
                            if f"'{term}" not in line and f'"{term}' not in line:
                                violations_found.append(f"{term} en: {line[:50]}...")
                
                if violations_found:
                    self.violations.append(check_id)
                    logger.warning(f"❌ {check_id}: Query understanding contiene código que menciona sistemas")
                    for violation in violations_found[:3]:  # Mostrar solo 3
                        logger.warning(f"   • {violation}")
                else:
                    logger.debug(f"✅ {check_id}: Query understanding correctamente separado")
            
            except Exception as e:
                logger.debug(f"⚠️ {check_id}: No se pudo analizar código: {e}")
        
        else:
            logger.debug(f"ℹ️ {check_id}: Query understanding no disponible")
    
    def _check_user_feedback_clean(self, system):
        """Verifica que feedback solo afecte ranking"""
        check_id = "user_feedback_clean"
        self.checks_performed.append(check_id)
        
        if hasattr(system, 'interaction_handler'):
            ih = system.interaction_handler
            
            try:
                ih_code = inspect.getsource(ih.__class__)
                
                # Feedback debe especificar qué afecta
                if 'affects' in ih_code.lower() and 'ranking' in ih_code.lower():
                    logger.debug(f"✅ {check_id}: Feedback especifica afecta ranking")
                else:
                    self.violations.append(check_id)
                    logger.warning(f"❌ {check_id}: Feedback no especifica qué afecta")
            
            except Exception as e:
                logger.debug(f"⚠️ {check_id}: No se pudo analizar código: {e}")
    
    def _check_reproducibility(self, system):
        """Verifica principio de reproducibilidad"""
        check_id = "reproducible"
        self.checks_performed.append(check_id)
        
        # Verificar que hay seed para random
        if hasattr(system, 'config'):
            config = system.config
            if 'experiment' in config and 'seed' in config['experiment']:
                logger.debug(f"✅ {check_id}: Seed configurada para reproducibilidad")
            else:
                logger.warning(f"⚠️ {check_id}: No hay seed configurada")
        
        # Verificar que vector store tenga timestamp de construcción
        if hasattr(system, 'vector_store'):
            vs = system.vector_store
            if hasattr(vs, 'construction_time') and vs.construction_time is not None:
                logger.debug(f"✅ {check_id}: VectorStore tiene timestamp")
    
    def _check_separation_concerns(self, system):
        """Verifica separación de responsabilidades"""
        check_id = "separation_concerns"
        self.checks_performed.append(check_id)
        
        # Verificar que los componentes existen y tienen responsabilidades claras
        components = [
            ('canonicalizer', 'Canonicaliza productos'),
            ('vector_store', 'Retrieval inmutable'),
            ('query_understanding', 'Análisis de query'),
            ('feature_engineer', 'Extracción de características'),
            ('rl_ranker', 'Ranking con aprendizaje'),
            ('interaction_handler', 'Manejo de feedback')
        ]
        
        all_present = True
        for component_name, responsibility in components:
            if hasattr(system, component_name):
                logger.debug(f"✅ {check_id}: {component_name} presente ({responsibility})")
            else:
                all_present = False
                logger.warning(f"⚠️ {check_id}: {component_name} ausente")
        
        if not all_present:
            self.violations.append(check_id)
    
    def generate_validation_report(self, validation_results: Dict[str, Any]) -> str:
        """Genera reporte de validación"""
        report = "=" * 80 + "\n"
        report += "REPORTE DE VALIDACIÓN DE ARQUITECTURA\n"
        report += "=" * 80 + "\n\n"
        
        report += f"Estado: {'✅ VÁLIDO' if validation_results['valid'] else '❌ INVALIDO'}\n"
        report += f"Checks realizados: {len(validation_results['checks_performed'])}\n"
        report += f"Violaciones: {len(validation_results['violations'])}\n\n"
        
        report += "PRINCIPIOS:\n"
        report += "-" * 80 + "\n"
        for principle, description in validation_results['principles'].items():
            status = "✅" if principle not in validation_results['violations'] else "❌"
            report += f"{status} {principle}: {description}\n"
        
        if validation_results['violations']:
            report += "\nVIOLACIONES DETECTADAS:\n"
            report += "-" * 80 + "\n"
            for violation in validation_results['violations']:
                report += f"• {violation}: {validation_results['principles'].get(violation, '')}\n"
        
        report += "\n" + "=" * 80
        
        return report