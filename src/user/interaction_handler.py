# src/user/interaction_handler.py
"""
Maneja interacción usuario → reward → aprendizaje
"""
import numpy as np
import logging
from typing import Dict, Any, Optional
from datetime import datetime
from collections import Counter
import json

logger = logging.getLogger(__name__)


class InteractionHandler:
    """
    Traduce acciones de usuario en señales de aprendizaje
    
    Acciones permitidas:
    1. Click/select → Reward positivo
    2. Ignore/scroll → Reward neutral
    3. Reject/skip → Reward negativo
    
    NO PERMITIDO:
    - Modificar productos
    - Cambiar embeddings
    - Alterar FAISS
    """
    
    def __init__(self):
        self.interaction_log = []
        self.user_sessions = {}
        self.interaction_history = []  # Para compatibilidad con nuevo código
        
        logger.info("👤 InteractionHandler inicializado")
    
    def process_interaction(self, interaction_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Procesa interacción - MÉTODO PRINCIPAL CORREGIDO
        Compatible con sistema principal y con real_experiment_runner.py
        """
        # Extraer datos del formato del sistema principal
        interaction_type = interaction_data.get('interaction_type', 'skip')
        context = interaction_data.get('context', {})
        
        # También manejar formato directo (para real_experiment_runner.py)
        if not interaction_type or interaction_type == 'skip':
            interaction_type = interaction_data.get('type', 'skip')
        
        # Llamar al método unificado
        return self._process_interaction_unified(interaction_type, context)
    
    def _process_interaction_unified(self, interaction_type: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Método unificado para procesar cualquier tipo de interacción
        """
        # Mapeo de rewards basado en tipo de interacción (compatible con ambos sistemas)
        reward_mapping = {
            # Interacciones positivas
            'click': 1.0,
            'purchase': 2.0,
            'add_to_cart': 1.5,
            'save': 1.2,
            'select': 1.5,
            'view_details': 0.8,
            
            # Interacciones neutrales
            'ignore': 0.0,
            'scroll': 0.0,
            'neutral_feedback': 0.0,
            
            # Interacciones negativas
            'skip': -0.5,
            'back': -1.0,
            'reject': -0.5,
            'skip_query': -1.0,
            'negative_feedback': -2.0
        }
        
        # Obtener reward base
        base_reward = reward_mapping.get(interaction_type, 0.0)
        
        # Ajustar reward basado en contexto
        if 'position' in context:
            # Penalizar productos muy abajo en ranking
            position = context.get('position', 0)
            if position > 0:
                position_factor = 1.0 / (1.0 + np.log1p(position))
                base_reward *= position_factor
        
        # Ajustar basado en CTR si está disponible
        if 'impressions' in context and 'clicks' in context:
            impressions = context.get('impressions', 1)
            clicks = context.get('clicks', 0)
            if impressions > 0:
                ctr = clicks / impressions
                base_reward *= (1.0 + ctr)  # Bonus por buen CTR
        
        # Verificar principios
        violates_principles = context.get('modifies_faiss', False) or \
                            context.get('modifies_embeddings', False)
        
        if violates_principles:
            logger.warning("⚠️  Interacción viola principios del sistema")
            base_reward = -3.0  # Penalización severa
        
        # Crear señal de aprendizaje
        learning_signal = {
            'reward': float(base_reward),
            'context': context,
            'interaction_type': interaction_type,
            'timestamp': np.datetime64('now'),
            'is_real': True,
            'processed_by': 'interaction_handler',
            'principles_violated': violates_principles
        }
        
        # Guardar en ambos logs para compatibilidad
        self.interaction_log.append(learning_signal)
        self.interaction_history.append(learning_signal)
        
        # Actualizar sesión de usuario si hay user_id
        user_id = context.get('user_id')
        if user_id:
            self._update_user_session(user_id, interaction_type, context, base_reward)
        
        logger.info(f"📝 Interacción procesada: {interaction_type} (reward={base_reward:.2f})")
        
        return learning_signal
    
    def _update_user_session(self, user_id: str, interaction_type: str, 
                            context: Dict[str, Any], reward: float):
        """Actualiza sesión del usuario"""
        if user_id not in self.user_sessions:
            self.user_sessions[user_id] = {
                'user_id': user_id,
                'first_seen': datetime.now().isoformat(),
                'last_seen': datetime.now().isoformat(),
                'total_interactions': 0,
                'total_reward': 0.0,
                'interaction_types': {},
                'clicked_products': []
            }
        
        user_session = self.user_sessions[user_id]
        user_session['last_seen'] = datetime.now().isoformat()
        user_session['total_interactions'] += 1
        user_session['total_reward'] += reward
        
        # Registrar tipo de interacción
        if interaction_type not in user_session['interaction_types']:
            user_session['interaction_types'][interaction_type] = 0
        user_session['interaction_types'][interaction_type] += 1
        
        # Guardar productos clickeados positivamente
        if reward > 0 and 'product_id' in context:
            user_session['clicked_products'].append({
                'product_id': context['product_id'],
                'timestamp': datetime.now().isoformat(),
                'reward': reward,
                'position': context.get('position', 0)
            })
    
    def get_user_session(self, user_id: str) -> Optional[Dict[str, Any]]:
        """Obtiene sesión de un usuario"""
        return self.user_sessions.get(user_id)
    
    def get_interaction_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas de interacciones (compatibilidad dual)"""
        # Usar interaction_log para mantener compatibilidad con código viejo
        if not self.interaction_log:
            return {'total_interactions': 0}
        
        total_reward = sum(entry['reward'] for entry in self.interaction_log)
        interaction_types = [entry['interaction_type'] for entry in self.interaction_log]
        
        type_counts = Counter(interaction_types)
        
        return {
            'total_interactions': len(self.interaction_log),
            'unique_users': len(self.user_sessions),
            'total_reward': total_reward,
            'avg_reward_per_interaction': total_reward / len(self.interaction_log) if self.interaction_log else 0,
            'interaction_type_counts': dict(type_counts),
            'principles_maintained': all(
                not entry.get('principles_violated', False)
                for entry in self.interaction_log
            )
        }
    
    def get_comprehensive_stats(self) -> Dict[str, Any]:
        """Estadísticas más detalladas (nuevo formato)"""
        if not self.interaction_history:
            return {"total": 0}
        
        rewards = [i['reward'] for i in self.interaction_history]
        return {
            "total_interactions": len(self.interaction_history),
            "avg_reward": float(np.mean(rewards)) if rewards else 0.0,
            "std_reward": float(np.std(rewards)) if len(rewards) > 1 else 0.0,
            "interaction_types": {
                itype: sum(1 for i in self.interaction_history if i['interaction_type'] == itype)
                for itype in set(i['interaction_type'] for i in self.interaction_history)
            }
        }
    
    def save_interaction_log(self, path: str):
        """Guarda log de interacciones para análisis"""
        save_data = {
            'metadata': {
                'export_time': datetime.now().isoformat(),
                'total_interactions': len(self.interaction_log),
                'principles': {
                    'affects': 'ranking_only',
                    'no_faiss_modification': True,
                    'no_embedding_modification': True
                }
            },
            'interactions': self.interaction_log,
            'user_sessions': self.user_sessions
        }
        
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(save_data, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Log de interacciones guardado: {path}")
    
    def load_interaction_log(self, path: str):
        """Carga log de interacciones desde archivo"""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        self.interaction_log = data.get('interactions', [])
        self.user_sessions = data.get('user_sessions', {})
        
        # También cargar en interaction_history para compatibilidad
        self.interaction_history = self.interaction_log.copy()
        
        logger.info(f"📂 Log de interacciones cargado: {len(self.interaction_log)} interacciones")
    
    def clear_logs(self):
        """Limpia todos los logs (para testing)"""
        self.interaction_log.clear()
        self.interaction_history.clear()
        self.user_sessions.clear()
        logger.info("🧹 Logs de interacciones limpiados")