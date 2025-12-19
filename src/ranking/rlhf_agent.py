# src/ranking/rlhf_agent.py
"""
Agente RLHF - Bandit contextual (LINUCB)
El CORAZÓN del paper
"""
import numpy as np
from typing import List, Dict, Any, Optional
import logging
import sys
import os

# Añadir src al path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

logger = logging.getLogger(__name__)

# Importación absoluta
try:
    from data.canonicalizer import CanonicalProduct
except ImportError:
    sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    from data.canonicalizer import CanonicalProduct

class LinUCB:
    """Bandit contextual LINUCB para ranking"""
    
    def __init__(self, alpha: float = 0.1):
        self.alpha = alpha  # Parámetro de exploración
        self.A = None  # Matriz A (d x d)
        self.b = None  # Vector b (d x 1)
        self.d = None  # Dimensión de características
        self.theta = None  # Vector de parámetros
        self.trained = False
    
    def initialize(self, feature_dim: int):
        """Inicializa el bandit"""
        self.d = feature_dim
        self.A = np.identity(feature_dim)
        self.b = np.zeros((feature_dim, 1))
        self.theta = np.zeros((feature_dim, 1))
        self.trained = False
        logger.info(f"🤖 LINUCB inicializado con d={feature_dim}, alpha={self.alpha}")
    
    def select_action(
        self,
        context_features: List[np.ndarray],
        candidate_indices: List[int]
    ) -> int:
        """
        Selecciona acción (producto) usando LINUCB
        
        Args:
            context_features: Lista de vectores de características [d]
            candidate_indices: Índices de productos candidatos
            
        Returns:
            Índice del producto seleccionado
        """
        if not self.trained:
            # Fase de exploración inicial
            return np.random.choice(candidate_indices)
        
        # Calcular scores UCB para cada candidato
        scores = []
        for idx in candidate_indices:
            x = context_features[idx].reshape(-1, 1)  # [d, 1]
            
            # Score esperado
            expected = self.theta.T @ x
            
            # Varianza/exploración
            A_inv = np.linalg.inv(self.A)
            exploration = self.alpha * np.sqrt(x.T @ A_inv @ x)
            
            # Score UCB
            ucb_score = expected + exploration
            scores.append(float(ucb_score))
        
        # Seleccionar acción con mayor score UCB
        best_idx = np.argmax(scores)
        return candidate_indices[best_idx]
    
    def update(self, chosen_idx: int, context_features: List[np.ndarray], reward: float):
        """Actualiza modelo con recompensa observada"""
        x = context_features[chosen_idx].reshape(-1, 1)  # [d, 1]
        
        # Actualizar matrices
        self.A += x @ x.T
        self.b += x * reward
        
        # Recalcular theta
        A_inv = np.linalg.inv(self.A)
        self.theta = A_inv @ self.b
        
        self.trained = True
    
    def get_weights(self) -> np.ndarray:
        """Obtiene pesos aprendidos"""
        return self.theta.flatten()


class RLHFAgent:
    """Agente RLHF para ranking adaptativo"""
    
    def __init__(self, feature_extractor, alpha: float = 0.1):
        self.feature_extractor = feature_extractor
        self.bandit = LinUCB(alpha=alpha)
        self.feature_dim = None
        self.history = []
        
    def initialize_for_query(self, query_features: Dict[str, float]):
        """Inicializa para una nueva query"""
        if self.feature_dim is None:
            # Determinar dimensión basado en extractor
            self.feature_dim = len(self._vectorize_features(query_features, {}))
            self.bandit.initialize(self.feature_dim)
    
    def select_ranking(
        self,
        query_features: Dict[str, float],
        products: List[CanonicalProduct],
        baseline_ranking: List[int]
    ) -> List[int]:
        """
        Selecciona ranking usando bandit contextual
        
        Args:
            query_features: Características de la query
            products: Lista de productos
            baseline_ranking: Ranking inicial del baseline
            
        Returns:
            Nuevo ranking (índices de productos)
        """
        # Extraer características de contexto
        context_features = []
        for product in products:
            features = self.feature_extractor(query_features, product)
            context_features.append(self._vectorize_features(query_features, features))
        
        # Seleccionar ranking usando bandit
        new_ranking = []
        remaining_indices = list(range(len(products)))
        
        for _ in range(min(10, len(products))):  # Top-10
            chosen_idx = self.bandit.select_action(context_features, remaining_indices)
            new_ranking.append(chosen_idx)
            remaining_indices.remove(chosen_idx)
        
        return new_ranking
    
    def update_with_feedback(
        self,
        query_features: Dict[str, float],
        products: List[CanonicalProduct],
        shown_indices: List[int],
        selected_idx: int,
        rating: float
    ):
        """Actualiza agente con feedback"""
        # Convertir rating a recompensa
        reward = self._rating_to_reward(rating)
        
        # Extraer características del producto seleccionado
        selected_product = products[selected_idx]
        features = self.feature_extractor(query_features, selected_product)
        context_features = []
        
        for product in products:
            prod_features = self.feature_extractor(query_features, product)
            context_features.append(self._vectorize_features(query_features, prod_features))
        
        # Actualizar bandit
        self.bandit.update(selected_idx, context_features, reward)
        
        # Guardar en historial
        self.history.append({
            "query_features": query_features,
            "selected_product": selected_product.id,
            "rating": rating,
            "reward": reward,
            "timestamp": np.datetime64('now')
        })
    
    def _vectorize_features(self, query_features: Dict, product_features: Dict) -> np.ndarray:
        """Convierte características a vector"""
        # Combinar query y producto features
        combined = {}
        combined.update({f"query_{k}": v for k, v in query_features.items()})
        combined.update({f"product_{k}": v for k, v in product_features.items()})
        
        # Ordenar para consistencia
        keys = sorted(combined.keys())
        vector = np.array([combined[k] for k in keys])
        
        return vector
    
    def _rating_to_reward(self, rating: float) -> float:
        """Convierte rating 1-5 a recompensa"""
        # Mapeo lineal: 1→0, 3→0.5, 5→1
        return max(0.0, min(1.0, (rating - 1) / 4.0))
    
    def get_learning_curve(self) -> List[float]:
        """Obtiene curva de aprendizaje (recompensas promedio)"""
        if not self.history:
            return []
        
        rewards = [h["reward"] for h in self.history]
        cumulative_avg = np.cumsum(rewards) / np.arange(1, len(rewards) + 1)
        
        return cumulative_avg.tolist()