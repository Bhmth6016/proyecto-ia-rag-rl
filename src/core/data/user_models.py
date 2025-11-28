from __future__ import annotations
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Dict, Optional, Any
from enum import Enum
import json
from pathlib import Path
import logging

logger = logging.getLogger(__name__)

class Gender(Enum):
    MALE = "male"
    FEMALE = "female" 
    OTHER = "other"
    PREFER_NOT_TO_SAY = "prefer_not_to_say"

class PriceSensitivity(Enum):
    LOW = "low"      # No le importa el precio
    MEDIUM = "medium" # Considera precio pero flexible  
    HIGH = "high"    # Muy sensible al precio

@dataclass
class SearchEvent:
    query: str
    timestamp: datetime
    results_count: int
    clicked_products: List[str] = field(default_factory=list)
    session_duration: float = 0.0

@dataclass  
class FeedbackEvent:
    query: str
    response: str
    rating: int
    timestamp: datetime
    products_shown: List[str] = field(default_factory=list)
    selected_product: Optional[str] = None

@dataclass
class PurchaseEvent:
    product_id: str
    timestamp: datetime 
    price: float
    category: str
    satisfaction: Optional[int] = None  # 1-5

@dataclass
class UserProfile:
    # Identificación
    user_id: str
    session_id: str
    
    # Datos demográficos (OBLIGATORIOS)
    age: int
    gender: Gender
    country: str
    language: str = "es"
    
    # Preferencias explícitas
    preferred_categories: List[str] = field(default_factory=list)
    preferred_brands: List[str] = field(default_factory=list)
    avoided_categories: List[str] = field(default_factory=list)
    price_sensitivity: PriceSensitivity = PriceSensitivity.MEDIUM
    preferred_price_range: Dict[str, float] = field(default_factory=lambda: {"min": 0, "max": 1000})
    
    # Comportamiento histórico
    search_history: List[SearchEvent] = field(default_factory=list)
    feedback_history: List[FeedbackEvent] = field(default_factory=list) 
    purchase_history: List[PurchaseEvent] = field(default_factory=list)
    
    # Metadata
    created_at: datetime = field(default_factory=datetime.now)
    last_active: datetime = field(default_factory=datetime.now)
    total_sessions: int = 1
    
    def calculate_similarity(self, other: UserProfile) -> float:
        """Calcula similitud robusta entre usuarios (0-1)"""
        try:
            if self.user_id == other.user_id:
                return 1.0
            
            similarity = 0.0
            total_weight = 0
            
            # 1. Edad (peso 25%) - diferencia máxima 40 años
            age_diff = abs(self.age - other.age)
            age_similarity = max(0, 1 - (age_diff / 40))
            similarity += age_similarity * 0.25
            total_weight += 0.25
            
            # 2. Género (peso 20%)
            if self.gender == other.gender:
                similarity += 1.0 * 0.20
            total_weight += 0.20
            
            # 3. País (peso 15%)
            if self.country.lower() == other.country.lower():
                similarity += 1.0 * 0.15
            total_weight += 0.15
            
            # 4. Categorías preferidas (peso 30%)
            if self.preferred_categories and other.preferred_categories:
                set_self = set(self.preferred_categories)
                set_other = set(other.preferred_categories)
                if set_self and set_other:
                    common_categories = set_self & set_other
                    category_similarity = len(common_categories) / max(len(set_self), len(set_other))
                    similarity += category_similarity * 0.30
            total_weight += 0.30
            
            # 5. Precio (peso 10%) - similitud en rango de precio preferido
            self_min, self_max = self.preferred_price_range.get('min', 0), self.preferred_price_range.get('max', 1000)
            other_min, other_max = other.preferred_price_range.get('min', 0), other.preferred_price_range.get('max', 1000)
            
            range_overlap = max(0, min(self_max, other_max) - max(self_min, other_min))
            range_union = max(self_max, other_max) - min(self_min, other_min)
            
            if range_union > 0:
                price_similarity = range_overlap / range_union
                similarity += price_similarity * 0.10
            total_weight += 0.10
            
            # Normalizar por el peso total aplicado
            return similarity / total_weight if total_weight > 0 else 0.0
            
        except Exception as e:
            logger.error(f"Error calculando similitud entre usuarios: {e}")
            return 0.0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convierte a dict para serialización"""
        return {
            "user_id": self.user_id,
            "session_id": self.session_id,
            "age": self.age,
            "gender": self.gender.value,
            "country": self.country,
            "language": self.language,
            "preferred_categories": self.preferred_categories,
            "preferred_brands": self.preferred_brands,
            "avoided_categories": self.avoided_categories,
            "price_sensitivity": self.price_sensitivity.value,
            "preferred_price_range": self.preferred_price_range,
            "search_history": [
                {
                    "query": event.query,
                    "timestamp": event.timestamp.isoformat(),
                    "results_count": event.results_count,
                    "clicked_products": event.clicked_products,
                    "session_duration": event.session_duration
                }
                for event in self.search_history
            ],
            "feedback_history": [
                {
                    "query": event.query,
                    "response": event.response,
                    "rating": event.rating,
                    "timestamp": event.timestamp.isoformat(),
                    "products_shown": event.products_shown,
                    "selected_product": event.selected_product
                }
                for event in self.feedback_history
            ],
            "purchase_history": [
                {
                    "product_id": event.product_id,
                    "timestamp": event.timestamp.isoformat(),
                    "price": event.price,
                    "category": event.category,
                    "satisfaction": event.satisfaction
                }
                for event in self.purchase_history
            ],
            "created_at": self.created_at.isoformat(),
            "last_active": self.last_active.isoformat(),
            "total_sessions": self.total_sessions
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> UserProfile:
        """Crea UserProfile desde dict"""
        profile = cls(
            user_id=data["user_id"],
            session_id=data["session_id"],
            age=data["age"],
            gender=Gender(data["gender"]),
            country=data["country"],
            language=data.get("language", "es"),
            preferred_categories=data.get("preferred_categories", []),
            preferred_brands=data.get("preferred_brands", []),
            avoided_categories=data.get("avoided_categories", []),
            price_sensitivity=PriceSensitivity(data.get("price_sensitivity", "medium")),
            preferred_price_range=data.get("preferred_price_range", {"min": 0, "max": 1000}),
            total_sessions=data.get("total_sessions", 1)
        )
        
        # Historial de búsquedas
        for event_data in data.get("search_history", []):
            profile.search_history.append(SearchEvent(
                query=event_data["query"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                results_count=event_data["results_count"],
                clicked_products=event_data.get("clicked_products", []),
                session_duration=event_data.get("session_duration", 0.0)
            ))
        
        # Historial de feedback
        for event_data in data.get("feedback_history", []):
            profile.feedback_history.append(FeedbackEvent(
                query=event_data["query"],
                response=event_data["response"],
                rating=event_data["rating"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                products_shown=event_data.get("products_shown", []),
                selected_product=event_data.get("selected_product")
            ))
        
        # Historial de compras
        for event_data in data.get("purchase_history", []):
            profile.purchase_history.append(PurchaseEvent(
                product_id=event_data["product_id"],
                timestamp=datetime.fromisoformat(event_data["timestamp"]),
                price=event_data["price"],
                category=event_data["category"],
                satisfaction=event_data.get("satisfaction")
            ))
        
        return profile
    
    def update_activity(self):
        """Actualiza última actividad"""
        self.last_active = datetime.now()
        self.total_sessions += 1
    
    def add_search_event(self, query: str, results_count: int, clicked_products: List[str] = None):
        """Añade evento de búsqueda"""
        self.search_history.append(SearchEvent(
            query=query,
            timestamp=datetime.now(),
            results_count=results_count,
            clicked_products=clicked_products or []
        ))
        self.update_activity()
    
    def add_feedback_event(self, query: str, response: str, rating: int, products_shown: List[str], selected_product: str = None):
        """Añade evento de feedback"""
        self.feedback_history.append(FeedbackEvent(
            query=query,
            response=response,
            rating=rating,
            timestamp=datetime.now(),
            products_shown=products_shown,
            selected_product=selected_product
        ))
        self.update_activity()