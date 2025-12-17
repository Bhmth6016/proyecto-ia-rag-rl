# src/core/data/user_manager.py
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime
import hashlib

from .user_models import UserProfile, Gender, PriceSensitivity

logger = logging.getLogger(__name__)

class UserManager:
    """Gestor centralizado de perfiles de usuario"""
    
    def __init__(self, storage_dir: Path = Path("data/users")):
        self.storage_dir = storage_dir
        self.storage_dir.mkdir(parents=True, exist_ok=True)
        self.active_profiles: Dict[str, UserProfile] = {}
        
    def generate_user_id(self, age: int, gender: str, country: str) -> str:
        """Genera ID único basado en datos demográficos"""
        base_string = f"{age}_{gender}_{country}_{datetime.now().timestamp()}"
        return hashlib.md5(base_string.encode()).hexdigest()[:12]
    
    def create_user_profile(self, 
                       age: int,
                       gender: str,
                       country: str,
                       language: str = "es",
                       preferred_categories: Optional[List[str]] = None,
                       preferred_brands: Optional[List[str]] = None) -> UserProfile:
        try:
            user_id = self.generate_user_id(age, gender, country)
            session_id = f"{user_id}_{int(datetime.now().timestamp())}"
            
            profile = UserProfile(
                user_id=user_id,
                session_id=session_id,
                age=age,
                gender=Gender(gender),
                country=country,
                language=language,
                preferred_categories=preferred_categories or [],
                preferred_brands=preferred_brands or []
            )
            
            # Guardar inmediatamente
            self.save_user_profile(profile)
            self.active_profiles[user_id] = profile
            
            logger.info(f"👤 Nuevo perfil creado: {user_id} ({age}{gender[0]}, {country})")
            return profile
            
        except Exception as e:
            logger.error(f"Error creando perfil de usuario: {e}")
            raise

    
    def get_user_profile(self, user_id: str) -> Optional[UserProfile]:
        """Obtiene perfil de usuario, cargando de disco si es necesario"""
        if user_id in self.active_profiles:
            return self.active_profiles[user_id]
        
        # Cargar desde almacenamiento
        profile_path = self.storage_dir / f"{user_id}.json"
        if profile_path.exists():
            try:
                with open(profile_path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                profile = UserProfile.from_dict(data)
                self.active_profiles[user_id] = profile
                return profile
            except Exception as e:
                logger.error(f"Error cargando perfil {user_id}: {e}")
        
        return None
    
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Guarda perfil de usuario en almacenamiento persistente"""
        try:
            profile_path = self.storage_dir / f"{profile.user_id}.json"
            with open(profile_path, 'w', encoding='utf-8') as f:
                json.dump(profile.to_dict(), f, ensure_ascii=False, indent=2)
            return True
        except Exception as e:
            logger.error(f"Error guardando perfil {profile.user_id}: {e}")
            return False
    
    def find_similar_users(self, target_profile: UserProfile, min_similarity: float = 0.6) -> List[UserProfile]:
        """Encuentra usuarios similares basado en demografía y preferencias"""
        similar_users = []
        
        # Buscar en archivos de usuarios
        for user_file in self.storage_dir.glob("*.json"):
            if user_file.name == f"{target_profile.user_id}.json":
                continue  # Saltar el usuario actual
                
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                other_profile = UserProfile.from_dict(data)
                
                similarity = target_profile.calculate_similarity(other_profile)
                if similarity >= min_similarity:
                    similar_users.append((similarity, other_profile))
                    
            except Exception as e:
                logger.debug(f"Error procesando {user_file}: {e}")
        
        # Ordenar por similitud descendente
        similar_users.sort(key=lambda x: x[0], reverse=True)
        return [profile for _, profile in similar_users]
    
    def get_demographic_stats(self) -> Dict[str, Any]:
        """Obtiene estadísticas demográficas de todos los usuarios"""
        stats = {
            "total_users": 0,
            "age_distribution": {},
            "gender_distribution": {},
            "country_distribution": {},
            "avg_sessions_per_user": 0,
            "total_searches": 0,
            "total_feedbacks": 0
        }
        
        total_sessions = 0
        total_searches = 0
        total_feedbacks = 0
        
        for user_file in self.storage_dir.glob("*.json"):
            try:
                with open(user_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                
                stats["total_users"] += 1
                
                # Edad
                age_group = f"{(data['age'] // 10) * 10}-{(data['age'] // 10) * 10 + 9}"
                stats["age_distribution"][age_group] = stats["age_distribution"].get(age_group, 0) + 1
                
                # Género
                gender = data['gender']
                stats["gender_distribution"][gender] = stats["gender_distribution"].get(gender, 0) + 1
                
                # País
                country = data['country']
                stats["country_distribution"][country] = stats["country_distribution"].get(country, 0) + 1
                
                # Sesiones y actividad
                total_sessions += data.get('total_sessions', 1)
                total_searches += len(data.get('search_history', []))
                total_feedbacks += len(data.get('feedback_history', []))
                
            except Exception as e:
                logger.debug(f"Error procesando stats de {user_file}: {e}")
        
        if stats["total_users"] > 0:
            stats["avg_sessions_per_user"] = total_sessions / stats["total_users"]
            stats["total_searches"] = total_searches
            stats["total_feedbacks"] = total_feedbacks
        
        return stats