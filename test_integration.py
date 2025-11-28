import sys
import os

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ""))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "src"))
# test_integration.py
from src.core.data.user_models import UserProfile, Gender
from src.core.scoring.score_normalizer import ScoreNormalizer

# Probar calculate_similarity
user1 = UserProfile("user1", "session1", 25, Gender.MALE, "Spain")
user2 = UserProfile("user2", "session2", 30, Gender.MALE, "Spain")

similarity = user1.calculate_similarity(user2)
print(f"✅ Similitud entre usuarios: {similarity}")

# Probar ScoreNormalizer
normalizer = ScoreNormalizer()
rag_score, rag_conf = normalizer.normalize_rag_score(0.8)
collab_score, collab_conf = normalizer.normalize_collaborative_score(0.9, 3)
final_score, final_conf = normalizer.calculate_final_score(
    rag_score, collab_score, 2.0, rag_conf, collab_conf
)
print(f"✅ Score final: {final_score:.3f} (confianza: {final_conf:.3f})")