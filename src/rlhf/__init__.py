"""
RLHF Package — Reinforcement Learning from Human Feedback
Componentes: PolicyModel, RewardModel, PPOTrainer, PreferenceCollector, RLHFPipeline
"""
from .policy_model import PolicyModel
from .reward_model import RewardModel
from .ppo_trainer import PPOTrainer, RankingExperience
from .preference_collector import PreferenceCollector
from .rlhf_pipeline import RLHFPipeline
from .tensor_utils import ProductTensorizer

__all__ = [
    "PolicyModel",
    "RewardModel",
    "PPOTrainer",
    "RankingExperience",
    "PreferenceCollector",
    "RLHFPipeline",
    "ProductTensorizer",
]