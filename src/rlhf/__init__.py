# src/rlhf/__init__.py
"""
RLHF Package — Reinforcement Learning from Human Feedback
Componentes: PolicyModel, PointwiseRewardModel, PPOTrainer,
             PreferenceCollector, RLHFPipeline, ProductTensorizer
"""
from .policy_model             import PolicyModel
from .pointwise_reward_model   import PointwiseRewardModel, PointwiseMarginLoss
from .ppo_trainer              import PPOTrainer
from .preference_collector     import PreferenceCollector
from .rlhf_pipeline            import RLHFPipeline
from .tensor_utils             import ProductTensorizer

__all__ = [
    "PolicyModel",
    "PointwiseRewardModel",
    "PointwiseMarginLoss",
    "PPOTrainer",
    "PreferenceCollector",
    "RLHFPipeline",
    "ProductTensorizer",
]