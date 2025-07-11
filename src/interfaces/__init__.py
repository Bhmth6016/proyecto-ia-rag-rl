#src/interfaces/__init__.py
from .ui import launch_ui as AmazonProductUI
from .cli import AmazonRecommendationCLI

__all__ = ['AmazonProductUI', 'AmazonRecommendationCLI']