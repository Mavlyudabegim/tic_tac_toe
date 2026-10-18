# __init__.py
"""Tic-Tac-Toe with tabular Q-learning (self-play, negamax targets, symmetry)."""
from .agents import MinimaxAgent, QAgent, RandomAgent
from .train import TrainConfig, train

__all__ = ["QAgent", "RandomAgent", "MinimaxAgent", "TrainConfig", "train"]