"""
Reinforcement Learning module for trading.

This module provides production-ready implementations of:
- SAC (Soft Actor-Critic) algorithm
- Neural network architectures (Actor, Critic)
- Experience replay buffers (basic and prioritized)
- Trading environment compatible with Gymnasium
"""

from .sac_agent import SACAgent
from .networks import MLPActor, MLPCritic, TwinCritic, StateNormalizer
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer, NStepReplayBuffer
from .trading_env import TradingEnv

__all__ = [
    # Main agent
    'SACAgent',

    # Neural networks
    'MLPActor',
    'MLPCritic',
    'TwinCritic',
    'StateNormalizer',

    # Replay buffers
    'ReplayBuffer',
    'PrioritizedReplayBuffer',
    'NStepReplayBuffer',

    # Environment
    'TradingEnv',
]

__version__ = '1.0.0'
