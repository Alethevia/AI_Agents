"""
Neural network architectures for SAC algorithm.

This module provides production-ready neural network implementations including:
- MLPActor: Policy network with tanh-squashed Gaussian distribution
- MLPCritic: Q-value networks for value estimation
- State normalization layers for stable training
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import Normal
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class StateNormalizer(nn.Module):
    """
    Online state normalization layer using running statistics.

    Normalizes states using running mean and standard deviation for stable training.
    """

    def __init__(self, state_dim: int, epsilon: float = 1e-8):
        """
        Initialize state normalizer.

        Args:
            state_dim: Dimension of the state space
            epsilon: Small constant for numerical stability
        """
        super().__init__()
        self.epsilon = epsilon

        # Running statistics (not trainable parameters)
        self.register_buffer('running_mean', torch.zeros(state_dim))
        self.register_buffer('running_var', torch.ones(state_dim))
        self.register_buffer('count', torch.tensor(0, dtype=torch.long))

    def forward(self, state: torch.Tensor, update_stats: bool = True) -> torch.Tensor:
        """
        Normalize state using running statistics.

        Args:
            state: Input state tensor
            update_stats: Whether to update running statistics

        Returns:
            Normalized state tensor
        """
        if update_stats and self.training:
            # Update running statistics
            batch_mean = state.mean(dim=0)
            batch_var = state.var(dim=0)
            batch_count = state.size(0)

            # Update count
            new_count = self.count + batch_count

            # Update running mean
            delta = batch_mean - self.running_mean
            self.running_mean += delta * batch_count / new_count

            # Update running variance
            m_a = self.running_var * self.count
            m_b = batch_var * batch_count
            M2 = m_a + m_b + delta ** 2 * self.count * batch_count / new_count
            self.running_var = M2 / new_count

            self.count = new_count

        # Normalize
        normalized = (state - self.running_mean) / (torch.sqrt(self.running_var) + self.epsilon)
        return normalized


def weights_init_(module: nn.Module, gain: float = 1.0):
    """
    Initialize neural network weights using orthogonal initialization.

    Args:
        module: Neural network module
        gain: Gain parameter for initialization
    """
    if isinstance(module, nn.Linear):
        nn.init.orthogonal_(module.weight, gain=gain)
        if module.bias is not None:
            nn.init.constant_(module.bias, 0.0)


class MLPActor(nn.Module):
    """
    Multi-Layer Perceptron Actor network with tanh-squashed Gaussian policy.

    Outputs mean and log_std for a diagonal Gaussian distribution.
    Actions are squashed using tanh to ensure they stay within [-1, 1].
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        log_std_min: float = -20.0,
        log_std_max: float = 2.0,
        use_state_norm: bool = True
    ):
        """
        Initialize MLPActor network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            log_std_min: Minimum log standard deviation
            log_std_max: Maximum log standard deviation
            use_state_norm: Whether to use state normalization
        """
        super().__init__()

        self.action_dim = action_dim
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

        # State normalization
        self.state_normalizer = StateNormalizer(state_dim) if use_state_norm else None

        # Build network layers
        layers = []
        input_dim = state_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        self.shared_net = nn.Sequential(*layers)

        # Separate heads for mean and log_std
        self.mean_head = nn.Linear(input_dim, action_dim)
        self.log_std_head = nn.Linear(input_dim, action_dim)

        # Initialize weights
        self.apply(lambda m: weights_init_(m, gain=1.0))
        # Initialize output layer with smaller weights
        weights_init_(self.mean_head, gain=0.01)
        weights_init_(self.log_std_head, gain=0.01)

        logger.info(f"Initialized MLPActor: state_dim={state_dim}, action_dim={action_dim}, "
                   f"hidden_dims={hidden_dims}")

    def forward(
        self,
        state: torch.Tensor,
        deterministic: bool = False,
        with_logprob: bool = True
    ) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """
        Forward pass to sample actions.

        Args:
            state: Current state
            deterministic: If True, return mean action without sampling
            with_logprob: If True, return log probability of sampled action

        Returns:
            action: Sampled action
            log_prob: Log probability of action (None if with_logprob=False)
        """
        # Normalize state if enabled
        if self.state_normalizer is not None:
            state = self.state_normalizer(state, update_stats=self.training)

        # Forward through shared network
        features = self.shared_net(state)

        # Get mean and log_std
        mean = self.mean_head(features)
        log_std = self.log_std_head(features)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        std = log_std.exp()

        # Create distribution
        if deterministic:
            # Deterministic action (mean)
            action = torch.tanh(mean)
            log_prob = None if not with_logprob else torch.zeros(
                mean.size(0), device=mean.device
            )
        else:
            # Sample from Gaussian distribution
            normal = Normal(mean, std)
            x_t = normal.rsample()  # Reparameterization trick
            action = torch.tanh(x_t)

            if with_logprob:
                # Compute log probability with tanh correction
                # log_prob = log_prob_gaussian - log(1 - tanh^2(x))
                log_prob = normal.log_prob(x_t)
                # Sum over action dimensions
                log_prob -= torch.log(1 - action.pow(2) + 1e-6)
                log_prob = log_prob.sum(dim=-1, keepdim=True)
            else:
                log_prob = None

        return action, log_prob

    def get_action(self, state: np.ndarray, deterministic: bool = False) -> np.ndarray:
        """
        Get action from state (for inference).

        Args:
            state: Current state as numpy array
            deterministic: If True, return mean action

        Returns:
            Action as numpy array
        """
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(
                next(self.parameters()).device
            )
            action, _ = self.forward(state_tensor, deterministic=deterministic, with_logprob=False)
            return action.cpu().numpy()[0]


class MLPCritic(nn.Module):
    """
    Multi-Layer Perceptron Critic network (Q-function).

    Estimates the Q-value Q(s,a) for state-action pairs.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_state_norm: bool = True
    ):
        """
        Initialize MLPCritic network.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            use_state_norm: Whether to use state normalization
        """
        super().__init__()

        # State normalization
        self.state_normalizer = StateNormalizer(state_dim) if use_state_norm else None

        # Build network layers
        layers = []
        input_dim = state_dim + action_dim

        for hidden_dim in hidden_dims:
            layers.append(nn.Linear(input_dim, hidden_dim))
            layers.append(nn.ReLU())
            input_dim = hidden_dim

        layers.append(nn.Linear(input_dim, 1))

        self.q_net = nn.Sequential(*layers)

        # Initialize weights
        self.apply(lambda m: weights_init_(m, gain=1.0))

        logger.info(f"Initialized MLPCritic: state_dim={state_dim}, action_dim={action_dim}, "
                   f"hidden_dims={hidden_dims}")

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Forward pass to compute Q-value.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Q-value Q(s,a)
        """
        # Normalize state if enabled
        if self.state_normalizer is not None:
            state = self.state_normalizer(state, update_stats=self.training)

        # Concatenate state and action
        x = torch.cat([state, action], dim=-1)

        # Forward through network
        q_value = self.q_net(x)

        return q_value


class TwinCritic(nn.Module):
    """
    Twin Q-networks for reducing overestimation bias.

    Uses two separate Q-networks and takes the minimum Q-value.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        use_state_norm: bool = True
    ):
        """
        Initialize twin critic networks.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Tuple of hidden layer dimensions
            use_state_norm: Whether to use state normalization
        """
        super().__init__()

        self.q1 = MLPCritic(state_dim, action_dim, hidden_dims, use_state_norm)
        self.q2 = MLPCritic(state_dim, action_dim, hidden_dims, use_state_norm)

        logger.info("Initialized TwinCritic with two Q-networks")

    def forward(self, state: torch.Tensor, action: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward pass through both Q-networks.

        Args:
            state: Current state
            action: Action taken

        Returns:
            q1_value: Q-value from first network
            q2_value: Q-value from second network
        """
        q1 = self.q1(state, action)
        q2 = self.q2(state, action)
        return q1, q2

    def q_min(self, state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """
        Get minimum Q-value from twin networks.

        Args:
            state: Current state
            action: Action taken

        Returns:
            Minimum Q-value
        """
        q1, q2 = self.forward(state, action)
        return torch.min(q1, q2)


if __name__ == "__main__":
    # Test the networks
    logging.basicConfig(level=logging.INFO)

    state_dim = 10
    action_dim = 3
    batch_size = 32

    # Test actor
    actor = MLPActor(state_dim, action_dim)
    states = torch.randn(batch_size, state_dim)
    actions, log_probs = actor(states)
    print(f"Actor - Actions shape: {actions.shape}, Log probs shape: {log_probs.shape}")
    print(f"Actions range: [{actions.min():.3f}, {actions.max():.3f}]")

    # Test critic
    critic = TwinCritic(state_dim, action_dim)
    q1, q2 = critic(states, actions)
    print(f"Critic - Q1 shape: {q1.shape}, Q2 shape: {q2.shape}")

    # Test state normalizer
    normalizer = StateNormalizer(state_dim)
    normalized_states = normalizer(states)
    print(f"Normalized states mean: {normalized_states.mean():.3f}, std: {normalized_states.std():.3f}")

    print("All network tests passed!")
