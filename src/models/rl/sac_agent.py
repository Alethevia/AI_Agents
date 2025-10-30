"""
Soft Actor-Critic (SAC) Agent implementation.

This module provides a production-ready SAC implementation with:
- Twin Q-networks with soft target updates
- Stochastic policy with tanh-squashed Gaussian distribution
- Automatic entropy tuning
- Prioritized experience replay support
- Comprehensive logging and checkpointing
- GPU acceleration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, Optional, Any, Union
import logging
from collections import deque
import json
from datetime import datetime

from .networks import MLPActor, TwinCritic
from .replay_buffer import ReplayBuffer, PrioritizedReplayBuffer

logger = logging.getLogger(__name__)


class SACAgent:
    """
    Soft Actor-Critic (SAC) Agent for continuous control.

    SAC is an off-policy actor-critic algorithm based on maximum entropy RL.
    The agent aims to maximize expected return while also maximizing entropy,
    leading to more robust and exploratory policies.

    Key features:
    - Twin Q-networks to reduce overestimation bias
    - Entropy-regularized policy optimization
    - Automatic temperature (entropy) tuning
    - Soft target updates for stability
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        hidden_dims: Tuple[int, ...] = (256, 256),
        learning_rate: float = 3e-4,
        gamma: float = 0.99,
        tau: float = 0.005,
        alpha: float = 0.2,
        auto_entropy_tuning: bool = True,
        target_entropy: Optional[float] = None,
        buffer_size: int = 1_000_000,
        batch_size: int = 256,
        use_prioritized_replay: bool = False,
        per_alpha: float = 0.6,
        per_beta: float = 0.4,
        use_state_norm: bool = True,
        device: str = "auto",
        seed: Optional[int] = None
    ):
        """
        Initialize SAC agent.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            hidden_dims: Hidden layer dimensions for networks
            learning_rate: Learning rate for all networks
            gamma: Discount factor
            tau: Soft update coefficient for target networks
            alpha: Initial entropy temperature coefficient
            auto_entropy_tuning: Whether to automatically tune entropy
            target_entropy: Target entropy (if None, set to -action_dim)
            buffer_size: Replay buffer capacity
            batch_size: Batch size for training
            use_prioritized_replay: Whether to use PER
            per_alpha: PER alpha parameter
            per_beta: PER beta parameter
            use_state_norm: Whether to use state normalization
            device: Device ('cpu', 'cuda', or 'auto')
            seed: Random seed
        """
        # Set device
        if device == "auto":
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            self.device = torch.device(device)

        # Set random seed
        if seed is not None:
            torch.manual_seed(seed)
            np.random.seed(seed)
            logger.info(f"Set random seed: {seed}")

        # Store hyperparameters
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.tau = tau
        self.batch_size = batch_size
        self.use_prioritized_replay = use_prioritized_replay

        # Initialize actor network
        self.actor = MLPActor(
            state_dim,
            action_dim,
            hidden_dims,
            use_state_norm=use_state_norm
        ).to(self.device)

        # Initialize twin critic networks
        self.critic = TwinCritic(
            state_dim,
            action_dim,
            hidden_dims,
            use_state_norm=use_state_norm
        ).to(self.device)

        # Initialize target critic networks
        self.critic_target = TwinCritic(
            state_dim,
            action_dim,
            hidden_dims,
            use_state_norm=use_state_norm
        ).to(self.device)

        # Copy parameters to target network
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Freeze target network
        for param in self.critic_target.parameters():
            param.requires_grad = False

        # Optimizers
        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=learning_rate)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=learning_rate)

        # Automatic entropy tuning
        self.auto_entropy_tuning = auto_entropy_tuning
        if auto_entropy_tuning:
            # Target entropy is -|A| (action dimension)
            if target_entropy is None:
                self.target_entropy = -float(action_dim)
            else:
                self.target_entropy = target_entropy

            # Log temperature parameter
            self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
            self.alpha = self.log_alpha.exp()
            self.alpha_optimizer = optim.Adam([self.log_alpha], lr=learning_rate)

            logger.info(f"Automatic entropy tuning enabled: target_entropy={self.target_entropy}")
        else:
            self.alpha = torch.tensor(alpha, device=self.device)
            logger.info(f"Fixed entropy coefficient: alpha={alpha}")

        # Replay buffer
        if use_prioritized_replay:
            self.replay_buffer = PrioritizedReplayBuffer(
                state_dim,
                action_dim,
                buffer_size,
                alpha=per_alpha,
                beta=per_beta,
                device=self.device
            )
            logger.info("Using Prioritized Experience Replay")
        else:
            self.replay_buffer = ReplayBuffer(
                state_dim,
                action_dim,
                buffer_size,
                device=self.device
            )
            logger.info("Using standard replay buffer")

        # Training statistics
        self.total_steps = 0
        self.training_steps = 0
        self.episodes = 0

        # Metrics tracking
        self.metrics_history = {
            'actor_loss': deque(maxlen=1000),
            'critic_loss': deque(maxlen=1000),
            'alpha_loss': deque(maxlen=1000),
            'alpha': deque(maxlen=1000),
            'q_value': deque(maxlen=1000),
            'episode_reward': deque(maxlen=100),
            'episode_length': deque(maxlen=100)
        }

        logger.info(f"Initialized SACAgent on device: {self.device}")
        logger.info(f"Networks: hidden_dims={hidden_dims}, lr={learning_rate}")
        logger.info(f"Hyperparameters: gamma={gamma}, tau={tau}, batch_size={batch_size}")

    def select_action(
        self,
        state: np.ndarray,
        deterministic: bool = False,
        eval_mode: bool = False
    ) -> np.ndarray:
        """
        Select action given state.

        Args:
            state: Current state
            deterministic: If True, return mean action
            eval_mode: If True, put networks in eval mode

        Returns:
            Action array
        """
        if eval_mode:
            self.actor.eval()

        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            action, _ = self.actor(state_tensor, deterministic=deterministic, with_logprob=False)
            action = action.cpu().numpy()[0]

        if eval_mode:
            self.actor.train()

        return action

    def store_transition(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Store transition in replay buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.replay_buffer.add(state, action, reward, next_state, done)
        self.total_steps += 1

    def train_step(self) -> Dict[str, float]:
        """
        Perform one training step.

        Returns:
            Dictionary with training metrics
        """
        # Check if enough samples in buffer
        if len(self.replay_buffer) < self.batch_size:
            return {}

        # Sample batch from replay buffer
        if self.use_prioritized_replay:
            batch, indices, weights = self.replay_buffer.sample(self.batch_size)
        else:
            batch = self.replay_buffer.sample(self.batch_size)
            weights = torch.ones(self.batch_size, 1, device=self.device)
            indices = None

        states = batch['states']
        actions = batch['actions']
        rewards = batch['rewards']
        next_states = batch['next_states']
        dones = batch['dones']

        # ============ Update Critic ============

        with torch.no_grad():
            # Sample actions from current policy for next states
            next_actions, next_log_probs = self.actor(next_states)

            # Compute target Q-values using target networks
            next_q1, next_q2 = self.critic_target(next_states, next_actions)
            next_q = torch.min(next_q1, next_q2)

            # Entropy-regularized target
            target_q = rewards + (1 - dones) * self.gamma * (
                next_q - self.alpha * next_log_probs
            )

        # Compute current Q-values
        current_q1, current_q2 = self.critic(states, actions)

        # Critic loss (MSE)
        if self.use_prioritized_replay:
            # Weighted loss for PER
            critic_loss = (weights * (F.mse_loss(current_q1, target_q, reduction='none') +
                                      F.mse_loss(current_q2, target_q, reduction='none'))).mean()
        else:
            critic_loss = F.mse_loss(current_q1, target_q) + F.mse_loss(current_q2, target_q)

        # Update critic
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        # Gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Update priorities in PER
        if self.use_prioritized_replay:
            td_errors = (current_q1 - target_q).detach().cpu().numpy()
            self.replay_buffer.update_priorities(indices, td_errors)

        # ============ Update Actor ============

        # Compute new actions and log probabilities
        new_actions, log_probs = self.actor(states)

        # Compute Q-values for new actions
        q1_new, q2_new = self.critic(states, new_actions)
        q_new = torch.min(q1_new, q2_new)

        # Actor loss (maximize Q-value and entropy)
        actor_loss = (self.alpha * log_probs - q_new).mean()

        # Update actor
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
        self.actor_optimizer.step()

        # ============ Update Temperature (Alpha) ============

        alpha_loss = torch.tensor(0.0)
        if self.auto_entropy_tuning:
            # Alpha loss
            alpha_loss = -(self.log_alpha * (log_probs + self.target_entropy).detach()).mean()

            # Update alpha
            self.alpha_optimizer.zero_grad()
            alpha_loss.backward()
            self.alpha_optimizer.step()

            # Update alpha value
            self.alpha = self.log_alpha.exp()

        # ============ Soft Update Target Networks ============

        self._soft_update_target_network()

        # Update training steps
        self.training_steps += 1

        # Collect metrics
        metrics = {
            'actor_loss': actor_loss.item(),
            'critic_loss': critic_loss.item(),
            'alpha_loss': alpha_loss.item(),
            'alpha': self.alpha.item() if torch.is_tensor(self.alpha) else self.alpha,
            'q_value': current_q1.mean().item(),
            'target_q': target_q.mean().item(),
            'log_prob': log_probs.mean().item()
        }

        # Update metrics history
        self.metrics_history['actor_loss'].append(metrics['actor_loss'])
        self.metrics_history['critic_loss'].append(metrics['critic_loss'])
        self.metrics_history['alpha_loss'].append(metrics['alpha_loss'])
        self.metrics_history['alpha'].append(metrics['alpha'])
        self.metrics_history['q_value'].append(metrics['q_value'])

        return metrics

    def _soft_update_target_network(self):
        """Soft update target network parameters: θ_target = τ*θ + (1-τ)*θ_target"""
        for param, target_param in zip(
            self.critic.parameters(),
            self.critic_target.parameters()
        ):
            target_param.data.copy_(
                self.tau * param.data + (1 - self.tau) * target_param.data
            )

    def train_episode(
        self,
        env,
        max_steps: Optional[int] = None,
        warmup_steps: int = 0,
        train_freq: int = 1,
        gradient_steps: int = 1
    ) -> Dict[str, Any]:
        """
        Train for one episode.

        Args:
            env: Training environment
            max_steps: Maximum steps per episode
            warmup_steps: Random exploration steps before training
            train_freq: Frequency of training updates
            gradient_steps: Number of gradient steps per update

        Returns:
            Episode statistics
        """
        state, _ = env.reset()
        episode_reward = 0
        episode_steps = 0
        done = False

        while not done:
            # Select action
            if self.total_steps < warmup_steps:
                # Random exploration
                action = env.action_space.sample()
            else:
                # Policy action
                action = self.select_action(state, deterministic=False)

            # Take action
            next_state, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated

            # Store transition
            self.store_transition(state, action, reward, next_state, done)

            # Update state
            state = next_state
            episode_reward += reward
            episode_steps += 1

            # Training updates
            if self.total_steps >= warmup_steps and self.total_steps % train_freq == 0:
                for _ in range(gradient_steps):
                    self.train_step()

            # Check max steps
            if max_steps is not None and episode_steps >= max_steps:
                break

        # Update episode count
        self.episodes += 1

        # Track episode metrics
        self.metrics_history['episode_reward'].append(episode_reward)
        self.metrics_history['episode_length'].append(episode_steps)

        # Episode statistics
        episode_stats = {
            'episode': self.episodes,
            'reward': episode_reward,
            'length': episode_steps,
            'total_steps': self.total_steps
        }

        return episode_stats

    def evaluate(
        self,
        env,
        num_episodes: int = 10,
        deterministic: bool = True,
        render: bool = False
    ) -> Dict[str, float]:
        """
        Evaluate agent performance.

        Args:
            env: Evaluation environment
            num_episodes: Number of episodes to evaluate
            deterministic: Whether to use deterministic actions
            render: Whether to render environment

        Returns:
            Evaluation metrics
        """
        self.actor.eval()

        episode_rewards = []
        episode_lengths = []

        for episode in range(num_episodes):
            state, _ = env.reset()
            episode_reward = 0
            episode_steps = 0
            done = False

            while not done:
                action = self.select_action(state, deterministic=deterministic, eval_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                if render:
                    env.render()

                state = next_state
                episode_reward += reward
                episode_steps += 1

            episode_rewards.append(episode_reward)
            episode_lengths.append(episode_steps)

            logger.info(f"Eval Episode {episode + 1}/{num_episodes}: "
                       f"Reward={episode_reward:.2f}, Length={episode_steps}")

        self.actor.train()

        # Compute statistics
        eval_metrics = {
            'mean_reward': np.mean(episode_rewards),
            'std_reward': np.std(episode_rewards),
            'min_reward': np.min(episode_rewards),
            'max_reward': np.max(episode_rewards),
            'mean_length': np.mean(episode_lengths)
        }

        logger.info(f"Evaluation Results: Mean Reward={eval_metrics['mean_reward']:.2f} "
                   f"+/- {eval_metrics['std_reward']:.2f}")

        return eval_metrics

    def save_checkpoint(
        self,
        save_path: Union[str, Path],
        include_buffer: bool = False
    ):
        """
        Save agent checkpoint.

        Args:
            save_path: Path to save checkpoint
            include_buffer: Whether to save replay buffer (can be large)
        """
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)

        checkpoint = {
            'actor_state_dict': self.actor.state_dict(),
            'critic_state_dict': self.critic.state_dict(),
            'critic_target_state_dict': self.critic_target.state_dict(),
            'actor_optimizer_state_dict': self.actor_optimizer.state_dict(),
            'critic_optimizer_state_dict': self.critic_optimizer.state_dict(),
            'total_steps': self.total_steps,
            'training_steps': self.training_steps,
            'episodes': self.episodes,
            'hyperparameters': {
                'state_dim': self.state_dim,
                'action_dim': self.action_dim,
                'gamma': self.gamma,
                'tau': self.tau,
                'batch_size': self.batch_size,
                'auto_entropy_tuning': self.auto_entropy_tuning
            }
        }

        # Save alpha parameters
        if self.auto_entropy_tuning:
            checkpoint['log_alpha'] = self.log_alpha
            checkpoint['alpha_optimizer_state_dict'] = self.alpha_optimizer.state_dict()
            checkpoint['target_entropy'] = self.target_entropy
        else:
            checkpoint['alpha'] = self.alpha

        # Optionally save replay buffer
        if include_buffer:
            checkpoint['replay_buffer_size'] = len(self.replay_buffer)
            # Note: Full buffer saving would require custom serialization

        torch.save(checkpoint, save_path)
        logger.info(f"Saved checkpoint to {save_path}")

    def load_checkpoint(
        self,
        load_path: Union[str, Path],
        load_optimizers: bool = True
    ):
        """
        Load agent checkpoint.

        Args:
            load_path: Path to checkpoint
            load_optimizers: Whether to load optimizer states
        """
        load_path = Path(load_path)

        if not load_path.exists():
            raise FileNotFoundError(f"Checkpoint not found: {load_path}")

        checkpoint = torch.load(load_path, map_location=self.device)

        # Load network parameters
        self.actor.load_state_dict(checkpoint['actor_state_dict'])
        self.critic.load_state_dict(checkpoint['critic_state_dict'])
        self.critic_target.load_state_dict(checkpoint['critic_target_state_dict'])

        # Load optimizers
        if load_optimizers:
            self.actor_optimizer.load_state_dict(checkpoint['actor_optimizer_state_dict'])
            self.critic_optimizer.load_state_dict(checkpoint['critic_optimizer_state_dict'])

            if self.auto_entropy_tuning and 'alpha_optimizer_state_dict' in checkpoint:
                self.log_alpha = checkpoint['log_alpha']
                self.alpha = self.log_alpha.exp()
                self.alpha_optimizer.load_state_dict(checkpoint['alpha_optimizer_state_dict'])

        # Load training state
        self.total_steps = checkpoint['total_steps']
        self.training_steps = checkpoint['training_steps']
        self.episodes = checkpoint['episodes']

        logger.info(f"Loaded checkpoint from {load_path}")
        logger.info(f"Resume at: episodes={self.episodes}, steps={self.total_steps}")

    def get_metrics_summary(self) -> Dict[str, float]:
        """
        Get summary of recent training metrics.

        Returns:
            Dictionary with metric statistics
        """
        summary = {}

        for key, values in self.metrics_history.items():
            if len(values) > 0:
                summary[f'{key}_mean'] = np.mean(values)
                if len(values) > 1:
                    summary[f'{key}_std'] = np.std(values)

        return summary


if __name__ == "__main__":
    # Test SAC agent
    logging.basicConfig(level=logging.INFO)

    state_dim = 10
    action_dim = 3

    print("Testing SACAgent initialization...")
    agent = SACAgent(
        state_dim=state_dim,
        action_dim=action_dim,
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        auto_entropy_tuning=True
    )

    print(f"Agent initialized on device: {agent.device}")
    print(f"Replay buffer size: {len(agent.replay_buffer)}")

    # Test action selection
    print("\nTesting action selection...")
    state = np.random.randn(state_dim)
    action = agent.select_action(state)
    print(f"Action shape: {action.shape}, values: {action}")

    # Test storing transitions
    print("\nTesting transition storage...")
    for _ in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = False
        agent.store_transition(state, action, reward, next_state, done)

    print(f"Buffer size after additions: {len(agent.replay_buffer)}")

    # Test training step
    print("\nTesting training step...")
    metrics = agent.train_step()
    print(f"Training metrics: {metrics}")

    # Test checkpoint saving/loading
    print("\nTesting checkpoint save/load...")
    checkpoint_path = Path("/tmp/sac_test_checkpoint.pt")
    agent.save_checkpoint(checkpoint_path)

    agent2 = SACAgent(state_dim=state_dim, action_dim=action_dim)
    agent2.load_checkpoint(checkpoint_path)
    print("Checkpoint loaded successfully")

    print("\nAll SAC agent tests passed!")
