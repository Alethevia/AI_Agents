"""
Experience replay buffers for reinforcement learning.

This module provides:
- ReplayBuffer: Basic uniform sampling replay buffer
- PrioritizedReplayBuffer: Prioritized experience replay (PER)
- NStepReplayBuffer: N-step returns for multi-step learning
"""

import numpy as np
import torch
from typing import Dict, Tuple, Optional, List
import logging
from collections import deque

logger = logging.getLogger(__name__)


class ReplayBuffer:
    """
    Basic replay buffer with uniform sampling.

    Stores transitions (state, action, reward, next_state, done) and
    samples batches uniformly for training.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        device: str = "cpu"
    ):
        """
        Initialize replay buffer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            capacity: Maximum buffer size
            device: Device for tensors ('cpu' or 'cuda')
        """
        self.capacity = capacity
        self.device = device
        self.position = 0
        self.size = 0

        # Pre-allocate memory for efficiency
        self.states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.next_states = np.zeros((capacity, state_dim), dtype=np.float32)
        self.dones = np.zeros((capacity, 1), dtype=np.float32)

        logger.info(f"Initialized ReplayBuffer with capacity={capacity}")

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add a transition to the buffer.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        self.states[self.position] = state
        self.actions[self.position] = action
        self.rewards[self.position] = reward
        self.next_states[self.position] = next_state
        self.dones[self.position] = float(done)

        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Dict[str, torch.Tensor]:
        """
        Sample a batch of transitions uniformly.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            Dictionary containing batch of transitions as tensors
        """
        if self.size < batch_size:
            logger.warning(f"Buffer has only {self.size} samples but requested {batch_size}")
            batch_size = self.size

        # Sample indices uniformly
        indices = np.random.randint(0, self.size, size=batch_size)

        # Get batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device)
        }

        return batch

    def __len__(self) -> int:
        """Return current buffer size."""
        return self.size

    def clear(self):
        """Clear the buffer."""
        self.position = 0
        self.size = 0
        logger.info("Cleared replay buffer")


class SumTree:
    """
    Sum tree data structure for efficient prioritized sampling.

    Binary tree where each parent node is the sum of its children.
    Enables O(log n) updates and sampling.
    """

    def __init__(self, capacity: int):
        """
        Initialize sum tree.

        Args:
            capacity: Maximum number of leaves
        """
        self.capacity = capacity
        self.tree = np.zeros(2 * capacity - 1)
        self.data_pointer = 0
        self.n_entries = 0

    def update(self, idx: int, priority: float):
        """
        Update priority of a leaf node.

        Args:
            idx: Data index
            priority: New priority value
        """
        tree_idx = idx + self.capacity - 1
        change = priority - self.tree[tree_idx]
        self.tree[tree_idx] = priority

        # Propagate change up the tree
        while tree_idx != 0:
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def add(self, priority: float):
        """
        Add new priority value.

        Args:
            priority: Priority value
        """
        idx = self.data_pointer
        self.update(idx, priority)

        self.data_pointer = (self.data_pointer + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)

    def get(self, value: float) -> int:
        """
        Get index for a given cumulative value.

        Args:
            value: Cumulative priority value

        Returns:
            Data index
        """
        parent_idx = 0

        while True:
            left_child_idx = 2 * parent_idx + 1
            right_child_idx = left_child_idx + 1

            # Leaf node reached
            if left_child_idx >= len(self.tree):
                leaf_idx = parent_idx
                break

            # Choose child
            if value <= self.tree[left_child_idx]:
                parent_idx = left_child_idx
            else:
                value -= self.tree[left_child_idx]
                parent_idx = right_child_idx

        data_idx = leaf_idx - self.capacity + 1
        return data_idx

    def total_priority(self) -> float:
        """Get total priority (root node value)."""
        return self.tree[0]

    def max_priority(self) -> float:
        """Get maximum priority in the tree."""
        return np.max(self.tree[-self.capacity:])


class PrioritizedReplayBuffer(ReplayBuffer):
    """
    Prioritized Experience Replay (PER) buffer.

    Samples transitions based on their TD error, giving priority to
    more "surprising" transitions that the agent can learn more from.
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        alpha: float = 0.6,
        beta: float = 0.4,
        beta_increment: float = 0.001,
        epsilon: float = 1e-6,
        device: str = "cpu"
    ):
        """
        Initialize prioritized replay buffer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            capacity: Maximum buffer size
            alpha: Priority exponent (0 = uniform, 1 = full prioritization)
            beta: Importance sampling exponent (0 = no correction, 1 = full correction)
            beta_increment: Increment for beta per sampling
            epsilon: Small constant to ensure non-zero priorities
            device: Device for tensors
        """
        super().__init__(state_dim, action_dim, capacity, device)

        self.alpha = alpha
        self.beta = beta
        self.beta_increment = beta_increment
        self.epsilon = epsilon

        # Sum tree for efficient prioritized sampling
        self.tree = SumTree(capacity)

        # Track max priority for new samples
        self.max_priority = 1.0

        logger.info(f"Initialized PrioritizedReplayBuffer: alpha={alpha}, beta={beta}")

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add transition with maximum priority.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Add to standard replay buffer
        super().add(state, action, reward, next_state, done)

        # Add to sum tree with max priority
        self.tree.add(self.max_priority)

    def sample(self, batch_size: int) -> Tuple[Dict[str, torch.Tensor], np.ndarray, np.ndarray]:
        """
        Sample batch based on priorities.

        Args:
            batch_size: Number of transitions to sample

        Returns:
            batch: Dictionary of transitions
            indices: Sampled indices for priority updates
            weights: Importance sampling weights
        """
        if self.size < batch_size:
            logger.warning(f"Buffer has only {self.size} samples but requested {batch_size}")
            batch_size = self.size

        indices = np.zeros(batch_size, dtype=np.int32)
        priorities = np.zeros(batch_size, dtype=np.float32)

        # Divide priority range into segments
        segment = self.tree.total_priority() / batch_size

        # Increment beta for importance sampling
        self.beta = min(1.0, self.beta + self.beta_increment)

        # Sample from each segment
        for i in range(batch_size):
            a = segment * i
            b = segment * (i + 1)
            value = np.random.uniform(a, b)
            idx = self.tree.get(value)
            indices[i] = idx
            priorities[i] = self.tree.tree[idx + self.tree.capacity - 1]

        # Compute importance sampling weights
        sampling_probs = priorities / self.tree.total_priority()
        weights = np.power(self.size * sampling_probs, -self.beta)
        weights /= weights.max()  # Normalize

        # Get batch
        batch = {
            'states': torch.FloatTensor(self.states[indices]).to(self.device),
            'actions': torch.FloatTensor(self.actions[indices]).to(self.device),
            'rewards': torch.FloatTensor(self.rewards[indices]).to(self.device),
            'next_states': torch.FloatTensor(self.next_states[indices]).to(self.device),
            'dones': torch.FloatTensor(self.dones[indices]).to(self.device),
            'weights': torch.FloatTensor(weights).unsqueeze(1).to(self.device)
        }

        return batch, indices, weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """
        Update priorities based on TD errors.

        Args:
            indices: Indices of samples
            td_errors: TD errors for priority computation
        """
        for idx, td_error in zip(indices, td_errors):
            # Compute priority: |TD error| ^ alpha
            priority = (np.abs(td_error) + self.epsilon) ** self.alpha
            self.tree.update(idx, priority)

            # Track max priority
            self.max_priority = max(self.max_priority, priority)


class NStepReplayBuffer(ReplayBuffer):
    """
    N-step replay buffer for multi-step returns.

    Computes n-step returns: R_t = r_t + γ*r_{t+1} + ... + γ^{n-1}*r_{t+n-1} + γ^n*V(s_{t+n})
    """

    def __init__(
        self,
        state_dim: int,
        action_dim: int,
        capacity: int = 1_000_000,
        n_step: int = 3,
        gamma: float = 0.99,
        device: str = "cpu"
    ):
        """
        Initialize n-step replay buffer.

        Args:
            state_dim: Dimension of state space
            action_dim: Dimension of action space
            capacity: Maximum buffer size
            n_step: Number of steps for return computation
            gamma: Discount factor
            device: Device for tensors
        """
        super().__init__(state_dim, action_dim, capacity, device)

        self.n_step = n_step
        self.gamma = gamma

        # Buffer for n-step transitions
        self.n_step_buffer = deque(maxlen=n_step)

        logger.info(f"Initialized NStepReplayBuffer: n_step={n_step}, gamma={gamma}")

    def add(
        self,
        state: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_state: np.ndarray,
        done: bool
    ):
        """
        Add transition and compute n-step returns.

        Args:
            state: Current state
            action: Action taken
            reward: Reward received
            next_state: Next state
            done: Whether episode terminated
        """
        # Add to n-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # Not enough steps yet
        if len(self.n_step_buffer) < self.n_step:
            return

        # Compute n-step return
        n_step_reward = 0.0
        discount = 1.0
        for i, (_, _, r, _, _) in enumerate(self.n_step_buffer):
            n_step_reward += discount * r
            discount *= self.gamma

        # Get first and last transitions
        first_state, first_action = self.n_step_buffer[0][:2]
        last_next_state, last_done = self.n_step_buffer[-1][3:]

        # Add to main buffer
        super().add(first_state, first_action, n_step_reward, last_next_state, last_done)

    def reset_episode(self):
        """Reset n-step buffer at episode end."""
        self.n_step_buffer.clear()


if __name__ == "__main__":
    # Test replay buffers
    logging.basicConfig(level=logging.INFO)

    state_dim = 10
    action_dim = 3
    capacity = 1000

    # Test basic replay buffer
    print("\nTesting ReplayBuffer...")
    buffer = ReplayBuffer(state_dim, action_dim, capacity)

    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 10 == 0
        buffer.add(state, action, reward, next_state, done)

    batch = buffer.sample(32)
    print(f"Sampled batch: states {batch['states'].shape}, rewards {batch['rewards'].shape}")
    print(f"Buffer size: {len(buffer)}")

    # Test prioritized replay buffer
    print("\nTesting PrioritizedReplayBuffer...")
    per_buffer = PrioritizedReplayBuffer(state_dim, action_dim, capacity)

    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = np.random.randn()
        next_state = np.random.randn(state_dim)
        done = i % 10 == 0
        per_buffer.add(state, action, reward, next_state, done)

    batch, indices, weights = per_buffer.sample(32)
    print(f"Sampled PER batch: states {batch['states'].shape}, weights {batch['weights'].shape}")
    print(f"Importance weights range: [{weights.min():.3f}, {weights.max():.3f}]")

    # Update priorities
    td_errors = np.random.randn(32)
    per_buffer.update_priorities(indices, td_errors)
    print("Updated priorities")

    # Test n-step buffer
    print("\nTesting NStepReplayBuffer...")
    nstep_buffer = NStepReplayBuffer(state_dim, action_dim, capacity, n_step=3)

    for i in range(100):
        state = np.random.randn(state_dim)
        action = np.random.randn(action_dim)
        reward = 1.0  # Fixed reward for testing
        next_state = np.random.randn(state_dim)
        done = i % 10 == 0
        nstep_buffer.add(state, action, reward, next_state, done)
        if done:
            nstep_buffer.reset_episode()

    if len(nstep_buffer) > 0:
        batch = nstep_buffer.sample(min(32, len(nstep_buffer)))
        print(f"N-step buffer size: {len(nstep_buffer)}")
        print(f"Sampled n-step batch: rewards {batch['rewards'].shape}")

    print("\nAll replay buffer tests passed!")
