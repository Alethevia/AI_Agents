# Soft Actor-Critic (SAC) for Trading

Production-ready implementation of the Soft Actor-Critic (SAC) reinforcement learning algorithm for algorithmic trading.

## Overview

This module provides a complete SAC implementation optimized for continuous control in trading environments, featuring:

- **Twin Q-networks** to reduce overestimation bias
- **Entropy-regularized policy** for better exploration
- **Automatic entropy tuning** for adaptive exploration
- **Prioritized Experience Replay (PER)** for efficient learning
- **GPU acceleration** with PyTorch
- **Comprehensive logging** and checkpointing
- **Trading-specific environment** with realistic market simulation

## Components

### 1. SAC Agent (`sac_agent.py`)

The main reinforcement learning agent implementing the SAC algorithm.

**Key Features:**
- Actor network with tanh-squashed Gaussian policy
- Twin critic networks with soft target updates
- Automatic temperature (alpha) tuning
- Support for prioritized and standard replay buffers
- Training and evaluation modes
- Checkpoint save/load functionality

**Example Usage:**
```python
from src.models.rl import SACAgent

agent = SACAgent(
    state_dim=10,
    action_dim=3,
    hidden_dims=(256, 256),
    learning_rate=3e-4,
    gamma=0.99,
    auto_entropy_tuning=True,
    use_prioritized_replay=True
)

# Select action
action = agent.select_action(state, deterministic=False)

# Store transition
agent.store_transition(state, action, reward, next_state, done)

# Train
metrics = agent.train_step()

# Save checkpoint
agent.save_checkpoint('checkpoints/agent.pt')
```

### 2. Neural Networks (`networks.py`)

Neural network architectures for actor and critic.

**Components:**
- `MLPActor`: Policy network with Gaussian distribution
- `MLPCritic`: Q-value estimation network
- `TwinCritic`: Dual Q-networks for stability
- `StateNormalizer`: Online state normalization layer

**Example Usage:**
```python
from src.models.rl import MLPActor, TwinCritic

# Create actor
actor = MLPActor(
    state_dim=10,
    action_dim=3,
    hidden_dims=(256, 256),
    use_state_norm=True
)

# Sample action
action, log_prob = actor(state_tensor)

# Create twin critics
critic = TwinCritic(
    state_dim=10,
    action_dim=3,
    hidden_dims=(256, 256)
)

# Get Q-values
q1, q2 = critic(state_tensor, action_tensor)
```

### 3. Replay Buffers (`replay_buffer.py`)

Experience replay implementations for efficient learning.

**Components:**
- `ReplayBuffer`: Standard uniform sampling
- `PrioritizedReplayBuffer`: Priority-based sampling
- `NStepReplayBuffer`: Multi-step returns
- `SumTree`: Efficient data structure for PER

**Example Usage:**
```python
from src.models.rl import ReplayBuffer, PrioritizedReplayBuffer

# Standard buffer
buffer = ReplayBuffer(
    state_dim=10,
    action_dim=3,
    capacity=1_000_000
)

# Prioritized buffer
per_buffer = PrioritizedReplayBuffer(
    state_dim=10,
    action_dim=3,
    capacity=1_000_000,
    alpha=0.6,
    beta=0.4
)

# Add transition
buffer.add(state, action, reward, next_state, done)

# Sample batch
batch = buffer.sample(batch_size=256)

# For PER, also get indices and weights
batch, indices, weights = per_buffer.sample(batch_size=256)

# Update priorities
per_buffer.update_priorities(indices, td_errors)
```

### 4. Trading Environment (`trading_env.py`)

Gymnasium-compatible trading environment for continuous actions.

**Features:**
- Realistic market simulation with transaction costs and slippage
- Multiple reward functions (Sharpe, Sortino, Calmar, returns)
- Portfolio management with position tracking
- Risk metrics and performance analytics
- Support for multiple features and lookback windows

**State Space:**
- Portfolio information (balance, positions, P&L)
- Market features (prices, volume, indicators)
- Historical data (lookback window)

**Action Space:**
- Continuous position sizing [-1, 1]
- -1: Full short position
- 0: Neutral (no position)
- +1: Full long position

**Example Usage:**
```python
from src.models.rl import TradingEnv
import pandas as pd

# Load market data
data = pd.read_csv('market_data.csv')

# Create environment
env = TradingEnv(
    data=data,
    initial_balance=100000,
    transaction_cost=0.001,
    slippage=0.0005,
    lookback_window=20,
    features=['close', 'volume', 'returns'],
    reward_type='sharpe'
)

# Reset environment
state, info = env.reset()

# Take action
action = agent.select_action(state)
next_state, reward, terminated, truncated, info = env.step(action)

# Get episode summary
episode_info = env.get_episode_info()
```

## Training a SAC Agent

### Quick Start

```python
from src.models.rl import SACAgent, TradingEnv
import pandas as pd

# Load data
data = pd.read_csv('your_data.csv')

# Create environment
env = TradingEnv(data=data, reward_type='sharpe')

# Create agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# Training loop
for episode in range(1000):
    episode_stats = agent.train_episode(
        env,
        warmup_steps=1000,
        train_freq=1,
        gradient_steps=1
    )
    print(f"Episode {episode}: Reward={episode_stats['reward']:.2f}")

# Evaluate
eval_metrics = agent.evaluate(env, num_episodes=10)
print(f"Eval Reward: {eval_metrics['mean_reward']:.2f}")
```

### Using the Training Script

```bash
# Train with default settings
python src/models/rl/train_sac_trading.py --episodes 1000

# Train with custom settings
python src/models/rl/train_sac_trading.py \
    --data-path data/market_data.csv \
    --episodes 2000 \
    --batch-size 256 \
    --lr 0.0003 \
    --use-per \
    --reward-type sharpe \
    --save-dir checkpoints/sac_v1

# Full options
python src/models/rl/train_sac_trading.py --help
```

### Training Configuration

**Environment Parameters:**
- `initial_balance`: Starting capital (default: 100,000)
- `transaction_cost`: Transaction cost as fraction (default: 0.001 = 0.1%)
- `slippage`: Slippage as fraction (default: 0.0005 = 0.05%)
- `reward_type`: Reward function ('sharpe', 'sortino', 'calmar', 'returns')

**Agent Parameters:**
- `learning_rate`: Learning rate for all networks (default: 3e-4)
- `gamma`: Discount factor (default: 0.99)
- `tau`: Soft update coefficient (default: 0.005)
- `batch_size`: Training batch size (default: 256)
- `buffer_size`: Replay buffer capacity (default: 1,000,000)

**Training Parameters:**
- `warmup_steps`: Random exploration before training (default: 1,000)
- `train_freq`: Steps between training updates (default: 1)
- `gradient_steps`: Gradient updates per training (default: 1)

## Performance Metrics

The implementation tracks comprehensive metrics:

### Training Metrics
- Actor loss
- Critic loss
- Alpha (entropy temperature)
- Q-values
- Log probabilities

### Episode Metrics
- Total reward
- Episode length
- Portfolio value
- Profit/Loss
- Max drawdown

### Evaluation Metrics
- Sharpe ratio
- Sortino ratio
- Win rate
- Total return
- Risk-adjusted returns

## Advanced Features

### Prioritized Experience Replay

```python
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    use_prioritized_replay=True,
    per_alpha=0.6,  # Priority exponent
    per_beta=0.4    # Importance sampling exponent
)
```

### N-Step Returns

```python
from src.models.rl import NStepReplayBuffer

buffer = NStepReplayBuffer(
    state_dim=state_dim,
    action_dim=action_dim,
    n_step=3,
    gamma=0.99
)
```

### Custom Reward Functions

```python
# Define custom reward in environment
class CustomTradingEnv(TradingEnv):
    def _compute_reward(self, portfolio_return, drawdown):
        # Custom reward logic
        sharpe = self._compute_sharpe()
        calmar = self._compute_calmar()
        return 0.5 * sharpe + 0.5 * calmar
```

### State Normalization

State normalization is enabled by default and helps with training stability:

```python
actor = MLPActor(
    state_dim=state_dim,
    action_dim=action_dim,
    use_state_norm=True  # Default: True
)
```

## Checkpointing and Loading

### Save Checkpoint

```python
# Save full checkpoint
agent.save_checkpoint(
    'checkpoints/agent_episode_1000.pt',
    include_buffer=False  # Buffer can be very large
)
```

### Load Checkpoint

```python
# Load checkpoint
agent.load_checkpoint(
    'checkpoints/agent_episode_1000.pt',
    load_optimizers=True  # Load optimizer states
)

# Continue training
for episode in range(1001, 2000):
    agent.train_episode(env)
```

## Testing

Each module includes comprehensive tests:

```bash
# Test networks
python src/models/rl/networks.py

# Test replay buffers
python src/models/rl/replay_buffer.py

# Test trading environment
python src/models/rl/trading_env.py

# Test SAC agent
python src/models/rl/sac_agent.py
```

## Requirements

- Python 3.8+
- PyTorch 1.12+
- NumPy
- Pandas
- Gymnasium (formerly OpenAI Gym)

Install dependencies:
```bash
pip install torch numpy pandas gymnasium
```

## Architecture

```
┌─────────────────────────────────────────────────┐
│                  SAC Agent                       │
├─────────────────────────────────────────────────┤
│                                                  │
│  ┌──────────────┐         ┌──────────────┐     │
│  │   Actor      │         │  Twin Critic │     │
│  │   Network    │         │   Networks   │     │
│  └──────────────┘         └──────────────┘     │
│         │                         │             │
│         │                         │             │
│  ┌──────▼─────────────────────────▼────────┐   │
│  │      Soft Target Update (τ)             │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
│  ┌──────────────────────────────────────────┐   │
│  │   Automatic Entropy Tuning (α)          │   │
│  └──────────────────────────────────────────┘   │
│                                                  │
└─────────────────────────────────────────────────┘
                     │
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│          Replay Buffer (PER/Standard)            │
└─────────────────────────────────────────────────┘
                     │
                     │
                     ▼
┌─────────────────────────────────────────────────┐
│            Trading Environment                   │
│  ┌───────────────────────────────────────────┐  │
│  │  State: Portfolio + Market Features       │  │
│  │  Action: Position Sizing [-1, 1]          │  │
│  │  Reward: Sharpe/Returns/Sortino/Calmar    │  │
│  └───────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

## References

1. Haarnoja et al. (2018). "Soft Actor-Critic: Off-Policy Maximum Entropy Deep Reinforcement Learning with a Stochastic Actor"
2. Haarnoja et al. (2018). "Soft Actor-Critic Algorithms and Applications"
3. Schaul et al. (2015). "Prioritized Experience Replay"

## License

This implementation is part of the AI_Agents trading system.

## Contributing

When extending this module:
1. Maintain comprehensive docstrings
2. Add tests for new features
3. Update this README
4. Follow the existing code style
5. Ensure GPU compatibility
6. Add logging for debugging

## Support

For issues or questions:
1. Check the test files for usage examples
2. Review the docstrings in each module
3. Examine the training script for complete workflow
