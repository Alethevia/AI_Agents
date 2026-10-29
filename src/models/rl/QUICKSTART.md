# SAC for Trading - Quick Start Guide

Get started with the SAC reinforcement learning implementation in 5 minutes.

## Installation

### 1. Install Dependencies

```bash
cd /home/user/AI_Agents/src/models/rl/
pip install -r requirements.txt
```

Required packages:
- PyTorch (≥1.12.0)
- NumPy (≥1.21.0)
- Pandas (≥1.3.0)
- Gymnasium (≥0.26.0)

### 2. Verify Installation

```bash
python -c "import torch; import gymnasium; print('Installation successful!')"
```

## Basic Usage

### Example 1: Simple Training Loop

```python
from src.models.rl import SACAgent, TradingEnv
import pandas as pd
import numpy as np

# 1. Create synthetic data (or load your own)
data = pd.DataFrame({
    'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
    'volume': np.random.randint(1000, 10000, 1000),
    'returns': np.random.randn(1000) * 0.01
})

# 2. Create trading environment
env = TradingEnv(
    data=data,
    initial_balance=100000,
    features=['close', 'volume', 'returns'],
    reward_type='sharpe'
)

# 3. Create SAC agent
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0]
)

# 4. Train
for episode in range(100):
    stats = agent.train_episode(env)
    print(f"Episode {episode}: Reward={stats['reward']:.2f}")

# 5. Evaluate
eval_metrics = agent.evaluate(env, num_episodes=10)
print(f"Eval Reward: {eval_metrics['mean_reward']:.2f}")

# 6. Save model
agent.save_checkpoint('my_agent.pt')
```

### Example 2: Using the Training Script

```bash
# Train with default settings (uses synthetic data)
python src/models/rl/train_sac_trading.py --episodes 1000

# Train with your own data
python src/models/rl/train_sac_trading.py \
    --data-path your_data.csv \
    --episodes 1000 \
    --reward-type sharpe \
    --save-dir checkpoints/my_run

# Train with advanced settings
python src/models/rl/train_sac_trading.py \
    --data-path your_data.csv \
    --episodes 2000 \
    --batch-size 256 \
    --lr 0.0003 \
    --use-per \
    --hidden-dims 256 256 \
    --gamma 0.99 \
    --save-dir checkpoints/advanced
```

### Example 3: Run Demo

```bash
# Run the example script (includes training and evaluation)
python src/models/rl/example_usage.py
```

## Key Components

### 1. Trading Environment (`TradingEnv`)

Configure your trading simulation:

```python
env = TradingEnv(
    data=your_dataframe,           # Market data
    initial_balance=100000,         # Starting capital
    transaction_cost=0.001,         # 0.1% transaction cost
    slippage=0.0005,               # 0.05% slippage
    lookback_window=20,            # Historical data in state
    features=['close', 'volume'],  # Features to use
    reward_type='sharpe'           # Reward function
)
```

### 2. SAC Agent (`SACAgent`)

Configure the learning algorithm:

```python
agent = SACAgent(
    state_dim=env.observation_space.shape[0],
    action_dim=env.action_space.shape[0],
    hidden_dims=(256, 256),        # Network architecture
    learning_rate=3e-4,            # Learning rate
    gamma=0.99,                    # Discount factor
    auto_entropy_tuning=True,      # Automatic exploration
    use_prioritized_replay=True    # Use PER
)
```

### 3. Training

```python
# Single episode
stats = agent.train_episode(
    env,
    warmup_steps=1000,    # Random exploration
    train_freq=1,         # Training frequency
    gradient_steps=1      # Gradient updates
)

# Multiple episodes
for episode in range(1000):
    stats = agent.train_episode(env)

    if episode % 50 == 0:
        eval_metrics = agent.evaluate(env, num_episodes=10)
        print(f"Eval: {eval_metrics['mean_reward']:.2f}")
```

### 4. Evaluation

```python
# Evaluate performance
metrics = agent.evaluate(
    env,
    num_episodes=10,
    deterministic=True  # Use mean actions
)

print(f"Mean Reward: {metrics['mean_reward']:.2f}")
print(f"Sharpe Ratio: {metrics.get('sharpe_ratio', 'N/A')}")
```

### 5. Save/Load

```python
# Save
agent.save_checkpoint('best_model.pt')

# Load
agent.load_checkpoint('best_model.pt')

# Use for inference
state, _ = env.reset()
action = agent.select_action(state, deterministic=True)
```

## Data Format

Your market data should be a pandas DataFrame with at minimum:

```python
data = pd.DataFrame({
    'close': [...],    # Closing prices (required)
    'volume': [...],   # Trading volume (recommended)
    # Add any additional features you want
    'returns': [...],
    'sma_10': [...],
    'volatility': [...]
})
```

## Reward Functions

Choose based on your objectives:

- **`sharpe`**: Risk-adjusted returns (recommended for most cases)
- **`sortino`**: Focus on downside risk
- **`calmar`**: Return vs max drawdown
- **`returns`**: Simple returns (can be unstable)

```python
env = TradingEnv(data=data, reward_type='sharpe')
```

## Advanced Features

### Prioritized Experience Replay

```python
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    use_prioritized_replay=True,
    per_alpha=0.6,
    per_beta=0.4
)
```

### Custom Network Architecture

```python
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    hidden_dims=(512, 512, 256)  # 3 layers
)
```

### GPU Acceleration

```python
agent = SACAgent(
    state_dim=state_dim,
    action_dim=action_dim,
    device='cuda'  # or 'auto' to auto-detect
)
```

## Troubleshooting

### Issue: "ModuleNotFoundError: No module named 'torch'"

**Solution**: Install PyTorch
```bash
pip install torch
```

### Issue: "ModuleNotFoundError: No module named 'gymnasium'"

**Solution**: Install Gymnasium
```bash
pip install gymnasium
```

### Issue: Training is unstable

**Solutions**:
1. Increase warmup steps: `warmup_steps=5000`
2. Use state normalization (enabled by default)
3. Reduce learning rate: `learning_rate=1e-4`
4. Use prioritized replay: `use_prioritized_replay=True`

### Issue: Poor performance

**Solutions**:
1. Train longer: `--episodes 2000`
2. Tune reward function: Try different `reward_type`
3. Adjust transaction costs to match reality
4. Add more features to state space
5. Increase network capacity: `hidden_dims=(512, 512)`

### Issue: Out of memory

**Solutions**:
1. Reduce buffer size: `buffer_size=100000`
2. Reduce batch size: `batch_size=128`
3. Use CPU: `device='cpu'`

## Performance Tips

1. **Use GPU**: 5-10x faster training with CUDA
2. **Prioritized Replay**: Better sample efficiency
3. **State Normalization**: Enabled by default, improves stability
4. **Larger Networks**: Better for complex problems (256+ hidden units)
5. **More Episodes**: 1000+ episodes for production models
6. **Hyperparameter Tuning**: Experiment with learning rates and network sizes

## Next Steps

1. **Read the full README**: `/home/user/AI_Agents/src/models/rl/README.md`
2. **Run the example**: `python src/models/rl/example_usage.py`
3. **Prepare your data**: Format your market data
4. **Start training**: Use the training script with your data
5. **Evaluate models**: Use the evaluation utilities
6. **Deploy**: Integrate with your trading system

## Common Workflows

### Workflow 1: Research & Development

```bash
# 1. Quick experiment with synthetic data
python src/models/rl/example_usage.py

# 2. Train on your data
python src/models/rl/train_sac_trading.py \
    --data-path data.csv \
    --episodes 500 \
    --eval-freq 25

# 3. Evaluate
python -c "
from src.models.rl import SACAgent, TradingEnv
from src.models.rl.evaluate import evaluate_model
# ... load and evaluate
"
```

### Workflow 2: Production Training

```bash
# Full training run with all features
python src/models/rl/train_sac_trading.py \
    --data-path production_data.csv \
    --episodes 2000 \
    --batch-size 256 \
    --use-per \
    --hidden-dims 512 512 \
    --reward-type sharpe \
    --save-dir checkpoints/production_v1 \
    --seed 42
```

### Workflow 3: Model Comparison

```python
from src.models.rl import SACAgent, TradingEnv
from src.models.rl.evaluate import ModelComparer

env = TradingEnv(data=test_data, ...)

comparer = ModelComparer()
comparer.add_model('Sharpe', agent1, 'checkpoints/sharpe_model.pt')
comparer.add_model('Sortino', agent2, 'checkpoints/sortino_model.pt')
comparer.add_model('Returns', agent3, 'checkpoints/returns_model.pt')

results = comparer.compare(env, num_episodes=100)
comparer.print_comparison()
```

## Support & Documentation

- **Full Documentation**: See `README.md`
- **Configuration Guide**: See `config_example.yaml`
- **Code Examples**: See `example_usage.py`
- **Training Script**: See `train_sac_trading.py`
- **Evaluation Tools**: See `evaluate.py`

## File Overview

```
src/models/rl/
├── __init__.py              # Module exports
├── sac_agent.py             # Main SAC agent
├── networks.py              # Actor & Critic networks
├── replay_buffer.py         # Experience replay
├── trading_env.py           # Trading environment
├── train_sac_trading.py     # Training script
├── evaluate.py              # Evaluation tools
├── example_usage.py         # Simple example
├── config_example.yaml      # Config template
├── requirements.txt         # Dependencies
├── README.md                # Full documentation
└── QUICKSTART.md           # This file
```

Happy Trading! 🚀
