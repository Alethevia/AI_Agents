"""
Simple example demonstrating SAC agent usage for trading.

This script shows the basic workflow:
1. Create synthetic market data
2. Initialize trading environment
3. Create and train SAC agent
4. Evaluate performance
5. Save model

Run: python example_usage.py
"""

import numpy as np
import pandas as pd
import logging
from pathlib import Path

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


def create_synthetic_data(n_steps=5000, seed=42):
    """Create synthetic market data for demonstration."""
    np.random.seed(seed)

    # Generate realistic price series using geometric Brownian motion
    mu = 0.0005  # Drift
    sigma = 0.02  # Volatility
    dt = 1  # Time step

    returns = np.random.normal(mu * dt, sigma * np.sqrt(dt), n_steps)
    price = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame with OHLCV data
    data = pd.DataFrame({
        'open': price * (1 + np.random.randn(n_steps) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n_steps)) * 0.015),
        'low': price * (1 - np.abs(np.random.randn(n_steps)) * 0.015),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, n_steps),
    })

    # Add technical indicators
    data['returns'] = data['close'].pct_change().fillna(0)
    data['sma_10'] = data['close'].rolling(10).mean().fillna(method='bfill')
    data['sma_30'] = data['close'].rolling(30).mean().fillna(method='bfill')
    data['volatility'] = data['returns'].rolling(20).std().fillna(0)
    data['rsi'] = compute_rsi(data['close'], 14)

    return data


def compute_rsi(prices, period=14):
    """Compute Relative Strength Index."""
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    rs = gain / (loss + 1e-10)
    rsi = 100 - (100 / (1 + rs))
    return rsi.fillna(50)


def main():
    """Main example function."""

    # Note: This example assumes the required packages are installed
    # Install with: pip install torch numpy pandas gymnasium

    try:
        from src.models.rl import SACAgent, TradingEnv
    except ImportError:
        logger.error("Required packages not installed. Please install: torch, numpy, pandas, gymnasium")
        logger.error("Run: pip install torch numpy pandas gymnasium")
        return

    logger.info("="*60)
    logger.info("SAC Trading Agent - Simple Example")
    logger.info("="*60)

    # Step 1: Create synthetic market data
    logger.info("\n1. Creating synthetic market data...")
    data = create_synthetic_data(n_steps=5000)
    logger.info(f"   Generated {len(data)} data points")
    logger.info(f"   Price range: ${data['close'].min():.2f} - ${data['close'].max():.2f}")

    # Split data into train/test
    split_idx = int(len(data) * 0.8)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()
    logger.info(f"   Train: {len(train_data)} samples, Test: {len(test_data)} samples")

    # Step 2: Create trading environments
    logger.info("\n2. Creating trading environments...")

    env_config = {
        'data': train_data,
        'initial_balance': 100000,
        'transaction_cost': 0.001,
        'slippage': 0.0005,
        'lookback_window': 20,
        'features': ['close', 'volume', 'returns', 'sma_10', 'sma_30', 'volatility', 'rsi'],
        'reward_type': 'sharpe',
        'reward_scaling': 100.0
    }

    train_env = TradingEnv(**env_config)
    test_env = TradingEnv(**{**env_config, 'data': test_data})

    logger.info(f"   State dimension: {train_env.observation_space.shape[0]}")
    logger.info(f"   Action dimension: {train_env.action_space.shape[0]}")

    # Step 3: Create SAC agent
    logger.info("\n3. Creating SAC agent...")

    agent = SACAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        hidden_dims=(256, 256),
        learning_rate=3e-4,
        gamma=0.99,
        tau=0.005,
        auto_entropy_tuning=True,
        batch_size=256,
        buffer_size=100000,  # Smaller for demo
        use_prioritized_replay=False,
        device='auto',
        seed=42
    )

    logger.info(f"   Agent initialized on device: {agent.device}")

    # Step 4: Train agent
    logger.info("\n4. Training agent...")
    logger.info("   This is a short demo with 50 episodes")
    logger.info("   For production, use 1000+ episodes\n")

    num_episodes = 50
    warmup_steps = 500
    eval_freq = 10

    best_eval_reward = -np.inf

    for episode in range(1, num_episodes + 1):
        # Train one episode
        episode_stats = agent.train_episode(
            train_env,
            warmup_steps=warmup_steps,
            train_freq=1,
            gradient_steps=1
        )

        # Log progress
        if episode % 5 == 0:
            logger.info(
                f"Episode {episode:3d}/{num_episodes} | "
                f"Reward: {episode_stats['reward']:7.2f} | "
                f"Steps: {episode_stats['length']:4d} | "
                f"Buffer: {len(agent.replay_buffer):6d}"
            )

        # Evaluate
        if episode % eval_freq == 0:
            logger.info("\n   Evaluating...")
            eval_metrics = agent.evaluate(test_env, num_episodes=3, deterministic=True)

            logger.info(f"   Eval Reward: {eval_metrics['mean_reward']:.2f} "
                       f"± {eval_metrics['std_reward']:.2f}")

            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                logger.info(f"   New best model! Reward: {best_eval_reward:.2f}\n")

    # Step 5: Final evaluation
    logger.info("\n5. Final evaluation on test set...")
    final_eval = agent.evaluate(test_env, num_episodes=10, deterministic=True)

    logger.info(f"\n   Final Test Results:")
    logger.info(f"   Mean Reward: {final_eval['mean_reward']:.2f}")
    logger.info(f"   Std Reward:  {final_eval['std_reward']:.2f}")
    logger.info(f"   Min Reward:  {final_eval['min_reward']:.2f}")
    logger.info(f"   Max Reward:  {final_eval['max_reward']:.2f}")

    # Step 6: Save model
    logger.info("\n6. Saving trained model...")
    save_dir = Path("./example_checkpoints")
    save_dir.mkdir(exist_ok=True)

    model_path = save_dir / "sac_agent_example.pt"
    agent.save_checkpoint(model_path, include_buffer=False)
    logger.info(f"   Model saved to: {model_path}")

    # Step 7: Demonstrate loading
    logger.info("\n7. Demonstrating model loading...")
    new_agent = SACAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0]
    )
    new_agent.load_checkpoint(model_path)
    logger.info("   Model loaded successfully!")

    # Test loaded model
    test_eval = new_agent.evaluate(test_env, num_episodes=3, deterministic=True)
    logger.info(f"   Loaded model test reward: {test_eval['mean_reward']:.2f}")

    # Summary
    logger.info("\n" + "="*60)
    logger.info("Example Complete!")
    logger.info("="*60)
    logger.info("\nNext steps:")
    logger.info("  1. Use your own market data instead of synthetic data")
    logger.info("  2. Increase training episodes (1000+)")
    logger.info("  3. Tune hyperparameters (learning rate, network size, etc.)")
    logger.info("  4. Experiment with different reward functions")
    logger.info("  5. Try prioritized experience replay")
    logger.info("\nFor full training, use:")
    logger.info("  python src/models/rl/train_sac_trading.py --episodes 1000")
    logger.info("="*60)


if __name__ == "__main__":
    main()
