"""
Training script for SAC agent on trading environment.

This script demonstrates how to:
1. Load and prepare market data
2. Create trading environment
3. Initialize SAC agent
4. Train the agent
5. Evaluate performance
6. Save checkpoints

Example usage:
    python train_sac_trading.py --episodes 1000 --eval-freq 50 --save-dir ./checkpoints
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from datetime import datetime
from typing import Dict, Any
import json

# Add parent directory to path if needed
sys.path.append(str(Path(__file__).parent.parent.parent.parent))

from src.models.rl import SACAgent, TradingEnv

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('sac_training.log')
    ]
)
logger = logging.getLogger(__name__)


def load_market_data(
    data_path: str,
    train_split: float = 0.8
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load and split market data into train/test sets.

    Args:
        data_path: Path to market data CSV
        train_split: Fraction of data for training

    Returns:
        train_data: Training data
        test_data: Testing data
    """
    logger.info(f"Loading market data from {data_path}")

    # Try to load data
    if Path(data_path).exists():
        data = pd.read_csv(data_path)
    else:
        # Generate synthetic data for demonstration
        logger.warning("Data file not found, generating synthetic market data")
        data = generate_synthetic_data(n_steps=10000)

    # Compute returns if not present
    if 'returns' not in data.columns and 'close' in data.columns:
        data['returns'] = data['close'].pct_change().fillna(0)

    # Compute additional features
    if 'close' in data.columns:
        # Simple moving averages
        data['sma_10'] = data['close'].rolling(10).mean().fillna(method='bfill')
        data['sma_30'] = data['close'].rolling(30).mean().fillna(method='bfill')

        # Volatility
        data['volatility'] = data['returns'].rolling(20).std().fillna(0)

    # Split data
    split_idx = int(len(data) * train_split)
    train_data = data.iloc[:split_idx].copy()
    test_data = data.iloc[split_idx:].copy()

    logger.info(f"Loaded {len(data)} samples: {len(train_data)} train, {len(test_data)} test")

    return train_data, test_data


def generate_synthetic_data(n_steps: int = 10000) -> pd.DataFrame:
    """
    Generate synthetic market data for testing.

    Args:
        n_steps: Number of time steps

    Returns:
        Market data DataFrame
    """
    np.random.seed(42)

    # Generate price using geometric Brownian motion
    returns = np.random.randn(n_steps) * 0.02  # 2% daily volatility
    price = 100 * np.exp(np.cumsum(returns))

    # Create DataFrame
    data = pd.DataFrame({
        'open': price * (1 + np.random.randn(n_steps) * 0.001),
        'high': price * (1 + np.abs(np.random.randn(n_steps)) * 0.01),
        'low': price * (1 - np.abs(np.random.randn(n_steps)) * 0.01),
        'close': price,
        'volume': np.random.randint(1000000, 10000000, n_steps),
        'returns': returns
    })

    return data


def create_environments(
    train_data: pd.DataFrame,
    test_data: pd.DataFrame,
    config: Dict[str, Any]
) -> tuple[TradingEnv, TradingEnv]:
    """
    Create training and evaluation environments.

    Args:
        train_data: Training data
        test_data: Testing data
        config: Environment configuration

    Returns:
        train_env: Training environment
        eval_env: Evaluation environment
    """
    # Select features
    features = config.get('features', ['close', 'volume', 'returns', 'sma_10', 'sma_30', 'volatility'])

    # Filter features that exist
    available_features = [f for f in features if f in train_data.columns]

    logger.info(f"Using features: {available_features}")

    # Create training environment
    train_env = TradingEnv(
        data=train_data,
        initial_balance=config.get('initial_balance', 100000),
        transaction_cost=config.get('transaction_cost', 0.001),
        slippage=config.get('slippage', 0.0005),
        max_position_size=config.get('max_position_size', 1.0),
        lookback_window=config.get('lookback_window', 20),
        features=available_features,
        reward_type=config.get('reward_type', 'sharpe'),
        risk_free_rate=config.get('risk_free_rate', 0.02),
        reward_scaling=config.get('reward_scaling', 100.0)
    )

    # Create evaluation environment
    eval_env = TradingEnv(
        data=test_data,
        initial_balance=config.get('initial_balance', 100000),
        transaction_cost=config.get('transaction_cost', 0.001),
        slippage=config.get('slippage', 0.0005),
        max_position_size=config.get('max_position_size', 1.0),
        lookback_window=config.get('lookback_window', 20),
        features=available_features,
        reward_type=config.get('reward_type', 'sharpe'),
        risk_free_rate=config.get('risk_free_rate', 0.02),
        reward_scaling=config.get('reward_scaling', 100.0)
    )

    logger.info(f"Created environments - State dim: {train_env.observation_space.shape[0]}, "
               f"Action dim: {train_env.action_space.shape[0]}")

    return train_env, eval_env


def train_sac_agent(
    agent: SACAgent,
    train_env: TradingEnv,
    eval_env: TradingEnv,
    config: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Train SAC agent on trading environment.

    Args:
        agent: SAC agent
        train_env: Training environment
        eval_env: Evaluation environment
        config: Training configuration

    Returns:
        Training history
    """
    num_episodes = config.get('num_episodes', 1000)
    warmup_steps = config.get('warmup_steps', 1000)
    eval_freq = config.get('eval_freq', 50)
    save_freq = config.get('save_freq', 100)
    train_freq = config.get('train_freq', 1)
    gradient_steps = config.get('gradient_steps', 1)
    save_dir = Path(config.get('save_dir', './checkpoints'))
    save_dir.mkdir(parents=True, exist_ok=True)

    # Training history
    history = {
        'episode_rewards': [],
        'episode_lengths': [],
        'eval_rewards': [],
        'eval_episodes': [],
        'actor_losses': [],
        'critic_losses': [],
        'alpha_values': []
    }

    logger.info(f"Starting training for {num_episodes} episodes")
    logger.info(f"Warmup steps: {warmup_steps}, Eval freq: {eval_freq}, Save freq: {save_freq}")

    best_eval_reward = -np.inf

    for episode in range(1, num_episodes + 1):
        # Train one episode
        episode_stats = agent.train_episode(
            train_env,
            warmup_steps=warmup_steps,
            train_freq=train_freq,
            gradient_steps=gradient_steps
        )

        # Record episode statistics
        history['episode_rewards'].append(episode_stats['reward'])
        history['episode_lengths'].append(episode_stats['length'])

        # Get training metrics
        metrics_summary = agent.get_metrics_summary()

        if 'actor_loss_mean' in metrics_summary:
            history['actor_losses'].append(metrics_summary['actor_loss_mean'])
            history['critic_losses'].append(metrics_summary['critic_loss_mean'])
            history['alpha_values'].append(metrics_summary['alpha_mean'])

        # Logging
        if episode % 10 == 0:
            recent_rewards = history['episode_rewards'][-10:]
            logger.info(
                f"Episode {episode}/{num_episodes} | "
                f"Reward: {episode_stats['reward']:.2f} | "
                f"Avg Reward (10 eps): {np.mean(recent_rewards):.2f} | "
                f"Steps: {episode_stats['length']} | "
                f"Total Steps: {agent.total_steps} | "
                f"Buffer Size: {len(agent.replay_buffer)}"
            )

            if 'actor_loss_mean' in metrics_summary:
                logger.info(
                    f"  Actor Loss: {metrics_summary['actor_loss_mean']:.4f} | "
                    f"Critic Loss: {metrics_summary['critic_loss_mean']:.4f} | "
                    f"Alpha: {metrics_summary['alpha_mean']:.4f}"
                )

        # Evaluation
        if episode % eval_freq == 0:
            logger.info(f"\n{'='*60}")
            logger.info(f"Evaluating at episode {episode}")
            eval_metrics = agent.evaluate(
                eval_env,
                num_episodes=5,
                deterministic=True
            )

            history['eval_rewards'].append(eval_metrics['mean_reward'])
            history['eval_episodes'].append(episode)

            logger.info(f"Evaluation Results:")
            logger.info(f"  Mean Reward: {eval_metrics['mean_reward']:.2f} +/- {eval_metrics['std_reward']:.2f}")
            logger.info(f"  Min/Max: {eval_metrics['min_reward']:.2f} / {eval_metrics['max_reward']:.2f}")
            logger.info(f"{'='*60}\n")

            # Save best model
            if eval_metrics['mean_reward'] > best_eval_reward:
                best_eval_reward = eval_metrics['mean_reward']
                best_model_path = save_dir / 'best_model.pt'
                agent.save_checkpoint(best_model_path)
                logger.info(f"New best model saved! Reward: {best_eval_reward:.2f}")

        # Save periodic checkpoint
        if episode % save_freq == 0:
            checkpoint_path = save_dir / f'checkpoint_ep{episode}.pt'
            agent.save_checkpoint(checkpoint_path)
            logger.info(f"Saved checkpoint to {checkpoint_path}")

    # Save final model
    final_model_path = save_dir / 'final_model.pt'
    agent.save_checkpoint(final_model_path)
    logger.info(f"Training complete! Final model saved to {final_model_path}")

    # Save training history
    history_path = save_dir / 'training_history.json'
    with open(history_path, 'w') as f:
        # Convert numpy types to Python types for JSON serialization
        json_history = {
            k: [float(v) if isinstance(v, (np.floating, np.integer)) else v for v in vals]
            for k, vals in history.items()
        }
        json.dump(json_history, f, indent=2)
    logger.info(f"Training history saved to {history_path}")

    return history


def main():
    """Main training function."""
    parser = argparse.ArgumentParser(description='Train SAC agent for trading')

    # Data arguments
    parser.add_argument('--data-path', type=str, default=None,
                       help='Path to market data CSV')
    parser.add_argument('--train-split', type=float, default=0.8,
                       help='Fraction of data for training')

    # Environment arguments
    parser.add_argument('--initial-balance', type=float, default=100000,
                       help='Initial trading balance')
    parser.add_argument('--transaction-cost', type=float, default=0.001,
                       help='Transaction cost fraction')
    parser.add_argument('--reward-type', type=str, default='sharpe',
                       choices=['returns', 'sharpe', 'sortino', 'calmar'],
                       help='Reward function type')

    # Agent arguments
    parser.add_argument('--hidden-dims', type=int, nargs='+', default=[256, 256],
                       help='Hidden layer dimensions')
    parser.add_argument('--lr', type=float, default=3e-4,
                       help='Learning rate')
    parser.add_argument('--gamma', type=float, default=0.99,
                       help='Discount factor')
    parser.add_argument('--tau', type=float, default=0.005,
                       help='Soft update coefficient')
    parser.add_argument('--batch-size', type=int, default=256,
                       help='Batch size')
    parser.add_argument('--buffer-size', type=int, default=1000000,
                       help='Replay buffer size')
    parser.add_argument('--use-per', action='store_true',
                       help='Use prioritized experience replay')

    # Training arguments
    parser.add_argument('--episodes', type=int, default=1000,
                       help='Number of training episodes')
    parser.add_argument('--warmup-steps', type=int, default=1000,
                       help='Random exploration steps')
    parser.add_argument('--eval-freq', type=int, default=50,
                       help='Evaluation frequency (episodes)')
    parser.add_argument('--save-freq', type=int, default=100,
                       help='Checkpoint save frequency (episodes)')
    parser.add_argument('--train-freq', type=int, default=1,
                       help='Training frequency (steps)')
    parser.add_argument('--gradient-steps', type=int, default=1,
                       help='Gradient steps per update')

    # Other arguments
    parser.add_argument('--save-dir', type=str, default='./checkpoints',
                       help='Directory to save checkpoints')
    parser.add_argument('--seed', type=int, default=None,
                       help='Random seed')
    parser.add_argument('--device', type=str, default='auto',
                       choices=['auto', 'cpu', 'cuda'],
                       help='Device to use')

    args = parser.parse_args()

    # Set random seed
    if args.seed is not None:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)

    # Load data
    train_data, test_data = load_market_data(
        args.data_path if args.data_path else 'data.csv',
        args.train_split
    )

    # Create environments
    env_config = {
        'initial_balance': args.initial_balance,
        'transaction_cost': args.transaction_cost,
        'reward_type': args.reward_type,
        'lookback_window': 20
    }
    train_env, eval_env = create_environments(train_data, test_data, env_config)

    # Create agent
    logger.info("Initializing SAC agent")
    agent = SACAgent(
        state_dim=train_env.observation_space.shape[0],
        action_dim=train_env.action_space.shape[0],
        hidden_dims=tuple(args.hidden_dims),
        learning_rate=args.lr,
        gamma=args.gamma,
        tau=args.tau,
        batch_size=args.batch_size,
        buffer_size=args.buffer_size,
        use_prioritized_replay=args.use_per,
        device=args.device,
        seed=args.seed
    )

    # Train agent
    training_config = {
        'num_episodes': args.episodes,
        'warmup_steps': args.warmup_steps,
        'eval_freq': args.eval_freq,
        'save_freq': args.save_freq,
        'train_freq': args.train_freq,
        'gradient_steps': args.gradient_steps,
        'save_dir': args.save_dir
    }

    history = train_sac_agent(agent, train_env, eval_env, training_config)

    logger.info("\n" + "="*60)
    logger.info("Training Summary")
    logger.info("="*60)
    logger.info(f"Total Episodes: {len(history['episode_rewards'])}")
    logger.info(f"Total Steps: {agent.total_steps}")
    logger.info(f"Final Avg Reward (100 eps): {np.mean(history['episode_rewards'][-100:]):.2f}")
    if history['eval_rewards']:
        logger.info(f"Best Eval Reward: {max(history['eval_rewards']):.2f}")
    logger.info("="*60)


if __name__ == "__main__":
    main()
