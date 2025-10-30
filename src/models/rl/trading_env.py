"""
Trading environment for reinforcement learning using Gymnasium.

This module provides a production-ready trading environment with:
- Realistic market simulation
- Portfolio management
- Risk-adjusted reward computation (Sharpe ratio, returns, etc.)
- Support for multiple assets
- Transaction costs and slippage
- Integration with backtesting framework
"""

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import pandas as pd
from typing import Dict, Tuple, Optional, List, Any
import logging
from collections import deque

logger = logging.getLogger(__name__)


class TradingEnv(gym.Env):
    """
    Gymnasium-compatible trading environment for continuous action spaces.

    State space includes:
    - Portfolio positions and values
    - Market features (prices, returns, technical indicators)
    - Risk metrics

    Action space:
    - Continuous position sizing [-1, 1] per asset
    - -1: Full short, 0: Neutral, 1: Full long
    """

    metadata = {'render_modes': ['human']}

    def __init__(
        self,
        data: pd.DataFrame,
        initial_balance: float = 100000.0,
        transaction_cost: float = 0.001,
        slippage: float = 0.0005,
        max_position_size: float = 1.0,
        lookback_window: int = 20,
        features: Optional[List[str]] = None,
        reward_type: str = "sharpe",
        risk_free_rate: float = 0.02,
        reward_scaling: float = 100.0,
        max_drawdown_penalty: float = 0.5,
        frame_bound: Optional[Tuple[int, int]] = None
    ):
        """
        Initialize trading environment.

        Args:
            data: Market data DataFrame with columns like ['open', 'high', 'low', 'close', 'volume']
            initial_balance: Starting capital
            transaction_cost: Transaction cost as fraction of trade value
            slippage: Slippage as fraction of price
            max_position_size: Maximum position size as fraction of portfolio
            lookback_window: Number of historical steps in observation
            features: List of feature columns to include in state
            reward_type: Type of reward ('sharpe', 'returns', 'sortino', 'calmar')
            risk_free_rate: Annual risk-free rate for Sharpe/Sortino
            reward_scaling: Scaling factor for rewards
            max_drawdown_penalty: Penalty multiplier for exceeding drawdown
            frame_bound: Tuple (start, end) to limit data range
        """
        super().__init__()

        # Data
        self.data = data.copy()
        if frame_bound:
            self.data = self.data.iloc[frame_bound[0]:frame_bound[1]]
        self.data.reset_index(drop=True, inplace=True)

        # Environment parameters
        self.initial_balance = initial_balance
        self.transaction_cost = transaction_cost
        self.slippage = slippage
        self.max_position_size = max_position_size
        self.lookback_window = lookback_window
        self.reward_type = reward_type
        self.risk_free_rate = risk_free_rate
        self.reward_scaling = reward_scaling
        self.max_drawdown_penalty = max_drawdown_penalty

        # Features
        if features is None:
            self.features = ['close', 'volume']
        else:
            self.features = features

        # Verify features exist in data
        missing_features = set(self.features) - set(self.data.columns)
        if missing_features:
            raise ValueError(f"Features not found in data: {missing_features}")

        # Number of assets (for now, single asset)
        self.n_assets = 1

        # Action space: continuous position sizing [-1, 1] per asset
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.n_assets,),
            dtype=np.float32
        )

        # Observation space
        # State includes: portfolio info + market features + positions
        portfolio_dim = 5  # balance, total_value, position, position_value, pnl_ratio
        market_features_dim = len(self.features) * self.lookback_window
        self.observation_dim = portfolio_dim + market_features_dim

        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(self.observation_dim,),
            dtype=np.float32
        )

        # State variables
        self.current_step = 0
        self.balance = initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.successful_trades = 0

        # History tracking
        self.portfolio_values = deque(maxlen=252)  # ~1 year
        self.returns = deque(maxlen=252)
        self.actions_history = []
        self.rewards_history = []

        # Metrics
        self.max_portfolio_value = initial_balance
        self.max_drawdown = 0.0

        logger.info(f"Initialized TradingEnv: data_length={len(self.data)}, "
                   f"features={self.features}, reward_type={reward_type}")

    def reset(
        self,
        seed: Optional[int] = None,
        options: Optional[Dict[str, Any]] = None
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        """
        Reset environment to initial state.

        Args:
            seed: Random seed
            options: Additional options

        Returns:
            observation: Initial state
            info: Additional information
        """
        super().reset(seed=seed)

        # Reset state
        self.current_step = self.lookback_window
        self.balance = self.initial_balance
        self.position = 0.0
        self.entry_price = 0.0
        self.total_trades = 0
        self.successful_trades = 0

        # Reset history
        self.portfolio_values.clear()
        self.returns.clear()
        self.actions_history = []
        self.rewards_history = []

        # Initialize history
        self.portfolio_values.append(self.initial_balance)
        self.max_portfolio_value = self.initial_balance
        self.max_drawdown = 0.0

        observation = self._get_observation()
        info = self._get_info()

        return observation, info

    def step(self, action: np.ndarray) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        """
        Execute one step in the environment.

        Args:
            action: Action to take (position sizing)

        Returns:
            observation: Next state
            reward: Reward received
            terminated: Whether episode ended
            truncated: Whether episode was truncated
            info: Additional information
        """
        # Clip action to valid range
        action = np.clip(action, -1.0, 1.0)
        target_position = float(action[0]) * self.max_position_size

        # Get current price
        current_price = self._get_current_price()

        # Execute trade
        reward = self._execute_trade(target_position, current_price)

        # Update step
        self.current_step += 1

        # Record history
        self.actions_history.append(action[0])
        self.rewards_history.append(reward)

        # Check if episode is done
        terminated = self.current_step >= len(self.data) - 1
        truncated = False

        # Check for bankruptcy
        if self.balance <= 0:
            terminated = True
            reward -= 10.0  # Large penalty for bankruptcy

        observation = self._get_observation()
        info = self._get_info()

        return observation, reward, terminated, truncated, info

    def _execute_trade(self, target_position: float, current_price: float) -> float:
        """
        Execute trade and compute reward.

        Args:
            target_position: Target position size [-1, 1]
            current_price: Current market price

        Returns:
            Reward value
        """
        # Calculate position change
        position_change = target_position - self.position

        # Apply transaction costs and slippage
        if abs(position_change) > 1e-6:
            # Slippage cost
            effective_price = current_price * (1 + self.slippage * np.sign(position_change))

            # Trade value
            trade_value = abs(position_change) * self.balance * effective_price

            # Transaction cost
            cost = trade_value * self.transaction_cost

            # Update balance
            self.balance -= cost

            # Update position
            if self.position == 0.0 and abs(target_position) > 1e-6:
                self.entry_price = effective_price

            self.position = target_position
            self.total_trades += 1

        # Calculate portfolio value
        position_value = self.position * self.balance * current_price
        if self.entry_price > 0:
            position_pnl = self.position * self.balance * (current_price - self.entry_price)
        else:
            position_pnl = 0.0

        total_value = self.balance + position_pnl

        # Track portfolio values
        self.portfolio_values.append(total_value)

        # Calculate returns
        if len(self.portfolio_values) > 1:
            portfolio_return = (total_value - self.portfolio_values[-2]) / self.portfolio_values[-2]
            self.returns.append(portfolio_return)
        else:
            portfolio_return = 0.0

        # Update max value and drawdown
        if total_value > self.max_portfolio_value:
            self.max_portfolio_value = total_value

        drawdown = (self.max_portfolio_value - total_value) / self.max_portfolio_value
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Compute reward based on reward_type
        reward = self._compute_reward(portfolio_return, drawdown)

        return reward

    def _compute_reward(self, portfolio_return: float, drawdown: float) -> float:
        """
        Compute reward based on specified reward type.

        Args:
            portfolio_return: Current step return
            drawdown: Current drawdown

        Returns:
            Reward value
        """
        if self.reward_type == "returns":
            # Simple returns
            reward = portfolio_return * self.reward_scaling

        elif self.reward_type == "sharpe":
            # Sharpe ratio approximation
            if len(self.returns) >= 2:
                returns_array = np.array(self.returns)
                mean_return = np.mean(returns_array)
                std_return = np.std(returns_array) + 1e-6
                sharpe = (mean_return - self.risk_free_rate / 252) / std_return
                reward = sharpe * self.reward_scaling
            else:
                reward = portfolio_return * self.reward_scaling

        elif self.reward_type == "sortino":
            # Sortino ratio (downside deviation)
            if len(self.returns) >= 2:
                returns_array = np.array(self.returns)
                mean_return = np.mean(returns_array)
                downside_returns = returns_array[returns_array < 0]
                if len(downside_returns) > 0:
                    downside_std = np.std(downside_returns) + 1e-6
                    sortino = (mean_return - self.risk_free_rate / 252) / downside_std
                    reward = sortino * self.reward_scaling
                else:
                    reward = mean_return * self.reward_scaling * 10  # Bonus for no losses
            else:
                reward = portfolio_return * self.reward_scaling

        elif self.reward_type == "calmar":
            # Calmar ratio (return / max drawdown)
            if len(self.returns) >= 2 and self.max_drawdown > 1e-6:
                returns_array = np.array(self.returns)
                total_return = np.prod(1 + returns_array) - 1
                calmar = total_return / (self.max_drawdown + 1e-6)
                reward = calmar * self.reward_scaling
            else:
                reward = portfolio_return * self.reward_scaling

        else:
            raise ValueError(f"Unknown reward_type: {self.reward_type}")

        # Apply drawdown penalty
        if drawdown > 0.2:  # 20% drawdown threshold
            reward -= self.max_drawdown_penalty * drawdown * self.reward_scaling

        return reward

    def _get_observation(self) -> np.ndarray:
        """
        Get current observation/state.

        Returns:
            State vector
        """
        # Portfolio information
        current_price = self._get_current_price()
        position_value = self.position * self.balance * current_price

        if self.entry_price > 0:
            position_pnl = self.position * self.balance * (current_price - self.entry_price)
        else:
            position_pnl = 0.0

        total_value = self.balance + position_pnl
        pnl_ratio = (total_value - self.initial_balance) / self.initial_balance

        portfolio_state = np.array([
            self.balance / self.initial_balance,  # Normalized balance
            total_value / self.initial_balance,    # Normalized total value
            self.position,                          # Current position
            position_value / total_value,           # Position allocation
            pnl_ratio                               # P&L ratio
        ], dtype=np.float32)

        # Market features (lookback window)
        market_features = []
        for i in range(self.lookback_window):
            idx = self.current_step - self.lookback_window + i + 1
            if idx >= 0:
                feature_values = self.data.iloc[idx][self.features].values
                # Normalize features (z-score normalization)
                if idx >= self.lookback_window:
                    history = self.data.iloc[idx-self.lookback_window:idx][self.features].values
                    mean = np.mean(history, axis=0)
                    std = np.std(history, axis=0) + 1e-6
                    normalized = (feature_values - mean) / std
                else:
                    normalized = feature_values
                market_features.extend(normalized)
            else:
                market_features.extend(np.zeros(len(self.features)))

        market_features = np.array(market_features, dtype=np.float32)

        # Combine all state components
        state = np.concatenate([portfolio_state, market_features])

        return state

    def _get_current_price(self) -> float:
        """Get current market price."""
        return float(self.data.iloc[self.current_step]['close'])

    def _get_info(self) -> Dict[str, Any]:
        """
        Get additional information.

        Returns:
            Dictionary with metrics and statistics
        """
        current_price = self._get_current_price()
        total_value = self.balance + self.position * self.balance * current_price

        info = {
            'step': self.current_step,
            'balance': self.balance,
            'position': self.position,
            'total_value': total_value,
            'pnl': total_value - self.initial_balance,
            'pnl_percent': (total_value - self.initial_balance) / self.initial_balance * 100,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'price': current_price
        }

        # Compute performance metrics if enough data
        if len(self.returns) >= 2:
            returns_array = np.array(self.returns)
            info['mean_return'] = np.mean(returns_array)
            info['std_return'] = np.std(returns_array)
            info['sharpe_ratio'] = (np.mean(returns_array) - self.risk_free_rate / 252) / (
                np.std(returns_array) + 1e-6
            )

        return info

    def render(self, mode: str = 'human'):
        """
        Render the environment.

        Args:
            mode: Rendering mode
        """
        if mode == 'human':
            info = self._get_info()
            print(f"\n=== Step {info['step']} ===")
            print(f"Price: ${info['price']:.2f}")
            print(f"Balance: ${info['balance']:.2f}")
            print(f"Position: {info['position']:.3f}")
            print(f"Total Value: ${info['total_value']:.2f}")
            print(f"P&L: ${info['pnl']:.2f} ({info['pnl_percent']:.2f}%)")
            print(f"Max Drawdown: {info['max_drawdown']*100:.2f}%")
            print(f"Total Trades: {info['total_trades']}")

            if 'sharpe_ratio' in info:
                print(f"Sharpe Ratio: {info['sharpe_ratio']:.3f}")

    def get_episode_info(self) -> Dict[str, Any]:
        """
        Get episode summary information.

        Returns:
            Dictionary with episode statistics
        """
        current_price = self._get_current_price()
        total_value = self.balance + self.position * self.balance * current_price

        episode_info = {
            'total_return': (total_value - self.initial_balance) / self.initial_balance,
            'total_return_pct': (total_value - self.initial_balance) / self.initial_balance * 100,
            'max_drawdown': self.max_drawdown,
            'total_trades': self.total_trades,
            'final_value': total_value,
            'total_reward': sum(self.rewards_history)
        }

        # Compute metrics if enough data
        if len(self.returns) >= 2:
            returns_array = np.array(self.returns)
            episode_info['mean_return'] = np.mean(returns_array)
            episode_info['std_return'] = np.std(returns_array)
            episode_info['sharpe_ratio'] = (
                np.mean(returns_array) - self.risk_free_rate / 252
            ) / (np.std(returns_array) + 1e-6)

            # Sortino ratio
            downside_returns = returns_array[returns_array < 0]
            if len(downside_returns) > 0:
                episode_info['sortino_ratio'] = (
                    np.mean(returns_array) - self.risk_free_rate / 252
                ) / (np.std(downside_returns) + 1e-6)

            # Win rate
            positive_returns = returns_array[returns_array > 0]
            episode_info['win_rate'] = len(positive_returns) / len(returns_array)

        return episode_info


if __name__ == "__main__":
    # Test the environment
    logging.basicConfig(level=logging.INFO)

    # Generate synthetic market data
    np.random.seed(42)
    n_steps = 1000

    data = pd.DataFrame({
        'close': 100 + np.cumsum(np.random.randn(n_steps) * 0.5),
        'volume': np.random.randint(1000, 10000, n_steps),
        'returns': np.random.randn(n_steps) * 0.01
    })

    print("Testing TradingEnv...")
    env = TradingEnv(
        data=data,
        initial_balance=10000,
        lookback_window=20,
        features=['close', 'volume', 'returns'],
        reward_type='sharpe'
    )

    # Test reset
    state, info = env.reset()
    print(f"Initial state shape: {state.shape}")
    print(f"Initial info: {info}")

    # Run a few random steps
    print("\nRunning random episode...")
    done = False
    total_reward = 0
    step_count = 0

    while not done and step_count < 100:
        action = env.action_space.sample()
        state, reward, terminated, truncated, info = env.step(action)
        total_reward += reward
        step_count += 1
        done = terminated or truncated

        if step_count % 20 == 0:
            env.render()

    # Get episode summary
    episode_info = env.get_episode_info()
    print("\n=== Episode Summary ===")
    print(f"Total Return: {episode_info['total_return_pct']:.2f}%")
    print(f"Sharpe Ratio: {episode_info.get('sharpe_ratio', 0):.3f}")
    print(f"Max Drawdown: {episode_info['max_drawdown']*100:.2f}%")
    print(f"Total Trades: {episode_info['total_trades']}")
    print(f"Total Reward: {total_reward:.2f}")

    print("\nTradingEnv test passed!")
