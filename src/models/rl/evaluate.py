"""
Evaluation and analysis utilities for trained SAC agents.

This module provides tools for:
- Evaluating trained models
- Analyzing trading performance
- Generating performance reports
- Comparing models
"""

import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Any, Optional
import json
import logging
from datetime import datetime

logger = logging.getLogger(__name__)


class PerformanceAnalyzer:
    """
    Analyze trading performance of RL agents.

    Computes comprehensive metrics including:
    - Returns and cumulative returns
    - Risk metrics (volatility, Sharpe, Sortino, max drawdown)
    - Trading statistics (win rate, profit factor, etc.)
    """

    def __init__(self):
        """Initialize performance analyzer."""
        self.metrics = {}

    def analyze_episode(self, env, info: Dict[str, Any]) -> Dict[str, float]:
        """
        Analyze a single episode's performance.

        Args:
            env: Trading environment
            info: Episode info from environment

        Returns:
            Dictionary of performance metrics
        """
        episode_info = env.get_episode_info()

        metrics = {
            'total_return': episode_info.get('total_return', 0),
            'total_return_pct': episode_info.get('total_return_pct', 0),
            'sharpe_ratio': episode_info.get('sharpe_ratio', 0),
            'sortino_ratio': episode_info.get('sortino_ratio', 0),
            'max_drawdown': episode_info.get('max_drawdown', 0),
            'total_trades': episode_info.get('total_trades', 0),
            'win_rate': episode_info.get('win_rate', 0),
            'final_value': episode_info.get('final_value', 0),
            'total_reward': episode_info.get('total_reward', 0)
        }

        return metrics

    def analyze_multiple_episodes(
        self,
        agent,
        env,
        num_episodes: int = 100,
        deterministic: bool = True
    ) -> Dict[str, Any]:
        """
        Analyze performance across multiple episodes.

        Args:
            agent: Trained SAC agent
            env: Trading environment
            num_episodes: Number of episodes to evaluate
            deterministic: Use deterministic policy

        Returns:
            Aggregated performance metrics
        """
        all_metrics = []

        logger.info(f"Analyzing {num_episodes} episodes...")

        for episode in range(num_episodes):
            state, _ = env.reset()
            done = False

            while not done:
                action = agent.select_action(state, deterministic=deterministic, eval_mode=True)
                next_state, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated
                state = next_state

            # Get episode metrics
            metrics = self.analyze_episode(env, info)
            all_metrics.append(metrics)

            if (episode + 1) % 10 == 0:
                logger.info(f"  Completed {episode + 1}/{num_episodes} episodes")

        # Aggregate metrics
        df = pd.DataFrame(all_metrics)

        aggregated = {
            'num_episodes': num_episodes,
            'mean_return': df['total_return'].mean(),
            'median_return': df['total_return'].median(),
            'std_return': df['total_return'].std(),
            'min_return': df['total_return'].min(),
            'max_return': df['total_return'].max(),
            'mean_sharpe': df['sharpe_ratio'].mean(),
            'mean_sortino': df['sortino_ratio'].mean(),
            'mean_max_drawdown': df['max_drawdown'].mean(),
            'max_drawdown': df['max_drawdown'].max(),
            'mean_trades': df['total_trades'].mean(),
            'mean_win_rate': df['win_rate'].mean(),
            'positive_return_pct': (df['total_return'] > 0).mean() * 100,
        }

        # Compute additional statistics
        returns = df['total_return'].values
        aggregated['var_95'] = np.percentile(returns, 5)  # 95% VaR
        aggregated['cvar_95'] = returns[returns <= aggregated['var_95']].mean()  # CVaR

        self.metrics = aggregated

        return aggregated

    def print_report(self, metrics: Optional[Dict[str, Any]] = None):
        """
        Print formatted performance report.

        Args:
            metrics: Metrics dictionary (uses stored if None)
        """
        if metrics is None:
            metrics = self.metrics

        if not metrics:
            logger.warning("No metrics available")
            return

        print("\n" + "="*70)
        print("PERFORMANCE ANALYSIS REPORT")
        print("="*70)

        print(f"\nNumber of Episodes: {metrics.get('num_episodes', 'N/A')}")

        print("\n--- Return Metrics ---")
        print(f"Mean Return:      {metrics.get('mean_return', 0)*100:8.2f}%")
        print(f"Median Return:    {metrics.get('median_return', 0)*100:8.2f}%")
        print(f"Std Return:       {metrics.get('std_return', 0)*100:8.2f}%")
        print(f"Min Return:       {metrics.get('min_return', 0)*100:8.2f}%")
        print(f"Max Return:       {metrics.get('max_return', 0)*100:8.2f}%")
        print(f"Positive Return:  {metrics.get('positive_return_pct', 0):8.2f}%")

        print("\n--- Risk Metrics ---")
        print(f"Mean Sharpe:      {metrics.get('mean_sharpe', 0):8.3f}")
        print(f"Mean Sortino:     {metrics.get('mean_sortino', 0):8.3f}")
        print(f"Mean Max DD:      {metrics.get('mean_max_drawdown', 0)*100:8.2f}%")
        print(f"Max Drawdown:     {metrics.get('max_drawdown', 0)*100:8.2f}%")
        print(f"VaR (95%):        {metrics.get('var_95', 0)*100:8.2f}%")
        print(f"CVaR (95%):       {metrics.get('cvar_95', 0)*100:8.2f}%")

        print("\n--- Trading Statistics ---")
        print(f"Mean Trades:      {metrics.get('mean_trades', 0):8.1f}")
        print(f"Mean Win Rate:    {metrics.get('mean_win_rate', 0)*100:8.2f}%")

        print("\n" + "="*70 + "\n")

    def save_report(self, filepath: str, metrics: Optional[Dict[str, Any]] = None):
        """
        Save performance report to file.

        Args:
            filepath: Path to save report
            metrics: Metrics dictionary (uses stored if None)
        """
        if metrics is None:
            metrics = self.metrics

        filepath = Path(filepath)
        filepath.parent.mkdir(parents=True, exist_ok=True)

        # Convert numpy types to Python types
        json_metrics = {
            k: float(v) if isinstance(v, (np.floating, np.integer)) else v
            for k, v in metrics.items()
        }

        with open(filepath, 'w') as f:
            json.dump(json_metrics, f, indent=2)

        logger.info(f"Report saved to {filepath}")


class ModelComparer:
    """
    Compare performance of multiple trained models.
    """

    def __init__(self):
        """Initialize model comparer."""
        self.models = {}
        self.results = {}

    def add_model(
        self,
        name: str,
        agent,
        checkpoint_path: Optional[str] = None
    ):
        """
        Add model to comparison.

        Args:
            name: Model name
            agent: SAC agent (will load checkpoint if path provided)
            checkpoint_path: Path to model checkpoint
        """
        if checkpoint_path is not None:
            agent.load_checkpoint(checkpoint_path, load_optimizers=False)
            logger.info(f"Loaded model '{name}' from {checkpoint_path}")

        self.models[name] = agent

    def compare(
        self,
        env,
        num_episodes: int = 50,
        deterministic: bool = True
    ) -> pd.DataFrame:
        """
        Compare all models on the same environment.

        Args:
            env: Trading environment
            num_episodes: Episodes per model
            deterministic: Use deterministic policy

        Returns:
            DataFrame with comparison results
        """
        analyzer = PerformanceAnalyzer()
        results = []

        for name, agent in self.models.items():
            logger.info(f"\nEvaluating model: {name}")

            metrics = analyzer.analyze_multiple_episodes(
                agent,
                env,
                num_episodes,
                deterministic
            )

            metrics['model_name'] = name
            results.append(metrics)

        df = pd.DataFrame(results)
        df = df.set_index('model_name')

        self.results = df

        return df

    def print_comparison(self, df: Optional[pd.DataFrame] = None):
        """
        Print formatted comparison table.

        Args:
            df: Comparison DataFrame (uses stored if None)
        """
        if df is None:
            df = self.results

        if df.empty:
            logger.warning("No comparison results available")
            return

        print("\n" + "="*100)
        print("MODEL COMPARISON")
        print("="*100)

        # Select key metrics
        key_metrics = [
            'mean_return',
            'std_return',
            'mean_sharpe',
            'mean_max_drawdown',
            'mean_win_rate',
            'positive_return_pct'
        ]

        display_df = df[key_metrics].copy()

        # Format percentages
        for col in ['mean_return', 'std_return', 'mean_max_drawdown', 'mean_win_rate', 'positive_return_pct']:
            if col in display_df.columns:
                display_df[col] = display_df[col] * 100

        # Rename columns for display
        display_df.columns = [
            'Mean Return (%)',
            'Std Return (%)',
            'Sharpe',
            'Max DD (%)',
            'Win Rate (%)',
            'Positive (%)'
        ]

        print("\n", display_df.to_string())
        print("\n" + "="*100 + "\n")

        # Highlight best model for each metric
        print("Best Models by Metric:")
        print(f"  Highest Return:    {display_df['Mean Return (%)'].idxmax()} "
              f"({display_df['Mean Return (%)'].max():.2f}%)")
        print(f"  Best Sharpe:       {display_df['Sharpe'].idxmax()} "
              f"({display_df['Sharpe'].max():.3f})")
        print(f"  Lowest Drawdown:   {display_df['Max DD (%)'].idxmin()} "
              f"({display_df['Max DD (%)'].min():.2f}%)")
        print(f"  Best Win Rate:     {display_df['Win Rate (%)'].idxmax()} "
              f"({display_df['Win Rate (%)'].max():.2f}%)")
        print()


def evaluate_model(
    agent,
    env,
    num_episodes: int = 100,
    checkpoint_path: Optional[str] = None,
    save_report: bool = True,
    report_dir: str = "./reports"
) -> Dict[str, Any]:
    """
    Evaluate a trained model and generate report.

    Args:
        agent: SAC agent
        env: Trading environment
        num_episodes: Number of evaluation episodes
        checkpoint_path: Path to model checkpoint (if loading)
        save_report: Whether to save report to file
        report_dir: Directory to save reports

    Returns:
        Performance metrics
    """
    # Load checkpoint if provided
    if checkpoint_path is not None:
        agent.load_checkpoint(checkpoint_path, load_optimizers=False)
        logger.info(f"Loaded model from {checkpoint_path}")

    # Analyze performance
    analyzer = PerformanceAnalyzer()
    metrics = analyzer.analyze_multiple_episodes(
        agent,
        env,
        num_episodes,
        deterministic=True
    )

    # Print report
    analyzer.print_report(metrics)

    # Save report
    if save_report:
        report_dir = Path(report_dir)
        report_dir.mkdir(parents=True, exist_ok=True)

        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_path = report_dir / f"evaluation_report_{timestamp}.json"

        analyzer.save_report(report_path, metrics)

    return metrics


if __name__ == "__main__":
    # Example usage
    logging.basicConfig(level=logging.INFO)

    print("Performance Analyzer - Example")
    print("\nThis module provides tools for evaluating trained SAC agents.")
    print("\nExample usage:")
    print("""
    from src.models.rl import SACAgent, TradingEnv
    from src.models.rl.evaluate import evaluate_model, ModelComparer

    # Load environment and agent
    env = TradingEnv(data=test_data, ...)
    agent = SACAgent(...)

    # Evaluate single model
    metrics = evaluate_model(
        agent,
        env,
        checkpoint_path='checkpoints/best_model.pt',
        num_episodes=100
    )

    # Compare multiple models
    comparer = ModelComparer()
    comparer.add_model('Model A', agent1, 'checkpoints/model_a.pt')
    comparer.add_model('Model B', agent2, 'checkpoints/model_b.pt')
    comparison = comparer.compare(env, num_episodes=50)
    comparer.print_comparison()
    """)
