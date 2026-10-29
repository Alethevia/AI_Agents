"""
Installation test script for SAC RL module.

This script verifies that all components are properly installed and functional.
Run this after installing dependencies to ensure everything works.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path

# Colors for terminal output
class Colors:
    GREEN = '\033[92m'
    RED = '\033[91m'
    YELLOW = '\033[93m'
    BLUE = '\033[94m'
    END = '\033[0m'


def print_header(text):
    """Print formatted header."""
    print(f"\n{Colors.BLUE}{'='*70}")
    print(f"{text}")
    print(f"{'='*70}{Colors.END}\n")


def print_success(text):
    """Print success message."""
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")


def print_error(text):
    """Print error message."""
    print(f"{Colors.RED}✗ {text}{Colors.END}")


def print_warning(text):
    """Print warning message."""
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")


def test_dependencies():
    """Test that all required dependencies are installed."""
    print_header("Testing Dependencies")

    dependencies = {
        'numpy': 'NumPy',
        'pandas': 'Pandas',
        'torch': 'PyTorch',
        'gymnasium': 'Gymnasium'
    }

    all_installed = True

    for module_name, display_name in dependencies.items():
        try:
            __import__(module_name)
            print_success(f"{display_name} is installed")
        except ImportError:
            print_error(f"{display_name} is NOT installed")
            all_installed = False

    return all_installed


def test_module_imports():
    """Test that all module components can be imported."""
    print_header("Testing Module Imports")

    try:
        from src.models.rl import (
            SACAgent,
            MLPActor,
            MLPCritic,
            TwinCritic,
            StateNormalizer,
            ReplayBuffer,
            PrioritizedReplayBuffer,
            NStepReplayBuffer,
            TradingEnv
        )
        print_success("All core components imported successfully")
        return True
    except ImportError as e:
        print_error(f"Failed to import components: {e}")
        return False


def test_network_creation():
    """Test neural network creation."""
    print_header("Testing Neural Networks")

    try:
        from src.models.rl import MLPActor, TwinCritic
        import torch

        state_dim = 10
        action_dim = 3

        # Test actor
        actor = MLPActor(state_dim, action_dim)
        print_success(f"Created MLPActor (state_dim={state_dim}, action_dim={action_dim})")

        # Test critic
        critic = TwinCritic(state_dim, action_dim)
        print_success(f"Created TwinCritic (state_dim={state_dim}, action_dim={action_dim})")

        # Test forward pass
        states = torch.randn(32, state_dim)
        actions, log_probs = actor(states)
        print_success(f"Actor forward pass: actions={actions.shape}, log_probs={log_probs.shape}")

        q1, q2 = critic(states, actions)
        print_success(f"Critic forward pass: Q1={q1.shape}, Q2={q2.shape}")

        return True
    except Exception as e:
        print_error(f"Network creation failed: {e}")
        return False


def test_replay_buffer():
    """Test replay buffer functionality."""
    print_header("Testing Replay Buffers")

    try:
        from src.models.rl import ReplayBuffer, PrioritizedReplayBuffer
        import numpy as np

        state_dim = 10
        action_dim = 3

        # Test basic buffer
        buffer = ReplayBuffer(state_dim, action_dim, capacity=1000)
        print_success(f"Created ReplayBuffer (capacity=1000)")

        # Add samples
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = False
            buffer.add(state, action, reward, next_state, done)

        print_success(f"Added 100 transitions (buffer size: {len(buffer)})")

        # Sample batch
        batch = buffer.sample(32)
        print_success(f"Sampled batch of 32 transitions")

        # Test PER buffer
        per_buffer = PrioritizedReplayBuffer(state_dim, action_dim, capacity=1000)
        print_success(f"Created PrioritizedReplayBuffer (capacity=1000)")

        # Add samples
        for _ in range(100):
            state = np.random.randn(state_dim)
            action = np.random.randn(action_dim)
            reward = np.random.randn()
            next_state = np.random.randn(state_dim)
            done = False
            per_buffer.add(state, action, reward, next_state, done)

        batch, indices, weights = per_buffer.sample(32)
        print_success(f"PER sampled batch with importance weights")

        return True
    except Exception as e:
        print_error(f"Replay buffer test failed: {e}")
        return False


def test_trading_env():
    """Test trading environment."""
    print_header("Testing Trading Environment")

    try:
        from src.models.rl import TradingEnv
        import pandas as pd
        import numpy as np

        # Create synthetic data
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(1000) * 0.5),
            'volume': np.random.randint(1000, 10000, 1000),
            'returns': np.random.randn(1000) * 0.01
        })
        print_success(f"Created synthetic market data (1000 samples)")

        # Create environment
        env = TradingEnv(
            data=data,
            initial_balance=100000,
            features=['close', 'volume', 'returns'],
            reward_type='sharpe'
        )
        print_success(f"Created TradingEnv (state_dim={env.observation_space.shape[0]})")

        # Test reset
        state, info = env.reset()
        print_success(f"Environment reset: state={state.shape}")

        # Test step
        action = env.action_space.sample()
        next_state, reward, terminated, truncated, info = env.step(action)
        print_success(f"Environment step: reward={reward:.2f}, done={terminated or truncated}")

        return True
    except Exception as e:
        print_error(f"Trading environment test failed: {e}")
        return False


def test_sac_agent():
    """Test SAC agent."""
    print_header("Testing SAC Agent")

    try:
        from src.models.rl import SACAgent, TradingEnv
        import pandas as pd
        import numpy as np

        # Create environment
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.randint(1000, 10000, 500),
            'returns': np.random.randn(500) * 0.01
        })

        env = TradingEnv(data=data, features=['close', 'volume', 'returns'])
        print_success(f"Created test environment")

        # Create agent
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            buffer_size=10000
        )
        print_success(f"Created SAC agent (device={agent.device})")

        # Test action selection
        state, _ = env.reset()
        action = agent.select_action(state, deterministic=False)
        print_success(f"Selected action: {action.shape}")

        # Test storing transition
        next_state, reward, terminated, truncated, info = env.step(action)
        agent.store_transition(state, action, reward, next_state, terminated)
        print_success(f"Stored transition (buffer size: {len(agent.replay_buffer)})")

        # Add more samples
        for _ in range(300):
            state, _ = env.reset()
            action = agent.select_action(state)
            next_state, reward, terminated, truncated, info = env.step(action)
            agent.store_transition(state, action, reward, next_state, terminated)

        print_success(f"Added 300 samples (buffer size: {len(agent.replay_buffer)})")

        # Test training step
        metrics = agent.train_step()
        print_success(f"Training step completed: actor_loss={metrics['actor_loss']:.4f}")

        # Test checkpoint save/load
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.pt', delete=False) as tmp:
            checkpoint_path = tmp.name

        agent.save_checkpoint(checkpoint_path)
        print_success(f"Saved checkpoint")

        agent.load_checkpoint(checkpoint_path)
        print_success(f"Loaded checkpoint")

        # Clean up
        Path(checkpoint_path).unlink()

        return True
    except Exception as e:
        print_error(f"SAC agent test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def test_training_loop():
    """Test a mini training loop."""
    print_header("Testing Training Loop")

    try:
        from src.models.rl import SACAgent, TradingEnv
        import pandas as pd
        import numpy as np

        # Create environment
        np.random.seed(42)
        data = pd.DataFrame({
            'close': 100 + np.cumsum(np.random.randn(500) * 0.5),
            'volume': np.random.randint(1000, 10000, 500),
            'returns': np.random.randn(500) * 0.01
        })

        env = TradingEnv(data=data, features=['close', 'volume', 'returns'])

        # Create agent
        agent = SACAgent(
            state_dim=env.observation_space.shape[0],
            action_dim=env.action_space.shape[0],
            buffer_size=10000
        )

        # Run 5 episodes
        print("Running 5 training episodes...")
        for episode in range(5):
            stats = agent.train_episode(env, warmup_steps=100)
            print(f"  Episode {episode + 1}: Reward={stats['reward']:7.2f}, "
                  f"Steps={stats['length']:3d}")

        print_success("Completed 5 training episodes")

        # Test evaluation
        eval_metrics = agent.evaluate(env, num_episodes=2)
        print_success(f"Evaluation: mean_reward={eval_metrics['mean_reward']:.2f}")

        return True
    except Exception as e:
        print_error(f"Training loop test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


def run_all_tests():
    """Run all tests."""
    print_header("SAC RL Module Installation Test")
    print("Testing all components of the SAC implementation...")

    tests = [
        ("Dependencies", test_dependencies),
        ("Module Imports", test_module_imports),
        ("Neural Networks", test_network_creation),
        ("Replay Buffers", test_replay_buffer),
        ("Trading Environment", test_trading_env),
        ("SAC Agent", test_sac_agent),
        ("Training Loop", test_training_loop),
    ]

    results = []

    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print_error(f"Test '{test_name}' crashed: {e}")
            results.append((test_name, False))

    # Print summary
    print_header("Test Summary")

    passed = sum(1 for _, result in results if result)
    total = len(results)

    for test_name, result in results:
        if result:
            print_success(f"{test_name}: PASSED")
        else:
            print_error(f"{test_name}: FAILED")

    print(f"\n{Colors.BLUE}{'='*70}")
    if passed == total:
        print(f"{Colors.GREEN}All tests passed! ({passed}/{total}){Colors.END}")
        print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
        print(f"\n{Colors.GREEN}✓ Installation is complete and working!{Colors.END}")
        print("\nNext steps:")
        print("  1. Run the example: python src/models/rl/example_usage.py")
        print("  2. Read QUICKSTART.md for usage guide")
        print("  3. Start training your own models!\n")
        return True
    else:
        print(f"{Colors.RED}Some tests failed ({passed}/{total} passed){Colors.END}")
        print(f"{Colors.BLUE}{'='*70}{Colors.END}\n")
        print(f"\n{Colors.YELLOW}Please check the errors above and ensure all dependencies are installed.{Colors.END}")
        print("\nTo install dependencies:")
        print("  pip install -r src/models/rl/requirements.txt\n")
        return False


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
