# Institutional AI Trading System 2.0

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

**Production-grade institutional AI trading system featuring state-of-the-art deep learning, reinforcement learning, and multi-agent coordination.**

Upgraded from Stock_Trading_Robot_RAC with cutting-edge 2025 AI technologies, replacing outdated approaches like XGBoost with modern Transformer architectures and Soft Actor-Critic reinforcement learning.

## 🚀 Key Features

### Advanced AI Models
- **Transformer-based Time Series Prediction**
  - Temporal Fusion Transformer (TFT) with variable selection and interpretability
  - Informer for efficient long-sequence modeling (O(L log L) complexity)
  - Autoformer with auto-correlation mechanism replacing self-attention
  - Multi-head attention with temporal and spatial mechanisms

- **Soft Actor-Critic (SAC) Reinforcement Learning**
  - Continuous action space for precise position sizing
  - Automatic entropy tuning for optimal exploration
  - Twin Q-networks to reduce overestimation bias
  - Prioritized experience replay and N-step returns
  - Realistic trading environment with transaction costs

### Multi-Agent System
- **6 Specialized Trading Agents**
  - **Market Maker**: Spread optimization, inventory management, multi-layer quoting
  - **Trend Follower**: Moving averages, ADX, breakout detection, multi-timeframe analysis
  - **Mean Reversion**: Bollinger Bands, pairs trading, z-score, cointegration testing
  - **Arbitrage**: Cross-exchange, statistical, triangular arbitrage detection
  - **Momentum**: Relative strength, sector rotation, cross-sectional ranking
  - **Coordinator**: Signal aggregation, conflict resolution, dynamic agent weighting

### Production Infrastructure
- **Real-time Data Pipeline**: YFinance, Alpha Vantage, Polygon.io, news sentiment, social media
- **Feature Engineering**: 145+ methods across technical, fundamental, microstructure, and sentiment
- **Risk Management**: VaR (5 methods), CVaR, drawdown monitoring, position limits, circuit breakers
- **Smart Execution**: TWAP, VWAP, POV, smart routing across brokers, slippage modeling
- **Portfolio Optimization**: Mean-variance, Black-Litterman, risk parity, hierarchical risk parity
- **Backtesting Framework**: Event-driven simulation with 31 performance metrics
- **REST & WebSocket APIs**: Real-time monitoring and control with authentication
- **Comprehensive Monitoring**: Prometheus metrics, Grafana dashboards, health checks, alerting

## 📊 System Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                    Trading System Core (~20K+ Lines)                │
├────────────────────────────────────────────────────────────────────┤
│                                                                      │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐  ┌──────────┐          │
│  │  Data    │  │ Feature  │  │  Model   │  │  Agent   │          │
│  │Pipeline  │→ │Engineer  │→ │Prediction│→ │ System   │          │
│  │  (7+)    │  │ (145+)   │  │(TFT/SAC) │  │   (6)    │          │
│  └──────────┘  └──────────┘  └──────────┘  └──────────┘          │
│       ↓              ↓              ↓              ↓               │
│  ┌─────────────────────────────────────────────────────┐          │
│  │         Strategy Generator & Optimizer               │          │
│  │    (Grid/Bayesian/Genetic/Walk-Forward)             │          │
│  └─────────────────────────────────────────────────────┘          │
│       ↓                                                            │
│  ┌──────────┐  ┌──────────┐  ┌──────────┐                       │
│  │   Risk   │→ │Execution │→ │Portfolio │                       │
│  │ Manager  │  │  Engine  │  │ Manager  │                       │
│  │(VaR/CVaR)│  │(TWAP/SOR)│  │(OptPort) │                       │
│  └──────────┘  └──────────┘  └──────────┘                       │
│       ↓              ↓              ↓                             │
│  ┌───────────────────────────────────────────────┐               │
│  │    Monitoring (Prometheus) & Alerting          │               │
│  └───────────────────────────────────────────────┘               │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘
```

## 🔧 Technology Stack

### Core Technologies
- **Python 3.11+** - Primary language with latest features
- **PyTorch 2.1+** - Deep learning framework with compile support
- **FastAPI** - High-performance REST API framework
- **WebSockets** - Real-time bidirectional communication
- **Polars & Pandas** - Fast data processing
- **Redis** - Caching and message broker
- **PostgreSQL 15+** - Relational database
- **MongoDB 6.0+** - Document database (optional)
- **Prometheus + Grafana** - Monitoring and visualization

### AI/ML Libraries
- **Transformers (Hugging Face)** - Latest transformer architectures
- **Stable-Baselines3** - State-of-the-art RL algorithms
- **Gymnasium** - Modern RL environments (successor to Gym)
- **Optuna** - Bayesian hyperparameter optimization
- **scikit-learn** - Traditional ML algorithms

### Trading & Finance
- **CCXT** - Cryptocurrency exchange integration
- **yfinance** - Yahoo Finance data
- **alpaca-py** - Alpaca broker API
- **ib_insync** - Interactive Brokers API
- **QuantStats** - Professional performance analytics
- **vectorbt** - Fast backtesting
- **cvxpy** - Convex optimization for portfolio

## 📦 Installation

### Prerequisites
```bash
- Python 3.11 or higher
- Redis 7.0+
- PostgreSQL 15+
- MongoDB 6.0+ (optional)
- 8GB+ RAM recommended
- GPU recommended for model training (CUDA 11.8+)
```

### Quick Start

```bash
# Clone the repository
git clone https://github.com/MauveAndromeda/AI_Agents.git
cd AI_Agents

# Create virtual environment
python3.11 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install poetry
poetry install

# Or with pip
pip install -r requirements.txt

# Copy and configure environment
cp .env.example .env
nano .env  # Add your API keys and settings

# Initialize database (if using PostgreSQL)
python scripts/init_db.py

# Start the system
python -m src.main --env development --log-level INFO
```

### Docker Deployment (Recommended)

```bash
# Build and start all services
docker-compose up -d

# View logs
docker-compose logs -f trading-system

# Access services
# - Trading System API: http://localhost:8000
# - Prometheus: http://localhost:9091
# - Grafana: http://localhost:3000 (admin/admin)

# Stop services
docker-compose down
```

## 🎯 Usage

### Command Line Interface

```bash
# Start the trading system
python -m src.cli start --env production

# Check system status (shows all components)
python -m src.cli status

# Run backtest
python -m src.cli backtest run \
    --strategy momentum \
    --symbols AAPL,TSLA,MSFT \
    --start 2023-01-01 \
    --end 2023-12-31

# Train Transformer model
python -m src.cli model train \
    --model transformer \
    --symbols AAPL,GOOGL \
    --epochs 100

# Train SAC RL agent
python -m src.cli model train \
    --model sac \
    --episodes 1000 \
    --use-per

# View real-time performance
python -m src.cli performance --period 30d

# View current positions
python -m src.cli positions

# View system logs
python -m src.cli logs --level error --tail 100
```

### Python API

```python
import asyncio
from src.config import get_settings
from src.data.collectors import YFinanceCollector
from src.features.extractors import TechnicalIndicators
from src.models.transformers import TemporalFusionTransformer
from src.models.rl import SACAgent, TradingEnv
from src.agents import TrendFollowerAgent, Coordinator
from src.execution import OrderManager
from src.risk import RiskMonitor

async def main():
    # Initialize settings
    settings = get_settings()

    # Data collection
    collector = YFinanceCollector()
    data = await collector.fetch_bars('AAPL', interval='1m', limit=1000)

    # Feature engineering
    features = TechnicalIndicators(symbol='AAPL')
    feature_data = features.calculate_all_indicators(data)

    # Transformer prediction
    transformer = TemporalFusionTransformer.load_checkpoint('models/tft_best.pt')
    price_prediction = transformer.predict(feature_data)

    # SAC RL decision
    sac_env = TradingEnv(data=data, features=feature_data)
    sac_agent = SACAgent.load_checkpoint('models/sac_best.pt')
    rl_action = sac_agent.select_action(sac_env.get_state(), evaluate=True)

    # Multi-agent coordination
    trend_agent = TrendFollowerAgent()
    coordinator = Coordinator()
    coordinator.register_agent(trend_agent)

    # Generate aggregated signal
    signal = await coordinator.generate_aggregated_signal(feature_data)

    # Risk check
    risk_monitor = RiskMonitor(settings.risk_config)
    risk_result = await risk_monitor.check_all_limits(portfolio)

    # Execute if approved
    if signal.strength >= SignalStrength.STRONG and not risk_result.has_violations:
        order_manager = OrderManager()
        order = await order_manager.create_order(
            symbol=signal.symbol,
            side=signal.type,
            quantity=signal.quantity,
            order_type='limit',
            limit_price=signal.price
        )
        await order_manager.submit_order(order.order_id)

# Run
asyncio.run(main())
```

### REST API Examples

```bash
# Health check
curl http://localhost:8000/health

# Login
TOKEN=$(curl -X POST http://localhost:8000/api/v1/auth/login \
  -H "Content-Type: application/json" \
  -d '{"username":"admin","password":"admin123"}' \
  | jq -r '.access_token')

# Get portfolio
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/portfolio | jq

# Create order
curl -X POST http://localhost:8000/api/v1/orders/create \
     -H "Authorization: Bearer $TOKEN" \
     -H "Content-Type: application/json" \
     -d '{
       "symbol": "AAPL",
       "side": "buy",
       "quantity": 100,
       "order_type": "limit",
       "limit_price": 150.00
     }' | jq

# Get risk metrics
curl -H "Authorization: Bearer $TOKEN" \
     http://localhost:8000/api/v1/risk/metrics | jq

# Start a strategy
curl -X POST "http://localhost:8000/api/v1/strategies/start?strategy_id=momentum_1" \
     -H "Authorization: Bearer $TOKEN"
```

### WebSocket Real-time Updates

```python
import asyncio
import websockets
import json

async def subscribe_updates():
    uri = f"ws://localhost:8000/ws?token={YOUR_JWT_TOKEN}"

    async with websockets.connect(uri) as websocket:
        # Subscribe to topics
        await websocket.send(json.dumps({
            "type": "subscribe",
            "data": {
                "topics": [
                    "market_data.ticker.AAPL",
                    "market_data.ticker.TSLA",
                    "orders.updates",
                    "portfolio.updates",
                    "risk.alerts"
                ]
            }
        }))

        # Receive real-time updates
        while True:
            message = await websocket.recv()
            data = json.loads(message)
            print(f"Update: {data['topic']}: {data['data']}")

asyncio.run(subscribe_updates())
```

## 📈 Code Statistics

### Total Implementation
- **~22,000+ lines** of production code
- **150+ Python modules**
- **300+ classes and functions**
- **100% type-hinted**
- **Comprehensive docstrings**
- **Full test coverage framework**

### Key Modules
| Module | Lines | Description |
|--------|-------|-------------|
| Models (Transformers) | 3,726 | TFT, Informer, Autoformer implementations |
| Models (RL/SAC) | 5,075 | Complete SAC with trading environment |
| Data Collectors | 4,008 | Multi-source data collection |
| Data Processors | 5,449 | Cleaning, normalization, aggregation |
| Feature Engineering | 4,221 | 145+ feature extraction methods |
| Multi-Agent System | 4,805 | 6 specialized trading agents |
| Strategy & Optimization | 4,740 | Signal generation and optimization |
| Risk Management | 4,373 | VaR, limits, monitoring, stress testing |
| Execution Engine | 4,993 | Order management and smart routing |
| Portfolio Management | 4,297 | Optimization, rebalancing, analytics |
| Backtesting Framework | 4,490 | Event-driven backtesting |
| API Services | 6,000+ | REST and WebSocket APIs |
| Core Infrastructure | 3,000+ | Logging, monitoring, error handling |

## 🔒 Security

- **JWT Authentication** - Secure token-based auth with expiration
- **API Key Management** - Rate-limited keys with role permissions
- **Role-Based Access Control (RBAC)** - Admin, trader, viewer roles
- **Rate Limiting** - Per-user, per-IP, per-endpoint limits
- **Input Validation** - Pydantic models with comprehensive validation
- **Secret Management** - Environment-based encrypted secrets
- **SQL Injection Protection** - SQLAlchemy ORM parameterization
- **CORS Configuration** - Configurable cross-origin policies

## 📊 Monitoring & Observability

### Metrics Collection
- **System**: CPU, memory, disk, network
- **Trading**: Trades, volume, P&L, positions, fills
- **Portfolio**: Value, returns, drawdown, exposure
- **Risk**: VaR, CVaR, volatility, beta, Sharpe
- **Models**: Accuracy, latency, predictions, updates
- **API**: Requests, latency, errors, rate limits
- **Agents**: Actions, signals, performance

### Dashboards
- **Grafana**: http://localhost:3000 (admin/admin)
- **API Docs**: http://localhost:8000/docs
- **Prometheus**: http://localhost:9091

### Alerting
- Risk limit breaches
- Circuit breaker triggers
- System errors and failures
- Model performance degradation
- Unusual trading activity

## 🧪 Testing

```bash
# Run all tests
pytest

# Run with coverage report
pytest --cov=src --cov-report=html --cov-report=term

# Run specific test suites
pytest tests/unit/                  # Unit tests
pytest tests/integration/           # Integration tests
pytest tests/performance/           # Performance tests

# Run with markers
pytest -m "not slow"                # Skip slow tests
pytest -m "integration"             # Only integration tests

# Generate HTML coverage report
open htmlcov/index.html  # View coverage
```

## 📚 Documentation

Comprehensive documentation available:

- **API Documentation**: http://localhost:8000/docs (Interactive Swagger UI)
- **Backtesting Guide**: [src/backtesting/README.md](src/backtesting/README.md)
- **SAC RL Guide**: [src/models/rl/README.md](src/models/rl/README.md)
- **Feature Engineering**: [src/features/README.md](src/features/README.md)
- **Type Definitions**: [docs/TYPE_DEFINITIONS.md](docs/TYPE_DEFINITIONS.md)

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure:
- Code follows black formatting
- Type hints are included
- Tests are added for new features
- Documentation is updated

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## ⚠️ Disclaimer

**IMPORTANT: This software is for educational and research purposes only.**

- Trading financial instruments carries significant risk of loss
- Past performance is not indicative of future results
- You may lose some or all of your invested capital
- Only trade with money you can afford to lose
- This system is NOT investment advice
- Always test thoroughly in paper trading before using real capital
- The developers assume no liability for trading losses

## 🙏 Acknowledgments

- **Original Inspiration**: Stock_Trading_Robot_RAC multi-agent framework
- **Research Papers**:
  - Temporal Fusion Transformers (Google Research)
  - Soft Actor-Critic (Berkeley AI Research)
  - Informer: Beyond Efficient Transformer
  - Autoformer: Decomposition Transformers
- **Community**: Open-source ML/trading communities

## 📞 Support

- **GitHub Issues**: [Report bugs and request features](https://github.com/MauveAndromeda/AI_Agents/issues)
- **GitHub Discussions**: [Ask questions and share ideas](https://github.com/MauveAndromeda/AI_Agents/discussions)

---

**Built with cutting-edge 2025 AI technologies for institutional-grade algorithmic trading** 🚀📈

*Disclaimer: Trading involves risk. Use responsibly.*
