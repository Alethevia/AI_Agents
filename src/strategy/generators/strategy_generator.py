"""
Strategy generator module for creating trading strategies.

Provides template-based strategy generation, parameter generation,
rule-based strategy creation, and ML-based strategy generation using GPT-4.
"""

import asyncio
import json
from datetime import datetime
from decimal import Decimal
from enum import Enum
from typing import Any, Dict, List, Optional, Type

from pydantic import BaseModel, Field

from src.core.logging.logger import get_logger, log_async_execution_time
from src.core.types.signals import SignalType, SignalStrength


logger = get_logger(__name__)


class StrategyType(str, Enum):
    """Strategy types."""

    TREND_FOLLOWING = "trend_following"
    MEAN_REVERSION = "mean_reversion"
    MOMENTUM = "momentum"
    ARBITRAGE = "arbitrage"
    MARKET_MAKING = "market_making"
    BREAKOUT = "breakout"
    PAIRS_TRADING = "pairs_trading"
    STATISTICAL_ARBITRAGE = "statistical_arbitrage"
    CUSTOM = "custom"


class ParameterType(str, Enum):
    """Parameter types."""

    INTEGER = "integer"
    FLOAT = "float"
    DECIMAL = "decimal"
    BOOLEAN = "boolean"
    STRING = "string"
    ENUM = "enum"
    LIST = "list"


class ParameterDefinition(BaseModel):
    """Definition of a strategy parameter."""

    name: str = Field(..., description="Parameter name")
    type: ParameterType = Field(..., description="Parameter type")
    default: Any = Field(..., description="Default value")
    min_value: Optional[Any] = Field(None, description="Minimum value")
    max_value: Optional[Any] = Field(None, description="Maximum value")
    choices: Optional[List[Any]] = Field(None, description="Valid choices")
    description: str = Field(..., description="Parameter description")
    required: bool = Field(default=True, description="Whether parameter is required")
    searchable: bool = Field(
        default=True,
        description="Whether parameter is optimizable"
    )


class StrategyRule(BaseModel):
    """Trading rule within a strategy."""

    rule_id: str = Field(..., description="Rule identifier")
    condition: str = Field(..., description="Rule condition")
    action: SignalType = Field(..., description="Action when condition is met")
    priority: int = Field(default=0, description="Rule priority")
    enabled: bool = Field(default=True, description="Whether rule is enabled")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class StrategyTemplate(BaseModel):
    """Template for creating strategies."""

    template_id: str = Field(..., description="Template identifier")
    name: str = Field(..., description="Template name")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    description: str = Field(..., description="Strategy description")
    parameters: List[ParameterDefinition] = Field(
        default_factory=list,
        description="Strategy parameters"
    )
    rules: List[StrategyRule] = Field(
        default_factory=list,
        description="Strategy rules"
    )
    required_indicators: List[str] = Field(
        default_factory=list,
        description="Required technical indicators"
    )
    required_features: List[str] = Field(
        default_factory=list,
        description="Required features"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Strategy tags"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class StrategyDefinition(BaseModel):
    """Complete strategy definition."""

    strategy_id: str = Field(..., description="Strategy identifier")
    name: str = Field(..., description="Strategy name")
    strategy_type: StrategyType = Field(..., description="Strategy type")
    description: str = Field(..., description="Strategy description")
    parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Strategy parameters"
    )
    rules: List[StrategyRule] = Field(
        default_factory=list,
        description="Strategy rules"
    )
    indicators: List[str] = Field(
        default_factory=list,
        description="Indicators used"
    )
    features: List[str] = Field(
        default_factory=list,
        description="Features used"
    )
    created_at: datetime = Field(
        default_factory=datetime.utcnow,
        description="Creation timestamp"
    )
    template_id: Optional[str] = Field(
        None,
        description="Template ID if created from template"
    )
    generation_method: str = Field(
        ...,
        description="How strategy was generated"
    )
    tags: List[str] = Field(
        default_factory=list,
        description="Strategy tags"
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class StrategyGeneratorConfig(BaseModel):
    """Configuration for strategy generator."""

    # ML settings
    enable_ml_generation: bool = Field(
        default=False,
        description="Enable ML-based strategy generation"
    )
    openai_api_key: Optional[str] = Field(
        None,
        description="OpenAI API key for GPT-4"
    )
    model_name: str = Field(
        default="gpt-4",
        description="OpenAI model name"
    )
    max_tokens: int = Field(
        default=2000,
        ge=100,
        description="Maximum tokens for generation"
    )
    temperature: float = Field(
        default=0.7,
        ge=0,
        le=2,
        description="Generation temperature"
    )

    # Template settings
    template_dir: Optional[str] = Field(
        None,
        description="Directory containing strategy templates"
    )

    # Parameter generation
    enable_random_parameters: bool = Field(
        default=True,
        description="Enable random parameter generation"
    )
    parameter_seed: Optional[int] = Field(
        None,
        description="Random seed for parameter generation"
    )


class StrategyGenerator:
    """
    Generate trading strategies using templates, rules, and ML.

    Supports multiple generation methods including template-based,
    rule-based, and ML-based strategy generation using GPT-4.
    """

    def __init__(self, config: Optional[StrategyGeneratorConfig] = None):
        """
        Initialize strategy generator.

        Args:
            config: Generator configuration
        """
        self.config = config or StrategyGeneratorConfig()
        self.logger = get_logger(f"{__name__}.StrategyGenerator")

        # Template registry
        self._templates: Dict[str, StrategyTemplate] = {}

        # Generated strategies
        self._strategies: Dict[str, StrategyDefinition] = {}

        # Load built-in templates
        self._load_builtin_templates()

        # Statistics
        self._stats = {
            'total_generated': 0,
            'template_based': 0,
            'rule_based': 0,
            'ml_based': 0,
            'errors': 0,
        }

    def _load_builtin_templates(self) -> None:
        """Load built-in strategy templates."""
        # Trend following template
        trend_following = StrategyTemplate(
            template_id="trend_following_basic",
            name="Basic Trend Following",
            strategy_type=StrategyType.TREND_FOLLOWING,
            description="Follow trends using moving averages and momentum",
            parameters=[
                ParameterDefinition(
                    name="fast_period",
                    type=ParameterType.INTEGER,
                    default=20,
                    min_value=5,
                    max_value=50,
                    description="Fast moving average period"
                ),
                ParameterDefinition(
                    name="slow_period",
                    type=ParameterType.INTEGER,
                    default=50,
                    min_value=20,
                    max_value=200,
                    description="Slow moving average period"
                ),
                ParameterDefinition(
                    name="momentum_threshold",
                    type=ParameterType.FLOAT,
                    default=0.02,
                    min_value=0.001,
                    max_value=0.1,
                    description="Minimum momentum for signal"
                ),
            ],
            rules=[
                StrategyRule(
                    rule_id="trend_buy",
                    condition="fast_ma > slow_ma AND momentum > momentum_threshold",
                    action=SignalType.BUY,
                    priority=1
                ),
                StrategyRule(
                    rule_id="trend_sell",
                    condition="fast_ma < slow_ma OR momentum < -momentum_threshold",
                    action=SignalType.SELL,
                    priority=1
                ),
            ],
            required_indicators=["sma", "momentum"],
            tags=["trend", "momentum", "moving_average"]
        )

        # Mean reversion template
        mean_reversion = StrategyTemplate(
            template_id="mean_reversion_basic",
            name="Basic Mean Reversion",
            strategy_type=StrategyType.MEAN_REVERSION,
            description="Trade reversions to the mean using Bollinger Bands",
            parameters=[
                ParameterDefinition(
                    name="period",
                    type=ParameterType.INTEGER,
                    default=20,
                    min_value=10,
                    max_value=100,
                    description="Bollinger Band period"
                ),
                ParameterDefinition(
                    name="std_dev",
                    type=ParameterType.FLOAT,
                    default=2.0,
                    min_value=1.0,
                    max_value=3.0,
                    description="Standard deviations"
                ),
                ParameterDefinition(
                    name="rsi_oversold",
                    type=ParameterType.INTEGER,
                    default=30,
                    min_value=10,
                    max_value=40,
                    description="RSI oversold level"
                ),
                ParameterDefinition(
                    name="rsi_overbought",
                    type=ParameterType.INTEGER,
                    default=70,
                    min_value=60,
                    max_value=90,
                    description="RSI overbought level"
                ),
            ],
            rules=[
                StrategyRule(
                    rule_id="mean_reversion_buy",
                    condition="price < bb_lower AND rsi < rsi_oversold",
                    action=SignalType.BUY,
                    priority=1
                ),
                StrategyRule(
                    rule_id="mean_reversion_sell",
                    condition="price > bb_upper AND rsi > rsi_overbought",
                    action=SignalType.SELL,
                    priority=1
                ),
            ],
            required_indicators=["bollinger_bands", "rsi"],
            tags=["mean_reversion", "bollinger", "rsi"]
        )

        # Momentum template
        momentum = StrategyTemplate(
            template_id="momentum_basic",
            name="Basic Momentum",
            strategy_type=StrategyType.MOMENTUM,
            description="Trade on price momentum and volume",
            parameters=[
                ParameterDefinition(
                    name="lookback_period",
                    type=ParameterType.INTEGER,
                    default=14,
                    min_value=5,
                    max_value=50,
                    description="Momentum lookback period"
                ),
                ParameterDefinition(
                    name="volume_threshold",
                    type=ParameterType.FLOAT,
                    default=1.5,
                    min_value=1.0,
                    max_value=3.0,
                    description="Volume multiplier threshold"
                ),
                ParameterDefinition(
                    name="min_momentum",
                    type=ParameterType.FLOAT,
                    default=0.05,
                    min_value=0.01,
                    max_value=0.2,
                    description="Minimum momentum percentage"
                ),
            ],
            rules=[
                StrategyRule(
                    rule_id="momentum_buy",
                    condition="momentum > min_momentum AND volume > avg_volume * volume_threshold",
                    action=SignalType.BUY,
                    priority=1
                ),
                StrategyRule(
                    rule_id="momentum_sell",
                    condition="momentum < -min_momentum",
                    action=SignalType.SELL,
                    priority=1
                ),
            ],
            required_indicators=["momentum", "volume"],
            tags=["momentum", "volume"]
        )

        # Register templates
        self._templates[trend_following.template_id] = trend_following
        self._templates[mean_reversion.template_id] = mean_reversion
        self._templates[momentum.template_id] = momentum

        self.logger.info(
            "Built-in templates loaded",
            template_count=len(self._templates)
        )

    @log_async_execution_time()
    async def generate_from_template(
        self,
        template_id: str,
        name: Optional[str] = None,
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[StrategyDefinition]:
        """
        Generate strategy from template.

        Args:
            template_id: Template identifier
            name: Optional strategy name
            parameters: Optional parameter overrides
            **kwargs: Additional metadata

        Returns:
            Generated strategy definition or None
        """
        try:
            if template_id not in self._templates:
                self.logger.error(
                    "Template not found",
                    template_id=template_id
                )
                return None

            template = self._templates[template_id]

            # Generate parameters
            strategy_params = await self._generate_parameters(
                template,
                parameters
            )

            # Create strategy ID
            strategy_id = f"{template.strategy_type.value}_{datetime.utcnow().timestamp()}"

            # Create strategy definition
            strategy = StrategyDefinition(
                strategy_id=strategy_id,
                name=name or f"{template.name} - {datetime.utcnow().strftime('%Y%m%d_%H%M%S')}",
                strategy_type=template.strategy_type,
                description=template.description,
                parameters=strategy_params,
                rules=template.rules.copy(),
                indicators=template.required_indicators.copy(),
                features=template.required_features.copy(),
                template_id=template_id,
                generation_method="template",
                tags=template.tags.copy(),
                metadata=kwargs
            )

            # Store strategy
            self._strategies[strategy_id] = strategy

            self._stats['total_generated'] += 1
            self._stats['template_based'] += 1

            self.logger.info(
                "Strategy generated from template",
                strategy_id=strategy_id,
                template_id=template_id,
                strategy_type=template.strategy_type.value
            )

            return strategy

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error generating strategy from template",
                template_id=template_id,
                error=str(e),
                exc_info=True
            )
            return None

    async def _generate_parameters(
        self,
        template: StrategyTemplate,
        overrides: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """Generate strategy parameters from template."""
        import random

        if self.config.parameter_seed is not None:
            random.seed(self.config.parameter_seed)

        parameters = {}
        overrides = overrides or {}

        for param_def in template.parameters:
            # Use override if provided
            if param_def.name in overrides:
                parameters[param_def.name] = overrides[param_def.name]
                continue

            # Use default if not searchable or random disabled
            if not self.config.enable_random_parameters or not param_def.searchable:
                parameters[param_def.name] = param_def.default
                continue

            # Generate random value based on type
            if param_def.type == ParameterType.INTEGER:
                min_val = param_def.min_value if param_def.min_value is not None else 0
                max_val = param_def.max_value if param_def.max_value is not None else 100
                parameters[param_def.name] = random.randint(min_val, max_val)

            elif param_def.type == ParameterType.FLOAT:
                min_val = param_def.min_value if param_def.min_value is not None else 0.0
                max_val = param_def.max_value if param_def.max_value is not None else 1.0
                parameters[param_def.name] = random.uniform(min_val, max_val)

            elif param_def.type == ParameterType.BOOLEAN:
                parameters[param_def.name] = random.choice([True, False])

            elif param_def.type == ParameterType.ENUM and param_def.choices:
                parameters[param_def.name] = random.choice(param_def.choices)

            else:
                parameters[param_def.name] = param_def.default

        return parameters

    @log_async_execution_time()
    async def generate_rule_based(
        self,
        name: str,
        strategy_type: StrategyType,
        rules: List[StrategyRule],
        indicators: List[str],
        parameters: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[StrategyDefinition]:
        """
        Generate rule-based strategy.

        Args:
            name: Strategy name
            strategy_type: Strategy type
            rules: Trading rules
            indicators: Required indicators
            parameters: Strategy parameters
            **kwargs: Additional metadata

        Returns:
            Generated strategy definition or None
        """
        try:
            strategy_id = f"rule_based_{datetime.utcnow().timestamp()}"

            strategy = StrategyDefinition(
                strategy_id=strategy_id,
                name=name,
                strategy_type=strategy_type,
                description=f"Rule-based {strategy_type.value} strategy",
                parameters=parameters or {},
                rules=rules,
                indicators=indicators,
                generation_method="rule_based",
                metadata=kwargs
            )

            self._strategies[strategy_id] = strategy

            self._stats['total_generated'] += 1
            self._stats['rule_based'] += 1

            self.logger.info(
                "Rule-based strategy generated",
                strategy_id=strategy_id,
                rule_count=len(rules)
            )

            return strategy

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error generating rule-based strategy",
                error=str(e),
                exc_info=True
            )
            return None

    @log_async_execution_time()
    async def generate_ml_based(
        self,
        prompt: str,
        strategy_type: Optional[StrategyType] = None,
        context: Optional[Dict[str, Any]] = None,
        **kwargs
    ) -> Optional[StrategyDefinition]:
        """
        Generate strategy using GPT-4.

        Args:
            prompt: Generation prompt
            strategy_type: Optional strategy type
            context: Optional context information
            **kwargs: Additional metadata

        Returns:
            Generated strategy definition or None
        """
        if not self.config.enable_ml_generation:
            self.logger.warning("ML-based generation is disabled")
            return None

        if not self.config.openai_api_key:
            self.logger.error("OpenAI API key not configured")
            return None

        try:
            import openai

            openai.api_key = self.config.openai_api_key

            # Construct system prompt
            system_prompt = """You are an expert quantitative trading strategy designer.
Generate a complete trading strategy in JSON format with the following structure:
{
    "name": "Strategy name",
    "strategy_type": "trend_following|mean_reversion|momentum|etc",
    "description": "Strategy description",
    "parameters": {"param_name": value},
    "rules": [
        {
            "rule_id": "unique_id",
            "condition": "condition expression",
            "action": "BUY|SELL|HOLD",
            "priority": 1
        }
    ],
    "indicators": ["indicator1", "indicator2"],
    "features": ["feature1", "feature2"]
}
"""

            # Add context to prompt
            full_prompt = prompt
            if context:
                full_prompt = f"Context: {json.dumps(context)}\n\n{prompt}"

            # Call GPT-4
            response = await asyncio.to_thread(
                openai.ChatCompletion.create,
                model=self.config.model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": full_prompt}
                ],
                max_tokens=self.config.max_tokens,
                temperature=self.config.temperature
            )

            # Parse response
            content = response.choices[0].message.content.strip()

            # Extract JSON (handle markdown code blocks)
            if "```json" in content:
                content = content.split("```json")[1].split("```")[0].strip()
            elif "```" in content:
                content = content.split("```")[1].split("```")[0].strip()

            strategy_data = json.loads(content)

            # Create strategy definition
            strategy_id = f"ml_based_{datetime.utcnow().timestamp()}"

            # Convert rules
            rules = []
            for rule_data in strategy_data.get("rules", []):
                rules.append(StrategyRule(**rule_data))

            # Determine strategy type
            if strategy_type:
                final_type = strategy_type
            else:
                type_str = strategy_data.get("strategy_type", "custom")
                try:
                    final_type = StrategyType(type_str)
                except ValueError:
                    final_type = StrategyType.CUSTOM

            strategy = StrategyDefinition(
                strategy_id=strategy_id,
                name=strategy_data.get("name", "ML Generated Strategy"),
                strategy_type=final_type,
                description=strategy_data.get("description", ""),
                parameters=strategy_data.get("parameters", {}),
                rules=rules,
                indicators=strategy_data.get("indicators", []),
                features=strategy_data.get("features", []),
                generation_method="ml_based",
                metadata={
                    **kwargs,
                    "prompt": prompt,
                    "model": self.config.model_name,
                    "temperature": self.config.temperature,
                }
            )

            self._strategies[strategy_id] = strategy

            self._stats['total_generated'] += 1
            self._stats['ml_based'] += 1

            self.logger.info(
                "ML-based strategy generated",
                strategy_id=strategy_id,
                model=self.config.model_name
            )

            return strategy

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error generating ML-based strategy",
                error=str(e),
                exc_info=True
            )
            return None

    def register_template(self, template: StrategyTemplate) -> None:
        """Register a strategy template."""
        self._templates[template.template_id] = template
        self.logger.info(
            "Template registered",
            template_id=template.template_id,
            name=template.name
        )

    def get_template(self, template_id: str) -> Optional[StrategyTemplate]:
        """Get a strategy template."""
        return self._templates.get(template_id)

    def list_templates(
        self,
        strategy_type: Optional[StrategyType] = None
    ) -> List[StrategyTemplate]:
        """List available templates."""
        templates = list(self._templates.values())

        if strategy_type:
            templates = [
                t for t in templates
                if t.strategy_type == strategy_type
            ]

        return templates

    def get_strategy(self, strategy_id: str) -> Optional[StrategyDefinition]:
        """Get a generated strategy."""
        return self._strategies.get(strategy_id)

    def list_strategies(
        self,
        strategy_type: Optional[StrategyType] = None
    ) -> List[StrategyDefinition]:
        """List generated strategies."""
        strategies = list(self._strategies.values())

        if strategy_type:
            strategies = [
                s for s in strategies
                if s.strategy_type == strategy_type
            ]

        return strategies

    def delete_strategy(self, strategy_id: str) -> bool:
        """Delete a generated strategy."""
        if strategy_id in self._strategies:
            del self._strategies[strategy_id]
            self.logger.info("Strategy deleted", strategy_id=strategy_id)
            return True
        return False

    def get_statistics(self) -> Dict[str, Any]:
        """Get generator statistics."""
        return {
            **self._stats,
            'templates_available': len(self._templates),
            'strategies_generated': len(self._strategies),
            'success_rate': (
                self._stats['total_generated'] /
                (self._stats['total_generated'] + self._stats['errors'])
                if (self._stats['total_generated'] + self._stats['errors']) > 0 else 0
            ),
        }

    def reset_statistics(self) -> None:
        """Reset generator statistics."""
        self._stats = {
            'total_generated': 0,
            'template_based': 0,
            'rule_based': 0,
            'ml_based': 0,
            'errors': 0,
        }
