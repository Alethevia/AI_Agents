"""
Model configuration for ML/DL models.
"""

from typing import List, Optional, Dict, Any
from enum import Enum

from pydantic import BaseModel, Field


class ModelType(str, Enum):
    """Model types."""
    TRANSFORMER = "transformer"
    SAC = "sac"
    PPO = "ppo"
    LSTM = "lstm"
    GRU = "gru"
    ENSEMBLE = "ensemble"


class OptimizerType(str, Enum):
    """Optimizer types."""
    ADAM = "adam"
    ADAMW = "adamw"
    SGDM = "sgdm"
    RMSPROP = "rmsprop"


class SchedulerType(str, Enum):
    """Learning rate scheduler types."""
    COSINE = "cosine"
    STEP = "step"
    EXPONENTIAL = "exponential"
    REDUCE_ON_PLATEAU = "reduce_on_plateau"


class TransformerConfig(BaseModel):
    """Transformer model configuration."""

    # Architecture
    d_model: int = Field(
        default=256,
        description="Model dimension",
        ge=64,
        le=2048
    )
    nhead: int = Field(
        default=8,
        description="Number of attention heads",
        ge=1,
        le=32
    )
    num_encoder_layers: int = Field(
        default=6,
        description="Number of encoder layers",
        ge=1,
        le=24
    )
    num_decoder_layers: int = Field(
        default=6,
        description="Number of decoder layers",
        ge=1,
        le=24
    )
    dim_feedforward: int = Field(
        default=1024,
        description="Feedforward dimension",
        ge=128,
        le=4096
    )
    dropout: float = Field(
        default=0.1,
        description="Dropout rate",
        ge=0.0,
        le=0.9
    )

    # Input/Output
    input_size: int = Field(
        default=50,
        description="Input feature dimension",
        ge=1
    )
    output_size: int = Field(
        default=1,
        description="Output dimension",
        ge=1
    )
    sequence_length: int = Field(
        default=60,
        description="Input sequence length",
        ge=10
    )
    prediction_horizon: int = Field(
        default=10,
        description="Prediction horizon",
        ge=1
    )

    # Positional encoding
    use_positional_encoding: bool = Field(
        default=True,
        description="Use positional encoding"
    )
    max_position_embeddings: int = Field(
        default=512,
        description="Maximum position embeddings",
        ge=100
    )

    # Attention mechanisms
    use_cross_attention: bool = Field(
        default=True,
        description="Use cross attention"
    )
    use_self_attention: bool = Field(
        default=True,
        description="Use self attention"
    )


class SACConfig(BaseModel):
    """SAC (Soft Actor-Critic) RL configuration."""

    # Network architecture
    actor_hidden_dims: List[int] = Field(
        default=[256, 256],
        description="Actor network hidden dimensions"
    )
    critic_hidden_dims: List[int] = Field(
        default=[256, 256],
        description="Critic network hidden dimensions"
    )

    # Learning rates
    actor_lr: float = Field(
        default=3e-4,
        description="Actor learning rate",
        gt=0,
        le=1.0
    )
    critic_lr: float = Field(
        default=3e-4,
        description="Critic learning rate",
        gt=0,
        le=1.0
    )
    alpha_lr: float = Field(
        default=3e-4,
        description="Temperature parameter learning rate",
        gt=0,
        le=1.0
    )

    # SAC hyperparameters
    gamma: float = Field(
        default=0.99,
        description="Discount factor",
        ge=0.0,
        le=1.0
    )
    tau: float = Field(
        default=0.005,
        description="Target network update rate",
        gt=0,
        le=1.0
    )
    alpha: float = Field(
        default=0.2,
        description="Temperature parameter",
        gt=0
    )
    auto_tune_alpha: bool = Field(
        default=True,
        description="Automatically tune temperature"
    )
    target_entropy: Optional[float] = Field(
        default=None,
        description="Target entropy for alpha tuning"
    )

    # Replay buffer
    buffer_size: int = Field(
        default=1_000_000,
        description="Replay buffer size",
        ge=10000
    )
    batch_size: int = Field(
        default=256,
        description="Batch size",
        ge=32,
        le=4096
    )

    # Training
    gradient_steps: int = Field(
        default=1,
        description="Gradient steps per environment step",
        ge=1
    )
    learning_starts: int = Field(
        default=10000,
        description="Steps before learning starts",
        ge=100
    )

    # Action space
    action_dim: int = Field(
        default=3,
        description="Action space dimension",
        ge=1
    )
    state_dim: int = Field(
        default=50,
        description="State space dimension",
        ge=1
    )


class ModelConfig(BaseModel):
    """General model configuration."""

    # Model selection
    model_type: ModelType = Field(
        default=ModelType.TRANSFORMER,
        description="Primary model type"
    )

    # Training
    batch_size: int = Field(
        default=256,
        description="Training batch size",
        ge=1
    )
    epochs: int = Field(
        default=100,
        description="Training epochs",
        ge=1
    )
    learning_rate: float = Field(
        default=1e-3,
        description="Learning rate",
        gt=0,
        le=1.0
    )
    weight_decay: float = Field(
        default=1e-5,
        description="Weight decay",
        ge=0
    )

    # Optimizer
    optimizer: OptimizerType = Field(
        default=OptimizerType.ADAMW,
        description="Optimizer type"
    )
    optimizer_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional optimizer parameters"
    )

    # Scheduler
    use_scheduler: bool = Field(
        default=True,
        description="Use learning rate scheduler"
    )
    scheduler: SchedulerType = Field(
        default=SchedulerType.COSINE,
        description="Scheduler type"
    )
    scheduler_params: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional scheduler parameters"
    )

    # Regularization
    dropout: float = Field(
        default=0.1,
        description="Dropout rate",
        ge=0.0,
        le=0.9
    )
    gradient_clip_norm: Optional[float] = Field(
        default=1.0,
        description="Gradient clipping norm"
    )
    early_stopping_patience: int = Field(
        default=10,
        description="Early stopping patience",
        ge=1
    )

    # Validation
    validation_split: float = Field(
        default=0.2,
        description="Validation set split",
        gt=0,
        lt=1.0
    )
    test_split: float = Field(
        default=0.1,
        description="Test set split",
        gt=0,
        lt=1.0
    )

    # Checkpointing
    save_checkpoints: bool = Field(
        default=True,
        description="Save model checkpoints"
    )
    checkpoint_frequency: int = Field(
        default=5,
        description="Checkpoint save frequency (epochs)",
        ge=1
    )
    keep_n_checkpoints: int = Field(
        default=5,
        description="Number of checkpoints to keep",
        ge=1
    )

    # Device
    device: str = Field(
        default="cuda",
        description="Training device (cuda/cpu)"
    )
    mixed_precision: bool = Field(
        default=True,
        description="Use mixed precision training"
    )
    compile_model: bool = Field(
        default=True,
        description="Compile model (PyTorch 2.0+)"
    )

    # Data loading
    num_workers: int = Field(
        default=4,
        description="DataLoader workers",
        ge=0
    )
    prefetch_factor: int = Field(
        default=2,
        description="DataLoader prefetch factor",
        ge=1
    )
    pin_memory: bool = Field(
        default=True,
        description="Pin memory for faster data transfer"
    )

    # Model-specific configs
    transformer_config: Optional[TransformerConfig] = Field(
        default=None,
        description="Transformer-specific configuration"
    )
    sac_config: Optional[SACConfig] = Field(
        default=None,
        description="SAC-specific configuration"
    )

    # Ensemble
    use_ensemble: bool = Field(
        default=True,
        description="Use ensemble of models"
    )
    ensemble_size: int = Field(
        default=5,
        description="Number of models in ensemble",
        ge=1
    )
    ensemble_method: str = Field(
        default="weighted_average",
        description="Ensemble aggregation method"
    )

    # Feature engineering
    use_technical_indicators: bool = Field(
        default=True,
        description="Use technical indicators"
    )
    use_sentiment_features: bool = Field(
        default=True,
        description="Use sentiment features"
    )
    use_fundamental_features: bool = Field(
        default=True,
        description="Use fundamental features"
    )

    # Update settings
    online_learning: bool = Field(
        default=True,
        description="Enable online learning"
    )
    update_frequency_minutes: int = Field(
        default=60,
        description="Model update frequency in minutes",
        ge=1
    )
    retrain_threshold: float = Field(
        default=0.1,
        description="Performance degradation threshold for retraining",
        ge=0
    )

    class Config:
        use_enum_values = True
