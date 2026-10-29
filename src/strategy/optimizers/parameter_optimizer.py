"""
Parameter optimizer module for optimizing strategy parameters.

Provides multiple optimization methods including grid search, random search,
Bayesian optimization using Optuna, genetic algorithms, and walk-forward optimization.
"""

import asyncio
import random
from datetime import datetime, timedelta
from decimal import Decimal
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
from pydantic import BaseModel, Field

from src.core.logging.logger import get_logger, log_async_execution_time


logger = get_logger(__name__)


class OptimizationMethod(str, Enum):
    """Optimization methods."""

    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN = "bayesian"
    GENETIC = "genetic"
    WALK_FORWARD = "walk_forward"
    PARTICLE_SWARM = "particle_swarm"


class ParameterSpace(BaseModel):
    """Parameter search space definition."""

    name: str = Field(..., description="Parameter name")
    type: str = Field(..., description="Parameter type (int/float/categorical)")
    min_value: Optional[float] = Field(None, description="Minimum value")
    max_value: Optional[float] = Field(None, description="Maximum value")
    step: Optional[float] = Field(None, description="Step size for grid search")
    values: Optional[List[Any]] = Field(None, description="Discrete values")
    log_scale: bool = Field(default=False, description="Use log scale")


class OptimizationResult(BaseModel):
    """Optimization result."""

    best_parameters: Dict[str, Any] = Field(..., description="Best parameters found")
    best_score: float = Field(..., description="Best score achieved")
    all_results: List[Dict[str, Any]] = Field(
        default_factory=list,
        description="All evaluation results"
    )
    optimization_method: OptimizationMethod = Field(
        ...,
        description="Optimization method used"
    )
    iterations: int = Field(..., description="Number of iterations")
    duration_seconds: float = Field(..., description="Optimization duration")
    metadata: Dict[str, Any] = Field(
        default_factory=dict,
        description="Additional metadata"
    )


class ParameterOptimizerConfig(BaseModel):
    """Configuration for parameter optimizer."""

    # General settings
    optimization_method: OptimizationMethod = Field(
        default=OptimizationMethod.BAYESIAN,
        description="Default optimization method"
    )
    maximize: bool = Field(
        default=True,
        description="Whether to maximize (True) or minimize (False) objective"
    )
    n_iterations: int = Field(
        default=100,
        ge=1,
        description="Number of optimization iterations"
    )
    n_jobs: int = Field(
        default=1,
        ge=-1,
        description="Number of parallel jobs (-1 for all CPUs)"
    )
    random_state: Optional[int] = Field(
        None,
        description="Random seed for reproducibility"
    )

    # Grid search
    grid_search_exhaustive: bool = Field(
        default=False,
        description="Exhaustive grid search vs smart grid"
    )

    # Random search
    random_search_n_samples: int = Field(
        default=100,
        ge=1,
        description="Number of random samples"
    )

    # Bayesian optimization (Optuna)
    bayesian_n_trials: int = Field(
        default=100,
        ge=1,
        description="Number of Bayesian optimization trials"
    )
    bayesian_sampler: str = Field(
        default="TPE",
        description="Sampler type (TPE/CmaEs/Random)"
    )

    # Genetic algorithm
    ga_population_size: int = Field(
        default=50,
        ge=10,
        description="GA population size"
    )
    ga_n_generations: int = Field(
        default=50,
        ge=1,
        description="GA number of generations"
    )
    ga_mutation_rate: float = Field(
        default=0.1,
        ge=0,
        le=1,
        description="GA mutation rate"
    )
    ga_crossover_rate: float = Field(
        default=0.8,
        ge=0,
        le=1,
        description="GA crossover rate"
    )
    ga_elite_size: int = Field(
        default=5,
        ge=0,
        description="GA elite size"
    )

    # Walk-forward optimization
    wf_train_size: int = Field(
        default=252,
        ge=1,
        description="Training window size in days"
    )
    wf_test_size: int = Field(
        default=63,
        ge=1,
        description="Testing window size in days"
    )
    wf_step_size: int = Field(
        default=21,
        ge=1,
        description="Step size between windows in days"
    )


class ParameterOptimizer:
    """
    Optimize strategy parameters using various methods.

    Supports grid search, random search, Bayesian optimization (Optuna),
    genetic algorithms, and walk-forward optimization.
    """

    def __init__(self, config: Optional[ParameterOptimizerConfig] = None):
        """
        Initialize parameter optimizer.

        Args:
            config: Optimizer configuration
        """
        self.config = config or ParameterOptimizerConfig()
        self.logger = get_logger(f"{__name__}.ParameterOptimizer")

        # Set random seed
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)

        # Optimization history
        self._history: List[OptimizationResult] = []

        # Statistics
        self._stats = {
            'total_optimizations': 0,
            'total_evaluations': 0,
            'best_score_overall': float('-inf') if self.config.maximize else float('inf'),
            'errors': 0,
        }

    @log_async_execution_time()
    async def optimize(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        method: Optional[OptimizationMethod] = None,
        **kwargs
    ) -> OptimizationResult:
        """
        Optimize parameters using specified method.

        Args:
            objective_func: Objective function to optimize (async callable)
            parameter_space: Parameter search space
            method: Optimization method (uses config default if None)
            **kwargs: Method-specific kwargs

        Returns:
            Optimization result
        """
        method = method or self.config.optimization_method
        start_time = datetime.utcnow()

        self.logger.info(
            "Starting parameter optimization",
            method=method.value,
            n_parameters=len(parameter_space)
        )

        try:
            if method == OptimizationMethod.GRID_SEARCH:
                result = await self._grid_search(objective_func, parameter_space, **kwargs)
            elif method == OptimizationMethod.RANDOM_SEARCH:
                result = await self._random_search(objective_func, parameter_space, **kwargs)
            elif method == OptimizationMethod.BAYESIAN:
                result = await self._bayesian_optimization(objective_func, parameter_space, **kwargs)
            elif method == OptimizationMethod.GENETIC:
                result = await self._genetic_algorithm(objective_func, parameter_space, **kwargs)
            elif method == OptimizationMethod.WALK_FORWARD:
                result = await self._walk_forward_optimization(objective_func, parameter_space, **kwargs)
            else:
                raise ValueError(f"Unknown optimization method: {method}")

            # Calculate duration
            duration = (datetime.utcnow() - start_time).total_seconds()
            result.duration_seconds = duration

            # Update statistics
            self._stats['total_optimizations'] += 1
            self._stats['total_evaluations'] += result.iterations

            if self.config.maximize:
                if result.best_score > self._stats['best_score_overall']:
                    self._stats['best_score_overall'] = result.best_score
            else:
                if result.best_score < self._stats['best_score_overall']:
                    self._stats['best_score_overall'] = result.best_score

            # Store in history
            self._history.append(result)

            self.logger.info(
                "Optimization completed",
                method=method.value,
                best_score=result.best_score,
                iterations=result.iterations,
                duration=duration
            )

            return result

        except Exception as e:
            self._stats['errors'] += 1
            self.logger.error(
                "Error during optimization",
                method=method.value,
                error=str(e),
                exc_info=True
            )
            raise

    async def _grid_search(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        **kwargs
    ) -> OptimizationResult:
        """Perform grid search optimization."""
        self.logger.info("Starting grid search")

        # Generate parameter grid
        grid = self._generate_parameter_grid(parameter_space)

        self.logger.info(
            "Grid search parameter combinations",
            total_combinations=len(grid)
        )

        # Evaluate all combinations
        results = []
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None

        for i, params in enumerate(grid):
            try:
                score = await objective_func(**params)

                results.append({
                    'parameters': params,
                    'score': score,
                    'iteration': i
                })

                # Update best
                if self.config.maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

                if (i + 1) % 10 == 0:
                    self.logger.debug(
                        "Grid search progress",
                        completed=i + 1,
                        total=len(grid),
                        best_score=best_score
                    )

            except Exception as e:
                self.logger.error(
                    "Error evaluating parameters",
                    params=params,
                    error=str(e)
                )
                continue

        return OptimizationResult(
            best_parameters=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_method=OptimizationMethod.GRID_SEARCH,
            iterations=len(results)
        )

    def _generate_parameter_grid(
        self,
        parameter_space: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """Generate parameter grid for grid search."""
        from itertools import product

        # Generate value lists for each parameter
        param_lists = {}

        for param in parameter_space:
            if param.values is not None:
                # Use provided discrete values
                param_lists[param.name] = param.values
            elif param.type == "categorical":
                raise ValueError(f"Categorical parameter {param.name} must have values")
            else:
                # Generate range
                if param.step is None:
                    # Use 10 points by default
                    param.step = (param.max_value - param.min_value) / 10

                if param.log_scale:
                    values = np.logspace(
                        np.log10(param.min_value),
                        np.log10(param.max_value),
                        num=int((np.log10(param.max_value) - np.log10(param.min_value)) / param.step) + 1
                    )
                else:
                    values = np.arange(
                        param.min_value,
                        param.max_value + param.step,
                        param.step
                    )

                if param.type == "int":
                    values = np.unique(values.astype(int))

                param_lists[param.name] = values.tolist()

        # Generate all combinations
        keys = list(param_lists.keys())
        value_combinations = list(product(*[param_lists[k] for k in keys]))

        grid = [
            dict(zip(keys, combo))
            for combo in value_combinations
        ]

        return grid

    async def _random_search(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        **kwargs
    ) -> OptimizationResult:
        """Perform random search optimization."""
        n_samples = kwargs.get('n_samples', self.config.random_search_n_samples)

        self.logger.info(
            "Starting random search",
            n_samples=n_samples
        )

        results = []
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None

        for i in range(n_samples):
            # Sample random parameters
            params = self._sample_parameters(parameter_space)

            try:
                score = await objective_func(**params)

                results.append({
                    'parameters': params,
                    'score': score,
                    'iteration': i
                })

                # Update best
                if self.config.maximize:
                    if score > best_score:
                        best_score = score
                        best_params = params
                else:
                    if score < best_score:
                        best_score = score
                        best_params = params

                if (i + 1) % 10 == 0:
                    self.logger.debug(
                        "Random search progress",
                        completed=i + 1,
                        total=n_samples,
                        best_score=best_score
                    )

            except Exception as e:
                self.logger.error(
                    "Error evaluating parameters",
                    params=params,
                    error=str(e)
                )
                continue

        return OptimizationResult(
            best_parameters=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_method=OptimizationMethod.RANDOM_SEARCH,
            iterations=len(results)
        )

    def _sample_parameters(
        self,
        parameter_space: List[ParameterSpace]
    ) -> Dict[str, Any]:
        """Sample random parameters from space."""
        params = {}

        for param in parameter_space:
            if param.values is not None:
                # Sample from discrete values
                params[param.name] = random.choice(param.values)
            elif param.type == "categorical":
                raise ValueError(f"Categorical parameter {param.name} must have values")
            elif param.log_scale:
                # Sample on log scale
                log_min = np.log10(param.min_value)
                log_max = np.log10(param.max_value)
                log_val = random.uniform(log_min, log_max)
                val = 10 ** log_val

                if param.type == "int":
                    params[param.name] = int(val)
                else:
                    params[param.name] = val
            else:
                # Sample uniformly
                val = random.uniform(param.min_value, param.max_value)

                if param.type == "int":
                    params[param.name] = int(val)
                else:
                    params[param.name] = val

        return params

    async def _bayesian_optimization(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        **kwargs
    ) -> OptimizationResult:
        """Perform Bayesian optimization using Optuna."""
        try:
            import optuna
            from optuna.samplers import TPESampler, CmaEsSampler, RandomSampler
        except ImportError:
            self.logger.error("Optuna not installed. Install with: pip install optuna")
            raise

        n_trials = kwargs.get('n_trials', self.config.bayesian_n_trials)

        self.logger.info(
            "Starting Bayesian optimization with Optuna",
            n_trials=n_trials,
            sampler=self.config.bayesian_sampler
        )

        # Create Optuna study
        direction = "maximize" if self.config.maximize else "minimize"

        # Select sampler
        if self.config.bayesian_sampler == "TPE":
            sampler = TPESampler(seed=self.config.random_state)
        elif self.config.bayesian_sampler == "CmaEs":
            sampler = CmaEsSampler(seed=self.config.random_state)
        else:
            sampler = RandomSampler(seed=self.config.random_state)

        study = optuna.create_study(
            direction=direction,
            sampler=sampler
        )

        # Store results
        results = []

        def optuna_objective(trial):
            """Optuna objective function."""
            # Suggest parameters
            params = {}
            for param in parameter_space:
                if param.values is not None:
                    params[param.name] = trial.suggest_categorical(
                        param.name,
                        param.values
                    )
                elif param.type == "int":
                    params[param.name] = trial.suggest_int(
                        param.name,
                        int(param.min_value),
                        int(param.max_value),
                        log=param.log_scale
                    )
                elif param.type == "float":
                    params[param.name] = trial.suggest_float(
                        param.name,
                        param.min_value,
                        param.max_value,
                        log=param.log_scale
                    )
                elif param.type == "categorical":
                    params[param.name] = trial.suggest_categorical(
                        param.name,
                        param.values
                    )

            # Evaluate
            try:
                # Run async function in sync context
                loop = asyncio.get_event_loop()
                score = loop.run_until_complete(objective_func(**params))

                results.append({
                    'parameters': params,
                    'score': score,
                    'iteration': trial.number
                })

                return score

            except Exception as e:
                self.logger.error(
                    "Error in Optuna trial",
                    trial_number=trial.number,
                    error=str(e)
                )
                raise

        # Optimize
        study.optimize(
            optuna_objective,
            n_trials=n_trials,
            n_jobs=self.config.n_jobs
        )

        return OptimizationResult(
            best_parameters=study.best_params,
            best_score=study.best_value,
            all_results=results,
            optimization_method=OptimizationMethod.BAYESIAN,
            iterations=len(study.trials),
            metadata={
                'n_trials': len(study.trials),
                'sampler': self.config.bayesian_sampler,
            }
        )

    async def _genetic_algorithm(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        **kwargs
    ) -> OptimizationResult:
        """Perform genetic algorithm optimization."""
        population_size = kwargs.get('population_size', self.config.ga_population_size)
        n_generations = kwargs.get('n_generations', self.config.ga_n_generations)

        self.logger.info(
            "Starting genetic algorithm",
            population_size=population_size,
            n_generations=n_generations
        )

        # Initialize population
        population = [
            self._sample_parameters(parameter_space)
            for _ in range(population_size)
        ]

        results = []
        best_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None

        for generation in range(n_generations):
            # Evaluate population
            scores = []
            for params in population:
                try:
                    score = await objective_func(**params)
                    scores.append(score)

                    results.append({
                        'parameters': params,
                        'score': score,
                        'iteration': generation * population_size + len(scores)
                    })

                    # Update best
                    if self.config.maximize:
                        if score > best_score:
                            best_score = score
                            best_params = params
                    else:
                        if score < best_score:
                            best_score = score
                            best_params = params

                except Exception as e:
                    self.logger.error(
                        "Error evaluating individual",
                        generation=generation,
                        error=str(e)
                    )
                    scores.append(float('-inf') if self.config.maximize else float('inf'))

            # Selection
            population = self._ga_selection(population, scores)

            # Crossover and mutation
            population = self._ga_crossover_mutation(population, parameter_space)

            if (generation + 1) % 10 == 0:
                self.logger.debug(
                    "GA generation progress",
                    generation=generation + 1,
                    total=n_generations,
                    best_score=best_score
                )

        return OptimizationResult(
            best_parameters=best_params or {},
            best_score=best_score,
            all_results=results,
            optimization_method=OptimizationMethod.GENETIC,
            iterations=len(results),
            metadata={
                'population_size': population_size,
                'n_generations': n_generations,
            }
        )

    def _ga_selection(
        self,
        population: List[Dict[str, Any]],
        scores: List[float]
    ) -> List[Dict[str, Any]]:
        """Genetic algorithm selection."""
        # Elitism - keep best individuals
        elite_size = self.config.ga_elite_size

        # Sort by score
        if self.config.maximize:
            sorted_indices = np.argsort(scores)[::-1]
        else:
            sorted_indices = np.argsort(scores)

        elite = [population[i] for i in sorted_indices[:elite_size]]

        # Tournament selection for rest
        new_population = elite.copy()

        while len(new_population) < len(population):
            # Tournament
            tournament_size = 3
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_scores = [scores[i] for i in tournament_indices]

            if self.config.maximize:
                winner_idx = tournament_indices[np.argmax(tournament_scores)]
            else:
                winner_idx = tournament_indices[np.argmin(tournament_scores)]

            new_population.append(population[winner_idx])

        return new_population

    def _ga_crossover_mutation(
        self,
        population: List[Dict[str, Any]],
        parameter_space: List[ParameterSpace]
    ) -> List[Dict[str, Any]]:
        """Genetic algorithm crossover and mutation."""
        new_population = []

        # Keep elite unchanged
        elite_size = self.config.ga_elite_size
        new_population.extend(population[:elite_size])

        # Crossover
        for i in range(elite_size, len(population), 2):
            if i + 1 < len(population) and random.random() < self.config.ga_crossover_rate:
                # Crossover
                parent1 = population[i]
                parent2 = population[i + 1]

                child1, child2 = {}, {}

                for param in parameter_space:
                    if random.random() < 0.5:
                        child1[param.name] = parent1[param.name]
                        child2[param.name] = parent2[param.name]
                    else:
                        child1[param.name] = parent2[param.name]
                        child2[param.name] = parent1[param.name]

                new_population.append(child1)
                if len(new_population) < len(population):
                    new_population.append(child2)
            else:
                new_population.append(population[i])
                if i + 1 < len(population) and len(new_population) < len(population):
                    new_population.append(population[i + 1])

        # Mutation
        for i in range(elite_size, len(new_population)):
            if random.random() < self.config.ga_mutation_rate:
                # Mutate random parameter
                param = random.choice(parameter_space)
                new_population[i][param.name] = self._sample_parameters([param])[param.name]

        return new_population

    async def _walk_forward_optimization(
        self,
        objective_func: Callable,
        parameter_space: List[ParameterSpace],
        **kwargs
    ) -> OptimizationResult:
        """Perform walk-forward optimization."""
        self.logger.info("Starting walk-forward optimization")

        # This requires data to be passed in kwargs
        data = kwargs.get('data')
        if data is None:
            raise ValueError("Walk-forward optimization requires 'data' in kwargs")

        train_size = kwargs.get('train_size', self.config.wf_train_size)
        test_size = kwargs.get('test_size', self.config.wf_test_size)
        step_size = kwargs.get('step_size', self.config.wf_step_size)

        results = []
        all_test_scores = []
        best_avg_score = float('-inf') if self.config.maximize else float('inf')
        best_params = None

        # Calculate number of windows
        total_size = len(data)
        n_windows = (total_size - train_size - test_size) // step_size + 1

        self.logger.info(
            "Walk-forward windows",
            n_windows=n_windows,
            train_size=train_size,
            test_size=test_size,
            step_size=step_size
        )

        # For each window
        for window in range(n_windows):
            start_idx = window * step_size
            train_end_idx = start_idx + train_size
            test_end_idx = train_end_idx + test_size

            if test_end_idx > total_size:
                break

            train_data = data[start_idx:train_end_idx]
            test_data = data[train_end_idx:test_end_idx]

            # Optimize on training data
            # Use Bayesian optimization for each window
            window_result = await self._bayesian_optimization(
                lambda **params: objective_func(data=train_data, **params),
                parameter_space,
                n_trials=20  # Fewer trials per window
            )

            # Evaluate on test data
            test_score = await objective_func(
                data=test_data,
                **window_result.best_parameters
            )

            all_test_scores.append(test_score)

            results.append({
                'window': window,
                'train_params': window_result.best_parameters,
                'train_score': window_result.best_score,
                'test_score': test_score,
            })

            self.logger.debug(
                "Walk-forward window completed",
                window=window + 1,
                total_windows=n_windows,
                test_score=test_score
            )

        # Calculate average test score
        avg_test_score = np.mean(all_test_scores)

        # Find parameters with best average test score
        # Use most recent parameters as best
        best_params = results[-1]['train_params'] if results else {}

        return OptimizationResult(
            best_parameters=best_params,
            best_score=avg_test_score,
            all_results=results,
            optimization_method=OptimizationMethod.WALK_FORWARD,
            iterations=len(results),
            metadata={
                'n_windows': len(results),
                'avg_test_score': avg_test_score,
                'test_scores': all_test_scores,
            }
        )

    def get_best_result(self) -> Optional[OptimizationResult]:
        """Get best optimization result from history."""
        if not self._history:
            return None

        if self.config.maximize:
            return max(self._history, key=lambda x: x.best_score)
        else:
            return min(self._history, key=lambda x: x.best_score)

    def get_history(
        self,
        method: Optional[OptimizationMethod] = None
    ) -> List[OptimizationResult]:
        """Get optimization history."""
        history = self._history

        if method:
            history = [r for r in history if r.optimization_method == method]

        return history

    def get_statistics(self) -> Dict[str, Any]:
        """Get optimizer statistics."""
        return {
            **self._stats,
            'optimizations_by_method': {
                method.value: sum(
                    1 for r in self._history
                    if r.optimization_method == method
                )
                for method in OptimizationMethod
            },
            'avg_iterations': (
                np.mean([r.iterations for r in self._history])
                if self._history else 0
            ),
            'avg_duration': (
                np.mean([r.duration_seconds for r in self._history])
                if self._history else 0
            ),
        }

    def reset_history(self) -> None:
        """Reset optimization history."""
        self._history.clear()
        self.logger.info("Optimization history reset")

    def reset_statistics(self) -> None:
        """Reset optimizer statistics."""
        self._stats = {
            'total_optimizations': 0,
            'total_evaluations': 0,
            'best_score_overall': float('-inf') if self.config.maximize else float('inf'),
            'errors': 0,
        }
