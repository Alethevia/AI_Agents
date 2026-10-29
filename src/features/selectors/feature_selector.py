"""
Feature Selection Module

This module provides comprehensive feature selection methods including
correlation-based, mutual information, L1 regularization, random forest,
sequential selection, and recursive feature elimination.

Author: AI Trading System
"""

import numpy as np
import pandas as pd
from typing import Optional, Union, List, Tuple, Dict
from sklearn.feature_selection import (
    SelectKBest, mutual_info_regression, mutual_info_classif,
    RFE, SequentialFeatureSelector
)
from sklearn.linear_model import Lasso, LogisticRegression
from sklearn.ensemble import RandomForestRegressor, RandomForestClassifier
from sklearn.metrics import mean_squared_error, accuracy_score
import warnings


class FeatureSelector:
    """
    Production-ready feature selection utilities.

    Supports:
    - Correlation-based selection
    - Mutual information
    - L1 regularization (Lasso)
    - Random forest importance
    - Sequential feature selection
    - Recursive feature elimination
    - Stability selection
    """

    def __init__(self, task: str = 'regression'):
        """
        Initialize feature selector.

        Args:
            task: Type of task ('regression' or 'classification')
        """
        self.task = task
        self.selected_features_ = []
        self.feature_scores_ = {}
        self.fitted_ = False

    # ==================== Correlation-Based Selection ====================

    @staticmethod
    def remove_low_variance(data: pd.DataFrame, threshold: float = 0.01) -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove features with low variance.

        Args:
            data: Input features DataFrame
            threshold: Variance threshold

        Returns:
            Tuple of (filtered data, removed features)
        """
        variances = data.var()
        low_variance = variances[variances < threshold].index.tolist()

        filtered_data = data.drop(columns=low_variance)

        return filtered_data, low_variance

    @staticmethod
    def remove_correlated_features(data: pd.DataFrame, threshold: float = 0.95,
                                  method: str = 'pearson') -> Tuple[pd.DataFrame, List[str]]:
        """
        Remove highly correlated features.

        Args:
            data: Input features DataFrame
            threshold: Correlation threshold (default: 0.95)
            method: Correlation method ('pearson', 'spearman', 'kendall')

        Returns:
            Tuple of (filtered data, removed features)
        """
        # Calculate correlation matrix
        corr_matrix = data.corr(method=method).abs()

        # Select upper triangle of correlation matrix
        upper_tri = corr_matrix.where(
            np.triu(np.ones(corr_matrix.shape), k=1).astype(bool)
        )

        # Find features with correlation greater than threshold
        to_drop = [column for column in upper_tri.columns
                  if any(upper_tri[column] > threshold)]

        filtered_data = data.drop(columns=to_drop)

        return filtered_data, to_drop

    @staticmethod
    def select_by_correlation_with_target(data: pd.DataFrame, target: pd.Series,
                                        k: int = 50, method: str = 'pearson') -> Tuple[pd.DataFrame, Dict]:
        """
        Select features based on correlation with target.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            method: Correlation method

        Returns:
            Tuple of (selected features, correlation scores)
        """
        # Calculate correlations with target
        correlations = data.corrwith(target, method=method).abs()

        # Sort and select top k
        top_features = correlations.nlargest(k).index.tolist()

        scores = correlations.to_dict()

        return data[top_features], scores

    # ==================== Mutual Information ====================

    def mutual_information_selection(self, data: pd.DataFrame, target: pd.Series,
                                    k: int = 50, random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """
        Select features using mutual information.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            random_state: Random seed

        Returns:
            Tuple of (selected features, MI scores)
        """
        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Calculate mutual information
        if self.task == 'classification':
            mi_scores = mutual_info_classif(data_clean, target, random_state=random_state)
        else:
            mi_scores = mutual_info_regression(data_clean, target, random_state=random_state)

        # Create scores dictionary
        scores = dict(zip(data.columns, mi_scores))

        # Select top k features
        top_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
        selected_features = [f[0] for f in top_features]

        return data[selected_features], scores

    # ==================== L1 Regularization ====================

    def lasso_selection(self, data: pd.DataFrame, target: pd.Series,
                       alpha: float = 0.01, k: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Select features using Lasso (L1) regularization.

        Args:
            data: Input features DataFrame
            target: Target variable
            alpha: Regularization strength (higher = more sparsity)
            k: Number of features to select (None = all non-zero)

        Returns:
            Tuple of (selected features, coefficient scores)
        """
        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Standardize features
        data_std = (data_clean - data_clean.mean()) / (data_clean.std() + 1e-10)

        # Fit Lasso
        if self.task == 'classification':
            model = LogisticRegression(penalty='l1', C=1/alpha, solver='liblinear',
                                      max_iter=1000, random_state=42)
        else:
            model = Lasso(alpha=alpha, max_iter=1000, random_state=42)

        model.fit(data_std, target)

        # Get feature coefficients
        if self.task == 'classification':
            coefficients = np.abs(model.coef_[0])
        else:
            coefficients = np.abs(model.coef_)

        # Create scores dictionary
        scores = dict(zip(data.columns, coefficients))

        # Select features with non-zero coefficients
        non_zero_features = [feat for feat, coef in scores.items() if coef > 1e-10]

        # If k specified, select top k
        if k is not None:
            sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
            selected_features = [f[0] for f in sorted_features[:k]]
        else:
            selected_features = non_zero_features

        return data[selected_features], scores

    # ==================== Random Forest Importance ====================

    def random_forest_selection(self, data: pd.DataFrame, target: pd.Series,
                               k: int = 50, n_estimators: int = 100,
                               random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """
        Select features using Random Forest feature importance.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            n_estimators: Number of trees
            random_state: Random seed

        Returns:
            Tuple of (selected features, importance scores)
        """
        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Fit Random Forest
        if self.task == 'classification':
            model = RandomForestClassifier(n_estimators=n_estimators,
                                         max_depth=10,
                                         min_samples_leaf=10,
                                         random_state=random_state,
                                         n_jobs=-1)
        else:
            model = RandomForestRegressor(n_estimators=n_estimators,
                                        max_depth=10,
                                        min_samples_leaf=10,
                                        random_state=random_state,
                                        n_jobs=-1)

        model.fit(data_clean, target)

        # Get feature importances
        importances = model.feature_importances_
        scores = dict(zip(data.columns, importances))

        # Select top k features
        sorted_features = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_features = [f[0] for f in sorted_features[:k]]

        return data[selected_features], scores

    # ==================== Sequential Feature Selection ====================

    def sequential_selection(self, data: pd.DataFrame, target: pd.Series,
                           k: int = 20, direction: str = 'forward',
                           cv: int = 5, scoring: Optional[str] = None) -> Tuple[pd.DataFrame, List[str]]:
        """
        Select features using sequential feature selection.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            direction: 'forward' or 'backward'
            cv: Number of cross-validation folds
            scoring: Scoring metric (None = default for task)

        Returns:
            Tuple of (selected features, selected feature names)
        """
        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Create estimator
        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, max_depth=10,
                                             random_state=42, n_jobs=-1)
            if scoring is None:
                scoring = 'accuracy'
        else:
            estimator = RandomForestRegressor(n_estimators=50, max_depth=10,
                                            random_state=42, n_jobs=-1)
            if scoring is None:
                scoring = 'neg_mean_squared_error'

        # Sequential Feature Selector
        sfs = SequentialFeatureSelector(
            estimator,
            n_features_to_select=k,
            direction=direction,
            scoring=scoring,
            cv=cv,
            n_jobs=-1
        )

        sfs.fit(data_clean, target)

        # Get selected features
        selected_mask = sfs.get_support()
        selected_features = data.columns[selected_mask].tolist()

        return data[selected_features], selected_features

    # ==================== Recursive Feature Elimination ====================

    def recursive_feature_elimination(self, data: pd.DataFrame, target: pd.Series,
                                     k: int = 20, step: int = 1,
                                     cv: Optional[int] = None) -> Tuple[pd.DataFrame, Dict]:
        """
        Select features using Recursive Feature Elimination.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            step: Number of features to remove at each iteration
            cv: Number of CV folds (None = no CV)

        Returns:
            Tuple of (selected features, ranking scores)
        """
        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Create estimator
        if self.task == 'classification':
            estimator = RandomForestClassifier(n_estimators=50, max_depth=10,
                                             random_state=42, n_jobs=-1)
        else:
            estimator = RandomForestRegressor(n_estimators=50, max_depth=10,
                                            random_state=42, n_jobs=-1)

        # RFE
        if cv is not None:
            from sklearn.feature_selection import RFECV
            rfe = RFECV(estimator, step=step, cv=cv, scoring=None, n_jobs=-1)
            rfe.fit(data_clean, target)
        else:
            rfe = RFE(estimator, n_features_to_select=k, step=step)
            rfe.fit(data_clean, target)

        # Get selected features and rankings
        selected_mask = rfe.support_
        rankings = rfe.ranking_

        selected_features = data.columns[selected_mask].tolist()
        scores = dict(zip(data.columns, 1 / rankings))  # Convert ranking to score

        return data[selected_features], scores

    # ==================== Stability Selection ====================

    def stability_selection(self, data: pd.DataFrame, target: pd.Series,
                          k: int = 50, n_iterations: int = 100,
                          sample_fraction: float = 0.5,
                          threshold: float = 0.6,
                          random_state: int = 42) -> Tuple[pd.DataFrame, Dict]:
        """
        Select features using stability selection.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Maximum number of features to select
            n_iterations: Number of iterations
            sample_fraction: Fraction of samples to use in each iteration
            threshold: Selection frequency threshold (0-1)
            random_state: Random seed

        Returns:
            Tuple of (selected features, selection frequency scores)
        """
        np.random.seed(random_state)

        # Handle missing values
        data_clean = data.fillna(data.mean())
        n_samples = len(data_clean)
        sample_size = int(n_samples * sample_fraction)

        # Track feature selection frequency
        selection_frequency = {col: 0 for col in data.columns}

        for i in range(n_iterations):
            # Random subsample
            sample_idx = np.random.choice(n_samples, sample_size, replace=False)
            X_sample = data_clean.iloc[sample_idx]
            y_sample = target.iloc[sample_idx]

            # Fit Lasso
            if self.task == 'classification':
                model = LogisticRegression(penalty='l1', C=0.1, solver='liblinear',
                                          max_iter=1000, random_state=i)
            else:
                model = Lasso(alpha=0.01, max_iter=1000, random_state=i)

            try:
                # Standardize
                X_std = (X_sample - X_sample.mean()) / (X_sample.std() + 1e-10)
                model.fit(X_std, y_sample)

                # Get selected features
                if self.task == 'classification':
                    coefficients = np.abs(model.coef_[0])
                else:
                    coefficients = np.abs(model.coef_)

                selected = np.where(coefficients > 1e-10)[0]

                # Update frequency
                for idx in selected:
                    feature = data.columns[idx]
                    selection_frequency[feature] += 1

            except Exception as e:
                warnings.warn(f"Iteration {i} failed: {str(e)}")
                continue

        # Convert to probability
        selection_prob = {feat: freq / n_iterations
                         for feat, freq in selection_frequency.items()}

        # Select features above threshold
        stable_features = [feat for feat, prob in selection_prob.items()
                          if prob >= threshold]

        # If too many features, select top k
        if len(stable_features) > k:
            sorted_features = sorted(selection_prob.items(),
                                   key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:k]]

        if len(stable_features) == 0:
            warnings.warn("No features passed threshold. Selecting top k by frequency.")
            sorted_features = sorted(selection_prob.items(),
                                   key=lambda x: x[1], reverse=True)
            stable_features = [f[0] for f in sorted_features[:k]]

        return data[stable_features], selection_prob

    # ==================== Permutation Importance ====================

    def permutation_importance(self, model, data: pd.DataFrame, target: pd.Series,
                             n_repeats: int = 10, random_state: int = 42) -> Dict:
        """
        Calculate permutation importance.

        Args:
            model: Fitted model
            data: Input features DataFrame
            target: Target variable
            n_repeats: Number of permutations
            random_state: Random seed

        Returns:
            Dictionary of importance scores
        """
        from sklearn.inspection import permutation_importance

        # Handle missing values
        data_clean = data.fillna(data.mean())

        # Calculate permutation importance
        result = permutation_importance(
            model, data_clean, target,
            n_repeats=n_repeats,
            random_state=random_state,
            n_jobs=-1
        )

        # Create scores dictionary
        scores = dict(zip(data.columns, result.importances_mean))

        return scores

    # ==================== Combined Selection ====================

    def combined_selection(self, data: pd.DataFrame, target: pd.Series,
                         k: int = 50, methods: Optional[List[str]] = None,
                         voting: str = 'intersection') -> Tuple[pd.DataFrame, Dict]:
        """
        Combine multiple feature selection methods.

        Args:
            data: Input features DataFrame
            target: Target variable
            k: Number of features to select
            methods: List of methods to use (None = all)
            voting: 'intersection', 'union', or 'majority'

        Returns:
            Tuple of (selected features, method scores)
        """
        if methods is None:
            methods = ['correlation', 'mutual_info', 'random_forest', 'lasso']

        # Store selected features from each method
        selected_by_method = {}
        method_scores = {}

        # Run each method
        for method in methods:
            try:
                if method == 'correlation':
                    _, scores = self.select_by_correlation_with_target(data, target, k=k)
                    selected = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:k]
                    selected_by_method[method] = [f[0] for f in selected]
                    method_scores[method] = scores

                elif method == 'mutual_info':
                    _, scores = self.mutual_information_selection(data, target, k=k)
                    selected_by_method[method] = sorted(scores.items(),
                                                       key=lambda x: x[1],
                                                       reverse=True)[:k]
                    selected_by_method[method] = [f[0] for f in selected_by_method[method]]
                    method_scores[method] = scores

                elif method == 'random_forest':
                    _, scores = self.random_forest_selection(data, target, k=k)
                    selected_by_method[method] = sorted(scores.items(),
                                                       key=lambda x: x[1],
                                                       reverse=True)[:k]
                    selected_by_method[method] = [f[0] for f in selected_by_method[method]]
                    method_scores[method] = scores

                elif method == 'lasso':
                    _, scores = self.lasso_selection(data, target, k=k)
                    selected_by_method[method] = sorted(scores.items(),
                                                       key=lambda x: x[1],
                                                       reverse=True)[:k]
                    selected_by_method[method] = [f[0] for f in selected_by_method[method]]
                    method_scores[method] = scores

            except Exception as e:
                warnings.warn(f"Method {method} failed: {str(e)}")
                continue

        # Combine results based on voting strategy
        if voting == 'intersection':
            # Select features selected by ALL methods
            selected = set(selected_by_method[methods[0]])
            for method in methods[1:]:
                if method in selected_by_method:
                    selected = selected.intersection(selected_by_method[method])
            selected = list(selected)

            # If intersection is too small, use majority voting
            if len(selected) < k // 2:
                voting = 'majority'

        if voting == 'union':
            # Select features selected by ANY method
            selected = set()
            for method in selected_by_method.values():
                selected = selected.union(method)
            selected = list(selected)[:k]

        if voting == 'majority':
            # Select features selected by majority of methods
            feature_votes = {}
            for method_features in selected_by_method.values():
                for feat in method_features:
                    feature_votes[feat] = feature_votes.get(feat, 0) + 1

            # Sort by votes and select top k
            sorted_features = sorted(feature_votes.items(),
                                   key=lambda x: x[1], reverse=True)
            selected = [f[0] for f in sorted_features[:k]]

        # Aggregate scores
        aggregated_scores = {}
        for feat in data.columns:
            scores_list = []
            for method, scores in method_scores.items():
                if feat in scores:
                    # Normalize scores to 0-1
                    max_score = max(scores.values())
                    min_score = min(scores.values())
                    if max_score > min_score:
                        normalized = (scores[feat] - min_score) / (max_score - min_score)
                        scores_list.append(normalized)

            if scores_list:
                aggregated_scores[feat] = np.mean(scores_list)
            else:
                aggregated_scores[feat] = 0

        return data[selected], aggregated_scores

    # ==================== Fit and Transform ====================

    def fit(self, data: pd.DataFrame, target: pd.Series,
           method: str = 'combined', k: int = 50, **kwargs):
        """
        Fit feature selector.

        Args:
            data: Input features DataFrame
            target: Target variable
            method: Selection method
            k: Number of features to select
            **kwargs: Additional arguments for specific methods

        Returns:
            Self
        """
        if method == 'correlation':
            _, self.feature_scores_ = self.select_by_correlation_with_target(
                data, target, k=k
            )
        elif method == 'mutual_info':
            _, self.feature_scores_ = self.mutual_information_selection(
                data, target, k=k
            )
        elif method == 'lasso':
            _, self.feature_scores_ = self.lasso_selection(
                data, target, k=k, **kwargs
            )
        elif method == 'random_forest':
            _, self.feature_scores_ = self.random_forest_selection(
                data, target, k=k, **kwargs
            )
        elif method == 'rfe':
            _, self.feature_scores_ = self.recursive_feature_elimination(
                data, target, k=k, **kwargs
            )
        elif method == 'stability':
            _, self.feature_scores_ = self.stability_selection(
                data, target, k=k, **kwargs
            )
        elif method == 'combined':
            _, self.feature_scores_ = self.combined_selection(
                data, target, k=k, **kwargs
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        # Select top k features by score
        sorted_features = sorted(self.feature_scores_.items(),
                               key=lambda x: x[1], reverse=True)
        self.selected_features_ = [f[0] for f in sorted_features[:k]
                                  if f[0] in data.columns]

        self.fitted_ = True
        return self

    def transform(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        Transform data using selected features.

        Args:
            data: Input features DataFrame

        Returns:
            DataFrame with selected features only
        """
        if not self.fitted_:
            raise ValueError("Selector must be fitted before transform")

        return data[self.selected_features_]

    def fit_transform(self, data: pd.DataFrame, target: pd.Series,
                     method: str = 'combined', k: int = 50, **kwargs) -> pd.DataFrame:
        """
        Fit and transform in one step.

        Args:
            data: Input features DataFrame
            target: Target variable
            method: Selection method
            k: Number of features to select
            **kwargs: Additional arguments

        Returns:
            Transformed DataFrame
        """
        self.fit(data, target, method=method, k=k, **kwargs)
        return self.transform(data)

    def get_feature_names(self) -> List[str]:
        """Get selected feature names."""
        return self.selected_features_

    def get_feature_scores(self) -> Dict:
        """Get feature importance scores."""
        return self.feature_scores_
