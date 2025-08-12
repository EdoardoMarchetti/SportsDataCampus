"""
Model training module for player rating prediction.

This module handles model training, validation, and hyperparameter tuning.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple, Optional
import logging
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, RandomizedSearchCV
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class PlayerRatingModel:
    """
    A class to handle player rating prediction models.
    """
    
    def __init__(self, model_type: str = 'random_forest', random_state: int = 42):
        """
        Initialize the model.
        
        Args:
            model_type (str): Type of model to use
            random_state (int): Random state for reproducibility
        """
        self.model_type = model_type
        self.random_state = random_state
        self.model = None
        self.scaler = StandardScaler()
        self.feature_names = None
        self.is_fitted = False
        
        # Initialize model based on type
        self._initialize_model()
    
    def _initialize_model(self):
        """Initialize the model based on model_type."""
        if self.model_type == 'linear':
            self.model = LinearRegression()
        elif self.model_type == 'ridge':
            self.model = Ridge(random_state=self.random_state)
        elif self.model_type == 'lasso':
            self.model = Lasso(random_state=self.random_state)
        elif self.model_type == 'random_forest':
            self.model = RandomForestRegressor(
                n_estimators=100,
                random_state=self.random_state,
                n_jobs=-1
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        logger.info(f"Initialized {self.model_type} model")
    
    def prepare_features(self, df: pd.DataFrame, target_col: str = 'rating') -> Tuple[pd.DataFrame, pd.Series]:
        """
        Prepare features and target for training.
        
        Args:
            df (pd.DataFrame): Input dataframe
            target_col (str): Name of the target column
            
        Returns:
            Tuple[pd.DataFrame, pd.Series]: Features and target
        """
        # Exclude non-feature columns
        exclude_cols = ['id', 'match_id', 'team_id', 'name', 'position', 'team_side', 
                       'rating', 'rating_original', 'rating_alternative']
        
        feature_cols = [col for col in df.columns if col not in exclude_cols]
        self.feature_names = feature_cols
        
        X = df[feature_cols]
        y = df[target_col]
        
        logger.info(f"Prepared {len(feature_cols)} features for training")
        
        return X, y
    
    def train(self, X: pd.DataFrame, y: pd.Series, 
              test_size: float = 0.2, 
              random_state: int = None) -> Dict[str, float]:
        """
        Train the model and return performance metrics.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            test_size (float): Proportion of data for testing
            random_state (int): Random state for train-test split
            
        Returns:
            Dict[str, float]: Performance metrics
        """
        if random_state is None:
            random_state = self.random_state
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        logger.info(f"Training {self.model_type} model...")
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = self.model.predict(X_train_scaled)
        y_test_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        metrics = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_mae': mean_absolute_error(y_train, y_train_pred),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_mae': mean_absolute_error(y_test, y_test_pred),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        self.is_fitted = True
        
        logger.info(f"Training completed. Test R²: {metrics['test_r2']:.3f}")
        
        return metrics
    
    def cross_validate(self, X: pd.DataFrame, y: pd.Series, 
                      cv: int = 5) -> Dict[str, List[float]]:
        """
        Perform cross-validation.
        
        Args:
            X (pd.DataFrame): Feature matrix
            y (pd.Series): Target variable
            cv (int): Number of cross-validation folds
            
        Returns:
            Dict[str, List[float]]: Cross-validation scores
        """
        X_scaled = self.scaler.fit_transform(X)
        
        # Perform cross-validation
        rmse_scores = np.sqrt(-cross_val_score(self.model, X_scaled, y, 
                                             scoring='neg_mean_squared_error', cv=cv))
        mae_scores = -cross_val_score(self.model, X_scaled, y, 
                                     scoring='neg_mean_absolute_error', cv=cv)
        r2_scores = cross_val_score(self.model, X_scaled, y, scoring='r2', cv=cv)
        
        cv_results = {
            'rmse_scores': rmse_scores.tolist(),
            'mae_scores': mae_scores.tolist(),
            'r2_scores': r2_scores.tolist(),
            'rmse_mean': rmse_scores.mean(),
            'rmse_std': rmse_scores.std(),
            'mae_mean': mae_scores.mean(),
            'mae_std': mae_scores.std(),
            'r2_mean': r2_scores.mean(),
            'r2_std': r2_scores.std()
        }
        
        logger.info(f"Cross-validation completed. Mean R²: {cv_results['r2_mean']:.3f} ± {cv_results['r2_std']:.3f}")
        
        return cv_results
    
    def get_feature_importance(self) -> pd.DataFrame:
        """
        Get feature importance scores.
        
        Returns:
            pd.DataFrame: Feature importance dataframe
        """
        if not self.is_fitted:
            raise ValueError("Model must be fitted before getting feature importance")
        
        if hasattr(self.model, 'feature_importances_'):
            importance = self.model.feature_importances_
        elif hasattr(self.model, 'coef_'):
            importance = np.abs(self.model.coef_)
        else:
            raise ValueError("Model does not support feature importance")
        
        importance_df = pd.DataFrame({
            'feature': self.feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return importance_df
    
    def save_model(self, filepath: str):
        """Save the trained model to disk."""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before saving")
        
        # Create directory if it doesn't exist
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        
        # Save model and scaler
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'feature_names': self.feature_names,
            'model_type': self.model_type
        }
        
        joblib.dump(model_data, filepath)
        logger.info(f"Model saved to {filepath}")
    
    def load_model(self, filepath: str):
        """Load a trained model from disk."""
        model_data = joblib.load(filepath)
        
        self.model = model_data['model']
        self.scaler = model_data['scaler']
        self.feature_names = model_data['feature_names']
        self.model_type = model_data['model_type']
        self.is_fitted = True
        
        logger.info(f"Model loaded from {filepath}")


def train_baseline_model(df: pd.DataFrame, target_col: str = 'rating') -> Dict[str, float]:
    """
    Train a simple baseline model (predicting mean rating).
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        
    Returns:
        Dict[str, float]: Baseline performance metrics
    """
    logger.info("Training baseline model (mean prediction)")
    
    # Calculate mean rating
    mean_rating = df[target_col].mean()
    
    # Create predictions (all same value)
    y_pred = np.full(len(df), mean_rating)
    y_true = df[target_col]
    
    # Calculate metrics
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred)
    }
    
    logger.info(f"Baseline model - RMSE: {metrics['rmse']:.3f}, R²: {metrics['r2']:.3f}")
    
    return metrics


def train_multiple_models(df: pd.DataFrame, 
                         target_col: str = 'rating',
                         test_size: float = 0.2) -> Dict[str, Dict[str, float]]:
    """
    Train multiple models and compare their performance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        target_col (str): Name of the target column
        test_size (float): Proportion of data for testing
        
    Returns:
        Dict[str, Dict[str, float]]: Performance metrics for each model
    """
    logger.info("Training multiple models for comparison")
    
    # Prepare features
    exclude_cols = ['id', 'match_id', 'team_id', 'name', 'position', 'team_side', 
                   'rating', 'rating_original', 'rating_alternative']
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    X = df[feature_cols]
    y = df[target_col]
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    # Define models to train
    models = {
        'linear': LinearRegression(),
        'ridge': Ridge(random_state=42),
        'lasso': Lasso(random_state=42),
        'random_forest': RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    }
    
    results = {}
    
    for name, model in models.items():
        logger.info(f"Training {name} model...")
        
        # Scale features
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Train model
        model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_train_pred = model.predict(X_train_scaled)
        y_test_pred = model.predict(X_test_scaled)
        
        # Calculate metrics
        results[name] = {
            'train_rmse': np.sqrt(mean_squared_error(y_train, y_train_pred)),
            'train_r2': r2_score(y_train, y_train_pred),
            'test_rmse': np.sqrt(mean_squared_error(y_test, y_test_pred)),
            'test_r2': r2_score(y_test, y_test_pred)
        }
        
        logger.info(f"{name} - Test R²: {results[name]['test_r2']:.3f}")
    
    return results


def hyperparameter_tuning(model_type: str, 
                         X: pd.DataFrame, 
                         y: pd.Series,
                         cv: int = 5) -> Dict[str, Any]:
    """
    Perform hyperparameter tuning using GridSearchCV.
    
    Args:
        model_type (str): Type of model to tune
        X (pd.DataFrame): Feature matrix
        y (pd.Series): Target variable
        cv (int): Number of cross-validation folds
        
    Returns:
        Dict[str, Any]: Best parameters and scores
    """
    logger.info(f"Performing hyperparameter tuning for {model_type}")
    
    # Define parameter grids
    param_grids = {
        'ridge': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'lasso': {
            'alpha': [0.001, 0.01, 0.1, 1, 10, 100]
        },
        'random_forest': {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20, 30],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4]
        }
    }
    
    if model_type not in param_grids:
        raise ValueError(f"No parameter grid defined for {model_type}")
    
    # Initialize model
    if model_type == 'ridge':
        model = Ridge(random_state=42)
    elif model_type == 'lasso':
        model = Lasso(random_state=42)
    elif model_type == 'random_forest':
        model = RandomForestRegressor(random_state=42, n_jobs=-1)
    
    # Perform grid search
    grid_search = GridSearchCV(
        model, param_grids[model_type], cv=cv, scoring='neg_mean_squared_error',
        n_jobs=-1, verbose=1
    )
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit grid search
    grid_search.fit(X_scaled, y)
    
    # Get best results
    best_params = grid_search.best_params_
    best_score = np.sqrt(-grid_search.best_score_)
    
    logger.info(f"Best parameters: {best_params}")
    logger.info(f"Best RMSE: {best_score:.3f}")
    
    return {
        'best_params': best_params,
        'best_score': best_score,
        'best_estimator': grid_search.best_estimator_,
        'cv_results': grid_search.cv_results_
    }


if __name__ == "__main__":
    # Example usage
    print("Model training module loaded successfully!")
