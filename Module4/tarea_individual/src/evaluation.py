"""
Model evaluation module for player rating prediction.

This module handles model evaluation, error analysis, and visualization.
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Any, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import os

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set style for plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")


def calculate_performance_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> Dict[str, float]:
    """
    Calculate comprehensive performance metrics.
    
    Args:
        y_true (np.ndarray): True values
        y_pred (np.ndarray): Predicted values
        
    Returns:
        Dict[str, float]: Dictionary of performance metrics
    """
    metrics = {
        'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
        'mae': mean_absolute_error(y_true, y_pred),
        'r2': r2_score(y_true, y_pred),
        'mape': np.mean(np.abs((y_true - y_pred) / y_true)) * 100,
        'max_error': np.max(np.abs(y_true - y_pred)),
        'mean_error': np.mean(y_true - y_pred)
    }
    
    return metrics


def analyze_errors_by_position(df: pd.DataFrame, 
                              y_true_col: str = 'rating',
                              y_pred_col: str = 'rating_pred') -> Dict[str, Dict[str, float]]:
    """
    Analyze prediction errors by player position.
    
    Args:
        df (pd.DataFrame): Dataframe with true and predicted values
        y_true_col (str): Column name for true values
        y_pred_col (str): Column name for predicted values
        
    Returns:
        Dict[str, Dict[str, float]]: Error analysis by position
    """
    logger.info("Analyzing errors by position")
    
    position_errors = {}
    
    for position in df['position'].unique():
        pos_mask = df['position'] == position
        pos_data = df[pos_mask]
        
        if len(pos_data) == 0:
            continue
            
        y_true = pos_data[y_true_col]
        y_pred = pos_data[y_pred_col]
        
        errors = y_true - y_pred
        
        position_errors[position] = {
            'count': len(pos_data),
            'rmse': np.sqrt(mean_squared_error(y_true, y_pred)),
            'mae': mean_absolute_error(y_true, y_pred),
            'r2': r2_score(y_true, y_pred),
            'mean_error': np.mean(errors),
            'std_error': np.std(errors),
            'max_error': np.max(np.abs(errors))
        }
    
    return position_errors


def create_error_visualizations(df: pd.DataFrame,
                               y_true_col: str = 'rating',
                               y_pred_col: str = 'rating_pred',
                               save_path: str = None) -> None:
    """
    Create comprehensive error analysis visualizations.
    
    Args:
        df (pd.DataFrame): Dataframe with true and predicted values
        y_true_col (str): Column name for true values
        y_pred_col (str): Column name for predicted values
        save_path (str): Path to save plots (optional)
    """
    logger.info("Creating error analysis visualizations")
    
    # Calculate errors
    df['error'] = df[y_true_col] - df[y_pred_col]
    df['abs_error'] = np.abs(df['error'])
    
    # Create figure with subplots
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('Player Rating Prediction - Error Analysis', fontsize=16, fontweight='bold')
    
    # 1. True vs Predicted scatter plot
    axes[0, 0].scatter(df[y_true_col], df[y_pred_col], alpha=0.6)
    axes[0, 0].plot([df[y_true_col].min(), df[y_true_col].max()], 
                     [df[y_true_col].min(), df[y_true_col].max()], 'r--', lw=2)
    axes[0, 0].set_xlabel('True Rating')
    axes[0, 0].set_ylabel('Predicted Rating')
    axes[0, 0].set_title('True vs Predicted Ratings')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. Error distribution histogram
    axes[0, 1].hist(df['error'], bins=30, alpha=0.7, edgecolor='black')
    axes[0, 1].axvline(df['error'].mean(), color='red', linestyle='--', 
                        label=f'Mean: {df["error"].mean():.3f}')
    axes[0, 1].set_xlabel('Prediction Error')
    axes[0, 1].set_ylabel('Frequency')
    axes[0, 1].set_title('Error Distribution')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. Absolute error by position
    position_errors = df.groupby('position')['abs_error'].mean().sort_values(ascending=False)
    axes[0, 2].bar(position_errors.index, position_errors.values, alpha=0.7)
    axes[0, 2].set_xlabel('Position')
    axes[0, 2].set_ylabel('Mean Absolute Error')
    axes[0, 2].set_title('Mean Absolute Error by Position')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. Error vs predicted rating
    axes[1, 0].scatter(df[y_pred_col], df['error'], alpha=0.6)
    axes[1, 0].axhline(y=0, color='red', linestyle='--')
    axes[1, 0].set_xlabel('Predicted Rating')
    axes[1, 0].set_ylabel('Prediction Error')
    axes[1, 0].set_title('Error vs Predicted Rating')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Error by minutes played
    axes[1, 1].scatter(df['minutes_played'], df['abs_error'], alpha=0.6)
    axes[1, 1].set_xlabel('Minutes Played')
    axes[1, 1].set_ylabel('Absolute Error')
    axes[1, 1].set_title('Error vs Minutes Played')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. Residuals Q-Q plot
    from scipy import stats
    stats.probplot(df['error'], dist="norm", plot=axes[1, 2])
    axes[1, 2].set_title('Q-Q Plot of Residuals')
    axes[1, 2].grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Error analysis plots saved to {save_path}")
    
    plt.show()


def create_feature_importance_plot(importance_df: pd.DataFrame,
                                  top_n: int = 20,
                                  save_path: str = None) -> None:
    """
    Create feature importance visualization.
    
    Args:
        importance_df (pd.DataFrame): Dataframe with feature importance
        top_n (int): Number of top features to display
        save_path (str): Path to save plot (optional)
    """
    logger.info("Creating feature importance visualization")
    
    # Get top N features
    top_features = importance_df.head(top_n)
    
    plt.figure(figsize=(12, 8))
    
    # Create horizontal bar plot
    bars = plt.barh(range(len(top_features)), top_features['importance'])
    
    # Customize plot
    plt.yticks(range(len(top_features)), top_features['feature'])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features', fontweight='bold')
    plt.grid(True, alpha=0.3)
    
    # Add value labels on bars
    for i, bar in enumerate(bars):
        width = bar.get_width()
        plt.text(width + 0.001, bar.get_y() + bar.get_height()/2, 
                f'{width:.3f}', ha='left', va='center')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Feature importance plot saved to {save_path}")
    
    plt.show()


def compare_models_performance(model_results: Dict[str, Dict[str, float]],
                              save_path: str = None) -> None:
    """
    Create model comparison visualization.
    
    Args:
        model_results (Dict): Dictionary with model performance results
        save_path (str): Path to save plot (optional)
    """
    logger.info("Creating model comparison visualization")
    
    # Prepare data for plotting
    models = list(model_results.keys())
    metrics = ['test_rmse', 'test_mae', 'test_r2']
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    fig.suptitle('Model Performance Comparison', fontsize=16, fontweight='bold')
    
    for i, metric in enumerate(metrics):
        values = [model_results[model][metric] for model in models]
        
        bars = axes[i].bar(models, values, alpha=0.7)
        axes[i].set_title(metric.replace('_', ' ').title())
        axes[i].set_ylabel(metric.replace('_', ' ').title())
        axes[i].grid(True, alpha=0.3)
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            axes[i].text(bar.get_x() + bar.get_width()/2., height + 0.001,
                        f'{value:.3f}', ha='center', va='bottom')
    
    plt.tight_layout()
    
    if save_path:
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        logger.info(f"Model comparison plot saved to {save_path}")
    
    plt.show()


def generate_evaluation_report(df: pd.DataFrame,
                              y_true_col: str = 'rating',
                              y_pred_col: str = 'rating_pred',
                              save_dir: str = 'results') -> Dict[str, Any]:
    """
    Generate comprehensive evaluation report.
    
    Args:
        df (pd.DataFrame): Dataframe with true and predicted values
        y_true_col (str): Column name for true values
        y_pred_col (str): Column name for predicted values
        save_dir (str): Directory to save results
        
    Returns:
        Dict[str, Any]: Comprehensive evaluation report
    """
    logger.info("Generating comprehensive evaluation report")
    
    # Calculate overall metrics
    y_true = df[y_true_col]
    y_pred = df[y_pred_col]
    
    overall_metrics = calculate_performance_metrics(y_true, y_pred)
    
    # Analyze errors by position
    position_errors = analyze_errors_by_position(df, y_true_col, y_pred_col)
    
    # Create visualizations
    os.makedirs(save_dir, exist_ok=True)
    
    # Error analysis plots
    error_plot_path = os.path.join(save_dir, 'error_analysis.png')
    create_error_visualizations(df, y_true_col, y_pred_col, error_plot_path)
    
    # Compile report
    report = {
        'overall_metrics': overall_metrics,
        'position_errors': position_errors,
        'data_summary': {
            'total_samples': len(df),
            'positions': df['position'].value_counts().to_dict(),
            'rating_range': (y_true.min(), y_true.max()),
            'rating_mean': y_true.mean(),
            'rating_std': y_true.std()
        },
        'plots_saved': {
            'error_analysis': error_plot_path
        }
    }
    
    # Save report to file
    report_path = os.path.join(save_dir, 'evaluation_report.txt')
    with open(report_path, 'w') as f:
        f.write("PLAYER RATING PREDICTION - EVALUATION REPORT\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("OVERALL PERFORMANCE METRICS:\n")
        f.write("-" * 30 + "\n")
        for metric, value in overall_metrics.items():
            f.write(f"{metric.upper()}: {value:.4f}\n")
        
        f.write("\nPOSITION-SPECIFIC ERRORS:\n")
        f.write("-" * 30 + "\n")
        for position, errors in position_errors.items():
            f.write(f"\n{position}:\n")
            for metric, value in errors.items():
                f.write(f"  {metric}: {value:.4f}\n")
        
        f.write("\nDATA SUMMARY:\n")
        f.write("-" * 20 + "\n")
        for key, value in report['data_summary'].items():
            f.write(f"{key}: {value}\n")
    
    logger.info(f"Evaluation report saved to {report_path}")
    
    return report


def analyze_prediction_outliers(df: pd.DataFrame,
                               y_true_col: str = 'rating',
                               y_pred_col: str = 'rating_pred',
                               threshold: float = 2.0) -> pd.DataFrame:
    """
    Identify and analyze prediction outliers.
    
    Args:
        df (pd.DataFrame): Dataframe with true and predicted values
        y_true_col (str): Column name for true values
        y_pred_col (str): Column name for predicted values
        threshold (float): Threshold for outlier detection (in standard deviations)
        
    Returns:
        pd.DataFrame: Dataframe with outlier records
    """
    logger.info("Analyzing prediction outliers")
    
    # Calculate errors and z-scores
    df['error'] = df[y_true_col] - df[y_pred_col]
    df['error_zscore'] = np.abs((df['error'] - df['error'].mean()) / df['error'].std())
    
    # Identify outliers
    outliers = df[df['error_zscore'] > threshold].copy()
    
    if len(outliers) > 0:
        logger.info(f"Found {len(outliers)} outliers (threshold: {threshold} std)")
        
        # Sort by error magnitude
        outliers = outliers.sort_values('error_zscore', ascending=False)
        
        # Add outlier analysis columns
        outliers['error_magnitude'] = np.abs(outliers['error'])
        outliers['prediction_accuracy'] = 1 - (outliers['error_magnitude'] / outliers[y_true_col])
        
    else:
        logger.info("No outliers found")
    
    return outliers


if __name__ == "__main__":
    # Example usage
    print("Evaluation module loaded successfully!")
