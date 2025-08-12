"""
Utility functions for player rating prediction project.

This module provides helper functions for data management and project utilities.
"""

import pandas as pd
import numpy as np
import os
import json
import logging
from typing import Dict, List, Any, Optional, Union
from datetime import datetime
import pickle

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def ensure_directory_exists(directory_path: str) -> None:
    """
    Ensure a directory exists, create it if it doesn't.
    
    Args:
        directory_path (str): Path to the directory
    """
    if not os.path.exists(directory_path):
        os.makedirs(directory_path)
        logger.info(f"Created directory: {directory_path}")
    else:
        logger.debug(f"Directory already exists: {directory_path}")


def save_dataframe(df: pd.DataFrame, 
                  filepath: str, 
                  format: str = 'csv',
                  **kwargs) -> None:
    """
    Save a dataframe to file in various formats.
    
    Args:
        df (pd.DataFrame): Dataframe to save
        filepath (str): Path to save the file
        format (str): File format ('csv', 'parquet', 'pickle', 'json')
        **kwargs: Additional arguments for the save function
    """
    ensure_directory_exists(os.path.dirname(filepath))
    
    try:
        if format.lower() == 'csv':
            df.to_csv(filepath, index=False, **kwargs)
        elif format.lower() == 'parquet':
            df.to_parquet(filepath, index=False, **kwargs)
        elif format.lower() == 'pickle':
            df.to_pickle(filepath, **kwargs)
        elif format.lower() == 'json':
            df.to_json(filepath, orient='records', **kwargs)
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataframe saved to {filepath} ({format} format)")
        
    except Exception as e:
        logger.error(f"Error saving dataframe: {e}")
        raise


def load_dataframe(filepath: str, 
                  format: str = None) -> pd.DataFrame:
    """
    Load a dataframe from file.
    
    Args:
        filepath (str): Path to the file
        format (str): File format (auto-detected if None)
        
    Returns:
        pd.DataFrame: Loaded dataframe
    """
    if format is None:
        # Auto-detect format from file extension
        _, ext = os.path.splitext(filepath)
        format = ext[1:].lower()
    
    try:
        if format == 'csv':
            df = pd.read_csv(filepath)
        elif format == 'parquet':
            df = pd.read_parquet(filepath)
        elif format == 'pickle':
            df = pd.read_pickle(filepath)
        elif format == 'json':
            df = pd.read_json(filepath, orient='records')
        else:
            raise ValueError(f"Unsupported format: {format}")
        
        logger.info(f"Dataframe loaded from {filepath} ({format} format)")
        return df
        
    except Exception as e:
        logger.error(f"Error loading dataframe: {e}")
        raise


def save_model_metadata(model_info: Dict[str, Any], 
                       filepath: str) -> None:
    """
    Save model metadata to JSON file.
    
    Args:
        model_info (Dict[str, Any]): Model information dictionary
        filepath (str): Path to save the metadata
    """
    ensure_directory_exists(os.path.dirname(filepath))
    
    try:
        with open(filepath, 'w') as f:
            json.dump(model_info, f, indent=2, default=str)
        
        logger.info(f"Model metadata saved to {filepath}")
        
    except Exception as e:
        logger.error(f"Error saving model metadata: {e}")
        raise


def load_model_metadata(filepath: str) -> Dict[str, Any]:
    """
    Load model metadata from JSON file.
    
    Args:
        filepath (str): Path to the metadata file
        
    Returns:
        Dict[str, Any]: Model metadata
    """
    try:
        with open(filepath, 'r') as f:
            metadata = json.load(f)
        
        logger.info(f"Model metadata loaded from {filepath}")
        return metadata
        
    except Exception as e:
        logger.error(f"Error loading model metadata: {e}")
        raise


def create_experiment_log(experiment_name: str,
                         base_dir: str = 'results') -> str:
    """
    Create a new experiment log directory.
    
    Args:
        experiment_name (str): Name of the experiment
        base_dir (str): Base directory for results
        
    Returns:
        str: Path to the experiment directory
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    experiment_dir = os.path.join(base_dir, f"{experiment_name}_{timestamp}")
    
    ensure_directory_exists(experiment_dir)
    
    # Create subdirectories
    subdirs = ['models', 'plots', 'reports', 'data']
    for subdir in subdirs:
        ensure_directory_exists(os.path.join(experiment_dir, subdir))
    
    logger.info(f"Created experiment directory: {experiment_dir}")
    
    return experiment_dir


def log_experiment_config(config: Dict[str, Any], 
                         experiment_dir: str) -> str:
    """
    Log experiment configuration to file.
    
    Args:
        config (Dict[str, Any]): Experiment configuration
        experiment_dir (str): Experiment directory path
        
    Returns:
        str: Path to the config file
    """
    config_file = os.path.join(experiment_dir, 'experiment_config.json')
    
    # Add timestamp
    config['timestamp'] = datetime.now().isoformat()
    
    save_model_metadata(config, config_file)
    
    return config_file


def get_data_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Generate comprehensive data summary.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        Dict[str, Any]: Data summary dictionary
    """
    summary = {
        'shape': df.shape,
        'columns': list(df.columns),
        'dtypes': df.dtypes.to_dict(),
        'missing_values': df.isnull().sum().to_dict(),
        'missing_percentage': (df.isnull().sum() / len(df) * 100).to_dict(),
        'numeric_columns': list(df.select_dtypes(include=[np.number]).columns),
        'categorical_columns': list(df.select_dtypes(include=['object']).columns),
        'memory_usage': df.memory_usage(deep=True).sum()
    }
    
    # Add basic statistics for numeric columns
    if summary['numeric_columns']:
        summary['numeric_stats'] = df[summary['numeric_columns']].describe().to_dict()
    
    # Add value counts for categorical columns
    if summary['categorical_columns']:
        summary['categorical_counts'] = {}
        for col in summary['categorical_columns']:
            summary['categorical_counts'][col] = df[col].value_counts().to_dict()
    
    return summary


def print_data_summary(df: pd.DataFrame, 
                      title: str = "Data Summary") -> None:
    """
    Print formatted data summary.
    
    Args:
        df (pd.DataFrame): Input dataframe
        title (str): Title for the summary
    """
    summary = get_data_summary(df)
    
    print(f"\n{title}")
    print("=" * len(title))
    print(f"Shape: {summary['shape']}")
    print(f"Memory Usage: {summary['memory_usage'] / 1024**2:.2f} MB")
    
    print(f"\nColumns ({len(summary['columns'])}):")
    for col in summary['columns']:
        dtype = summary['dtypes'][col]
        missing = summary['missing_values'][col]
        missing_pct = summary['missing_percentage'][col]
        print(f"  {col}: {dtype} (missing: {missing} ({missing_pct:.1f}%))")
    
    if summary['numeric_columns']:
        print(f"\nNumeric Columns ({len(summary['numeric_columns'])}):")
        for col in summary['numeric_columns']:
            stats = summary['numeric_stats'][col]
            print(f"  {col}: mean={stats['mean']:.3f}, std={stats['std']:.3f}, "
                  f"min={stats['min']:.3f}, max={stats['max']:.3f}")


def validate_dataframe_structure(df: pd.DataFrame, 
                                required_columns: List[str],
                                optional_columns: List[str] = None) -> bool:
    """
    Validate that dataframe has required structure.
    
    Args:
        df (pd.DataFrame): Dataframe to validate
        required_columns (List[str]): List of required columns
        optional_columns (List[str]): List of optional columns
        
    Returns:
        bool: True if validation passes
    """
    if optional_columns is None:
        optional_columns = []
    
    missing_required = [col for col in required_columns if col not in df.columns]
    
    if missing_required:
        logger.error(f"Missing required columns: {missing_required}")
        return False
    
    logger.info("Dataframe structure validation passed")
    return True


def create_feature_summary(df: pd.DataFrame, 
                          exclude_cols: List[str] = None) -> pd.DataFrame:
    """
    Create a summary of all features in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        exclude_cols (List[str]): Columns to exclude from summary
        
    Returns:
        pd.DataFrame: Feature summary dataframe
    """
    if exclude_cols is None:
        exclude_cols = ['id', 'match_id', 'team_id', 'name']
    
    feature_cols = [col for col in df.columns if col not in exclude_cols]
    
    summary_data = []
    
    for col in feature_cols:
        col_data = df[col]
        
        if col_data.dtype in ['int64', 'float64']:
            summary_data.append({
                'feature': col,
                'type': 'numeric',
                'missing': col_data.isnull().sum(),
                'missing_pct': col_data.isnull().sum() / len(df) * 100,
                'unique': col_data.nunique(),
                'mean': col_data.mean() if col_data.dtype == 'float64' else None,
                'std': col_data.std() if col_data.dtype == 'float64' else None,
                'min': col_data.min(),
                'max': col_data.max()
            })
        else:
            summary_data.append({
                'feature': col,
                'type': 'categorical',
                'missing': col_data.isnull().sum(),
                'missing_pct': col_data.isnull().sum() / len(df) * 100,
                'unique': col_data.nunique(),
                'top_value': col_data.value_counts().index[0] if len(col_data) > 0 else None,
                'top_freq': col_data.value_counts().iloc[0] if len(col_data) > 0 else None
            })
    
    return pd.DataFrame(summary_data)


def save_feature_summary(df: pd.DataFrame, 
                        filepath: str,
                        exclude_cols: List[str] = None) -> None:
    """
    Create and save feature summary to file.
    
    Args:
        df (pd.DataFrame): Input dataframe
        filepath (str): Path to save the summary
        exclude_cols (List[str]): Columns to exclude from summary
    """
    summary_df = create_feature_summary(df, exclude_cols)
    save_dataframe(summary_df, filepath, format='csv')
    
    logger.info(f"Feature summary saved to {filepath}")


if __name__ == "__main__":
    # Example usage
    print("Utils module loaded successfully!")
