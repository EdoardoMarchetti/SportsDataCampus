"""
Data preprocessing module for player rating prediction.

This module handles data cleaning, validation, and preparation for the ML pipeline.
"""

import pandas as pd
import numpy as np
from typing import Tuple, Dict, Any
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_raw_data(filepath: str) -> pd.DataFrame:
    """
    Load raw player statistics data from CSV file.
    
    Args:
        filepath (str): Path to the CSV file
        
    Returns:
        pd.DataFrame: Raw data loaded from CSV
    """
    try:
        df = pd.read_csv(filepath)
        logger.info(f"Loaded {len(df)} records from {filepath}")
        return df
    except Exception as e:
        logger.error(f"Error loading data: {e}")
        raise


def filter_valid_ratings(df: pd.DataFrame, rating_col: str = 'rating') -> pd.DataFrame:
    """
    Filter records with valid ratings (not null, between 0-10).
    
    Args:
        df (pd.DataFrame): Input dataframe
        rating_col (str): Name of the rating column
        
    Returns:
        pd.DataFrame: Filtered dataframe with valid ratings
    """
    initial_count = len(df)
    
    # Filter out null ratings
    df_filtered = df.dropna(subset=[rating_col])
    
    # Filter ratings between 0-10
    df_filtered = df_filtered[
        (df_filtered[rating_col] >= 0) & 
        (df_filtered[rating_col] <= 10)
    ]
    
    final_count = len(df_filtered)
    logger.info(f"Filtered ratings: {initial_count} -> {final_count} records")
    
    return df_filtered


def separate_goalkeepers(df: pd.DataFrame, position_col: str = 'position') -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Separate goalkeepers from field players.
    
    Args:
        df (pd.DataFrame): Input dataframe
        position_col (str): Name of the position column
        
    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (goalkeepers_df, field_players_df)
    """
    goalkeepers = df[df[position_col] == 'G'].copy()
    field_players = df[df[position_col] != 'G'].copy()
    
    logger.info(f"Separated players: {len(goalkeepers)} goalkeepers, {len(field_players)} field players")
    
    return goalkeepers, field_players


def handle_missing_values(df: pd.DataFrame, fill_value: float = 0.0) -> pd.DataFrame:
    """
    Fill missing values in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        fill_value (float): Value to fill missing numeric columns with
        
    Returns:
        pd.DataFrame: Dataframe with filled missing values
    """
    # Fill numeric columns with fill_value
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    df[numeric_columns] = df[numeric_columns].fillna(fill_value)
    
    # Fill categorical columns with 'Unknown'
    categorical_columns = df.select_dtypes(include=['object']).columns
    df[categorical_columns] = df[categorical_columns].fillna('Unknown')
    
    logger.info(f"Filled missing values in {len(numeric_columns)} numeric and {len(categorical_columns)} categorical columns")
    
    return df


def validate_data_quality(df: pd.DataFrame, rating_col: str = 'rating') -> Dict[str, Any]:
    """
    Validate data quality and return summary statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe
        rating_col (str): Name of the rating column
        
    Returns:
        Dict[str, Any]: Data quality summary
    """
    quality_report = {
        'total_records': len(df),
        'missing_ratings': df[rating_col].isnull().sum(),
        'rating_range': (df[rating_col].min(), df[rating_col].max()),
        'rating_mean': df[rating_col].mean(),
        'rating_std': df[rating_col].std(),
        'position_distribution': df['position'].value_counts().to_dict(),
        'teams_count': df['team_id'].nunique(),
        'players_count': df['id'].nunique(),
        'matches_count': df['match_id'].nunique()
    }
    
    logger.info("Data quality validation completed")
    return quality_report


def create_clean_dataset(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create a clean, structured dataset ready for feature engineering.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Clean dataset
    """
    logger.info("Starting data cleaning process...")
    
    # Filter valid ratings
    df_clean = filter_valid_ratings(df)
    
    # Handle missing values
    df_clean = handle_missing_values(df_clean)
    
    # Ensure proper data types
    df_clean['match_id'] = df_clean['match_id'].astype(int)
    df_clean['id'] = df_clean['id'].astype(int)
    df_clean['team_id'] = df_clean['team_id'].astype(int)
    
    logger.info(f"Clean dataset created with {len(df_clean)} records")
    
    return df_clean


if __name__ == "__main__":
    # Example usage
    print("Data preprocessing module loaded successfully!")
