"""
Feature engineering module for player rating prediction.

This module creates features from performance statistics and historical context.
"""

import pandas as pd
import numpy as np
from typing import List, Dict, Any, Optional
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def extract_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract performance statistics features from the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe with player statistics
        
    Returns:
        pd.DataFrame: Dataframe with performance features
    """
    logger.info("Extracting performance features...")
    
    # Get all numeric columns that are performance statistics
    # Exclude metadata columns like id, match_id, team_id, etc.
    exclude_cols = ['id', 'match_id', 'team_id', 'shirt_number', 'rating', 
                    'rating_original', 'rating_alternative', 'minutes_played']
    
    performance_cols = [col for col in df.select_dtypes(include=[np.number]).columns 
                       if col not in exclude_cols]
    
    logger.info(f"Identified {len(performance_cols)} performance features")
    
    return df[performance_cols]


def calculate_rolling_averages(df: pd.DataFrame, 
                             window_size: int = 3,
                             min_matches: int = 2) -> pd.DataFrame:
    """
    Calculate rolling averages from previous matches for each player.
    
    Args:
        df (pd.DataFrame): Input dataframe
        window_size (int): Number of previous matches to consider
        min_matches (int): Minimum matches required to calculate rolling average
        
    Returns:
        pd.DataFrame: Dataframe with rolling average features
    """
    logger.info(f"Calculating rolling averages with window size {window_size}")
    
    df_rolling = df.copy()
    
    # Sort by player and match timestamp (if available) or match_id
    df_rolling = df_rolling.sort_values(['id', 'match_id'])
    
    # Get performance columns
    performance_cols = extract_performance_features(df_rolling).columns
    
    # Calculate rolling averages for each player
    for col in performance_cols:
        rolling_col_name = f"{col}_rolling_{window_size}"
        
        # Group by player and calculate rolling mean
        df_rolling[rolling_col_name] = df_rolling.groupby('id')[col].transform(
            lambda x: x.rolling(window=window_size, min_periods=min_matches).mean()
        )
        
        # Fill NaN values with 0 (for players with insufficient history)
        df_rolling[rolling_col_name] = df_rolling[rolling_col_name].fillna(0)
    
    logger.info(f"Created {len(performance_cols)} rolling average features")
    
    return df_rolling


def create_match_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create match context features (minutes played, substitute status, etc.).
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with context features
    """
    logger.info("Creating match context features...")
    
    df_context = df.copy()
    
    # Minutes played features
    df_context['minutes_played_pct'] = df_context['minutes_played'] / 90.0
    df_context['is_starter'] = (df_context['minutes_played'] >= 60).astype(int)
    df_context['is_substitute'] = (df_context['minutes_played'] < 60).astype(int)
    
    # Captain status
    df_context['is_captain'] = df_context['captain'].astype(int)
    
    # Team side (home/away)
    df_context['is_home'] = (df_context['team_side'] == 'home').astype(int)
    
    # Position encoding
    position_mapping = {'G': 0, 'D': 1, 'M': 2, 'F': 3}
    df_context['position_encoded'] = df_context['position'].map(position_mapping)
    
    logger.info("Created match context features")
    
    return df_context


def create_position_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create position-specific features for different player roles.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with position-specific features
    """
    logger.info("Creating position-specific features...")
    
    df_pos = df.copy()
    
    # Goalkeeper-specific features
    gk_mask = df_pos['position'] == 'G'
    df_pos.loc[gk_mask, 'save_rate'] = (
        df_pos.loc[gk_mask, 'saves'] / 
        (df_pos.loc[gk_mask, 'saves'] + df_pos.loc[gk_mask, 'goals_conceded']).replace(0, 1)
    )
    
    # Defender-specific features
    def_mask = df_pos['position'] == 'D'
    df_pos.loc[def_mask, 'defensive_actions'] = (
        df_pos.loc[def_mask, 'total_tackle'] + 
        df_pos.loc[def_mask, 'interception_won'] + 
        df_pos.loc[def_mask, 'total_clearance']
    )
    
    # Midfielder-specific features
    mid_mask = df_pos['position'] == 'M'
    df_pos.loc[mid_mask, 'midfield_actions'] = (
        df_pos.loc[mid_mask, 'key_pass'] + 
        df_pos.loc[mid_mask, 'goal_assist'] + 
        df_pos.loc[mid_mask, 'total_pass']
    )
    
    # Forward-specific features
    fwd_mask = df_pos['position'] == 'F'
    df_pos.loc[fwd_mask, 'attacking_actions'] = (
        df_pos.loc[fwd_mask, 'on_target_scoring_attempt'] + 
        df_pos.loc[fwd_mask, 'goal_assist'] + 
        df_pos.loc[fwd_mask, 'expected_goals']
    )
    
    # Fill NaN values with 0
    df_pos = df_pos.fillna(0)
    
    logger.info("Created position-specific features")
    
    return df_pos


def normalize_features(df: pd.DataFrame, 
                     method: str = 'standard',
                     exclude_cols: Optional[List[str]] = None) -> pd.DataFrame:
    """
    Normalize numerical features for better model performance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Normalization method ('standard', 'minmax', 'robust')
        exclude_cols (List[str]): Columns to exclude from normalization
        
    Returns:
        pd.DataFrame: Dataframe with normalized features
    """
    logger.info(f"Normalizing features using {method} method")
    
    if exclude_cols is None:
        exclude_cols = ['id', 'match_id', 'team_id', 'rating', 'rating_original', 'rating_alternative']
    
    df_norm = df.copy()
    
    # Get numerical columns to normalize
    numeric_cols = df_norm.select_dtypes(include=[np.number]).columns
    normalize_cols = [col for col in numeric_cols if col not in exclude_cols]
    
    if method == 'standard':
        # Z-score normalization
        df_norm[normalize_cols] = (df_norm[normalize_cols] - df_norm[normalize_cols].mean()) / df_norm[normalize_cols].std()
    elif method == 'minmax':
        # Min-max scaling to [0,1]
        df_norm[normalize_cols] = (df_norm[normalize_cols] - df_norm[normalize_cols].min()) / (df_norm[normalize_cols].max() - df_norm[normalize_cols].min())
    elif method == 'robust':
        # Robust scaling using median and IQR
        Q1 = df_norm[normalize_cols].quantile(0.25)
        Q3 = df_norm[normalize_cols].quantile(0.75)
        IQR = Q3 - Q1
        df_norm[normalize_cols] = (df_norm[normalize_cols] - df_norm[normalize_cols].median()) / IQR
    
    # Fill NaN values with 0 (for columns with zero variance)
    df_norm[normalize_cols] = df_norm[normalize_cols].fillna(0)
    
    logger.info(f"Normalized {len(normalize_cols)} features")
    
    return df_norm


def engineer_all_features(df: pd.DataFrame, 
                         rolling_window: int = 3,
                         normalize: bool = True) -> pd.DataFrame:
    """
    Apply all feature engineering steps to create the final feature set.
    
    Args:
        df (pd.DataFrame): Input dataframe
        rolling_window (int): Window size for rolling averages
        normalize (bool): Whether to normalize features
        
    Returns:
        pd.DataFrame: Dataframe with all engineered features
    """
    logger.info("Starting comprehensive feature engineering...")
    
    # Step 1: Calculate rolling averages
    df_features = calculate_rolling_averages(df, window_size=rolling_window)
    
    # Step 2: Create match context features
    df_features = create_match_context_features(df_features)
    
    # Step 3: Create position-specific features
    df_features = create_position_specific_features(df_features)
    
    # Step 4: Normalize features if requested
    if normalize:
        df_features = normalize_features(df_features)
    
    logger.info(f"Feature engineering completed. Final shape: {df_features.shape}")
    
    return df_features


if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully!")
