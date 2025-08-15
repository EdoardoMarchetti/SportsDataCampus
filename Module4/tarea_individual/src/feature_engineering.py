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




def add_ratio_features(df, ratio_specs):
    """Return a new DataFrame with ratio features added (does not modify input df)."""
    df_new = df.copy()
    for new_f, spec in ratio_specs.items():
        num = spec["num"]
        den = spec["den"]
        def ratio_row(row):
            num_sum = sum([row.get(col, 0) for col in num])
            den_sum = sum([row.get(col, 0) for col in den])
            if den_sum == 0:
                return np.nan
            return num_sum / den_sum
        df_new[new_f] = df_new.apply(ratio_row, axis=1)
    return df_new


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





if __name__ == "__main__":
    # Example usage
    print("Feature engineering module loaded successfully!")
