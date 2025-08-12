# Player Rating Prediction - Feature Engineering Module
"""
This module handles feature engineering for player rating prediction.
Features are derived ONLY from current match performance data.
"""

import pandas as pd
import numpy as np
import logging
from typing import List, Dict, Any, Tuple
from sklearn.preprocessing import StandardScaler, MinMaxScaler, RobustScaler

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def extract_performance_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Extract performance-related features from the dataframe.
    Only uses current match statistics.
    
    Args:
        df (pd.DataFrame): Input dataframe with player statistics
        
    Returns:
        pd.DataFrame: Dataframe with performance features only
    """
    logger.info("Extracting performance features from current match data")
    
    # Get all numeric columns that represent performance stats
    # Exclude non-performance columns
    exclude_cols = ['id', 'match_id', 'team_id', 'shirt_number', 'position', 
                    'substitute', 'captain', 'team_side', 'rating', 'minutes_played']
    
    performance_cols = [col for col in df.columns 
                       if col not in exclude_cols and 
                       df[col].dtype in ['int64', 'float64']]
    
    logger.info(f"Found {len(performance_cols)} performance features")
    
    return df[performance_cols]

def create_match_context_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create match context features from current match data only.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with match context features
    """
    logger.info("Creating match context features")
    
    df_context = df.copy()
    
    # Minutes played percentage (assuming 90 minutes for full match)
    df_context['minutes_played_pct'] = df_context['minutes_played'] / 90.0
    
    # Starter vs substitute (1 for starter, 0 for substitute)
    df_context['is_starter'] = (~df_context['substitute']).astype(int)
    
    # Captain status (1 for captain, 0 for not)
    df_context['is_captain'] = df_context['captain'].astype(int)
    
    # Home vs away team (1 for home, 0 for away)
    df_context['is_home'] = (df_context['team_side'] == 'home').astype(int)
    
    # Position encoding (one-hot encoding)
    position_dummies = pd.get_dummies(df_context['position'], prefix='position')
    df_context = pd.concat([df_context, position_dummies], axis=1)
    
    logger.info("Created match context features")
    
    return df_context

def create_position_specific_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Create position-specific features based on current match performance.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        pd.DataFrame: Dataframe with position-specific features
    """
    logger.info("Creating position-specific features")
    
    df_pos = df.copy()
    
    # Goalkeeper-specific features
    gk_mask = df_pos['position'] == 'G'
    if gk_mask.any():
        # Save rate (saves / total shots faced)
        df_pos.loc[gk_mask, 'save_rate'] = (
            df_pos.loc[gk_mask, 'saves'] / 
            (df_pos.loc[gk_mask, 'saves'] + df_pos.loc[gk_mask, 'goals_conceded']).replace(0, 1)
        )
        
        # Clean sheet (1 if no goals conceded, 0 otherwise)
        df_pos.loc[gk_mask, 'clean_sheet'] = (
            df_pos.loc[gk_mask, 'goals_conceded'] == 0
        ).astype(int)
    
    # Defender-specific features
    def_mask = df_pos['position'] == 'D'
    if def_mask.any():
        # Defensive actions per 90 minutes
        defensive_cols = ['total_tackle', 'total_clearance', 'interception_won', 'blocked_scoring_attempt']
        df_pos.loc[def_mask, 'defensive_actions_per_90'] = (
            df_pos.loc[def_mask, defensive_cols].sum(axis=1) * 90 / 
            df_pos.loc[def_mask, 'minutes_played'].replace(0, 1)
        )
    
    # Midfielder-specific features
    mid_mask = df_pos['position'] == 'M'
    if mid_mask.any():
        # Pass accuracy
        df_pos.loc[mid_mask, 'pass_accuracy'] = (
            df_pos.loc[mid_mask, 'accurate_pass'] / 
            df_pos.loc[mid_mask, 'total_pass'].replace(0, 1)
        )
        
        # Key passes per 90 minutes
        df_pos.loc[mid_mask, 'key_passes_per_90'] = (
            df_pos.loc[mid_mask, 'key_pass'] * 90 / 
            df_pos.loc[mid_mask, 'minutes_played'].replace(0, 1)
        )
    
    # Forward-specific features
    fwd_mask = df_pos['position'] == 'F'
    if fwd_mask.any():
        # Goals per 90 minutes
        df_pos.loc[fwd_mask, 'goals_per_90'] = (
            df_pos.loc[fwd_mask, 'goals'] * 90 / 
            df_pos.loc[fwd_mask, 'minutes_played'].replace(0, 1)
        )
        
        # Expected goals conversion rate
        df_pos.loc[fwd_mask, 'xg_conversion_rate'] = (
            df_pos.loc[fwd_mask, 'goals'] / 
            df_pos.loc[fwd_mask, 'expected_goals'].replace(0, 1)
        )
    
    # Fill NaN values with 0 for position-specific features
    position_features = ['save_rate', 'clean_sheet', 'defensive_actions_per_90', 
                        'pass_accuracy', 'key_passes_per_90', 'goals_per_90', 'xg_conversion_rate']
    
    for feature in position_features:
        if feature in df_pos.columns:
            df_pos[feature] = df_pos[feature].fillna(0)
    
    logger.info("Created position-specific features")
    
    return df_pos

def normalize_features(df: pd.DataFrame, 
                     method: str = 'standard',
                     exclude_cols: List[str] = None) -> Tuple[pd.DataFrame, Any]:
    """
    Normalize numerical features using specified method.
    
    Args:
        df (pd.DataFrame): Input dataframe
        method (str): Normalization method ('standard', 'minmax', 'robust')
        exclude_cols (List[str]): Columns to exclude from normalization
        
    Returns:
        Tuple[pd.DataFrame, Any]: Normalized dataframe and scaler object
    """
    logger.info(f"Normalizing features using {method} scaling")
    
    if exclude_cols is None:
        exclude_cols = ['id', 'match_id', 'team_id', 'shirt_number', 'position', 
                       'substitute', 'captain', 'team_side', 'rating', 'minutes_played']
    
    # Get columns to normalize
    normalize_cols = [col for col in df.columns 
                     if col not in exclude_cols and 
                     df[col].dtype in ['int64', 'float64']]
    
    if not normalize_cols:
        logger.warning("No columns to normalize")
        return df, None
    
    df_norm = df.copy()
    
    # Choose scaler
    if method == 'standard':
        scaler = StandardScaler()
    elif method == 'minmax':
        scaler = MinMaxScaler()
    elif method == 'robust':
        scaler = RobustScaler()
    else:
        raise ValueError(f"Unknown normalization method: {method}")
    
    # Fit and transform
    df_norm[normalize_cols] = scaler.fit_transform(df_norm[normalize_cols])
    
    logger.info(f"Normalized {len(normalize_cols)} features using {method} scaling")
    
    return df_norm, scaler

def engineer_all_features(df: pd.DataFrame, 
                         normalize: bool = True,
                         normalization_method: str = 'standard') -> Tuple[pd.DataFrame, Any]:
    """
    Engineer all features from current match data only.
    NO historical context or rolling averages.
    
    Args:
        df (pd.DataFrame): Input dataframe with raw player statistics
        normalize (bool): Whether to normalize features
        normalization_method (str): Normalization method to use
        
    Returns:
        Tuple[pd.DataFrame, Any]: Engineered dataframe and scaler object
    """
    logger.info("Starting feature engineering pipeline (current match only)")
    
    # Step 1: Extract performance features (current match stats only)
    df_features = extract_performance_features(df)
    
    # Step 2: Add match context features
    df_features = create_match_context_features(df_features)
    
    # Step 3: Add position-specific features
    df_features = create_position_specific_features(df_features)
    
    # Step 4: Normalize features if requested
    scaler = None
    if normalize:
        df_features, scaler = normalize_features(df_features, normalization_method)
    
    # Step 5: Add back essential columns
    essential_cols = ['id', 'match_id', 'team_id', 'position', 'rating', 'minutes_played']
    for col in essential_cols:
        if col in df.columns:
            df_features[col] = df[col]
    
    logger.info(f"Feature engineering complete. Final shape: {df_features.shape}")
    
    return df_features, scaler

def get_feature_summary(df: pd.DataFrame) -> Dict[str, Any]:
    """
    Get summary of engineered features.
    
    Args:
        df (pd.DataFrame): Engineered dataframe
        
    Returns:
        Dict[str, Any]: Feature summary
    """
    summary = {
        'total_features': len(df.columns),
        'feature_types': {},
        'missing_values': {},
        'feature_ranges': {}
    }
    
    for col in df.columns:
        # Feature type
        summary['feature_types'][col] = str(df[col].dtype)
        
        # Missing values
        summary['missing_values'][col] = df[col].isnull().sum()
        
        # Feature range (for numeric columns)
        if df[col].dtype in ['int64', 'float64']:
            summary['feature_ranges'][col] = {
                'min': df[col].min(),
                'max': df[col].max(),
                'mean': df[col].mean(),
                'std': df[col].std()
            }
    
    return summary
