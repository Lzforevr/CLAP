"""
Data Preprocessing Module
Handles data loading, format conversion, and filtering
"""

import pandas as pd
import numpy as np
from pathlib import Path
import config


def csv_generator(directory: Path):
    """Generator for reading CSV files"""
    for file in directory.glob('*.csv'):
        df_temp = pd.read_csv(file)
        if 'day' in df_temp.columns:
            df_temp = df_temp.drop(columns='day')
        df_temp = df_temp.fillna(0)
        yield df_temp


def read_and_preprocess_data(directory: Path):
    """
    Read and preprocess invocation data
    
    Args:
        directory: Path to directory containing CSV files
        
    Returns:
        DataFrame with minute_code as time index
    """
    df = pd.concat((df_temp for df_temp in csv_generator(directory)), ignore_index=True)
    df['minute_code'] = df['time'] // 60 
    df = df.drop(columns=['time'])
    return df


def convert_to_long_format(df):
    """Convert wide format to long format"""
    func_cols = [col for col in df.columns if col != 'minute_code']
    df_long = df.melt(id_vars=['minute_code'], value_vars=func_cols, 
                     var_name='funcID', value_name='call_count')
    return df_long


def build_daily_pattern(df_long: pd.DataFrame):
    """Build daily pattern matrix with funcID as rows and minutes as columns"""
    daily_pattern_pivot = df_long.pivot(index='funcID', columns='minute_code', 
                                       values='call_count').fillna(0) 
    return daily_pattern_pivot


def filter_low_total_calls(daily_pattern_pivot, threshold=None):
    """
    Filter out functions with total calls below threshold
    
    Args:
        daily_pattern_pivot: Pivot table of daily patterns
        threshold: Minimum total calls threshold (default from config)
        
    Returns:
        Filtered DataFrame
    """
    if threshold is None:
        threshold = config.MIN_CALL_THRESHOLD
    
    total_call = daily_pattern_pivot.sum(axis=1)
    filtered_daily_pattern_pivot = daily_pattern_pivot[total_call >= threshold]
    return filtered_daily_pattern_pivot
