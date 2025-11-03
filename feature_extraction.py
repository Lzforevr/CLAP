"""
Module 1: Multifaceted Feature Extraction
Extracts time-domain, frequency-domain, anomaly, and shape features from invocation sequences
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from scipy.fft import fft
import config


def extract_features(daily_pattern_pivot: pd.DataFrame):
    """
    Extract multi-dimensional features from daily call patterns
    
    Args:
        daily_pattern_pivot: DataFrame with funcID as index and time as columns
        
    Returns:
        Tuple of extracted features: (total_call, shape_features, calls_std, calls_mean, 
        calls_median, calls_max, calls_min, fft_peak, fft_freq, anomaly_ratios, burst_ratios)
    """
    # Time-domain statistical features
    total_call = daily_pattern_pivot.sum(axis=1).to_frame(name='total_call')
    calls_std = daily_pattern_pivot.std(axis=1).to_frame(name='calls_std')
    calls_mean = daily_pattern_pivot.mean(axis=1).to_frame(name='calls_mean')
    calls_median = daily_pattern_pivot.median(axis=1).to_frame(name='calls_median')
    calls_max = daily_pattern_pivot.max(axis=1).to_frame(name='calls_max')
    calls_min = daily_pattern_pivot.min(axis=1).to_frame(name='calls_min')
    
    # Frequency-domain features via FFT
    fft_vals_matrix = np.abs(fft(daily_pattern_pivot.values, axis=1))
    fft_peak_indices = np.argmax(fft_vals_matrix[:, 1:fft_vals_matrix.shape[1]//2], axis=1)
    fft_peak = np.max(fft_vals_matrix[:, 1:fft_vals_matrix.shape[1]//2], axis=1).reshape(-1, 1)
    fft_freq = fft_peak_indices.reshape(-1, 1)
    
    # Anomaly-related features
    z_scores = (daily_pattern_pivot.values - daily_pattern_pivot.mean(axis=1).values.reshape(-1, 1)) / \
              (daily_pattern_pivot.std(axis=1).values.reshape(-1, 1) + 1e-8)
    anomaly_ratios = (np.abs(z_scores) > 3).mean(axis=1).reshape(-1, 1)
    
    # Burstiness features
    diffs = np.diff(daily_pattern_pivot.values, axis=1)
    std_vals = daily_pattern_pivot.std(axis=1).values.reshape(-1, 1)
    burst_ratios = (np.abs(diffs) > 2*std_vals).mean(axis=1).reshape(-1, 1)
    
    # Shape features (normalized patterns)
    shape_features = daily_pattern_pivot.div(total_call['total_call'], axis=0).fillna(0)
    
    return (total_call, shape_features, calls_std, calls_mean, calls_median, 
            calls_max, calls_min, fft_peak, fft_freq, 
            anomaly_ratios, burst_ratios)


def reduce_and_combine_features(total_call, shape_features, calls_std, calls_mean, 
                              calls_median, calls_max, calls_min, fft_peak,
                              fft_freq, anomaly_ratios, burst_ratios):
    """
    Reduce feature dimensionality and combine all features
    
    Args:
        Feature components from extract_features()
        
    Returns:
        combined_features: Normalized and combined feature matrix
        shape_features_pca: PCA-transformed shape features
    """
    # PCA for shape features
    n_components = min(5, shape_features.shape[1], shape_features.shape[0])
    pca = PCA(n_components=n_components, random_state=config.RANDOM_SEED)
    shape_features_pca = pca.fit_transform(shape_features)
    
    # Standardize numerical features
    scaler = StandardScaler()
    values = np.hstack([total_call, calls_std, calls_mean, calls_median, 
                      calls_max, calls_min, fft_peak, fft_freq,
                      anomaly_ratios, burst_ratios])
    values = np.nan_to_num(values, nan=0, posinf=np.max(np.isfinite(values)), 
                          neginf=np.min(np.isfinite(values)))
    stats_features = scaler.fit_transform(values)
    
    # Combine all features
    combined_features = np.hstack([shape_features_pca, stats_features])
    combined_features = np.nan_to_num(combined_features, nan=0)
    
    return combined_features, shape_features_pca
