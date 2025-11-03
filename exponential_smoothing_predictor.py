"""
Module 3b: Hybrid Temporal Network Learning - Exponential Smoothing Component
Adaptive exponential smoothing for simple pattern prediction
"""

import numpy as np
import pandas as pd
from statsmodels.tsa.holtwinters import SimpleExpSmoothing, Holt, ExponentialSmoothing
from sklearn.metrics import mean_squared_error
import logging
import warnings
import config

warnings.filterwarnings('ignore')


def load_cluster_data(file_path):
    """Load and split cluster data for exponential smoothing"""
    df = pd.read_csv(file_path, usecols=['ds', 'y'])
    df['ds'] = pd.to_datetime(df['ds'], utc=False)
    
    train_end = pd.to_datetime(config.TRAIN_END_DATE, utc=False)
    train_df = df[df['ds'] <= train_end].copy()
    test_df = df[df['ds'] > train_end].copy()
    
    return train_df, test_df


def check_seasonality(series, logger=None):
    """
    Detect seasonality in time series
    
    Returns:
        seasonal_period: Detected seasonal period or None
    """
    if len(series) >= 1440 * 3:  # Daily seasonality
        return config.ES_SEASONAL_PERIODS_DAILY
    elif len(series) >= 60 * 6:  # Hourly seasonality
        return config.ES_SEASONAL_PERIODS_HOURLY
    else:
        return None


def train_simple_exponential_smoothing(train_series, logger=None):
    """
    Train Simple Exponential Smoothing model
    
    Returns:
        model_fit: Fitted model
        train_time: Training time
    """
    import time
    if logger:
        logger.info("Training Simple Exponential Smoothing...")
    
    train_start = time.time()
    model = SimpleExpSmoothing(train_series)
    model_fit = model.fit(optimized=True)
    train_time = time.time() - train_start
    
    if logger:
        logger.info(f"SES training completed in {train_time:.4f}s")
        logger.info(f"Optimal alpha: {model_fit.params['smoothing_level']:.4f}")
    
    return model_fit, train_time


def train_holt_model(train_series, logger=None):
    """
    Train Holt's Linear Trend model
    
    Returns:
        model_fit: Fitted model
        train_time: Training time
    """
    import time
    if logger:
        logger.info("Training Holt's Linear Trend model...")
    
    train_start = time.time()
    model = Holt(train_series)
    model_fit = model.fit(optimized=True)
    train_time = time.time() - train_start
    
    if logger:
        logger.info(f"Holt training completed in {train_time:.4f}s")
    
    return model_fit, train_time


def train_holt_winters_model(train_series, seasonal_periods, logger=None):
    """
    Train Holt-Winters Seasonal model
    
    Returns:
        model_fit: Fitted model or None if failed
        train_time: Training time
    """
    import time
    if logger:
        logger.info("Training Holt-Winters Seasonal model...")
    
    train_start = time.time()
    
    try:
        # Try additive model first
        model = ExponentialSmoothing(
            train_series,
            seasonal_periods=seasonal_periods,
            trend='add',
            seasonal='add'
        )
        model_fit = model.fit(optimized=True)
        train_time = time.time() - train_start
        
        if logger:
            logger.info(f"Holt-Winters (additive) training completed in {train_time:.4f}s")
        
        return model_fit, train_time
    
    except:
        try:
            # Fallback to multiplicative
            model = ExponentialSmoothing(
                train_series,
                seasonal_periods=seasonal_periods,
                trend='add',
                seasonal='mul'
            )
            model_fit = model.fit(optimized=True)
            train_time = time.time() - train_start
            
            if logger:
                logger.info(f"Holt-Winters (multiplicative) training completed in {train_time:.4f}s")
            
            return model_fit, train_time
        
        except Exception as e:
            if logger:
                logger.warning(f"Holt-Winters training failed: {str(e)}")
            return None, time.time() - train_start


def select_best_es_model(train_series, test_series, logger=None):
    """
    Select best exponential smoothing model based on validation error
    
    Returns:
        best_model: Best fitted model
        best_model_name: Name of best model
        train_time: Total training time
    """
    seasonal_period = check_seasonality(train_series, logger)
    
    models = []
    
    # Train Simple Exponential Smoothing
    ses_model, ses_time = train_simple_exponential_smoothing(train_series, logger)
    models.append(('SES', ses_model, ses_time))
    
    # Train Holt's Linear Trend
    holt_model, holt_time = train_holt_model(train_series, logger)
    models.append(('Holt', holt_model, holt_time))
    
    # Train Holt-Winters if seasonal pattern detected
    if seasonal_period and len(train_series) >= 2 * seasonal_period:
        hw_model, hw_time = train_holt_winters_model(train_series, seasonal_period, logger)
        if hw_model is not None:
            models.append(('HoltWinters', hw_model, hw_time))
    
    # Evaluate models on validation set
    best_mse = float('inf')
    best_model = None
    best_model_name = None
    total_train_time = sum(t for _, _, t in models)
    
    validation_size = min(len(test_series), len(train_series) // 10)
    
    for model_name, model_fit, _ in models:
        try:
            forecast = model_fit.forecast(steps=validation_size)
            mse = mean_squared_error(test_series[:validation_size], forecast)
            
            if logger:
                logger.info(f"{model_name} validation MSE: {mse:.4f}")
            
            if mse < best_mse:
                best_mse = mse
                best_model = model_fit
                best_model_name = model_name
        
        except Exception as e:
            if logger:
                logger.warning(f"{model_name} validation failed: {str(e)}")
    
    if best_model is None:
        best_model = models[0][1]
        best_model_name = models[0][0]
    
    if logger:
        logger.info(f"Selected model: {best_model_name} (MSE: {best_mse:.4f})")
    
    return best_model, best_model_name, total_train_time


def predict_exponential_smoothing(model, steps, logger=None):
    """
    Make predictions using exponential smoothing model
    
    Returns:
        predictions: Numpy array of predictions
    """
    import time
    
    pred_start = time.time()
    forecast = model.forecast(steps=steps)
    pred_time = time.time() - pred_start
    
    if logger:
        logger.info(f"Prediction completed in {pred_time:.4f}s")
    
    # Round to integers (invocation counts)
    predictions = np.round(forecast.values).astype(int)
    predictions = np.maximum(predictions, 0)  # Ensure non-negative
    
    return predictions, pred_time


def run_exponential_smoothing_prediction(cluster_file, logger=None):
    """
    Complete exponential smoothing prediction pipeline for a cluster
    
    Returns:
        predictions: Numpy array of predictions
        model_name: Name of selected model
        train_time: Training time
        pred_time: Prediction time
    """
    # Load data
    train_df, test_df = load_cluster_data(cluster_file)
    train_series = train_df['y'].values
    test_series = test_df['y'].values
    
    # Select and train best model
    model, model_name, train_time = select_best_es_model(train_series, test_series, logger)
    
    # Make predictions
    predictions, pred_time = predict_exponential_smoothing(model, len(test_series), logger)
    
    return predictions, model_name, train_time, pred_time
