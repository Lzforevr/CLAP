"""
Main Script for Hybrid Temporal Network Learning
Combines LSTM for complex patterns and Exponential Smoothing for simple patterns
"""

import numpy as np
import pandas as pd
from pathlib import Path
import logging
import time
import config
from pattern_classifier import classify_clusters, load_ground_truth
from lstm_predictor import (
    load_cluster_data as load_lstm_data,
    prepare_lstm_data,
    LSTMModel,
    train_lstm_model,
    predict_lstm,
    save_lstm_model
)
from exponential_smoothing_predictor import run_exponential_smoothing_prediction


def setup_prediction_logging():
    """Setup logging for prediction module"""
    config.LOG_DIR.mkdir(parents=True, exist_ok=True)
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        filename=config.LOG_DIR / 'hybrid_prediction.log'
    )
    
    logger = logging.getLogger(__name__)
    return logger


def load_cluster_classifications(classification_file=None):
    """
    Load cluster pattern classifications
    
    Returns:
        Dictionary mapping cluster_id to pattern type ('simple' or 'complex')
    """
    if classification_file is None:
        classification_file = config.CLASSIFICATION_RESULTS_DIR / "classification_results.csv"
    
    if not classification_file.exists():
        raise FileNotFoundError(
            f"Classification results not found: {classification_file}\n"
            "Please run pattern_classifier.py first."
        )
    
    df = pd.read_csv(classification_file)
    classifications = dict(zip(df['cluster_id'], df['predicted']))
    return classifications


def train_lstm_for_complex_clusters(complex_cluster_files, logger):
    """
    Train LSTM model for all complex clusters
    
    Returns:
        model: Trained LSTM model
        num_clusters: Number of clusters
    """
    logger.info("="*60)
    logger.info("Training LSTM for Complex Clusters")
    logger.info("="*60)
    
    # Load all cluster data
    all_train_dfs = []
    all_test_dfs = []
    
    for file in complex_cluster_files:
        train_df, test_df = load_lstm_data(file)
        all_train_dfs.append(train_df)
        all_test_dfs.append(test_df)
    
    combined_train_df = pd.concat(all_train_dfs, ignore_index=True)
    combined_test_df = pd.concat(all_test_dfs, ignore_index=True)
    
    num_clusters = len(complex_cluster_files)
    logger.info(f"Training on {num_clusters} complex clusters")
    logger.info(f"Training samples: {len(combined_train_df)}")
    
    # Prepare data
    train_loader, test_loader, scalers = prepare_lstm_data(combined_train_df, combined_test_df)
    
    # Create model
    model = LSTMModel(
        config.LSTM_INPUT_SIZE,
        config.LSTM_HIDDEN_SIZE,
        config.LSTM_NUM_LAYERS,
        config.LSTM_OUTPUT_SIZE,
        num_clusters
    )
    model.to(config.DEVICE)
    
    # Train model
    criterion = torch.nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=config.LSTM_LEARNING_RATE)
    
    train_start = time.time()
    train_lstm_model(model, train_loader, criterion, optimizer, config.LSTM_EPOCHS, logger)
    train_time = time.time() - train_start
    
    logger.info(f"LSTM training completed in {train_time:.2f}s")
    
    # Save model
    save_lstm_model(model)
    logger.info(f"Model saved to {config.LSTM_MODEL_PATH}")
    
    return model, num_clusters


def predict_all_clusters(logger):
    """
    Main prediction pipeline: classify clusters and apply appropriate models
    
    Returns:
        results_df: DataFrame with prediction results and metrics
    """
    logger.info("\n" + "="*70)
    logger.info("CLAP Hybrid Prediction Module")
    logger.info("="*70)
    
    # Load classifications
    logger.info("\nStep 1: Loading Cluster Classifications")
    classifications = load_cluster_classifications()
    logger.info(f"Loaded classifications for {len(classifications)} clusters")
    
    simple_count = sum(1 for v in classifications.values() if v == 'simple')
    complex_count = len(classifications) - simple_count
    logger.info(f"  Simple clusters: {simple_count}")
    logger.info(f"  Complex clusters: {complex_count}")
    
    # Get cluster files
    cluster_files = sorted(config.CLUSTERED_DATA_DIR.glob("cluster-*_merged_data.csv"))
    
    simple_files = []
    complex_files = []
    
    for file in cluster_files:
        cluster_id = file.stem.split('_')[0]
        pattern = classifications.get(cluster_id, 'complex')  # Default to complex if unknown
        
        if pattern == 'simple':
            simple_files.append(file)
        else:
            complex_files.append(file)
    
    logger.info(f"\nFound {len(simple_files)} simple and {len(complex_files)} complex cluster files")
    
    # Train LSTM for complex clusters
    if complex_files:
        logger.info("\nStep 2: Training LSTM for Complex Clusters")
        lstm_model, num_clusters = train_lstm_for_complex_clusters(complex_files, logger)
    else:
        logger.warning("No complex clusters found!")
        lstm_model = None
    
    # Predict for all clusters
    logger.info("\nStep 3: Making Predictions")
    results = []
    
    # Predict simple clusters using Exponential Smoothing
    logger.info("\nPredicting Simple Clusters (Exponential Smoothing):")
    for file in simple_files:
        cluster_id = file.stem.split('_')[0]
        logger.info(f"  Processing {cluster_id}...")
        
        try:
            predictions, model_name, train_time, pred_time = \
                run_exponential_smoothing_prediction(file, logger)
            
            # Save predictions
            pred_df = pd.DataFrame({
                'cluster_id': cluster_id,
                'prediction': predictions
            })
            pred_file = config.PREDICTION_RESULTS_DIR / f"{cluster_id}_predictions.csv"
            pred_df.to_csv(pred_file, index=False)
            
            results.append({
                'cluster_id': cluster_id,
                'pattern': 'simple',
                'model': model_name,
                'train_time': train_time,
                'pred_time': pred_time,
                'num_predictions': len(predictions)
            })
            
        except Exception as e:
            logger.error(f"    Failed: {str(e)}")
    
    # Predict complex clusters using LSTM
    if lstm_model and complex_files:
        logger.info("\nPredicting Complex Clusters (LSTM):")
        
        for file in complex_files:
            cluster_id = file.stem.split('_')[0]
            cluster_idx = int(cluster_id.split('-')[1])
            logger.info(f"  Processing {cluster_id}...")
            
            try:
                train_df, test_df = load_lstm_data(file)
                _, test_loader, scalers = prepare_lstm_data(train_df, test_df)
                
                pred_start = time.time()
                predictions = predict_lstm(lstm_model, test_loader, scalers, target_cluster_id=cluster_idx)
                pred_time = time.time() - pred_start
                
                # Save predictions
                pred_df = pd.DataFrame({
                    'cluster_id': cluster_id,
                    'prediction': predictions
                })
                pred_file = config.PREDICTION_RESULTS_DIR / f"{cluster_id}_predictions.csv"
                pred_df.to_csv(pred_file, index=False)
                
                results.append({
                    'cluster_id': cluster_id,
                    'pattern': 'complex',
                    'model': 'LSTM',
                    'train_time': 0,  # Shared training time
                    'pred_time': pred_time,
                    'num_predictions': len(predictions)
                })
                
            except Exception as e:
                logger.error(f"    Failed: {str(e)}")
    
    # Save summary results
    results_df = pd.DataFrame(results)
    summary_file = config.PREDICTION_RESULTS_DIR / "prediction_summary.csv"
    results_df.to_csv(summary_file, index=False)
    
    logger.info("\n" + "="*70)
    logger.info("Prediction Summary:")
    logger.info(f"  Total clusters processed: {len(results)}")
    logger.info(f"  Simple (ES): {sum(1 for r in results if r['pattern'] == 'simple')}")
    logger.info(f"  Complex (LSTM): {sum(1 for r in results if r['pattern'] == 'complex')}")
    logger.info(f"  Results saved to: {summary_file}")
    logger.info("="*70)
    
    return results_df


if __name__ == "__main__":
    import torch
    
    print(f"Using device: {config.DEVICE}")
    
    logger = setup_prediction_logging()
    results = predict_all_clusters(logger)
    
    print("\nHybrid prediction completed!")
    print(f"Results saved to: {config.PREDICTION_RESULTS_DIR}")
