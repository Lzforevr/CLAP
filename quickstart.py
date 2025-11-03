"""
Quick Start Script for CLAP Framework
Demonstrates the complete workflow from clustering to prediction
"""

import sys
from pathlib import Path

# Add CLAP to path
sys.path.insert(0, str(Path(__file__).parent))

import config
from run_clustering import run_hierarchical_clustering
from pattern_classifier import run_pattern_classification
from run_hybrid_prediction import predict_all_clusters, setup_prediction_logging


def main():
    """Run complete CLAP pipeline"""
    
    print("="*70)
    print("CLAP Framework - Quick Start")
    print("="*70)
    
    # Ensure directories exist
    config.ensure_directories()
    
    # Step 1: Hierarchical Clustering
    print("\n" + "="*70)
    print("STEP 1: Hierarchical Clustering")
    print("="*70)
    
    try:
        clusters, best_k = run_hierarchical_clustering()
        print(f"\n✓ Clustering completed: {best_k} clusters discovered")
    except Exception as e:
        print(f"\n✗ Clustering failed: {e}")
        print("Please check your data in the 'data/raw_requests' directory")
        return
    
    # Step 2: Pattern Classification
    print("\n" + "="*70)
    print("STEP 2: Pattern Classification")
    print("="*70)
    
    try:
        results = run_pattern_classification()
        print(f"\n✓ Classification completed: {len(results)} clusters classified")
    except Exception as e:
        print(f"\n✗ Classification failed: {e}")
        print("Please ensure 'data/cluster_pattern.csv' exists with ground truth labels")
        print("Or check that 'data/clustered_requests/' contains cluster data")
        return
    
    # Step 3: Hybrid Prediction
    print("\n" + "="*70)
    print("STEP 3: Hybrid Prediction")
    print("="*70)
    
    try:
        logger = setup_prediction_logging()
        pred_results = predict_all_clusters(logger)
        print(f"\n✓ Prediction completed: {len(pred_results)} clusters predicted")
    except Exception as e:
        print(f"\n✗ Prediction failed: {e}")
        return
    
    # Summary
    print("\n" + "="*70)
    print("CLAP Pipeline Completed Successfully!")
    print("="*70)
    print(f"\nResults saved to: {config.OUTPUT_DIR}")
    print(f"  - Clustering results: {config.CLUSTER_RESULTS_DIR}")
    print(f"  - Classification results: {config.CLASSIFICATION_RESULTS_DIR}")
    print(f"  - Prediction results: {config.PREDICTION_RESULTS_DIR}")
    print(f"  - Visualizations: {config.VISUALIZATION_DIR}")
    print(f"  - Logs: {config.LOG_DIR}")
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
