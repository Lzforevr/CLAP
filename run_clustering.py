"""
Main Script for Hierarchical Clustering Analysis
Module 2: Optimal Hierarchical Clustering
"""

import time
from pathlib import Path
import config
from data_preprocessing import (
    read_and_preprocess_data, 
    convert_to_long_format,
    build_daily_pattern,
    filter_low_total_calls
)
from feature_extraction import extract_features, reduce_and_combine_features
from hierarchical_clustering import (
    find_optimal_clusters_adaptive,
    print_cluster_info
)
from clustering_utils import setup_logging, load_trigger_mapping, export_cluster_results


def run_hierarchical_clustering(data_dir=None, meta_file=None, output_dir=None, 
                               max_clusters=None, min_threshold=None):
    """
    Run complete hierarchical clustering pipeline
    
    Args:
        data_dir: Directory containing invocation CSV files
        meta_file: Metadata file with funcID to triggerType mapping
        output_dir: Output directory for results
        max_clusters: Maximum number of clusters
        min_threshold: Minimum call volume threshold
        
    Returns:
        clusters: Final clustering result
        best_k: Optimal number of clusters
    """
    # Use default values from config if not provided
    if data_dir is None:
        data_dir = config.DATA_DIR
    if meta_file is None:
        meta_file = config.META_FILE
    if output_dir is None:
        output_dir = config.CLUSTER_RESULTS_DIR
    if max_clusters is None:
        max_clusters = config.MAX_CLUSTERS
    if min_threshold is None:
        min_threshold = config.MIN_CALL_THRESHOLD
    
    # Setup logging
    logger = setup_logging()
    logger.info("=" * 60)
    logger.info("Starting Hierarchical Clustering Analysis")
    logger.info("=" * 60)
    
    try:
        # Create output directory
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
        
        # Load trigger type mapping
        trigger_mapping = load_trigger_mapping(meta_file)
        logger.info(f"Loaded {len(trigger_mapping)} function trigger mappings")
        
        # Step 1: Data Preprocessing
        logger.info("\nStep 1: Data Preprocessing")
        start_time = time.time()
        
        df = read_and_preprocess_data(Path(data_dir))
        df_funcs = df[[col for col in df.columns if col not in ['minute_code', 'day']]]
        
        df_long = convert_to_long_format(df[["minute_code"] + list(df_funcs.columns)])
        daily_pattern_pivot = build_daily_pattern(df_long)
        daily_pattern_pivot = filter_low_total_calls(daily_pattern_pivot, threshold=min_threshold)
        
        logger.info(f"Preprocessing completed in {time.time() - start_time:.2f}s")
        logger.info(f"Functions after filtering: {len(daily_pattern_pivot)}")
        
        # Step 2: Feature Extraction
        logger.info("\nStep 2: Feature Extraction")
        start_time = time.time()
        
        (total_call, shape_features, calls_std, calls_mean, calls_median, 
         calls_max, calls_min, fft_peak, fft_freq, 
         anomaly_ratios, burst_ratios) = extract_features(daily_pattern_pivot)
        
        combined_features, shape_features_pca = reduce_and_combine_features(
            total_call, shape_features, calls_std, calls_mean, calls_median,
            calls_max, calls_min, fft_peak, fft_freq,
            anomaly_ratios, burst_ratios
        )

        logger.info(f"Feature extraction completed in {time.time() - start_time:.2f}s")
        logger.info(f"Feature dimensions: {combined_features.shape[1]}")
        
        # Step 3: Clustering Analysis
        logger.info("\nStep 3: Clustering Analysis")
        start_time = time.time()
        
        clusters, best_k, best_silhouette, linkage_matrix, all_clusters = find_optimal_clusters_adaptive(
            combined_features, 
            min_clusters=config.MIN_CLUSTERS, 
            max_clusters=max_clusters,
            output_dir=output_dir
        )
        
        logger.info(f"Clustering completed in {time.time() - start_time:.2f}s")
        logger.info(f"Optimal clusters: {best_k}, Score: {best_silhouette:.4f}")
        
        # Print cluster information
        print_cluster_info(best_k, combined_features, clusters, total_call)
        
        # Step 4: Export Results
        logger.info("\nStep 4: Exporting Results")
        start_time = time.time()
        
        export_cluster_results(clusters, daily_pattern_pivot, trigger_mapping, output_dir)
        
        logger.info(f"Results exported in {time.time() - start_time:.2f}s")
        logger.info(f"Results saved to: {output_dir}")
        
        logger.info("\n" + "=" * 60)
        logger.info("Hierarchical Clustering Analysis Completed Successfully")
        logger.info("=" * 60)
        
        return clusters, best_k
    
    except Exception as e:
        logger.error(f"Error during clustering: {str(e)}")
        raise


if __name__ == "__main__":
    # Run clustering with default configuration
    clusters, best_k = run_hierarchical_clustering()
    print(f"\nClustering completed: {best_k} clusters found")
    print(f"Results saved to: {config.CLUSTER_RESULTS_DIR}")
