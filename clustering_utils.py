"""
Utility Functions for Clustering Module
"""

import logging
import pandas as pd
from pathlib import Path
import config


def setup_logging(log_file=None):
    """
    Configure logging system
    
    Args:
        log_file: Path to log file
        
    Returns:
        Logger instance
    """
    if log_file is None:
        log_file = config.LOG_DIR / "clustering.log"
    
    config.LOG_DIR.mkdir(exist_ok=True, parents=True)
    
    logging.basicConfig(
        level=logging.INFO, 
        format='%(asctime)s - %(message)s',
        filename=log_file
    )
    
    logger = logging.getLogger(__name__)
    return logger


def load_trigger_mapping(file_path=None):
    """
    Load funcID to triggerType mapping
    
    Args:
        file_path: Path to metadata CSV file
        
    Returns:
        Dictionary mapping funcID to triggerType
    """
    if file_path is None:
        file_path = config.META_FILE
    
    meta_df = pd.read_csv(file_path, usecols=['funcID', 'triggerType-invocationType'])
    trigger_mapping = meta_df.set_index('funcID')['triggerType-invocationType'].to_dict()
    return trigger_mapping


def export_cluster_results(clusters, daily_pattern_pivot, trigger_mapping, output_dir=None):
    """
    Export clustering results to CSV files
    
    Args:
        clusters: Cluster assignments
        daily_pattern_pivot: Original data pivot table
        trigger_mapping: funcID to triggerType mapping
        output_dir: Output directory
    """
    if output_dir is None:
        output_dir = config.CLUSTER_RESULTS_DIR
    
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True, parents=True)
    
    # Export overall results
    results = pd.DataFrame({
        'funcID': daily_pattern_pivot.index,
        'cluster': clusters,
        'triggerType': [trigger_mapping.get(func_id, 'Unknown') 
                       for func_id in daily_pattern_pivot.index]
    })
    results.to_csv(output_dir / "final_cluster_results.csv", index=False)
    
    # Export individual cluster results
    for cluster in range(len(pd.unique(clusters))):
        cluster_results = results[results['cluster'] == cluster]
        cluster_results.to_csv(
            output_dir / f"final_cluster-{cluster}-results.csv", 
            index=False
        )
        
        # Export cluster statistics
        cluster_stats = {
            'cluster': cluster,
            'num_functions': len(cluster_results),
            'trigger_types': cluster_results['triggerType'].value_counts().to_dict(),
        }
        with open(output_dir / f"final_cluster-{cluster}-stats.txt", 'w') as f:
            for key, value in cluster_stats.items():
                f.write(f"{key}: {value}\n")
