"""
Module 2: Optimal Hierarchical Clustering (OHC)
Implements adaptive hierarchical clustering with optimal cluster number selection
"""

import numpy as np
import pandas as pd
import logging
from sklearn.cluster import AgglomerativeClustering
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import silhouette_score, davies_bouldin_score
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist
import matplotlib.pyplot as plt
from pathlib import Path
import config

logger = logging.getLogger(__name__)


def perform_clustering(combined_features, n_clusters):
    """
    Perform hierarchical clustering
    
    Args:
        combined_features: Feature matrix
        n_clusters: Number of clusters
        
    Returns:
        clusters: Cluster labels
        model: Trained clustering model
    """
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    
    model = AgglomerativeClustering(
        n_clusters=n_clusters,
        metric=config.CLUSTERING_METRIC,
        linkage=config.CLUSTERING_LINKAGE,
        compute_distances=True
    )
    
    clusters = model.fit_predict(scaled_features)
    return clusters, model


def extract_all_levels_clusters(linkage_matrix, n_samples, max_clusters=None):
    """
    Extract clustering results for all hierarchical levels
    
    Args:
        linkage_matrix: Hierarchical linkage matrix
        n_samples: Number of samples
        max_clusters: Maximum number of clusters
        
    Returns:
        Dictionary of cluster assignments for each level
    """
    if max_clusters is None:
        max_clusters = config.MAX_CLUSTERS
    
    all_clusters = {}
    for num_clusters in range(2, max_clusters + 1):
        clusters = fcluster(linkage_matrix, t=num_clusters, criterion='maxclust')
        clusters = clusters - 1  # Convert to 0-indexed
        all_clusters[num_clusters] = clusters
    
    return all_clusters


def adaptive_cluster_selection(k_values, combined_scores, silhouette_scores, 
                               db_scores, all_clusters, alpha=0.5, beta=0.5, 
                               output_dir=None):
    """
    Select optimal cluster number using adaptive strategy
    
    Args:
        k_values: List of candidate cluster numbers
        combined_scores: Combined evaluation scores
        silhouette_scores: Silhouette scores
        db_scores: Davies-Bouldin scores
        all_clusters: Dictionary of clustering results
        alpha: Weight for score component
        beta: Weight for simplicity component
        output_dir: Directory for saving visualizations
        
    Returns:
        best_k: Optimal number of clusters
        best_clusters: Optimal clustering result
        best_score: Optimal combined score
    """
    if not k_values:
        return None, None, -1
    
    k_array = np.array(k_values)
    score_array = np.array(combined_scores)
    
    global_max_index = np.argmax(score_array)
    global_max_k = k_array[global_max_index]
    global_max_score = score_array[global_max_index]
    
    if len(k_array) < 5:
        best_k, best_score = global_max_k, global_max_score
    else:
        score_diffs = abs(score_array - global_max_score)
        candidate_indices = np.where((score_diffs < 0.02) & (k_array < global_max_k))[0]
        
        if len(candidate_indices) > 0:
            candidate_k_values = k_array[candidate_indices]
            candidate_scores = score_array[candidate_indices]
            weight_scores = alpha * candidate_scores + beta / candidate_k_values
            candidate_idx = np.argmax(weight_scores)
            best_k = candidate_k_values[candidate_idx]
            best_score = candidate_scores[candidate_idx]
        else:
            best_k = global_max_k
            best_score = global_max_score

    if output_dir:
        plot_cluster_selection(k_array, score_array, silhouette_scores, 
                             db_scores, best_k, output_dir)
    
    best_clusters = all_clusters.get(best_k)
    return best_k, best_clusters, best_score


def find_optimal_clusters_adaptive(combined_features, min_clusters=None, 
                                  max_clusters=None, output_dir=None):
    """
    Find optimal number of clusters using adaptive approach
    
    Args:
        combined_features: Feature matrix
        min_clusters: Minimum number of clusters
        max_clusters: Maximum number of clusters
        output_dir: Directory for saving results
        
    Returns:
        best_clusters: Optimal clustering result
        best_k: Optimal number of clusters
        best_silhouette: Best combined score
        linkage_matrix: Hierarchical linkage matrix
        all_clusters: All clustering results
    """
    if min_clusters is None:
        min_clusters = config.MIN_CLUSTERS
    if max_clusters is None:
        max_clusters = config.MAX_CLUSTERS
    
    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(exist_ok=True, parents=True)
    
    n_samples = combined_features.shape[0]
    adjusted_max = max_clusters
    
    logger.info(f"Cluster range: [{min_clusters}, {adjusted_max}]")
    
    # Compute linkage matrix
    scaler = StandardScaler()
    scaled_features = scaler.fit_transform(combined_features)
    dist_matrix = pdist(scaled_features, metric=config.CLUSTERING_METRIC)
    linkage_matrix = linkage(dist_matrix, method=config.CLUSTERING_LINKAGE)
    
    # Extract all levels
    all_clusters = extract_all_levels_clusters(linkage_matrix, n_samples, max_clusters=adjusted_max)
    
    # Evaluate each level
    k_values = []
    combined_scores = []
    silhouette_scores = []
    db_scores = []
    
    for k in range(min_clusters, adjusted_max + 1):
        try:
            if k not in all_clusters:
                continue
            
            clusters = all_clusters[k]
            unique_clusters = len(np.unique(clusters))
            if unique_clusters < 2:
                continue
                
            silhouette = silhouette_score(combined_features, clusters)
            db_score = davies_bouldin_score(combined_features, clusters)
            
            k_values.append(k)
            silhouette_scores.append(silhouette)
            db_scores.append(db_score)
            
            combined_score = silhouette / (1 + db_score) + 505 / unique_clusters
            combined_scores.append(combined_score)
            
        except Exception as e:
            logger.warning(f"Cluster {k} evaluation failed: {str(e)}")
            continue
    
    # Apply adaptive selection
    best_k, best_clusters, best_score = adaptive_cluster_selection(
        k_values, combined_scores, silhouette_scores, db_scores, 
        all_clusters, alpha=0.5, beta=0.5, output_dir=output_dir
    )
    
    if best_k is None:
        logger.warning("Could not find optimal clusters, using default k=5")
        best_k = 5
        best_clusters = all_clusters.get(5) or perform_clustering(combined_features, 5)[0]
        best_score = -1
    
    logger.info(f"Optimal clusters: {best_k}, Score: {best_score:.4f}")
    return best_clusters, best_k, best_score, linkage_matrix, all_clusters


def plot_cluster_selection(k_array, score_array, silhouette_scores, 
                          db_scores, best_k, output_dir):
    """Plot cluster selection visualization"""
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = ['Times New Roman']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    p1, = ax.plot(k_array, score_array, marker='o', linestyle='-', 
                 markersize=5, label='Combined Score')
    p2, = ax.plot(k_array, silhouette_scores, linestyle='--', 
                 label='Silhouette Score')
    ax.set_xlabel('Number of Clusters (k)', fontsize=14)
    ax.set_ylabel('Score', fontsize=14)
    ax.tick_params(axis='both', labelsize=12)

    ax2 = ax.twinx()
    p3, = ax2.plot(k_array, db_scores, linestyle='-.', color='green', 
                  label='DB Index')
    ax2.set_ylabel('DB Index', fontsize=14)
    ax2.tick_params(axis='y', labelsize=12)

    ax.axvline(x=best_k, color='crimson', linestyle=':', linewidth=2, 
              label=f'Selected k={best_k}')
    
    ax.invert_xaxis()
    ax2.invert_xaxis()

    lines = [p1, p2, p3]
    ax.legend(lines, [l.get_label() for l in lines], fontsize=12, loc='upper right')
    
    ax.grid(True, linestyle='--', linewidth=0.5)
    
    plt.tight_layout()
    plt.savefig(output_dir / "adaptive_cluster_selection.png", dpi=300, bbox_inches='tight')
    plt.close(fig)


def print_cluster_info(best_k, combined_features, clusters, total_call):
    """Print clustering statistics"""
    logger.info("=" * 50)
    logger.info("Cluster Analysis Results")
    logger.info("=" * 50)
    
    silhouette = silhouette_score(combined_features, clusters)
    logger.info(f"Optimal Clusters: {best_k}")
    logger.info(f"Silhouette Score: {silhouette:.4f}")
    
    cluster_counts = pd.Series(clusters).value_counts().sort_index()
    logger.info("\nCluster Size Distribution:")
    logger.info(cluster_counts.to_string())
    
    cluster_avg_calls = total_call.groupby(clusters).mean()
    logger.info("\nAverage Total Calls per Cluster:")
    logger.info(cluster_avg_calls.to_string())
    logger.info("=" * 50)
