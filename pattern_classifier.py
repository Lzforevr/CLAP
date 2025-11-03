"""
Module 3a: Cluster Pattern Classifier
Rule-based binary classifier for distinguishing simple vs complex patterns
Combines time-domain, frequency-domain, and statistical features
"""

from __future__ import annotations
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, List, Tuple, Any, Optional
from tqdm import tqdm
from scipy.stats import skew, kurtosis
from scipy import signal
from scipy.stats import entropy as scipy_entropy
from scipy.fft import fft
import config


# ========================================================================
# Part 1: Feature Computation Functions (from patternAna2.py)
# ========================================================================

def compute_time_domain_features(x: np.ndarray) -> Dict[str, float]:
    """Compute time-domain statistical features"""
    x = x[x != 0]
    mean_val = np.mean(x)
    std_val = np.std(x)
    
    cv = std_val / (abs(mean_val) + 1e-10)
    
    # Segment-based stability analysis
    n_segments = 10
    segment_size = len(x) // n_segments
    segment_means = []
    segment_stds = []
    
    for i in range(n_segments):
        seg = x[i*segment_size:(i+1)*segment_size]
        if len(seg) > 0:
            segment_means.append(np.mean(seg))
            segment_stds.append(np.std(seg))
    
    mean_stability = np.std(segment_means) / (abs(np.mean(segment_means)) + 1e-10)
    std_stability = np.std(segment_stds) / (abs(np.mean(segment_stds)) + 1e-10)
    
    # Trend analysis
    if len(x) > 1:
        time_indices = np.arange(len(x))
        trend_coef = np.polyfit(time_indices, x, 1)[0]
        trend_strength = abs(trend_coef) / (abs(mean_val) + 1e-10)
    else:
        trend_strength = 0.0
    
    # Autocorrelation
    if len(x) > 1:
        autocorr = np.corrcoef(x[:-1], x[1:])[0, 1]
        autocorr_lag1 = autocorr if not np.isnan(autocorr) else 0.0
    else:
        autocorr_lag1 = 0.0
    
    return {
        'mean': mean_val,
        'std': std_val,
        'coefficient_of_variation': cv,
        'mean_stability': mean_stability,
        'std_stability': std_stability,
        'trend_strength': trend_strength,
        'autocorr_lag1': autocorr_lag1
    }


def compute_magnitude_spectrum(x, fs=1.0, nperseg=2048, noverlap=None, window='hann'):
    """Compute magnitude spectrum using Welch's method"""
    if noverlap is None:
        noverlap = nperseg // 2
    
    try:
        f, Pxx = signal.welch(x, fs=fs, nperseg=min(nperseg, len(x)), 
                             noverlap=min(noverlap, len(x)//2), window=window)
        Mxx = np.sqrt(Pxx)  # Magnitude spectrum
        return f, Mxx
    except:
        return np.array([0]), np.array([0])


def spectral_entropy_magnitude(Mxx):
    """Compute spectral entropy from magnitude spectrum"""
    Mxx_norm = Mxx / (np.sum(Mxx) + 1e-12)
    return scipy_entropy(Mxx_norm + 1e-12)


def peak_energy_ratio_magnitude(Mxx, top_k=1):
    """Compute peak energy ratio"""
    sorted_mag = np.sort(Mxx)[::-1]
    top_energy = np.sum(sorted_mag[:top_k])
    total_energy = np.sum(Mxx)
    return top_energy / (total_energy + 1e-12)


def number_of_significant_peaks_magnitude(f, Mxx, rel_thresh=0.05):
    """Count significant peaks in magnitude spectrum"""
    peaks, _ = signal.find_peaks(Mxx, height=rel_thresh * np.max(Mxx))
    return len(peaks)


def extract_spectral_features_magnitude(x, fs=1.0, nperseg=2048, topk=3):
    """Extract basic spectral features"""
    f, Mxx = compute_magnitude_spectrum(x, fs, nperseg)
    
    features = {
        'PER': peak_energy_ratio_magnitude(Mxx, top_k=topk),
        'SpecEntropy': spectral_entropy_magnitude(Mxx),
        'NumPeaks': number_of_significant_peaks_magnitude(f, Mxx, rel_thresh=0.05),
        'Flatness': scipy_entropy(Mxx + 1e-12) / np.log(len(Mxx) + 1e-12)
    }
    return features


def enhanced_spectral_features_magnitude(x, fs=1.0, nperseg=2048):
    """Extract enhanced spectral features"""
    f, Mxx = compute_magnitude_spectrum(x, fs, nperseg)
    
    # Significant peaks
    num_peaks = number_of_significant_peaks_magnitude(f, Mxx, rel_thresh=0.05)
    
    # Peak energy concentration
    peak_concentration = peak_energy_ratio_magnitude(Mxx, top_k=3)
    
    # Local flatness
    window_size = max(5, len(Mxx) // 20)
    local_flatness_vals = []
    for i in range(0, len(Mxx) - window_size + 1, window_size // 2):
        window = Mxx[i:i+window_size]
        geom_mean = np.exp(np.mean(np.log(window + 1e-12)))
        arith_mean = np.mean(window)
        local_flatness_vals.append(geom_mean / (arith_mean + 1e-12))
    avg_local_flatness = np.mean(local_flatness_vals) if local_flatness_vals else 0.0
    
    # Sparsity
    threshold = 0.1 * np.max(Mxx)
    sparsity_ratio = np.sum(Mxx < threshold) / len(Mxx)
    
    return {
        'num_significant_peaks': num_peaks,
        'peak_energy_concentration': peak_concentration,
        'avg_local_flatness': avg_local_flatness,
        'sparsity_ratio': sparsity_ratio
    }


# ========================================================================
# Part 2: Pattern Classification & Optimization (from optimize_pattern_classifier.py)
# ========================================================================

def load_ground_truth(path: Path = None) -> Dict[str, str]:
    """Load ground truth cluster pattern labels"""
    if path is None:
        path = config.CLUSTER_PATTERN_FILE
    
    if not path.exists():
        raise FileNotFoundError(f"Ground truth file not found: {path}")
    
    gt_df = pd.read_csv(path)
    mapping = dict(zip(gt_df['cluster_id'], gt_df['pattern']))
    return mapping


def extract_all_features(data_dir: Path = None) -> pd.DataFrame:
    """
    Extract all classification features from cluster data
    
    Returns:
        DataFrame with cluster_id, label, and all feature columns
    """
    if data_dir is None:
        data_dir = config.CLUSTERED_DATA_DIR
    
    gt_map = load_ground_truth()
    rows: List[Dict[str, Any]] = []

    if not data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {data_dir}")

    files = sorted(data_dir.glob("cluster-*_merged_data.csv"))
    if not files:
        files = sorted(data_dir.glob("*.csv"))
    
    for f in tqdm(files, desc="Extracting features"):
        try:
            df = pd.read_csv(f, usecols=["ds", "y"])
            series = df['y'].to_numpy()
            stem = f.stem
            cluster_id = stem.split('_')[0]

            time_feats = compute_time_domain_features(series)
            base_feats = extract_spectral_features_magnitude(series, fs=1440.0)
            enh_feats = enhanced_spectral_features_magnitude(series, fs=1440.0)

            # Compute skewness and kurtosis
            vals = series.astype(float)
            if len(vals) > 0:
                skew_v = float(skew(vals))
                kurt_v = float(kurtosis(vals, fisher=False))  # Pearson's kurtosis
            else:
                skew_v = 0.0
                kurt_v = 3.0

            row = {
                'cluster_id': cluster_id,
                'label': gt_map.get(cluster_id),
                'cv': time_feats['coefficient_of_variation'],
                'mean_stability': time_feats['mean_stability'],
                'std_stability': time_feats['std_stability'],
                'trend_strength': time_feats['trend_strength'],
                'autocorr_lag1': time_feats['autocorr_lag1'],
                'skewness': abs(skew_v),
                'kurtosis': kurt_v,
                'PER': base_feats['PER'],
                'SpecEntropy': base_feats['SpecEntropy'],
                'NumPeaks_basic': base_feats['NumPeaks'],
                'Flatness_base': base_feats['Flatness'],
                'num_significant_peaks': enh_feats['num_significant_peaks'],
                'peak_energy_concentration': enh_feats['peak_energy_concentration'],
                'avg_local_flatness': enh_feats['avg_local_flatness'],
                'sparsity_ratio': enh_feats['sparsity_ratio'],
            }
            rows.append(row)
        except Exception as e:
            print(f"[WARN] Failed to process {f.name}: {e}")

    feat_df = pd.DataFrame(rows)
    
    if 'label' in feat_df.columns:
        before = len(feat_df)
        feat_df = feat_df.dropna(subset=['label'])
        after = len(feat_df)
        if after == 0:
            raise ValueError("All samples missing ground truth labels!")
        print(f"[INFO] Dropped {before-after} unlabeled samples, kept {after} samples.")
    
    return feat_df


def optimize_single_feature_threshold(df: pd.DataFrame, feature: str, 
                                     direction: str) -> Dict[str, Any]:
    """
    Find optimal threshold for a single feature
    
    Args:
        df: DataFrame with features and labels
        feature: Feature name
        direction: 'low', 'high', or 'range'
        
    Returns:
        Dictionary with optimal threshold and accuracy
    """
    y_true = (df['label'] == 'simple').astype(int).values
    values = df[feature].fillna(df[feature].median()).to_numpy()
    
    if direction in ('low', 'high'):
        # Test quantile-based thresholds
        quantiles = np.linspace(0.05, 0.95, 19)
        candidates = np.quantile(values, quantiles)
        
        best_acc = -1
        best_t = None
        
        for t in candidates:
            if direction == 'low':
                y_pred = (values <= t).astype(int)
            else:
                y_pred = (values >= t).astype(int)
            acc = (y_pred == y_true).mean()
            if acc > best_acc:
                best_acc = acc
                best_t = t
        
        return {
            'feature': feature,
            'direction': direction,
            'threshold': float(best_t),
            'accuracy': float(best_acc)
        }
    
    elif direction == 'range':
        # Test range-based thresholds
        vmin, vmax = int(values.min()), int(values.max())
        best_acc = -1
        best_range = (vmin, vmax)
        
        for lo in range(vmin, vmax + 1):
            for hi in range(lo, min(vmax + 1, lo + 20)):
                y_pred = ((values >= lo) & (values <= hi)).astype(int)
                acc = (y_pred == y_true).mean()
                if acc > best_acc:
                    best_acc = acc
                    best_range = (lo, hi)
        
        return {
            'feature': feature,
            'direction': direction,
            'threshold': best_range,
            'accuracy': float(best_acc)
        }


def greedy_feature_selection(df: pd.DataFrame, 
                            feature_definitions: Dict = None) -> Tuple[List[str], Dict[str, Any]]:
    """
    Greedy feature selection based on accuracy improvement
    
    Returns:
        selected_features: List of selected feature names
        feature_rules: Dictionary of optimal rules for each feature
    """
    if feature_definitions is None:
        feature_definitions = config.FEATURE_DEFINITIONS
    
    print("\n" + "="*60)
    print("Starting Greedy Feature Selection")
    print("="*60)
    
    # Evaluate all features individually
    single_results = []
    for feat_name, feat_config in feature_definitions.items():
        if feat_name not in df.columns:
            continue
        
        result = optimize_single_feature_threshold(
            df, feat_name, feat_config['direction']
        )
        single_results.append(result)
        print(f"{feat_name:30s} | Acc: {result['accuracy']:.4f}")
    
    # Sort by accuracy
    single_results.sort(key=lambda x: x['accuracy'], reverse=True)
    
    # Greedy selection
    selected_features = []
    feature_rules = {}
    best_acc = 0.0
    
    for result in single_results:
        feat_name = result['feature']
        
        # Try adding this feature
        test_features = selected_features + [feat_name]
        test_rules = {**feature_rules, feat_name: result}
        
        # Evaluate combined performance
        y_true = (df['label'] == 'simple').astype(int).values
        votes = np.zeros(len(df))
        
        for fname, rule in test_rules.items():
            values = df[fname].fillna(df[fname].median()).to_numpy()
            direction = rule['direction']
            threshold = rule['threshold']
            
            if direction == 'low':
                votes += (values <= threshold).astype(int)
            elif direction == 'high':
                votes += (values >= threshold).astype(int)
            elif direction == 'range':
                lo, hi = threshold
                votes += ((values >= lo) & (values <= hi)).astype(int)
        
        # Simple voting: >= half of features agree
        y_pred = (votes >= len(test_features) / 2).astype(int)
        acc = (y_pred == y_true).mean()
        
        # Accept if improvement is significant
        if acc > best_acc + 1e-4:
            selected_features.append(feat_name)
            feature_rules[feat_name] = result
            best_acc = acc
            print(f"  ✓ Added {feat_name:25s} | New Acc: {acc:.4f}")
        else:
            print(f"  ✗ Skipped {feat_name:25s} | Would be: {acc:.4f}")
    
    print("="*60)
    print(f"Final selected features: {len(selected_features)}")
    print(f"Final accuracy: {best_acc:.4f}")
    print("="*60)
    
    return selected_features, feature_rules


def classify_clusters(df: pd.DataFrame, feature_rules: Dict[str, Any], 
                     decision_fraction: float = 0.5) -> pd.DataFrame:
    """
    Classify clusters using learned rules
    
    Args:
        df: DataFrame with features
        feature_rules: Dictionary of classification rules
        decision_fraction: Voting threshold
        
    Returns:
        DataFrame with predictions
    """
    votes = np.zeros(len(df))
    
    for feat_name, rule in feature_rules.items():
        values = df[feat_name].fillna(df[feat_name].median()).to_numpy()
        direction = rule['direction']
        threshold = rule['threshold']
        
        if direction == 'low':
            votes += (values <= threshold).astype(int)
        elif direction == 'high':
            votes += (values >= threshold).astype(int)
        elif direction == 'range':
            lo, hi = threshold
            votes += ((values >= lo) & (values <= hi)).astype(int)
    
    required_votes = decision_fraction * len(feature_rules)
    predictions = (votes >= required_votes).astype(int)
    predicted_labels = ['simple' if p == 1 else 'complex' for p in predictions]
    
    result_df = df.copy()
    result_df['predicted'] = predicted_labels
    result_df['votes'] = votes
    result_df['required_votes'] = required_votes
    
    return result_df


def run_pattern_classification(output_csv: Path = None) -> pd.DataFrame:
    """
    Main function to run complete pattern classification pipeline
    
    Returns:
        DataFrame with classification results
    """
    if output_csv is None:
        output_csv = config.CLASSIFICATION_RESULTS_DIR / "classification_results.csv"
    
    print("\n" + "="*70)
    print("CLAP Pattern Classification Module")
    print("="*70)
    
    # Extract features
    print("\nStep 1: Feature Extraction")
    df = extract_all_features()
    print(f"Extracted features for {len(df)} clusters")
    
    # Feature selection
    print("\nStep 2: Greedy Feature Selection")
    selected_features, feature_rules = greedy_feature_selection(df)
    
    # Classify
    print("\nStep 3: Classification")
    results = classify_clusters(df, feature_rules, decision_fraction=0.5)
    
    # Evaluate
    if 'label' in results.columns:
        y_true = results['label'].values
        y_pred = results['predicted'].values
        accuracy = (y_true == y_pred).mean()
        print(f"\nClassification Accuracy: {accuracy:.4f}")
    
    # Save results
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    results.to_csv(output_csv, index=False)
    print(f"\nResults saved to: {output_csv}")
    
    # Save feature rules
    rules_file = output_csv.parent / "feature_rules.txt"
    with open(rules_file, 'w') as f:
        f.write("Selected Features and Rules:\n")
        f.write("="*60 + "\n")
        for feat_name, rule in feature_rules.items():
            f.write(f"\n{feat_name}:\n")
            f.write(f"  Direction: {rule['direction']}\n")
            f.write(f"  Threshold: {rule['threshold']}\n")
            f.write(f"  Accuracy: {rule['accuracy']:.4f}\n")
    
    print(f"Feature rules saved to: {rules_file}")
    
    return results


if __name__ == "__main__":
    results = run_pattern_classification()
    print("\nPattern classification completed successfully!")
