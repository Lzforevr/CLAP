"""
CLAP Framework Configuration File
Centralized configuration for all paths and parameters
"""

from pathlib import Path

# Base directories (relative paths for portability)
BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parent

# Data directories
DATA_DIR = BASE_DIR / "data" / "raw_requests"
META_FILE = BASE_DIR / "data" / "meta_funcID_runtime_triggerType.csv"
CLUSTER_PATTERN_FILE = BASE_DIR / "data" / "cluster_pattern.csv"
CLUSTERED_DATA_DIR = BASE_DIR / "data" / "clustered_requests"

# Output directories
OUTPUT_DIR = BASE_DIR / "output"
CLUSTER_RESULTS_DIR = OUTPUT_DIR / "clustering_results"
CLASSIFICATION_RESULTS_DIR = OUTPUT_DIR / "classification_results"
PREDICTION_RESULTS_DIR = OUTPUT_DIR / "prediction_results"
LOG_DIR = OUTPUT_DIR / "logs"
VISUALIZATION_DIR = OUTPUT_DIR / "visualizations"

# Model directories
MODEL_DIR = BASE_DIR / "models"
LSTM_MODEL_PATH = MODEL_DIR / "lstm_model.pth"

# ============================================================
# Module 1: Feature Extraction Parameters
# ============================================================
MIN_CALL_THRESHOLD = 24 * 31  # Minimum total calls threshold (24 hours * 31 days)
PCA_VARIANCE_RATIO = 0.95  # PCA variance ratio to retain

# ============================================================
# Module 2: Hierarchical Clustering Parameters
# ============================================================
MIN_CLUSTERS = 3
MAX_CLUSTERS = 100  # Will be dynamically adjusted based on data size
CLUSTERING_METRIC = 'euclidean'
CLUSTERING_LINKAGE = 'ward'

# ============================================================
# Module 3a: Pattern Classification Parameters
# ============================================================
RANDOM_SEED = 2025
DECISION_FRACTIONS = [0.083, 0.167, 0.25, 0.333, 0.417, 0.5, 0.583, 0.667, 0.75, 0.833, 0.917, 1.0]

# Feature definitions for classification
FEATURE_DEFINITIONS = {
    'total_calls': {'direction': 'high', 'desc': 'Total number of calls'},
    'cv': {'direction': 'low', 'desc': 'Coefficient of Variation'},
    'mean_stability': {'direction': 'low'},
    'std_stability': {'direction': 'low'},
    'trend_strength': {'direction': 'low'},
    'autocorr_lag1': {'direction': 'high'},
    'skewness': {'direction': 'low'},
    'kurtosis': {'direction': 'low'},
    'PER': {'direction': 'high'},
    'SpecEntropy': {'direction': 'low'},
    'num_significant_peaks': {'direction': 'range'},
    'peak_energy_concentration': {'direction': 'high'},
    'avg_local_flatness': {'direction': 'low'},
    'sparsity_ratio': {'direction': 'low'},
}

RANGE_FEATURE_DEFAULT_BOUNDS = {
    'num_significant_peaks': (1, 12)
}

# ============================================================
# Module 3b: Prediction Model Parameters
# ============================================================

# Common parameters
TRAIN_END_DATE = '2025-01-28'  # Training set ends on this date
TEST_DAYS = 3  # Number of days reserved for testing

# Exponential Smoothing parameters
ES_SEASONAL_PERIODS_DAILY = 1440  # Minutes in a day for daily seasonality
ES_SEASONAL_PERIODS_HOURLY = 60   # Minutes in an hour for hourly seasonality

# LSTM parameters
LSTM_INPUT_SIZE = 1
LSTM_HIDDEN_SIZE = 128
LSTM_NUM_LAYERS = 4
LSTM_OUTPUT_SIZE = 1
LSTM_LOOK_BACK = 10
LSTM_BATCH_SIZE = 512
LSTM_LEARNING_RATE = 0.001
LSTM_EPOCHS = 15

# Device configuration
import torch
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

if DEVICE == "cuda":
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True
    torch.backends.cudnn.deterministic = False

# ============================================================
# Utility Functions
# ============================================================

def ensure_directories():
    """Create all necessary directories"""
    dirs = [
        DATA_DIR,
        OUTPUT_DIR,
        CLUSTER_RESULTS_DIR,
        CLASSIFICATION_RESULTS_DIR,
        PREDICTION_RESULTS_DIR,
        LOG_DIR,
        VISUALIZATION_DIR,
        MODEL_DIR,
        CLUSTERED_DATA_DIR
    ]
    for dir_path in dirs:
        dir_path.mkdir(parents=True, exist_ok=True)

if __name__ == "__main__":
    ensure_directories()
    print("CLAP configuration initialized successfully!")
    print(f"Base directory: {BASE_DIR}")
    print(f"Data directory: {DATA_DIR}")
    print(f"Output directory: {OUTPUT_DIR}")
