"""
CLAP: Cluster-guided Learning for Adaptive Prediction

A hybrid time series forecasting framework for serverless function invocation prediction.

Modules:
- feature_extraction: Multifaceted feature extraction from invocation sequences
- hierarchical_clustering: Optimal hierarchical clustering with adaptive selection
- pattern_classifier: Rule-based binary classifier for pattern complexity
- lstm_predictor: LSTM model with cluster embeddings for complex patterns
- exponential_smoothing_predictor: Adaptive ES models for simple patterns
"""

__version__ = "1.0.0"
__author__ = "CLAP Development Team"

from . import config
from . import feature_extraction
from . import data_preprocessing
from . import hierarchical_clustering
from . import clustering_utils
from . import pattern_classifier
from . import lstm_predictor
from . import exponential_smoothing_predictor

__all__ = [
    'config',
    'feature_extraction',
    'data_preprocessing',
    'hierarchical_clustering',
    'clustering_utils',
    'pattern_classifier',
    'lstm_predictor',
    'exponential_smoothing_predictor'
]
