# CLAP: Cluster-guided Learning for Adaptive Prediction

**CLAP** (Cluster-guided Learning for Adaptive Prediction) is a hybrid time series forecasting framework designed for serverless function invocation prediction. It combines hierarchical clustering, pattern classification, and adaptive model selection to achieve accurate and efficient predictions.

## Overview

CLAP addresses the challenge of predicting invocation patterns for serverless functions by:
1. **Extracting multifaceted features** from high-dimensional invocation sequences
2. **Discovering optimal clusters** of functions with similar patterns
3. **Classifying cluster complexity** using rule-based methods
4. **Applying adaptive prediction models**: LSTM for complex patterns, Exponential Smoothing for simple patterns

## Framework Architecture

```
CLAP Framework
├── Module 1: Multifaceted Feature Extraction
│   ├── Time-domain statistical features
│   ├── Frequency-domain FFT features
│   ├── Anomaly & burstiness features
│   └── Shape features with PCA dimensionality reduction
│
├── Module 2: Optimal Hierarchical Clustering (OHC)
│   ├── Agglomerative hierarchical clustering (Ward linkage)
│   ├── Adaptive cluster number selection (Silhouette + DB Index)
│   └── Cluster-level sequence aggregation
│
├── Module 3a: Pattern Classification
│   ├── Binary classifier (Simple vs Complex patterns)
│   ├── Greedy feature selection
│   └── Rule-based voting mechanism
│
└── Module 3b: Hybrid Temporal Network Learning
    ├── Exponential Smoothing for simple patterns
    │   └── Auto-selection: SES / Holt / Holt-Winters
    └── Embed-LSTM for complex patterns
        └── Cluster-specific embeddings
```

## Requirements

### Environment
- Python 3.8 or higher
- CUDA-capable GPU (optional, for LSTM training acceleration)

### Dependencies

Install required Python packages:

```bash
pip install -r requirements.txt
```

**Core Dependencies:**
- `numpy >= 1.21.0`
- `pandas >= 1.3.0`
- `scikit-learn >= 1.0.0`
- `scipy >= 1.7.0`
- `matplotlib >= 3.4.0`
- `statsmodels >= 0.13.0`
- `torch >= 1.10.0`
- `tqdm >= 4.62.0`

## Project Structure

```
CLAP/
├── config.py                              # Unified configuration file
├── data/                                  # Data directory
│   ├── raw_requests/                      # Raw invocation data (*.csv)
│   ├── clustered_requests/                # Clustered data
│   ├── cluster_pattern.csv                # Ground truth labels
│   └── meta_funcID_runtime_triggerType.csv
├── output/                                # Output directory
│   ├── clustering_results/                # Clustering results
│   ├── classification_results/            # Classification results
│   ├── prediction_results/                # Prediction results
│   ├── visualizations/                    # Visualization plots
│   └── logs/                              # Log files
├── models/                                # Trained models
│   └── lstm_model.pth
├── feature_extraction.py                  # Module 1: Feature extraction
├── data_preprocessing.py                  # Data preprocessing utilities
├── hierarchical_clustering.py             # Module 2: Hierarchical clustering
├── clustering_utils.py                    # Clustering utilities
├── run_clustering.py                      # Main script for clustering
├── pattern_classifier.py                  # Module 3a: Pattern classifier
├── lstm_predictor.py                      # Module 3b: LSTM predictor
├── exponential_smoothing_predictor.py     # Module 3b: ES predictor
├── run_hybrid_prediction.py               # Main script for prediction
└── README.md                              # This file
```

## Data Format

### Description
We adopt Huawei Public Cloud Trace 2025 (https://github.com/sir-lab/data-release/blob/main/README_data_release_2025.md) as our primary dataset for function invocation patterns. Considering the scalability and diversity of cloud applications, this dataset provides a rich source of real-world invocation traces. 


### Input Data Format

**Raw invocation data** (`data/raw_requests/*.csv`):
```csv
time,funcID_1,funcID_2,...,funcID_N
0,10,5,...,8
60,12,7,...,9
...
```
- `time`: Unix timestamp in seconds
- Each column represents invocation count for a specific function

**Metadata** (`data/meta_funcID_runtime_triggerType.csv`):
```csv
funcID,triggerType-invocationType,...
func_001,HTTP-sync,...
func_002,Timer-async,...
...
```

**Ground truth labels** (`data/cluster_pattern.csv`) (optional, for training classifier):
```csv
cluster_id,pattern
cluster-0,simple
cluster-1,complex
...
```

## Usage

### Step 1: Configure Paths

Edit `config.py` to set your data paths:

```python
# Modify these paths according to your data location
DATA_DIR = BASE_DIR / "data" / "raw_requests"
META_FILE = BASE_DIR / "data" / "meta_funcID_runtime_triggerType.csv"
```

### Step 2: Run Hierarchical Clustering

Execute clustering to discover function groups with similar patterns:

```bash
python run_clustering.py
```

**Output:**
- `output/clustering_results/final_cluster_results.csv` - Cluster assignments for all functions
- `output/clustering_results/final_cluster-{X}-results.csv` - Individual cluster members
- `output/visualizations/adaptive_cluster_selection.png` - Cluster selection visualization

### Step 3: Prepare Clustered Data

After clustering, aggregate invocation sequences for each cluster. You can use the provided `reconstructData_Hier.py` script (from original codebase) or create cluster-level aggregated time series manually.

Expected format in `data/clustered_requests/`:
```
cluster-0_merged_data.csv
cluster-1_merged_data.csv
...
```

Each file should contain:
```csv
ds,y
2025-01-01 00:00:00,150
2025-01-01 00:01:00,165
...
```

### Step 4: Run Pattern Classification

Classify clusters into simple vs complex patterns:

```bash
python pattern_classifier.py
```

**Output:**
- `output/classification_results/classification_results.csv` - Cluster pattern labels
- `output/classification_results/feature_rules.txt` - Selected classification rules

### Step 5: Run Hybrid Prediction

Execute the hybrid prediction pipeline:

```bash
python run_hybrid_prediction.py
```

**Output:**
- `output/prediction_results/cluster-{X}_predictions.csv` - Predictions for each cluster
- `output/prediction_results/prediction_summary.csv` - Summary of all predictions
- `models/lstm_model.pth` - Trained LSTM model

## Configuration Parameters

Key parameters in `config.py`:

### Feature Extraction
- `MIN_CALL_THRESHOLD = 24 * 31` - Minimum total calls to include a function
- `PCA_VARIANCE_RATIO = 0.95` - Variance ratio for PCA dimensionality reduction

### Hierarchical Clustering
- `MIN_CLUSTERS = 3` - Minimum number of clusters
- `MAX_CLUSTERS = 505` - Maximum number of clusters (dynamically adjusted)
- `CLUSTERING_LINKAGE = 'ward'` - Linkage method for hierarchical clustering

### Pattern Classification
- `DECISION_FRACTIONS` - Voting thresholds for rule-based classification

### LSTM Parameters
- `LSTM_HIDDEN_SIZE = 128` - Hidden layer size
- `LSTM_NUM_LAYERS = 4` - Number of LSTM layers
- `LSTM_BATCH_SIZE = 512` - Training batch size
- `LSTM_EPOCHS = 15` - Number of training epochs
- `LSTM_LEARNING_RATE = 0.001` - Learning rate

### Exponential Smoothing Parameters
- `ES_SEASONAL_PERIODS_DAILY = 1440` - Daily seasonality (minutes)
- `ES_SEASONAL_PERIODS_HOURLY = 60` - Hourly seasonality (minutes)

## Output Interpretation

### Clustering Results

`final_cluster_results.csv`:
```csv
funcID,cluster,triggerType
func_001,0,HTTP-sync
func_002,0,HTTP-sync
func_003,1,Timer-async
...
```

### Classification Results

`classification_results.csv`:
```csv
cluster_id,label,predicted,votes,required_votes
cluster-0,simple,simple,8.0,6.0
cluster-1,complex,complex,3.0,6.0
...
```

### Prediction Results

`cluster-{X}_predictions.csv`:
```csv
cluster_id,prediction
cluster-0,145
cluster-0,152
cluster-0,148
...
```

`prediction_summary.csv`:
```csv
cluster_id,pattern,model,train_time,pred_time,num_predictions
cluster-0,simple,HoltWinters,2.34,0.12,4320
cluster-1,complex,LSTM,0.00,1.56,4320
...
```

<!-- ## Citation

If you use CLAP in your research, please cite:

```bibtex
@article{clap2025,
  title={CLAP: Cluster-guided Learning for Adaptive Prediction of Serverless Function Invocations},
  author={[Your Name]},
  journal={[Journal Name]},
  year={2025}
}
``` -->

## License

This project is licensed under the MIT License - see the LICENSE file for details.

---

**Note:** This is a research prototype. For production use, additional validation and optimization may be required.
