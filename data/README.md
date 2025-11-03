# CLAP Data Directory

This directory contains data files required for the CLAP framework.

## Directory Structure

```
data/
├── cluster_pattern.csv              # Ground truth cluster pattern labels
├── meta_funcID_runtime_triggerType.csv  # Function metadata with trigger types
├── raw_requests/                    # Raw invocation data (CSV files)
│   └── *.csv                        # Per-day request data
└── clustered_requests/              # Clustered invocation data
    └── cluster-*_merged_data.csv    # Per-cluster aggregated data
```

## Data Files

### cluster_pattern.csv
Ground truth labels for cluster patterns (simple vs complex).

**Format:**
- `cluster_id`: Cluster identifier (e.g., "cluster-0")
- `pattern`: Pattern type ("simple" or "complex")

### meta_funcID_runtime_triggerType.csv
Metadata for serverless functions.

**Format:**
- `funcID`: Function identifier
- `triggerType-invocationType`: Trigger type information

### raw_requests/
Directory containing raw invocation data files. Each CSV file represents one day of data (day_00.csv to day_30.csv for 31 days).

**Expected format:**
- `day`: Day index (0-30)
- `time`: Timestamp in seconds (0-86340, minute-level granularity: 0, 60, 120, ..., 86340)
- `funcID_1`, `funcID_2`, ...: Column names in format `{appID}---{runtime}---{pool}`, values are invocation counts for each function at that minute

**Example:**
```
day,time,400---418---pool22-300-128,1531---418---pool22-300-128,...
0,0,1.0,,2.0,...
0,60,,,3.0,...
```

### clustered_requests/
Directory containing clustered and aggregated invocation data. Each file represents one cluster (cluster-0_merged_data.csv to cluster-N_merged_data.csv).

**Expected format:**
- `ds`: Datetime timestamp (e.g., "2025-01-01 00:00:00")
- `y`: Aggregated invocation count for the cluster at this timestamp
- `weight_*`: Weight of each function in the cluster (optional columns, sum to 1.0 per row)

## Data Preparation

If you're using your own data:

1. **Raw Data**: Place per-day invocation CSV files in `raw_requests/` (e.g., day_00.csv, day_01.csv, ...)
   - Each file should contain minute-level data (1440 rows per day)
   - Columns should be: `day,time,funcID_1,funcID_2,...`
   - Function IDs formatted as: `{appID}---{runtime}---{pool}`

2. **Metadata**: Update `meta_funcID_runtime_triggerType.csv` with your function metadata
   - Columns: `funcID`, `triggerType-invocationType`
   - One row per unique function

3. **Ground Truth** (optional): If you have labeled cluster patterns, update `cluster_pattern.csv`
   - Columns: `cluster_id`, `pattern`
   - Pattern values: "simple" or "complex"

4. **Clustered Data**: After running clustering (Module 2), the results will be saved in `clustered_requests/`
   - Format: cluster-X_merged_data.csv where X is cluster number
   - Contains aggregated time series for each cluster

## Using the Provided Dataset

For the cold start dataset provided:

**Source directories:**
- Raw requests: `R2/requests/` → Copy to `CLAP/data/raw_requests/`
- Clustered results: Run clustering script `run_clustering.py` to generate `CLAP/data/clustered_requests/`


## Notes

- The framework expects minute-level invocation data (1440 minutes per day)
- Time values should be in seconds: 0, 60, 120, ..., 86340 (for 00:00 to 23:59)
- Missing values in raw data should be left empty or filled with 0 for invocation counts
- Clustered data uses ISO format datetime strings (YYYY-MM-DD HH:MM:SS)
