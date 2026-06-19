# Workflow Energy Recommender

## Overview

Workflow Energy Recommender is a machine learning pipeline that predicts the total power consumption of scientific workflows and recommends the most energy-efficient compute cluster for execution.

Given a workflow description (JSON), the system extracts structural graph features, applies trained regression models, and ranks available cluster tiers by predicted energy consumption. This enables researchers and system operators to make data-driven scheduling decisions that minimize energy use across heterogeneous HPC environments.

---

## Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Pipeline Overview](#pipeline-overview)
- [Installation](#installation)
- [Usage](#usage)
  - [1. Generate Workflows](#1-generate-workflows)
  - [2. Preprocess Data](#2-preprocess-data)
  - [3. Train Models](#3-train-models)
  - [4. Run the Recommender](#4-run-the-recommender)
- [Supported Models](#supported-models)
- [Cluster Tiers](#cluster-tiers)
- [Workflow Feature Set](#workflow-feature-set)
- [Output](#output)

---

## Features

- Automated workflow generation using [WfCommons](https://wfcommons.org/) recipes (BLAST, BWA, Cycles, Epigenomics, Genome, Montage, Seismology)
- Structural feature extraction from workflow DAGs (task count, edge count, depth, file sizes, degree statistics, etc.)
- Multi-target regression for `total_power`
- 10 regression algorithms supported, from linear baselines to gradient boosting and symbolic regression
- Per-cluster model training across four cluster tiers
- Energy-aware recommendation — ranks clusters by predicted energy consumption
- Batch evaluation mode — processes 700 workflows (100 per type × 7 types) and reports ranking statistics
- Visualization — actual vs. predicted scatter plots and MAE/RMSE comparison bar charts

---

## Project Structure

```
.
├── workflowGenerator.py       # Generate synthetic workflow JSON files via WfCommons
├── preprocess.py              # Orchestrates the full feature extraction -> merge pipeline
├── _featureExtractor.py       # Extracts structural graph features from workflow JSONs
├── _executionProcessor.py     # Aggregates per-task execution data into per-workflow summaries
├── _dataMerger.py             # Merges extracted features with execution summaries
├── modelTraining.py           # Trains and evaluates regression models; saves to disk
├── recommender.py             # Loads trained models and recommends the best cluster
│
├── workflows/                 # Generated workflow JSON files (one subfolder per type)
│   ├── blast/
│   ├── bwa/
│   ├── cycles/
│   ├── epigenomics/
│   ├── genome/
│   ├── montage/
│   └── seismology/
│
├── data/                      # Intermediate and merged CSV datasets
│   ├── extracted_features.csv
│   ├── execution_per_workflow.csv
│   └── merged_data/
│       └── merged_data_<tier>.csv
│
├── models/                    # Trained model files (.pkl), one subfolder per cluster tier
│   ├── low_tier/
│   ├── mid_tier/
│   ├── high_tier/
│   └── extra_high_tier/
│
└── figures/                   # Evaluation plots and feature importance reports
```

---

## Installation

```bash
# Clone the repository
git clone <repository-url>
cd workflow-energy-recommender

# Install dependencies
pip install pandas numpy scikit-learn xgboost catboost lightgbm pysr matplotlib wfcommons
```

> Note: PySR (symbolic regression) requires Julia to be installed. See the [PySR documentation](https://astroautomata.com/PySR/) for setup instructions. If you do not need symbolic regression, you can omit it and exclude `sr` from model selection.

### Install the WRENCH Simulation Framework

WRENCH is a C++ library and must be built from source. See the [official documentation](https://wrench-project.org/wrench/2.8/getting_started.html) for full prerequisites and platform-specific instructions.

```bash
# Clone the WRENCH repository
git clone https://github.com/wrench-project/wrench.git
cd wrench

# Build and install
mkdir build && cd build
cmake ..
make
sudo make install
```

> Note: WRENCH requires [SimGrid](https://simgrid.org) as a dependency. Ensure it is installed before building WRENCH.

---

## Usage

### 1. Generate Workflows

Generates 200 synthetic workflow JSON files per type (7 types, 1400 total) using WfCommons recipes.

```bash
python workflowGenerator.py
```

Workflows are saved to `workflows/<type>/` with filenames following the pattern `<type>-workflow-<num_tasks>-0.json`.

---

### 2. Simulate Workflows

Runs WRENCH simulations on the generated workflow JSON files to produce per-task execution data.

```bash
start_simulation_single.sh.py
```
or
```bash
start_simulation_multiple.sh.py
```

Simulations are run per workflow type and tier. Results are saved to `data/execution_output_<tier>.csv`.

> Note: Ensure WRENCH and SimGrid are installed and accessible before running this step. See [Install the WRENCH Simulation Framework](#install-the-wrench-simulation-framework).

---

### 3. Preprocess Data

Runs the full feature extraction -> execution processing -> merge pipeline. Requires a CSV file containing per-task execution output (power readings).

```bash
python preprocess.py --execution_output data/execution_output_<tier>.csv
```

Steps performed:
1. Feature Extraction — parses all workflow JSONs and writes `data/extracted_features.csv`
2. Execution Processing — aggregates per-task power readings into per-workflow summaries (`data/execution_per_workflow.csv`)
3. Data Merge — joins features with execution summaries into `data/merged_data/merged_data_<tier>.csv`

---

### 4. Train Models

Trains regression models on the merged dataset. Run once per cluster tier.

```bash
# Train all models
python modelTraining.py data/merged_data/merged_data_<tier>.csv

# Train specific models only
python modelTraining.py data/merged_data/merged_data_<tier>.csv --models rf xgb catboost lgbm
```

Trained models are saved to `models/<tier>/`. Evaluation figures are saved to `figures/<tier>/`.

---

### 5. Run the Recommender

#### Single workflow
```bash
python recommender.py --workflow workflows/epigenomics/epigenomics-workflow-190-0.json --model rf
```

#### Random workflow
```bash
python recommender.py --random --model xgb
```

#### Batch mode (700 workflows, ranking statistics)
```bash
python recommender.py --batch --workflows_dir workflows --model rf --output_csv results/choices.csv
```

CLI arguments:

| Argument | Description | Default |
|---|---|---|
| `--workflow` | Path to a single workflow JSON | — |
| `--random` | Pick a random workflow from `--workflows_dir` | — |
| `--batch` | Process 100 workflows per type (700 total) | — |
| `--workflows_dir` | Root directory of workflow JSON files | `workflows/` |
| `--model` | ML model key to use for prediction | `rf` |
| `--models_dir` | Root directory of trained model files | `models/` |
| `--output_csv` | Output CSV path for batch results | `batch_choices.csv` |

---

## Supported Models

| Key | Algorithm |
|---|---|
| `lr` | Linear Regression |
| `pr` | Power Regression (`y = a · x^b`) |
| `er` | Exponential Regression (`y = a · e^(bx)`) |
| `poly` | Polynomial Regression (degree 2) |
| `rf` | Random Forest |
| `xgb` | XGBoost |
| `gbr` | Gradient Boosting Regressor |
| `catboost` | CatBoost |
| `lgbm` | LightGBM |
| `sr` | Symbolic Regression (PySR) |

Models are trained independently for the total power target.  
Evaluation metrics reported: MSE, MAE, RMSE, MAPE, R².

---

## Cluster Tiers

The recommender supports four cluster tiers. Models must be trained separately for each tier using the corresponding execution output CSV:

| Tier | Directory |
|---|---|
| Low tier | `models/low_tier/` |
| Mid tier | `models/mid_tier/` |
| High tier | `models/high_tier/` |
| Extra high tier | `models/extra_high_tier/` |

---

## Workflow Feature Set

Features extracted from each workflow JSON for model input:

| Feature | Description |
|---|---|
| `workflow_type` | Workflow category (folder name) |
| `task_count` | Total number of tasks |
| `edge_count` | Total dependency edges in the DAG |
| `workflow_depth` | Longest path (critical path length) |
| `unique_task_types` | Number of distinct task types |
| `avg_file_size` | Average file size in bytes |
| `max_file_size` | Maximum file size in bytes |
| `avg_in_degree` / `max_in_degree` | Task in-degree statistics |
| `avg_out_degree` / `max_out_degree` | Task out-degree statistics |
| `total_input_bytes` / `total_output_bytes` | Total data volume |
| `declared_core_count_avg/max` | Declared CPU core requirements |
| `type_count__<name>` | Per-task-type occurrence counts |

---

## Output

| File | Description |
|---|---|
| `data/extracted_features.csv` | Structural features for all workflows |
| `data/merged_data/merged_data_<tier>.csv` | Features joined with execution metrics |
| `models/<tier>/<model>_total_power.pkl` | Serialized trained models |
| `batch_choices.csv` | Per-workflow cluster recommendations (batch mode) |

---

> Keywords: scientific workflow scheduling, energy-efficient computing, cluster recommendation, machine learning regression, workflow feature extraction, power consumption prediction, HPC workload optimization, XGBoost, Random Forest, CatBoost, LightGBM, symbolic regression, WfCommons, Python
