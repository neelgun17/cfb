# QB Archetype Analysis Pipeline

A modular, production-ready pipeline for analyzing quarterback archetypes using machine learning clustering and classification.

## 🏗️ Project Structure

```
PFF_Data/
├── config.py                 # Central configuration and parameters
├── data_processing.py        # Data cleaning and feature engineering
├── clustering.py            # Clustering analysis and archetype discovery
├── model_training.py        # Classification model training
├── main_pipeline.py         # Main orchestration script
├── README.md               # This file
├── requirements.txt        # Python dependencies
│
├── raw_data/              # Raw CSV files from PFF
│   ├── passing_summary.csv
│   ├── rushing_summary.csv
│   ├── passing_concept.csv
│   ├── passing_depth.csv
│   ├── passing_pressure.csv
│   └── time_in_pocket.csv
│
├── processed_data/         # Cleaned and processed data
│   ├── qb_player_merged_summary.csv
│   ├── qb_player_summary.csv
│   ├── qb_player_rushing_summary.csv
│   └── ...
│
├── analysis/              # Clustering results and analysis
│   ├── hierarchical_player_assignments_k4.csv
│   ├── hierarchical_profiles_k4.csv
│   ├── cluster_data_scaled_df.csv
│   ├── feature_importance.png
│   └── archetype_distribution.png
│
├── models/                # Trained models
│   ├── recall_optimized/
│   │   ├── recall_optimized_qb_archetype_rf_model.joblib
│   │   ├── recall_archetype_label_encoder.joblib
│   │   ├── recall_archetype_feature_scaler.joblib
│   │   └── recall_archetype_feature_imputer.joblib
│   └── standard/
│       ├── best_qb_archetype_rf_model.joblib
│       └── ...
│
├── validation_data/       # Validation datasets
├── results/              # Final results and reports
└── logs/                 # Pipeline logs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt

# Run configuration to create directories
python config.py
```

### 2. Run Full Pipeline

```bash
# Run the complete pipeline (data processing → clustering → model training)
python main_pipeline.py --step full

# Run with specific models
python main_pipeline.py --step full --models recall_optimized standard
```

### 3. Run Individual Steps

```bash
# Data processing only
python main_pipeline.py --step data_processing

# Clustering only
python main_pipeline.py --step clustering

# Model training only
python main_pipeline.py --step model_training --models recall_optimized
```

## 📋 Pipeline Steps

### 1. Data Processing (`data_processing.py`)
- **Input**: Raw PFF CSV files
- **Process**: 
  - Filter for QB position
  - Clean and merge multiple data sources
  - Calculate derived features (rates, ratios)
  - Handle missing data
- **Output**: Clean, merged dataset ready for analysis

### 2. Clustering Analysis (`clustering.py`)
- **Input**: Processed QB data
- **Process**:
  - Feature preprocessing (imputation, scaling)
  - PCA dimensionality reduction
  - Hierarchical clustering
  - Archetype profile generation
- **Output**: QB archetype assignments and profiles

### 3. Model Training (`model_training.py`)
- **Input**: QB data with archetype labels
- **Process**:
  - Train Random Forest classifiers
  - Hyperparameter optimization
  - Model evaluation and analysis
- **Output**: Trained models for archetype prediction

## 🎯 QB Archetypes

The pipeline discovers 4 distinct QB archetypes:

1. **Scrambling Survivors**: High pressure, high scramble rate, risky passing
2. **Pocket Managers**: Low mobility, efficient passing, low risk
3. **Dynamic Dual-Threats**: High designed run rate, explosive rushing, aggressive passing
4. **Mobile Pocket Passer**: Balanced mobility, high passing efficiency, low turnover risk

## ⚙️ Configuration

All parameters are centralized in `config.py`:

```python
# Data processing parameters
DATA_PARAMS = {
    'min_dropbacks': 150,
    'position_filter': 'QB',
    'position_column': 'position'
}

# Clustering parameters
CLUSTERING_PARAMS = {
    'n_clusters': 4,
    'random_state': 42,
    'pca_variance': 0.90
}

# Model parameters
MODEL_PARAMS = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,
    'scoring': 'recall_macro'
}
```

## 📊 Key Features

### Feature Engineering
- **33 engineered features** covering passing, rushing, pressure handling, and PFF grades
- **Calculated rates** (designed run rate, scramble rate, pressure rate, etc.)
- **Advanced metrics** (TD/INT ratio, YAC per rush, etc.)

### Model Performance
- **Recall-optimized model**: 88.89% accuracy, 85.71% macro recall
- **Feature importance analysis** shows PFF grades and rushing metrics are most important
- **Robust validation** with separate test sets and cross-validation

### Modular Design
- **Separation of concerns**: Each step is independent and can be run separately
- **Configurable**: All parameters centralized in config file
- **Logging**: Comprehensive logging for debugging and monitoring
- **Error handling**: Robust error handling throughout the pipeline

## 🔧 Usage Examples

### Basic Usage
```bash
# Run everything
python main_pipeline.py

# Run with verbose logging
python main_pipeline.py --log-level DEBUG
```

### Advanced Usage
```bash
# Train only recall-optimized model
python main_pipeline.py --step model_training --models recall_optimized

# Run clustering with custom parameters (edit config.py first)
python main_pipeline.py --step clustering
```

### Validation
```bash
# Use the feature importance analysis script
python feature_importance_analysis.py

# Use the recall-optimized classification script
python recall_optimized_classification.py
```

## 📈 Results Interpretation

### Feature Importance
The top 5 most important features for QB archetype classification:
1. `grades_offense` (0.105) - Overall PFF offensive grade
2. `grades_pass` (0.095) - PFF passing grade
3. `grades_run` (0.088) - PFF rushing grade
4. `qb_rush_attempt_rate` (0.078) - QB rush attempts per dropback
5. `pct_total_yards_rushing` (0.070) - Rushing yards as % of total yards

### Model Performance
- **Overall Accuracy**: 88.89%
- **Macro Recall**: 85.71%
- **Best performing archetype**: Pocket Managers (100% recall)
- **Most challenging**: Mobile Pocket Passer and Scrambling Survivors (71% recall each)

## 🛠️ Development

### Adding New Features
1. Add feature to `FEATURES_FOR_CLUSTERING` in `config.py`
2. Ensure feature is calculated in `data_processing.py`
3. Re-run pipeline to test

### Modifying Archetypes
1. Update `ARCHETYPE_MAP` in `config.py`
2. Adjust `n_clusters` in `CLUSTERING_PARAMS`
3. Re-run clustering step

### Adding New Models
1. Create new model class in `model_training.py`
2. Add model type to `MODEL_FILES` in `config.py`
3. Update `main_pipeline.py` to include new model

## 📝 Logging

The pipeline provides comprehensive logging:
- **File logging**: All logs saved to `pipeline.log`
- **Console logging**: Real-time progress updates
- **Step timing**: Performance metrics for each pipeline step
- **Error tracking**: Detailed error messages and stack traces

## 🔍 Troubleshooting

### Common Issues

1. **Missing raw data files**
   - Ensure all CSV files are in `raw_data/` directory
   - Check file names match those in `config.py`

2. **Memory issues**
   - Reduce `n_estimators` in model parameters
   - Use smaller `cv_folds` for cross-validation

3. **Feature mismatch**
   - Check that all features in `FEATURES_FOR_CLUSTERING` exist in processed data
   - Verify column names match exactly

### Debug Mode
```bash
python main_pipeline.py --log-level DEBUG
```

## 📚 Dependencies

```
pandas>=1.3.0
numpy>=1.21.0
scikit-learn>=1.0.0
matplotlib>=3.4.0
seaborn>=0.11.0
joblib>=1.1.0
```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- PFF (Pro Football Focus) for providing the data
- Scikit-learn community for the excellent ML tools
- The football analytics community for inspiration and feedback 