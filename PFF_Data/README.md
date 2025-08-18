# QB Archetype Analysis Pipeline

A modular, production-ready pipeline for analyzing quarterback archetypes using machine learning clustering and classification.

## 🏗️ Project Structure

```
PFF_Data/
├── main_pipeline.py          # Main orchestration script
├── pipeline_steps.py         # Individual pipeline step classes
├── pipeline_summary.py       # Pipeline summary and reporting
├── config.py                 # Central configuration and parameters
├── data_processing.py        # Data cleaning and feature engineering
├── clustering.py             # Clustering analysis and archetype discovery
├── model_training.py         # Classification model training
├── enhanced_api.py           # Enhanced REST API with AI analysis
├── ai_analysis_service.py    # Qwen3:8B LLM-based AI analysis
├── lightweight_ai_analyzer.py # Rule-based AI analysis service
├── switch_ai_analyzer.py     # AI analyzer switching utility
├── test_ai_integration.py    # AI integration tests
├── README.md                 # This file
├── requirements.txt          # Python dependencies
│
├── data/                     # Data directory (year-based structure)
│   ├── raw/                  # Raw PFF CSV files
│   │   └── 2024/            # Year-specific raw data
│   ├── processed/            # Cleaned and merged data
│   │   └── 2024/            # Year-specific processed data
│   └── analysis/             # Analysis results
│       └── 2024/            # Year-specific analysis results
│
├── models/                   # Trained models
│   └── single_year/         # Year-specific models
│       └── 2024/            # Models for 2024
│
└── logs/                     # Pipeline logs
```

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Install dependencies
pip install -r requirements.txt
```

### 2. Run Full Pipeline

```bash
# Run the complete pipeline (default year 2024)
python3 main_pipeline.py

# Run for a specific year
python3 main_pipeline.py --year 2024

# Run with specific models
python3 main_pipeline.py --models recall_optimized standard --year 2024

# Force retrain models (ignore existing models)
python3 main_pipeline.py --force-retrain --year 2024
```

### 3. Run Individual Steps

```bash
# Data processing only
python3 main_pipeline.py --step data_processing --year 2024

# Clustering only
python3 main_pipeline.py --step clustering --year 2024

# Model training only
python3 main_pipeline.py --step model_training --year 2024


```

### 4. Advanced Options

```bash
# Train both recall-optimized and standard models
python3 main_pipeline.py --models recall_optimized standard --year 2024

# Train only standard model
python3 main_pipeline.py --models standard --year 2024

# Run with debug logging
python3 main_pipeline.py --log-level DEBUG --year 2024

# Run with minimal logging
python3 main_pipeline.py --log-level WARNING --year 2024
```

### 5. Enhanced QB Archetype Prediction API

The project includes a comprehensive REST API for QB archetype analysis with both **player lookup** and **stats-based prediction** capabilities:

#### Features
- **Player Lookup**: Find archetypes for existing players in the dataset
- **Stats-based Prediction**: Predict archetypes for new/unknown QBs or hypothetical scenarios
- **Player Search**: Search for players by name with partial matching
- **Analytics**: Get archetype distributions and top players by archetype
- **Batch Processing**: Predict archetypes for multiple QBs at once

#### Start the Enhanced API
```bash
python3 enhanced_api.py
```

#### Test the Enhanced API
```bash
python3 test_enhanced_api.py
```

#### Example Usage
```bash
python3 enhanced_example_usage.py
```

#### API Endpoints
- `GET /health` - Health check and model status
- `GET /features` - Get required features for prediction
- `GET /models` - Get model information
- `GET /search?q={query}` - Search for players by name
- `GET /archetypes/distribution` - Get archetype distribution
- `GET /archetypes/top-players` - Get top players by archetype
- `POST /predict` - Predict archetype from stats
- `POST /predict/batch` - Batch prediction for multiple QBs
- `POST /compare` - Compare two QBs (with optional AI analysis)
- `GET /ai/config` - Get AI analyzer configuration
- `POST /ai/config` - Set AI analyzer configuration
- `POST /analyze/ai/qb` - AI-powered individual QB analysis
- `POST /compare/ai` - AI-powered QB comparison
- `POST /analyze/ai/strategy` - AI-powered strategic insights

For detailed API documentation, see [ENHANCED_API_DOCUMENTATION.md](ENHANCED_API_DOCUMENTATION.md).

## 🤖 AI-Powered QB Analysis

The project now includes **intelligent AI-powered analysis** that provides deep insights into quarterback performance using rule-based statistical analysis and archetype comparisons.

### 🧠 AI Analysis Features

#### 1. **Individual QB Performance Analysis**
Get comprehensive analysis of any quarterback's performance relative to their archetype:

- **Archetype Fit Analysis**: How well the QB matches their archetype (with percentage scores)
- **Performance Strengths**: Identifies superior statistical performance areas
- **Performance Weaknesses**: Highlights areas of concern and improvement opportunities
- **Strategic Insights**: Provides actionable defensive and offensive strategies

#### 2. **AI-Powered QB Comparison**
Compare two quarterbacks with intelligent statistical analysis:

- **Key Statistical Differences**: Identifies significant performance gaps
- **Strategic Implications**: Provides game-planning insights
- **Head-to-Head Analysis**: Detailed comparison of strengths and weaknesses

#### 3. **Strategic Insights**
Generate game-specific strategic recommendations:

- **Offensive Strategy**: Tailored play-calling recommendations
- **Defensive Strategy**: How to approach defending against the QB
- **Situational Analysis**: Critical situation handling recommendations

### 🤖 Dual AI Analyzer Options

The system supports two AI analyzer types that you can switch between:

#### ⚡ Lightweight Analyzer (Default)
- **Speed**: Instant (milliseconds)
- **Quality**: Consistent, statistical insights
- **Best for**: Real-time API usage, multiple concurrent requests
- **Type**: Rule-based analysis using statistical deviations

#### 🧠 Qwen3:8B Analyzer
- **Speed**: Slow (60-120 seconds)
- **Quality**: Nuanced, natural language insights
- **Best for**: Detailed analysis, one-off deep dives
- **Type**: Large Language Model (Qwen3:8B via Ollama)

### 🔄 Switching Between Analyzers

**Option 1: Configuration File**
```bash
# Edit config.py and change:
AI_ANALYZER_TYPE = "lightweight"  # or "qwen"
# Then restart the API
```

**Option 2: Convenience Script**
```bash
# Check current analyzer
python3 switch_ai_analyzer.py status

# Switch to lightweight (fast)
python3 switch_ai_analyzer.py lightweight

# Switch to Qwen (detailed)
python3 switch_ai_analyzer.py qwen
```

**Option 3: API Endpoint**
```bash
# Check configuration
curl -X GET http://localhost:5001/ai/config

# Set configuration (requires restart)
curl -X POST http://localhost:5001/ai/config \
  -H "Content-Type: application/json" \
  -d '{"analyzer_type": "qwen"}'
```

### 🚀 AI Analysis Endpoints

#### Individual QB Analysis
```bash
curl -X POST http://localhost:5001/analyze/ai/qb \
  -H "Content-Type: application/json" \
  -d '{"qb_name": "Dillon Gabriel"}'
```

#### QB Comparison
```bash
curl -X POST http://localhost:5001/compare/ai \
  -H "Content-Type: application/json" \
  -d '{"qb1": "Dillon Gabriel", "qb2": "Will Howard"}'
```

#### Strategic Insights
```bash
curl -X POST http://localhost:5001/analyze/ai/strategy \
  -H "Content-Type: application/json" \
  -d '{"qb_name": "Dillon Gabriel", "context": "upcoming_game_against_strong_pass_rush"}'
```

#### Enhanced Comparison (with AI)
```bash
curl -X POST http://localhost:5001/compare \
  -H "Content-Type: application/json" \
  -d '{"qb1": "Dillon Gabriel", "qb2": "Will Howard", "include_ai": true}'
```

### 📊 Example AI Analysis Output

#### Individual QB Analysis
```
# AI Analysis: Dillon Gabriel (OREGON)

**Archetype**: Pocket Managers

**Archetype Fit Analysis**
QB shows **Good** alignment with the Pocket Managers archetype (67% fit).

**Key Deviations:**
• Accuracy Percent: 81.9 (above archetype avg by 9.2%)
• Scramble Rate: 0.0 (below archetype avg by 99.3%)
• Pressure Accuracy Percent: 67.8 (above archetype avg by 8.6%)

**Performance Strengths**
• **Superior Accuracy**: 81.9% (above archetype average)
• **Strong Pressure Handling**: 67.8% accuracy under pressure
• **Elite Passing Grade**: PFF Pass Grade of 86.3
• **Efficient Decision Making**: TD/INT ratio of 5.0

**Strategic Insights**
• **Defensive Strategy**: Focus on coverage over pressure - QB is highly accurate
• **Offensive Strategy**: Emphasize quick, accurate passing game
```

#### QB Comparison
```
# AI Comparison: Dillon Gabriel vs Will Howard

**Key Statistical Differences**
• **Pressure Handling**: Will Howard (+5.4%)
• **Efficiency**: Dillon Gabriel (+1.5 TD/INT ratio)

**Strategic Implications**
• Defenses should pressure Dillon Gabriel more aggressively
```

### ⚡ Performance Benefits

- **Instant Response**: No more 30-120 second timeouts
- **Resource Efficient**: Uses minimal CPU/memory
- **Reliable**: No external service dependencies
- **Scalable**: Can handle multiple requests simultaneously
- **Intelligent**: Rule-based analysis with statistical significance

### 🔧 Technical Implementation

- **Statistical Analysis**: Compares QB stats to archetype averages
- **Rule-Based Logic**: Uses predefined thresholds for significance
- **Error Handling**: Robust error handling and logging
- **Integration**: Seamlessly integrated with existing Flask API

### 🧪 Testing AI Features

Test all AI functionality:
```bash
python3 test_ai_integration.py
```

This will test:
- ✅ Individual QB analysis
- ✅ QB comparison
- ✅ Strategic insights
- ✅ Enhanced comparison with AI

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
  - **Smart Loading**: Check if trained models already exist
  - **Conditional Training**: Only retrain if models don't exist or `--force-retrain` is used
  - **Grid Search**: Hyperparameter optimization (when training)
  - **Model Evaluation**: Performance analysis and feature importance
- **Output**: Trained models for archetype prediction

### 4. Final Merged CSV Creation
- **Input**: Processed QB data + archetype assignments
- **Process**:
  - Merge all statistical features with archetype labels
  - Filter to players with ≥150 dropbacks
  - Reorder columns for optimal analysis
- **Output**: Complete dataset with all stats and archetypes (`final_merged_qb_data_with_archetypes.csv`)

## 🧠 Smart Model Loading

The pipeline includes intelligent model management to save time and computational resources:

### How It Works
1. **Model Check**: Before training, the pipeline checks if trained models already exist
2. **Conditional Training**: 
   - If models exist → Load existing models (fast)
   - If models don't exist → Train new models (slow)
   - If `--force-retrain` is used → Always train new models
3. **Evaluation**: Always runs model evaluation and analysis regardless of loading method

### Benefits
- **Time Savings**: Skip expensive grid search when models already exist
- **Consistency**: Use the same best-performing models across runs
- **Control**: Force retraining when needed with `--force-retrain` flag
- **Reliability**: Always get model evaluation results

### Usage Examples
```bash
# First run - trains new models (slow)
python3 main_pipeline.py --year 2023

# Subsequent runs - loads existing models (fast)
python3 main_pipeline.py --year 2023

# Force retrain - ignores existing models (slow)
python3 main_pipeline.py --force-retrain --year 2023
```

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
# Run everything (default year 2023)
python3 main_pipeline.py

# Run for a specific year
python3 main_pipeline.py --year 2023

# Run with verbose logging
python3 main_pipeline.py --log-level DEBUG --year 2023
```

### Advanced Usage
```bash
# Train only recall-optimized model
python3 main_pipeline.py --step model_training --models recall_optimized --year 2023

# Force retrain a specific model (ignore existing model)
python3 main_pipeline.py --step model_training --models recall_optimized --force-retrain --year 2023

# Run clustering with custom parameters (edit config.py first)
python3 main_pipeline.py --step clustering --year 2023

# Train both model types
python3 main_pipeline.py --models recall_optimized standard --year 2023
```

### Most Common Usage
```bash
# For typical workflow (recommended)
python3 main_pipeline.py --year 2024
```

This will run the full pipeline (data processing → clustering → model training) with recall-optimized models for 2024 data.

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
python3 main_pipeline.py --log-level DEBUG
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