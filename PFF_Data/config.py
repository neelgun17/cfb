"""
Configuration file for QB Archetype Analysis Project
Centralizes all paths, parameters, and settings
"""

import os
from pathlib import Path

# --- Project Root Directory ---
PROJECT_ROOT = Path("/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data")

# --- Year Configuration ---
YEARS = [2019, 2023]  # Add new years here
CURRENT_YEAR = 2023   # Default year for processing

# --- Data Directories (New Clean Structure) ---
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
ANALYSIS_DIR = DATA_DIR / "analysis"
VALIDATION_DIR = DATA_DIR / "validation"

# --- Model and Results Directories ---
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"

# --- Dynamic Path Generation Functions ---
def get_data_paths(year=None):
    """Get data paths for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    return {
        'raw': RAW_DATA_DIR / str(year),
        'processed': PROCESSED_DATA_DIR / str(year),
        'analysis': ANALYSIS_DIR / str(year),
        'validation': VALIDATION_DIR / str(year),
        'models': MODELS_DIR / "single_year" / str(year),
        'results': RESULTS_DIR / "yearly_reports" / str(year)
    }

def get_raw_data_files(year=None):
    """Get raw data file paths for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    raw_dir = RAW_DATA_DIR / str(year)
    
    return {
        'passing_summary': raw_dir / "passing_summary.csv",
        'rushing_summary': raw_dir / "rushing_summary.csv", 
        'passing_concept': raw_dir / "passing_concept.csv",
        'passing_depth': raw_dir / "passing_depth.csv",
        'passing_pressure': raw_dir / "passing_pressure.csv",
        'time_in_pocket': raw_dir / "time_in_pocket.csv"
    }

def get_processed_data_files(year=None):
    """Get processed data file paths for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    processed_dir = PROCESSED_DATA_DIR / str(year)
    
    return {
        'merged_summary': processed_dir / "qb_player_merged_summary.csv",
        'passing_summary': processed_dir / "qb_player_summary.csv",
        'rushing_summary': processed_dir / "qb_player_rushing_summary.csv",
        'concept_summary': processed_dir / "qb_player_concept_summary.csv",
        'depth_summary': processed_dir / "qb_player_depth_summary.csv",
        'pressure_summary': processed_dir / "qb_player_pressure_summary.csv",
        'pocket_time_summary': processed_dir / "qb_player_pocket_time_summary.csv"
    }

def get_analysis_files(year=None):
    """Get analysis file paths for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    analysis_dir = ANALYSIS_DIR / str(year)
    
    return {
        'player_assignments': analysis_dir / "hierarchical_player_assignments_k4.csv",
        'cluster_profiles': analysis_dir / "hierarchical_profiles_k4.csv",
        'cluster_data_scaled': analysis_dir / "cluster_data_scaled_df.csv"
    }

def get_model_files(year=None, model_type='recall_optimized'):
    """Get model file paths for a specific year and model type."""
    if year is None:
        year = CURRENT_YEAR
    
    if model_type == 'cross_year':
        model_dir = MODELS_DIR / "cross_year"
    else:
        model_dir = MODELS_DIR / "single_year" / str(year)
    
    return {
        'model': model_dir / f"{model_type}_qb_archetype_rf_model.joblib",
        'label_encoder': model_dir / f"{model_type}_archetype_label_encoder.joblib",
        'scaler': model_dir / f"{model_type}_archetype_feature_scaler.joblib",
        'imputer': model_dir / f"{model_type}_archetype_feature_imputer.joblib"
    }

# --- Legacy Support (for backward compatibility) ---
# These will be deprecated but kept for now
RAW_DATA_FILES = get_raw_data_files()
PROCESSED_DATA_FILES = get_processed_data_files()
ANALYSIS_FILES = get_analysis_files()
MODEL_FILES = {
    'recall_optimized': get_model_files(model_type='recall_optimized'),
    'standard': get_model_files(model_type='standard')
}

# --- Results Files ---
RESULTS_FILES = {
    'cluster_profiles_excel': RESULTS_DIR / "qb_cluster_profiles.xlsx",
    'feature_importance_plot': RESULTS_DIR / "feature_importance.png",
    'validation_report': RESULTS_DIR / "validation_report.txt"
}

# --- Data Processing Parameters ---
DATA_PARAMS = {
    'min_dropbacks': 150,
    'position_filter': 'QB',
    'position_column': 'position'
}

# --- Clustering Parameters ---
CLUSTERING_PARAMS = {
    'n_clusters': 4,
    'random_state': 42,
    'pca_variance': 0.90,
    'min_dropbacks': 150
}

# --- Model Parameters ---
MODEL_PARAMS = {
    'test_size': 0.25,
    'random_state': 42,
    'cv_folds': 5,
    'scoring': 'recall_macro'  # For recall-optimized model
}

# --- Feature Lists ---
FEATURES_FOR_CLUSTERING = [
    # Overall Passing Style/Efficiency
    'accuracy_percent', 'avg_depth_of_target', 'avg_time_to_throw', 'btt_rate',
    'completion_percent', 'sack_percent', 'twp_rate', 'ypa', 'td_int_ratio',
    'comp_pct_diff', 'ypa_diff',
    
    # Rushing/Mobility
    'designed_run_rate', 'scramble_rate', 'elusive_rating', 'ypa_rushing', 
    'breakaway_percent', 'YAC_per_rush_attempt', 'pct_total_yards_rushing', 
    'qb_rush_attempt_rate',
    
    # PFF Grades
    'grades_offense', 'grades_pass', 'grades_run',
    
    # Play Types
    'pa_rate', 'pa_ypa', 'screen_rate', 'deep_attempt_rate', 
    'deep_accuracy_percent', 'deep_twp_rate',
    
    # Pressure Handling
    'pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate', 
    'pressure_accuracy_percent', 'quick_throw_rate'
]

# --- Archetype Mapping ---
ARCHETYPE_MAP = {
    0: "Scrambling Survivors",
    1: "Pocket Managers", 
    2: "Dynamic Dual-Threats",
    3: "Mobile Pocket Passer"
}

# --- Columns to Drop During Processing ---
COLUMNS_TO_DROP = {
    'passing': ['bats', 'thrown_aways', 'declined_penalties', 'hit_as_threw', 
                'spikes', 'penalties', 'pressure_to_sack_rate', 'position', 
                'franchise_id', 'aimed_passes', 'drop_rate', 'completions', 
                'avg_time_to_throw', 'grades_hands_fumble'],
    'rushing': ['position', 'team_name', 'player_game_count', 'declined_penalties', 
                'drops', 'penalties', 'rec_yards', 'receptions', 'routes',
                'zone_attempts', 'grades_pass_block', 'gap_attempts', 'elu_recv_mtf',
                'targets', 'grades_offense_penalty', 'grades_pass_route', 'def_gen_pressures',
                'grades_offense', 'grades_run_block', 'yco_attempt', 'elu_rush_mtf', 
                'grades_hands_fumble', 'franchise_id', 'scrambles', 'grades_pass', 'grades_run']
}

# --- Create Directories ---
def create_directories(year=None):
    """Create all necessary directories for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    paths = get_data_paths(year)
    
    directories = [
        DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR, ANALYSIS_DIR, VALIDATION_DIR,
        MODELS_DIR, RESULTS_DIR,
        paths['raw'], paths['processed'], paths['analysis'], paths['validation'],
        paths['models'], paths['results'],
        MODELS_DIR / "cross_year",
        RESULTS_DIR / "comparisons"
    ]
    
    for directory in directories:
        directory.mkdir(parents=True, exist_ok=True)
        print(f"Created directory: {directory}")

# --- Validation Functions ---
def validate_paths(year=None):
    """Validate that all required paths exist for a specific year."""
    if year is None:
        year = CURRENT_YEAR
    
    missing_paths = []
    
    # Check raw data files
    raw_files = get_raw_data_files(year)
    for name, path in raw_files.items():
        if not path.exists():
            missing_paths.append(f"Raw data ({year}): {name} -> {path}")
    
    # Check if processed data directory exists
    processed_dir = get_data_paths(year)['processed']
    if not processed_dir.exists():
        missing_paths.append(f"Processed data directory ({year}): {processed_dir}")
    
    if missing_paths:
        print(f"Missing required paths for year {year}:")
        for path in missing_paths:
            print(f"  - {path}")
        return False
    
    return True

if __name__ == "__main__":
    # Create directories and validate paths when config is run directly
    create_directories()
    validate_paths() 