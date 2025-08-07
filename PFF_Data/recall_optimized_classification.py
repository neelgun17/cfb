import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Configuration & Constants ---
PLAYER_ASSIGNMENTS_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_player_assignments_k4.csv'
ORIGINAL_STATS_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv'

FEATURES_FOR_CLUSTERING = [
    'accuracy_percent', 'avg_depth_of_target', 'avg_time_to_throw', 'btt_rate',
    'completion_percent', 'sack_percent', 'twp_rate', 'ypa', 'td_int_ratio',
    'comp_pct_diff', 'ypa_diff', 'designed_run_rate', 'scramble_rate',
    'elusive_rating', 'ypa_rushing', 'breakaway_percent', 'YAC_per_rush_attempt',
    'pct_total_yards_rushing', 'qb_rush_attempt_rate', 'grades_offense',
    'grades_pass', 'grades_run', 'pa_rate', 'pa_ypa', 'screen_rate',
    'deep_attempt_rate', 'deep_accuracy_percent', 'deep_twp_rate',
    'pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate',
    'pressure_accuracy_percent', 'quick_throw_rate'
]

def load_and_prepare_data():
    """Load and prepare data for recall-optimized training."""
    print("--- 1. Loading and Preparing Data ---")
    try:
        # Load player assignments
        df_player_assignments = pd.read_csv(PLAYER_ASSIGNMENTS_PATH)
        if not all(col in df_player_assignments.columns for col in ['player_id', 'archetype_name', 'player', 'team_name']):
            raise ValueError("Player assignments CSV is missing required columns.")

        # Load original stats
        df_original_stats = pd.read_csv(ORIGINAL_STATS_PATH)
        df_original_stats.columns = df_original_stats.columns.str.rstrip('_')
        df_original_stats.rename(columns=lambda c: c.replace('.', '_'), inplace=True)

        # Filter by dropbacks
        df_filtered_stats = df_original_stats[df_original_stats["dropbacks"].astype(int) >= 150].copy()

        # Merge to get features AND labels
        df_merged_for_modeling = pd.merge(
            df_filtered_stats,
            df_player_assignments[['player_id', 'archetype_name']],
            on='player_id',
            how='inner'
        )

        if df_merged_for_modeling.empty:
            raise ValueError("Merged DataFrame for modeling is empty.")

        print(f"Shape of merged data for modeling: {df_merged_for_modeling.shape}")

        # Prepare features and targets
        X_unscaled = df_merged_for_modeling[FEATURES_FOR_CLUSTERING].copy()
        identifiers_all_clustered = df_merged_for_modeling[['player', 'team_name', 'player_id']].copy()
        y_str_labels = df_merged_for_modeling['archetype_name'].copy()

        # Preprocess features
        X_unscaled.replace([np.inf, -np.inf], np.nan, inplace=True)
        imputer = SimpleImputer(strategy='median')
        X_imputed_array = imputer.fit_transform(X_unscaled)
        X_imputed_df_full = pd.DataFrame(X_imputed_array, columns=FEATURES_FOR_CLUSTERING, index=X_unscaled.index)

        scaler = StandardScaler()
        X_scaled_array = scaler.fit_transform(X_imputed_df_full)
        X = pd.DataFrame(X_scaled_array, columns=FEATURES_FOR_CLUSTERING, index=X_unscaled.index)

        print(f"Features shape (X): {X.shape}")
        print(f"Target shape (y_str_labels): {y_str_labels.shape}")
        print(f"Unique archetypes in y: {y_str_labels.unique()}")
        print(f"Archetype distribution:\n{y_str_labels.value_counts()}")

        return X, y_str_labels, identifiers_all_clustered, imputer, scaler, X_imputed_df_full

    except FileNotFoundError as e:
        print(f"Error loading files: {e}")
        return None, None, None, None, None, None
    except (KeyError, ValueError) as e:
        print(f"Error in data preparation: {e}")
        return None, None, None, None, None, None

def train_recall_optimized_model(X, y_str_labels, identifiers_all_clustered):
    """Train a Random Forest model optimized for recall."""
    print("\n--- 2. Training Recall-Optimized Model ---")
    
    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_str_labels)
    for i, class_name in enumerate(label_encoder.classes_):
        print(f"Numerical label {i} -> Archetype: {class_name}")

    # Split data
    X_train, X_test, y_train_encoded, y_test_encoded, identifiers_train, identifiers_test = train_test_split(
        X, y_encoded, identifiers_all_clustered,
        test_size=0.25,
        random_state=42,
        stratify=y_encoded
    )
    print(f"X_train shape: {X_train.shape}, y_train_encoded shape: {y_train_encoded.shape}")
    print(f"X_test shape: {X_test.shape}, y_test_encoded shape: {y_test_encoded.shape}")

    # Hyperparameter tuning optimized for recall
    print("\n--- 3. Hyperparameter Tuning for Recall Optimization ---")
    
    # Expanded parameter grid for recall optimization
    param_grid_rf_recall = {
        'n_estimators': [200, 300, 400],
        'max_depth': [None, 10, 15, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4],
        'class_weight': ['balanced', 'balanced_subsample'],
        'max_features': ['sqrt', 'log2', None]
    }
    
    cv_stratified = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    
    # Use recall_macro as scoring metric to optimize for recall
    grid_search_rf_recall = GridSearchCV(
        estimator=RandomForestClassifier(random_state=42),
        param_grid=param_grid_rf_recall,
        scoring='recall_macro',  # Optimize for macro-averaged recall
        cv=cv_stratified,
        verbose=1,
        n_jobs=-1
    )
    
    grid_search_rf_recall.fit(X_train, y_train_encoded)
    best_rf_model_recall = grid_search_rf_recall.best_estimator_
    
    print("\nBest Random Forest Parameters for Recall:")
    print(grid_search_rf_recall.best_params_)
    print(f"Best cross-validated recall_macro score: {grid_search_rf_recall.best_score_:.4f}")

    return best_rf_model_recall, label_encoder, X_test, y_test_encoded, identifiers_test

def evaluate_recall_optimized_model(model, label_encoder, X_test, y_test_encoded, identifiers_test):
    """Evaluate the recall-optimized model."""
    print("\n--- 4. Evaluating Recall-Optimized Model ---")
    
    y_pred_recall = model.predict(X_test)
    accuracy_recall = accuracy_score(y_test_encoded, y_pred_recall)
    recall_macro = recall_score(y_test_encoded, y_pred_recall, average='macro')
    
    report_recall = classification_report(y_test_encoded, y_pred_recall, target_names=label_encoder.classes_, zero_division=0)
    cm_recall = confusion_matrix(y_test_encoded, y_pred_recall)

    print(f"Accuracy: {accuracy_recall:.4f}")
    print(f"Macro-Averaged Recall: {recall_macro:.4f}")
    print("Classification Report:")
    print(report_recall)
    
    # Confusion matrix
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_recall, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Recall-Optimized Random Forest')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()

    # Per-class recall analysis
    print("\n--- Per-Class Recall Analysis ---")
    recall_per_class = recall_score(y_test_encoded, y_pred_recall, average=None)
    for i, (class_name, recall_val) in enumerate(zip(label_encoder.classes_, recall_per_class)):
        print(f"{class_name:<25}: {recall_val:.4f}")

    # Analyze misclassifications
    print("\n--- Misclassification Analysis ---")
    misclassified_indices = np.where(y_test_encoded != y_pred_recall)[0]
    misclassified_df = identifiers_test.iloc[misclassified_indices].copy()
    misclassified_df['true_archetype'] = label_encoder.inverse_transform(y_test_encoded[misclassified_indices])
    misclassified_df['predicted_archetype'] = label_encoder.inverse_transform(y_pred_recall[misclassified_indices])
    
    print(f"Total misclassifications: {len(misclassified_df)}")
    print("\nMisclassified players:")
    print(misclassified_df[['player', 'team_name', 'true_archetype', 'predicted_archetype']])

    return accuracy_recall, recall_macro, report_recall

def save_recall_optimized_model(model, label_encoder, imputer, scaler):
    """Save the recall-optimized model and preprocessors."""
    print("\n--- 5. Saving Recall-Optimized Model ---")
    
    base_path = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/recall_optimized_model/"
    os.makedirs(base_path, exist_ok=True)
    
    joblib.dump(model, base_path + 'recall_optimized_qb_archetype_rf_model.joblib')
    joblib.dump(label_encoder, base_path + 'recall_archetype_label_encoder.joblib')
    joblib.dump(scaler, base_path + 'recall_archetype_feature_scaler.joblib')
    joblib.dump(imputer, base_path + 'recall_archetype_feature_imputer.joblib')
    
    print(f"Recall-optimized model and preprocessors saved to: {base_path}")

def main():
    """Main function to run recall-optimized training."""
    # Load and prepare data
    X, y_str_labels, identifiers_all_clustered, imputer, scaler, X_imputed_df_full = load_and_prepare_data()
    if X is None:
        return

    # Train recall-optimized model
    best_rf_model_recall, label_encoder, X_test, y_test_encoded, identifiers_test = train_recall_optimized_model(
        X, y_str_labels, identifiers_all_clustered
    )

    # Evaluate model
    accuracy_recall, recall_macro, report_recall = evaluate_recall_optimized_model(
        best_rf_model_recall, label_encoder, X_test, y_test_encoded, identifiers_test
    )

    # Save model
    save_recall_optimized_model(best_rf_model_recall, label_encoder, imputer, scaler)

    print(f"\n=== FINAL RESULTS ===")
    print(f"Recall-Optimized Model Accuracy: {accuracy_recall:.4f}")
    print(f"Recall-Optimized Model Macro Recall: {recall_macro:.4f}")

if __name__ == "__main__":
    main() 