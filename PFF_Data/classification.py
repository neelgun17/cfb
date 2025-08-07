import os
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

# --- Configuration & Constants ---
PLAYER_ASSIGNMENTS_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_player_assignments_k4.csv'
ORIGINAL_STATS_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv'
# HIERARCHICAL_PROFILES_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_profiles_k4.csv' # For error analysis later

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

# --- 1. Load and Prepare Data ---
print("--- 1. Loading and Preparing Data ---")
try:
    # df_player_assignments contains players who were clustered, their labels, and identifiers
    df_player_assignments = pd.read_csv(PLAYER_ASSIGNMENTS_PATH)
    if not all(col in df_player_assignments.columns for col in ['player_id', 'archetype_name', 'player', 'team_name']):
        raise ValueError("Player assignments CSV is missing required identifier or archetype columns.")

    # df_original contains all raw stats for all players
    df_original_stats = pd.read_csv(ORIGINAL_STATS_PATH)
    df_original_stats.columns = df_original_stats.columns.str.rstrip('_')
    df_original_stats.rename(columns=lambda c: c.replace('.', '_'), inplace=True)

    # Filter original stats (e.g., by dropbacks) - this should match the filter used BEFORE clustering
    df_filtered_stats = df_original_stats[df_original_stats["dropbacks"].astype(int) >= 150].copy()

    # Merge to get features AND labels for the CLUSTERED players ONLY
    # This ensures X and y are perfectly aligned and only include players who have an archetype
    df_merged_for_modeling = pd.merge(
        df_filtered_stats,
        df_player_assignments[['player_id', 'archetype_name']], # Select necessary columns from assignments
        on='player_id', # Assuming player_id is the common unique key
        how='inner'     # Use 'inner' to keep only players present in both (i.e., clustered and meeting stat filter)
    )
    # If player names/team names in df_player_assignments are more up-to-date, you can prioritize them:
    # df_merged_for_modeling.drop(columns=['player_x', 'team_name_x'], inplace=True, errors='ignore')
    # df_merged_for_modeling.rename(columns={'player_y': 'player', 'team_name_y': 'team_name'}, inplace=True, errors='ignore')
    print(df_merged_for_modeling.head())

    if df_merged_for_modeling.empty:
        raise ValueError("Merged DataFrame for modeling is empty. Check filters and merge keys.")

    print(f"Shape of merged data for modeling: {df_merged_for_modeling.shape}")

    # Prepare features (X) - Unscaled first
    X_unscaled = df_merged_for_modeling[FEATURES_FOR_CLUSTERING].copy()
    
    # Prepare identifiers for error analysis later
    identifiers_all_clustered = df_merged_for_modeling[['player', 'team_name', 'player_id']].copy() # Ensure these column names are correct

    # Prepare target (y) - String labels
    y_str_labels = df_merged_for_modeling['archetype_name'].copy()

    # Impute and Scale features
    X_unscaled.replace([np.inf, -np.inf], np.nan, inplace=True)
    imputer = SimpleImputer(strategy='median')
    X_imputed_array = imputer.fit_transform(X_unscaled)
    X_imputed_df_full = pd.DataFrame(X_imputed_array, columns=FEATURES_FOR_CLUSTERING, index=X_unscaled.index) # For error analysis

    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_imputed_df_full)
    X = pd.DataFrame(X_scaled_array, columns=FEATURES_FOR_CLUSTERING, index=X_unscaled.index)

    print(f"Features shape (X): {X.shape}")
    print(f"Target shape (y_str_labels): {y_str_labels.shape}")
    print(f"Unique archetypes in y: {y_str_labels.unique()}")
    print(f"Archetype distribution:\n{y_str_labels.value_counts()}")

except FileNotFoundError as e:
    print(f"Error loading files: {e}")
    exit()
except (KeyError, ValueError) as e:
    print(f"Error in data preparation: {e}")
    exit()

# --- 2. Encode Target Labels ---
print("\n--- 2. Encoding Target Labels ---")
label_encoder = LabelEncoder()
y_encoded = label_encoder.fit_transform(y_str_labels)
for i, class_name in enumerate(label_encoder.classes_): # Use enumerate for cleaner mapping display
    print(f"Numerical label {i} -> Archetype: {class_name}") # Assumes transform gives 0 to N-1

# --- 3. Split Data ---
print("\n--- 3. Splitting Data ---")
X_train, X_test, y_train_encoded, y_test_encoded, identifiers_train, identifiers_test = train_test_split(
    X, y_encoded, identifiers_all_clustered, # Pass the identifiers DF
    test_size=0.25,
    random_state=42,
    stratify=y_encoded
)
print(f"X_train shape: {X_train.shape}, y_train_encoded shape: {y_train_encoded.shape}")
print(f"X_test shape: {X_test.shape}, y_test_encoded shape: {y_test_encoded.shape}")

# --- 4. Hyperparameter Tuning for Random Forest (Example) ---
print("\n--- 4. Hyperparameter Tuning for Random Forest ---")
param_grid_rf = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 5, 10, 15],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4],
    'class_weight': ['balanced']
}
cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42)
grid_search_rf = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid_rf,
    scoring='f1_weighted', # Or 'f1_macro' if you want to give more weight to smaller classes
    cv=cv_stratified,
    verbose=1,
    n_jobs=-1
)
grid_search_rf.fit(X_train, y_train_encoded)
best_rf_model = grid_search_rf.best_estimator_
print("\nBest Random Forest Parameters Found:")
print(grid_search_rf.best_params_)
print(f"Best cross-validated f1_weighted score: {grid_search_rf.best_score_:.4f}")

# --- 5. Evaluate Best Tuned Model ---
print("\n--- 5. Evaluating Best Tuned Random Forest Model on Test Set ---")
y_pred_best_rf = best_rf_model.predict(X_test)
accuracy_best_rf = accuracy_score(y_test_encoded, y_pred_best_rf)
report_best_rf = classification_report(y_test_encoded, y_pred_best_rf, target_names=label_encoder.classes_, zero_division=0)
cm_best_rf = confusion_matrix(y_test_encoded, y_pred_best_rf)

print(f"Accuracy: {accuracy_best_rf:.4f}")
print("Classification Report:")
print(report_best_rf)
print("Confusion Matrix:")
plt.figure(figsize=(8, 6))
sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues',
            xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.title(f'Confusion Matrix - Best Random Forest')
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.show()

# --- 6. Feature Importances for Best Tuned Model ---
print("\n--- 6. Feature Importances for Best Tuned Random Forest Model ---")
importances_best_rf = best_rf_model.feature_importances_
feature_names_rf = X.columns # Ensure X is a DataFrame
best_rf_importances = pd.Series(importances_best_rf, index=feature_names_rf).sort_values(ascending=False)
plt.figure(figsize=(10, 8))
sns.barplot(x=best_rf_importances.head(15), y=best_rf_importances.head(15).index)
plt.title('Top 15 Feature Importances (Best Random Forest)')
plt.xlabel('Mean Decrease in Impurity')
plt.ylabel('Feature')
plt.tight_layout()
plt.show()
print("Top 15 Important Features (Best Random Forest):") # Changed to 15 to match plot
print(best_rf_importances.head(15))

# --- 7. Detailed Error Analysis for Best Tuned Model ---
print("\n--- 7. Detailed Error Analysis for Best Tuned Random Forest Model ---")
misclassified_row_indices_in_test = np.where(y_test_encoded != y_pred_best_rf)[0]
original_df_indices_for_misclassified = X_test.iloc[misclassified_row_indices_in_test].index

misclassified_df_identifiers = identifiers_test.iloc[misclassified_row_indices_in_test].copy()
misclassified_df_identifiers['true_archetype_encoded'] = y_test_encoded[misclassified_row_indices_in_test]
misclassified_df_identifiers['predicted_archetype_encoded'] = y_pred_best_rf[misclassified_row_indices_in_test]
misclassified_df_identifiers['true_archetype_name'] = label_encoder.inverse_transform(misclassified_df_identifiers['true_archetype_encoded'])
misclassified_df_identifiers['predicted_archetype_name'] = label_encoder.inverse_transform(misclassified_df_identifiers['predicted_archetype_encoded'])

print("\nMisclassified Players Summary:")
print(misclassified_df_identifiers[['player', 'team_name', 'true_archetype_name', 'predicted_archetype_name']])

# Get original (unscaled) stats for misclassified players
misclassified_stats_unscaled = X_imputed_df_full.loc[original_df_indices_for_misclassified].copy()

# Combine identifiers with unscaled stats for full analysis DataFrame
misclassified_analysis_df = pd.concat([
    misclassified_df_identifiers.reset_index(drop=True),
    misclassified_stats_unscaled.reset_index(drop=True)
], axis=1)

print("\nStats of All Misclassified Players (Original Scale) - First 5 rows:")
# pd.set_option('display.max_columns', None) # Uncomment to see all columns if needed
print(misclassified_analysis_df.head())

# Example: Filter for specific misclassifications to print their full stats
ddt_misclassified_detailed = misclassified_analysis_df[
    misclassified_analysis_df['true_archetype_name'] == 'Dynamic Dual-Threats'
]
print("\n--- Dynamic Dual-Threats Misclassified (Full Stats for Analysis) ---")
if not ddt_misclassified_detailed.empty:
    print(ddt_misclassified_detailed[['player', 'team_name', 'predicted_archetype_name'] + FEATURES_FOR_CLUSTERING])
else:
    print("No Dynamic Dual-Threats were misclassified (or none were in the test set and misclassified).")

mpp_misclassified_detailed = misclassified_analysis_df[
    misclassified_analysis_df['true_archetype_name'] == 'Mobile Pocket Passer'
]
print("\n--- Mobile Pocket Passers Misclassified (Full Stats for Analysis) ---")
if not mpp_misclassified_detailed.empty:
    print(mpp_misclassified_detailed[['player', 'team_name', 'predicted_archetype_name'] + FEATURES_FOR_CLUSTERING])
else:
    print("No Mobile Pocket Passers were misclassified (or none were in the test set and misclassified).")

# --- [End of Script or Add Tuning for other models like XGBoost here] ---
# If you want to tune XGBoost, you would add a similar GridSearchCV block for it.
# Remember to handle class imbalance (e.g., SMOTE on X_train_smote, y_train_smote_encoded or sample_weight)
# if you are tuning XGBoost specifically for that.


base_path ="/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/"
os.makedirs(base_path, exist_ok=True)
joblib.dump(best_rf_model,base_path + 'best_qb_archetype_rf_model.joblib')
joblib.dump(label_encoder, base_path + 'archetype_label_encoder.joblib') # Save the encoder too!
joblib.dump(scaler, base_path + 'archetype_feature_scaler.joblib') # Save the scaler!
joblib.dump(imputer, base_path + 'archetype_feature_imputer.joblib') # Save the imputer!