# import pandas as pd
# import numpy as np
# from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
# from sklearn.preprocessing import LabelEncoder
# from sklearn.linear_model import LogisticRegression
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.svm import SVC
# import xgboost as xgb 
# from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
# import matplotlib.pyplot as plt
# import seaborn as sns
# from imblearn.over_sampling import SMOTE

# # --- 1. Load and Prepare Data ---
# # Assume you have these DataFrames from your previous clustering script:
# # df_filtered_hierarchical_k4: Contains 'player', 'team_name', 'archetype_name', and original stats.
# df_filtered_hierarchical_k4 = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/qb_data_with_archetypes_k4.csv"
# # cluster_data_scaled_df: Contains the SCALED features used for clustering.
# cluster_data_scaled_df = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/cluster_data_scaled_df.csv"
# # hierarchical_profiles_k4: Contains the mean profiles (not directly used here but for context).
# hierarchical_profiles_k4 = "/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_profiles_k4.csv"
# # features_for_clustering: The list of feature column names.
# features_for_clustering = [
#     # --- Overall Passing Style/Efficiency (Core) ---
#     'accuracy_percent',         # Overall accuracy
#     'avg_depth_of_target',      # Passing aggressiveness/style
#     'avg_time_to_throw',    # Pocket presence/style
#     'btt_rate',                 # Big play ability (passing)
#     'completion_percent',       # Basic efficiency
#     # 'qb_rating',              # Often redundant with other efficiency metrics
#     'sack_percent',             # Sack avoidance on dropbacks
#     'twp_rate',                 # Risk aversion (passing)
#     'ypa',                      # Overall passing efficiency
#     'td_int_ratio',             # Outcome ratio
#     'comp_pct_diff',            # Accuracy vs Expected (Advanced)
#     'ypa_diff',                 # YPA vs Expected (Advanced)

#     # --- Rushing Style/Efficiency/Impact (Enhanced) ---
#     'designed_run_rate',        # Tendency: Designed runs % of QB plays
#     'scramble_rate',            # Tendency: Scrambles % of dropbacks
#     'elusive_rating',           # Quality: Rushing ability metric
#     'ypa_rushing',              # Efficiency: Yards per rush attempt
#     'breakaway_percent',        # Quality: Explosive run plays %
#     'YAC_per_rush_attempt',     # CALCULATED: Quality: Yards After Contact / Rush Attempt
#     # 'designed_YPA',           # CALCULATED: Efficiency on designed runs (Optional, maybe covered by ypa_rushing)
#     # 'scramble_YPA',           # CALCULATED: Efficiency on scrambles (Optional)
#     'pct_total_yards_rushing', # CALCULATED & ADDED: **IMPACT**: Rushing Yards / (Pass Yards + Rush Yards)
#     'qb_rush_attempt_rate',    # CALCULATED & ADDED: **IMPACT**: QB Rushes / (Dropbacks + Designed Runs)

#     # --- PFF Grades (Holistic Eval) ---
#     'grades_offense',
#     'grades_pass',
#     'grades_run',

#     # --- Play Type Tendencies & *Selective* Performance ---
#     'pa_rate',                  # Tendency: Play-action usage rate
#     'pa_ypa',                   # Performance: Play-action effectiveness (YPA)
#     'screen_rate',              # Tendency: Screen usage rate
#     # 'screen_ypa',             # (Optional: screen effectiveness less differentiating?)

#     # --- Depth Tendencies & *Selective* Performance ---
#     'deep_attempt_rate',        # Tendency: Deep passing frequency
#     'deep_accuracy_percent',    # Performance: Deep ball skill/placement
#     'deep_twp_rate',            # Performance: Deep ball risk
#     # 'medium_attempt_rate',    # (Optional: Can be inferred if others included)
#     # 'short_attempt_rate',     # (Covered by complement of deep/medium?)
#     # 'behind_los_attempt_rate',# (Less critical maybe?)

#     # --- Pressure Handling (Crucial Split) ---
#     'pressure_rate',            # Tendency: How often pressured (Context)
#     'pressure_sack_percent',    # Performance: Sack avoidance *under pressure*
#     'pressure_twp_rate',        # Performance: Decision making *under pressure*
#     'pressure_accuracy_percent',# Performance: Accuracy *under pressure*
#     # 'pressure_ypa',           # (Optional: Maybe less critical than TWP/Sack%/Acc%)

#     # --- Time in Pocket Tendency ---
#     'quick_throw_rate',         # Tendency: % throws < 2.5s (Style)
#     # Remove performance metrics based on time (less critical than pressure/depth)

#     # --- Other ---
#     # 'yprr',                   # (If deemed relevant and reliable for QBs)
# ]

# # For demonstration, let's re-create or load the necessary components.
# # If you have them in memory, you can skip these loading steps.
# try:
#     # Path to the file containing player assignments and archetypes
#     player_assignments_path = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_player_assignments_k4.csv'
#     df_player_assignments = pd.read_csv(player_assignments_path)

#     # Path to the original CSV from which features were derived for scaling
#     # This is needed to reconstruct the scaled features in the same order if cluster_data_scaled_df isn't saved/available
#     original_data_path = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv'
#     df_original = pd.read_csv(original_data_path)
#     df_original.columns = df_original.columns.str.rstrip('_')
#     df_original.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
#     df_filtered_for_features = df_original[df_original["dropbacks"].astype(int) >= 125].copy() # Apply same initial filter
#     df_full_identifiers_and_labels = pd.read_csv(player_assignments_path)

#     # Re-create the scaled data ensuring alignment with archetype labels
#     # This assumes df_player_assignments has players in the same order as df_filtered_for_features
#     # If indices are not aligned, you'd need to merge/join carefully.
#     # For simplicity, let's assume df_player_assignments corresponds row-wise to the players
#     # that would make up cluster_data_scaled_df.

#     # Ensure df_filtered_for_features has the same players as df_player_assignments
#     # This might require merging based on player_id if order is not guaranteed
#     # For now, let's assume direct correspondence for feature selection
#     if len(df_filtered_for_features) != len(df_player_assignments):
#         print("Warning: Length mismatch between feature source and assignment file. Ensure correct player alignment.")
#         # Potentially, merge df_player_assignments with df_filtered_for_features on 'player_id'
#         # to ensure you're getting features for the correct players who have archetypes.
#         # Example (if needed, adjust columns):
#         # df_merged = pd.merge(df_player_assignments[['player_id', 'archetype_name']],
#         #                      df_filtered_for_features, on='player_id', how='inner')
#         # X_features_unscaled = df_merged[features_for_clustering].copy()
#         # y_labels_series = df_merged['archetype_name']
#         # identifiers_for_analysis = df_merged[['player', 'team_name']].copy() # Assuming player, team_name in df_merged
#     else: # Assuming direct row-wise correspondence
#         X_features_unscaled = df_filtered_for_features[features_for_clustering].copy()
#         y_labels_series = df_player_assignments['archetype_name'].copy()
#         identifiers_for_analysis = df_player_assignments[['player', 'team_name']].copy()


#     # Preprocessing (Imputation and Scaling - should mirror your clustering script)
#     from sklearn.impute import SimpleImputer
#     from sklearn.preprocessing import StandardScaler

#     X_features_unscaled.replace([np.inf, -np.inf], np.nan, inplace=True)
#     imputer = SimpleImputer(strategy='median')
#     X_imputed = imputer.fit_transform(X_features_unscaled)
#     X_imputed_df = pd.DataFrame(X_imputed, columns=features_for_clustering, index=X_features_unscaled.index)

#     scaler = StandardScaler()
#     X_scaled_array = scaler.fit_transform(X_imputed_df)
#     X = pd.DataFrame(X_scaled_array, columns=features_for_clustering, index=X_imputed_df.index)

#     y = y_labels_series # This is your target variable

#     print(f"Features shape (X): {X.shape}")
#     print(f"Target shape (y): {y.shape}")
#     print(f"Unique archetypes in y: {y.unique()}")
#     print(f"Archetype distribution:\n{y.value_counts()}")

# except FileNotFoundError as e:
#     print(f"Error loading files: {e}")
#     print("Please ensure the paths to your CSV files are correct and they exist.")
#     print("And that 'features_for_clustering' list is defined.")
#     exit()
# except KeyError as e:
#     print(f"KeyError: {e}. A specified column might be missing from a loaded DataFrame.")
#     print("Please check your CSV files and feature list.")
#     exit()


# # --- 2. Encode Target Labels (Archetype Names to Numbers) ---
# # Although many sklearn models handle string targets, encoding is good practice.
# label_encoder = LabelEncoder()
# y_encoded = label_encoder.fit_transform(y)
# # To see the mapping: print(dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_))))

# print("\n--- Label Encoding ---")
# for i, class_name in enumerate(label_encoder.classes_):
#     print(f"Numerical label {label_encoder.transform([class_name])[0]} -> Archetype: {class_name}")


# # --- 3. Split Data into Training and Testing sets ---
# # Stratify ensures that the proportion of archetypes is similar in train and test sets
# X_train, X_test, y_train, y_test, identifiers_train, identifiers_test = train_test_split(
#     X, y_encoded, identifiers_for_analysis,
#     test_size=0.25, # 25% for testing
#     random_state=42, # For reproducibility
#     stratify=y_encoded # Important for potentially imbalanced classes
# )


# # --- Define Parameter Grid for Random Forest ---
# # We are focusing on parameters that control tree complexity and number of trees
# # to get good generalization, especially given small sample sizes for some classes.
# param_grid_rf = {
#     'n_estimators': [100, 200, 300],  # Number of trees
#     'max_depth': [None, 5, 10, 15],   # Max depth of trees. None means nodes expanded until all leaves are pure or contain less than min_samples_split
#                                       # Start with relatively shallow depths (5, 10, 15) to prevent overfitting on small classes
#     'min_samples_split': [2, 5, 10],  # Minimum number of samples required to split an internal node
#     'min_samples_leaf': [1, 2, 4],    # Minimum number of samples required to be at a leaf node
#     'class_weight': ['balanced']      # Keep this as it showed some promise
#     # 'max_features': ['sqrt', 'log2'] # You mentioned this didn't make a difference, so we can omit for now or add back later
# }

# # --- Initialize GridSearchCV ---
# # We'll use StratifiedKFold for cross-validation to maintain class proportions in folds.
# # Scoring: 'f1_weighted' or 'f1_macro' are good general choices for multiclass.
# # 'f1_weighted' accounts for class imbalance in the metric itself.
# # 'f1_macro' gives equal weight to each class's F1-score. Given our goal, 'f1_weighted' might be slightly preferred initially.
# # Or, you can use 'accuracy' if overall accuracy is the primary driver now. Let's try 'f1_weighted'.

# cv_stratified = StratifiedKFold(n_splits=3, shuffle=True, random_state=42) # 3-5 splits is common

# grid_search_rf = GridSearchCV(
#     estimator=RandomForestClassifier(random_state=42),
#     param_grid=param_grid_rf,
#     scoring='f1_weighted', # or 'accuracy', or 'f1_macro'
#     cv=cv_stratified,
#     verbose=1,  # Shows progress
#     n_jobs=-1   # Use all available CPU cores
# )

# print("\nStarting GridSearchCV for Random Forest...")
# # Fit GridSearchCV on the training data
# grid_search_rf.fit(X_train, y_train)

# # --- Get the Best Model and Parameters ---
# best_rf_model = grid_search_rf.best_estimator_
# print("\nBest Random Forest Parameters Found:")
# print(grid_search_rf.best_params_)
# print(f"Best cross-validated f1_weighted score: {grid_search_rf.best_score_:.4f}")


# # --- Evaluate the Best Model on the Test Set ---
# y_pred_best_rf = best_rf_model.predict(X_test)

# accuracy_best_rf = accuracy_score(y_test, y_pred_best_rf)
# report_best_rf = classification_report(y_test, y_pred_best_rf, target_names=label_encoder.classes_, zero_division=0)
# cm_best_rf = confusion_matrix(y_test, y_pred_best_rf)

# print(f"\n--- Best Random Forest (from GridSearchCV) Results on Test Set ---")
# print(f"Accuracy: {accuracy_best_rf:.4f}")
# print("Classification Report:")
# print(report_best_rf)
# print("Confusion Matrix:")

# plt.figure(figsize=(8, 6))
# sns.heatmap(cm_best_rf, annot=True, fmt='d', cmap='Blues',
#             xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
# plt.title(f'Confusion Matrix - Best Random Forest')
# plt.xlabel('Predicted Label')
# plt.ylabel('True Label')
# plt.show()

# # --- (Optional) Feature Importances from the Best Random Forest Model ---
# print("\n--- Best Random Forest - Feature Importances ---")
# importances_best_rf = best_rf_model.feature_importances_
# feature_names = X.columns
# best_rf_importances = pd.Series(importances_best_rf, index=feature_names).sort_values(ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(x=best_rf_importances.head(15), y=best_rf_importances.head(15).index)
# plt.title('Top 15 Feature Importances (Best Random Forest)')
# plt.xlabel('Mean Decrease in Impurity')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()
# print("\nBest Random Forest - Top 10 Important Features:")
# print(best_rf_importances.head(15))

# print(f"\nX_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
# print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")




# misclassified_row_indices_in_test_arrays = np.where(y_test != y_pred_best_rf)[0]

# # --- 2. Get Original Player Identifiers and True/Predicted Archetypes for Misclassified Players ---
# # Use these row indices to select from identifiers_test, y_test_encoded, and y_pred_best_rf
# # Since identifiers_test was created by train_test_split alongside X_test and y_test_encoded,
# # their rows are aligned. We can use .iloc for row-based selection.

# misclassified_df = identifiers_test.iloc[misclassified_row_indices_in_test_arrays].copy()
# # Add true and predicted labels. These correspond to the order in misclassified_row_indices_in_test_arrays
# misclassified_df['true_archetype_encoded'] = y_test[misclassified_row_indices_in_test_arrays]
# misclassified_df['predicted_archetype_encoded'] = y_pred_best_rf[misclassified_row_indices_in_test_arrays]

# misclassified_df['true_archetype_name'] = label_encoder.inverse_transform(misclassified_df['true_archetype_encoded'])
# misclassified_df['predicted_archetype_name'] = label_encoder.inverse_transform(misclassified_df['predicted_archetype_encoded'])

# print("--- Misclassified Players ---")
# print(misclassified_df[['player', 'team_name', 'true_archetype_name', 'predicted_archetype_name']])

# # --- 3. Get Original (Unscaled) Stats for Misclassified Players ---
# # To get the stats, we need the ORIGINAL DataFrame indices that these misclassified test samples correspond to.
# # These original indices are stored in the index of X_test.
# original_df_indices_for_misclassified = X_test.iloc[misclassified_row_indices_in_test_arrays].index

# # Now use these original DataFrame indices to get rows from X_imputed_df
# misclassified_stats_df = X_imputed_df.loc[original_df_indices_for_misclassified, features_for_clustering].copy()

# # Combine with player info for easier viewing
# # We reset index on both to ensure clean concatenation based on row order
# misclassified_analysis_df = pd.concat([
#     misclassified_df.reset_index(drop=True),
#     misclassified_stats_df.reset_index(drop=True)
# ], axis=1)

# print("\n--- Stats of Misclassified Players (Original Scale) ---")
# # print(misclassified_analysis_df.head())

# # --- 4. Focus on Specific Misclassifications (as you listed) ---
# # (The rest of the code for filtering and comparing with profiles remains the same)
# # Example: Dynamic Dual-Threats misclassified
# ddt_misclassified = misclassified_analysis_df[
#     (misclassified_analysis_df['true_archetype_name'] == 'Dynamic Dual-Threats')
# ]
# print("\n--- Dynamic Dual-Threats Misclassified ---")
# # Make sure features_for_clustering is defined
# print(ddt_misclassified[['player', 'team_name', 'predicted_archetype_name'] + features_for_clustering])


# # Example: Mobile Pocket Passers misclassified
# mpp_misclassified = misclassified_analysis_df[
#     misclassified_analysis_df['true_archetype_name'] == 'Mobile Pocket Passer'
# ]
# print("\n--- Mobile Pocket Passers Misclassified ---")
# print(mpp_misclassified[['player', 'team_name', 'predicted_archetype_name'] + features_for_clustering])

# # pd.set_option('display.max_columns', None)  # Show all columns
# # pd.set_option('display.width', 1000)  

# # print("\n--- Dynamic Dual-Threats Misclassified (Full Stats) ---")
# # print(ddt_misclassified[['player', 'team_name', 'predicted_archetype_name'] + features_for_clustering]) # features_for_clustering has all stat names

# # print("\n--- Mobile Pocket Passers Misclassified (Full Stats) ---")
# # print(mpp_misclassified[['player', 'team_name', 'predicted_archetype_name'] + features_for_clustering])





# # --- 4. Define Models ---
# models = {
#     # "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42, solver='liblinear'), # liblinear for smaller datasets
#     # "Random Forest": RandomForestClassifier(random_state=42, n_estimators=100,class_weight='balanced'),
#     # "SVC": SVC(random_state=42, probability=True), # probability=True for consistency if needed later, slightly slower
#     # "XGBoost": xgb.XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='mlogloss') # use_label_encoder=False for newer XGBoost
# }
# results = {}
# # --- 5 & 6. Train and Evaluate Models ---
# print("\n--- Model Training and Evaluation ---")
# for name, model in models.items():
#     print(f"\nTraining {name}...")
#     model.fit(X_train, y_train)
#     y_pred = model.predict(X_test)

#     accuracy = accuracy_score(y_test, y_pred)
#     report = classification_report(y_test, y_pred, target_names=label_encoder.classes_, zero_division=0)
#     cm = confusion_matrix(y_test, y_pred)

#     results[name] = {"accuracy": accuracy, "report": report, "cm": cm, "model": model, "y_pred": y_pred}

#     print(f"--- {name} Results ---")
#     print(f"Accuracy: {accuracy:.4f}")
#     print("Classification Report:")
#     print(report)
#     print("Confusion Matrix:")
#     # Plotting Confusion Matrix
#     plt.figure(figsize=(8, 6))
#     sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
#                 xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
#     plt.title(f'Confusion Matrix - {name}')
#     plt.xlabel('Predicted Label')
#     plt.ylabel('True Label')
#     plt.show()

# # --- 7. Feature Importance (for Random Forest and XGBoost) ---
# print("\n--- Feature Importances ---")

# # Random Forest
# rf_model = results["Random Forest"]["model"]
# importances_rf = rf_model.feature_importances_
# feature_names = X.columns
# forest_importances = pd.Series(importances_rf, index=feature_names).sort_values(ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(x=forest_importances.head(15), y=forest_importances.head(15).index) # Top 15
# plt.title('Top 15 Feature Importances (Random Forest)')
# plt.xlabel('Mean Decrease in Impurity')
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()
# print("\nRandom Forest - Top 10 Important Features:")
# print(forest_importances.head(10))

# # XGBoost
# xgb_model = results["XGBoost"]["model"]
# importances_xgb = xgb_model.feature_importances_
# xgb_importances = pd.Series(importances_xgb, index=feature_names).sort_values(ascending=False)

# plt.figure(figsize=(10, 8))
# sns.barplot(x=xgb_importances.head(15), y=xgb_importances.head(15).index) # Top 15
# plt.title('Top 15 Feature Importances (XGBoost)')
# plt.xlabel('F score') # or 'weight', 'gain', etc. depending on importance_type
# plt.ylabel('Feature')
# plt.tight_layout()
# plt.show()
# print("\nXGBoost - Top 10 Important Features:")
# print(xgb_importances.head(10))


# # --- 8. (Optional) Analyze Misclassifications for a specific model ---
# # Let's analyze for Random Forest as an example
# print("\n--- Analyzing Misclassifications (Random Forest Example) ---")
# y_pred_rf = results["Random Forest"]["y_pred"]
# misclassified_indices = np.where(y_test != y_pred_rf)[0]

# # Get original indices from X_test
# original_indices_misclassified = X_test.iloc[misclassified_indices].index

# # Get player info for misclassified samples
# # Ensure identifiers_test was created with original indices from identifiers_for_analysis
# misclassified_players = identifiers_test.loc[original_indices_misclassified].copy() # Use .loc for safety
# misclassified_players['true_archetype'] = label_encoder.inverse_transform(y_test[misclassified_indices])
# misclassified_players['predicted_archetype'] = label_encoder.inverse_transform(y_pred_rf[misclassified_indices])

# print(f"Number of misclassified players by Random Forest: {len(misclassified_players)}")
# if not misclassified_players.empty:
#     print("Examples of misclassified players:")
#     print(misclassified_players.head())
# else:
#     print("No misclassifications by Random Forest on the test set (or test set was too small).")
import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
# Import other models like SVC, XGBoost if you plan to tune them similarly later
# import xgboost as xgb
# from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
# from imblearn.over_sampling import SMOTE # Keep if you plan to use SMOTE with XGBoost tuning

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
    df_filtered_stats = df_original_stats[df_original_stats["dropbacks"].astype(int) >= 125].copy()

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

import joblib
base_path =" /Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/"
joblib.dump(best_rf_model,base_path + 'best_qb_archetype_rf_model.joblib')
joblib.dump(label_encoder, base_path + 'archetype_label_encoder.joblib') # Save the encoder too!
joblib.dump(scaler, base_path + 'archetype_feature_scaler.joblib') # Save the scaler!
joblib.dump(imputer, base_path + 'archetype_feature_imputer.joblib') # Save the imputer!