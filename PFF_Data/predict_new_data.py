import seaborn as sns
from matplotlib import pyplot as plt
import pandas as pd
import numpy as np
import joblib # For loading saved model and preprocessors
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler # Just for type hinting, actual object is loaded
from sklearn.impute import SimpleImputer # Just for type hinting
from sklearn.ensemble import RandomForestClassifier # Just for type hinting
from sklearn.preprocessing import LabelEncoder # Just for type hinting

# --- Configuration & Constants ---
MODEL_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/best_qb_archetype_rf_model.joblib'
LABEL_ENCODER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_label_encoder.joblib'
SCALER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_feature_scaler.joblib'
IMPUTER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_feature_imputer.joblib'

NEW_DATA_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data_2024/qb_player_merged_summary.csv' # IMPORTANT: Update this path
OUTPUT_PREDICTIONS_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis_2024/predicted_archetypes_2024.csv' # IMPORTANT: Update

# This MUST be the exact same list of raw feature names your model was trained on
# before any new feature engineering specific to this prediction step.
# It's the list of columns to select from the new data before imputation/scaling.
RAW_FEATURES_FOR_MODEL = [
    'accuracy_percent', 'avg_depth_of_target', 'avg_time_to_throw', 'btt_rate',
    'completion_percent', 'sack_percent', 'twp_rate', 'ypa', 'td_int_ratio',
    'comp_pct_diff', 'ypa_diff', 'designed_run_rate', 'scramble_rate',
    'elusive_rating', 'ypa_rushing', 'breakaway_percent', 'YAC_per_rush_attempt',
    'pct_total_yards_rushing', 'qb_rush_attempt_rate', 'grades_offense',
    'grades_pass', 'grades_run', 'pa_rate', 'pa_ypa', 'screen_rate',
    'deep_attempt_rate', 'deep_accuracy_percent', 'deep_twp_rate',
    'pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate',
    'pressure_accuracy_percent', 'quick_throw_rate'
    # Add any other raw features that were part of FEATURES_FOR_CLUSTERING in the training script
]

# Define identifier columns you want to keep from the new data
IDENTIFIER_COLUMNS = ['player', 'player_id', 'team_name', 'season', 'dropbacks'] # Adjust as needed, 'season' might be good

# --- 1. Load Saved Model and Preprocessing Objects ---
print("--- 1. Loading Model and Preprocessors ---")
try:
    model: RandomForestClassifier = joblib.load(MODEL_PATH)
    label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)
    scaler: StandardScaler = joblib.load(SCALER_PATH)
    imputer: SimpleImputer = joblib.load(IMPUTER_PATH)
    print("Model and preprocessors loaded successfully.")
except FileNotFoundError as e:
    print(f"Error loading saved objects: {e}. Ensure paths are correct and files exist.")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading objects: {e}")
    exit()

# --- 2. Load New 2024 Data ---
print("\n--- 2. Loading New 2024 QB Data ---")
try:
    df_new_data = pd.read_csv(NEW_DATA_PATH)
    # Basic cleaning of column names if needed (mirroring training script)
    df_new_data.columns = df_new_data.columns.str.rstrip('_')
    df_new_data.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
    print(f"Loaded 2024 data with shape: {df_new_data.shape}")
except FileNotFoundError:
    print(f"Error: New data file not found at {NEW_DATA_PATH}")
    exit()
except Exception as e:
    print(f"An unexpected error occurred while loading new data: {e}")
    exit()

# --- 3. Preprocess the 2024 Data ---
print("\n--- 3. Preprocessing 2024 Data ---")

# Make a copy to avoid modifying the original loaded DataFrame
df_new_data_processed = df_new_data.copy()

# a) Apply initial filters (e.g., dropbacks) - MUST BE THE SAME AS TRAINING
# Ensure 'dropbacks' column exists and is of a comparable type
if 'dropbacks' in df_new_data_processed.columns:
    try:
        # Convert to numeric, coercing errors if any non-numeric values exist
        df_new_data_processed['dropbacks'] = pd.to_numeric(df_new_data_processed['dropbacks'], errors='coerce')
        df_new_data_filtered = df_new_data_processed[df_new_data_processed["dropbacks"].notna() & (df_new_data_processed["dropbacks"].astype(int) >= 125)].copy()
        if df_new_data_filtered.empty:
            print("Warning: No players in the 2024 data meet the dropback filter criteria.")
            exit()
        print(f"Shape after dropback filter: {df_new_data_filtered.shape}")
    except Exception as e:
        print(f"Error during dropback filtering: {e}. Ensure 'dropbacks' column is suitable.")
        # Potentially exit or handle, depending on how critical this filter is
        df_new_data_filtered = df_new_data_processed # Fallback to using all if filter fails, with a warning
        print("Warning: Proceeding without dropback filter due to an error.")

else:
    print("Warning: 'dropbacks' column not found in new data. Skipping dropback filter.")
    df_new_data_filtered = df_new_data_processed # Use all data if no dropback column

# b) Select only the features the model was trained on
# First, ensure all necessary feature columns are present
missing_features = [col for col in RAW_FEATURES_FOR_MODEL if col not in df_new_data_filtered.columns]
if missing_features:
    print(f"Error: The new data is missing the following required features: {missing_features}")
    exit()

X_new_unscaled = df_new_data_filtered[RAW_FEATURES_FOR_MODEL].copy()

# c) Store identifiers before features are transformed
# Ensure IDENTIFIER_COLUMNS exist in df_new_data_filtered
actual_identifier_cols = [col for col in IDENTIFIER_COLUMNS if col in df_new_data_filtered.columns]
df_new_data_identifiers = df_new_data_filtered[actual_identifier_cols].copy()
if df_new_data_identifiers.empty and not X_new_unscaled.empty :
    print("Warning: No identifier columns were found or selected. Output will lack player info.")


# d) Handle infinite values
X_new_unscaled.replace([np.inf, -np.inf], np.nan, inplace=True)

# e) Impute missing values using the *loaded (already fit)* imputer
print("Applying loaded imputer...")
X_new_imputed_array = imputer.transform(X_new_unscaled) # Use .transform() NOT .fit_transform()
X_new_imputed_df = pd.DataFrame(X_new_imputed_array, columns=RAW_FEATURES_FOR_MODEL, index=X_new_unscaled.index)

# f) Scale features using the *loaded (already fit)* scaler
print("Applying loaded scaler...")
X_new_scaled_array = scaler.transform(X_new_imputed_df)   # Use .transform() NOT .fit_transform()
X_new_scaled = pd.DataFrame(X_new_scaled_array, columns=RAW_FEATURES_FOR_MODEL, index=X_new_imputed_df.index)

print(f"Shape of preprocessed 2024 features for prediction (X_new_scaled): {X_new_scaled.shape}")

if X_new_scaled.empty:
    print("Error: No data remains after preprocessing. Cannot make predictions.")
    exit()

# --- 4. Make Predictions ---
print("\n--- 4. Making Predictions on 2024 Data ---")
try:
    predictions_encoded = model.predict(X_new_scaled)
except Exception as e:
    print(f"Error during model prediction: {e}")
    print("This could be due to a mismatch in feature number/order if preprocessing changed columns.")
    exit()

# --- 5. Decode Predictions to Archetype Names ---
print("\n--- 5. Decoding Predictions ---")
try:
    predictions_names = label_encoder.inverse_transform(predictions_encoded)
except Exception as e:
    print(f"Error decoding predictions: {e}")
    # Fallback if decoding fails for some reason
    predictions_names = [f"Encoded_{p}" for p in predictions_encoded]


# --- 6. Analyze and Store 2024 Predictions ---
print("\n--- 6. Analyzing and Storing 2024 Predictions ---")

# Add predictions to the identifier DataFrame
# Ensure indices align if df_new_data_identifiers was subsetted
# Best to re-align using the index from X_new_scaled (which matches df_new_data_identifiers if created correctly)
df_predictions_output = df_new_data_identifiers.loc[X_new_scaled.index].copy() # Re-align using X_new_scaled's index
df_predictions_output['predicted_archetype_encoded'] = predictions_encoded
df_predictions_output['predicted_archetype_name'] = predictions_names

print("\nDistribution of Predicted Archetypes for 2024 Season:")
print(df_predictions_output['predicted_archetype_name'].value_counts())

print("\nFirst few 2024 players with their predicted archetypes:")
print(df_predictions_output.head())

# Optional: Spot-check some well-known QBs from 2024 if you know their IDs or names
# Example:
# joe_burrow_2024 = df_predictions_output[df_predictions_output['player'] == 'Joe Burrow'] # Adjust if name format differs
# if not joe_burrow_2024.empty:
#     print("\nJoe Burrow (2024) Predicted Archetype:")
#     print(joe_burrow_2024[['player', 'team_name', 'predicted_archetype_name']])

# --- 7. Save the 2024 Data with Predicted Archetypes ---
print("\n--- 7. Saving 2024 Predictions ---")
try:
    df_predictions_output.to_csv(OUTPUT_PREDICTIONS_PATH, index=False)
    print(f"2024 predictions saved to: {OUTPUT_PREDICTIONS_PATH}")
except Exception as e:
    print(f"Error saving predictions CSV: {e}")

print("\n--- Script Finished ---")
print("\n\n--- Starting Comparison with 2024 Clustered Archetypes ---")

# Define the path to your file containing 2024 stats WITH archetypes derived from CLUSTERING 2024 data
PATH_TO_2024_CLUSTERED_ARCHETYPES = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis_2024/merged_summary_with_archetypes.csv'
# Define column names for clarity (these should match your actual CSV headers)
PLAYER_ID_COL_IN_BOTH = 'player_id' # Must be common and unique identifier
TRUE_ARCHETYPE_COL_2024 = 'archetype_name' # Column name in your 2024 clustered file
PREDICTED_ARCHETYPE_COL_FROM_MODEL = 'predicted_archetype_name' # Column name in df_predictions_output

try:
    df_true_2024_clustered = pd.read_csv(PATH_TO_2024_CLUSTERED_ARCHETYPES)
print(f"Loaded 2024 clustered data (true labels) with shape: {df_true_2024_clustered.shape}")

# Ensure necessary columns exist
if PLAYER_ID_COL_IN_BOTH not in df_true_2024_clustered.columns or \
   TRUE_ARCHETYPE_COL_2024 not in df_true_2024_clustered.columns:
    raise ValueError(f"Missing key columns in 2024 clustered data file: {PATH_TO_2024_CLUSTERED_ARCHETYPES}")

    if PLAYER_ID_COL_IN_BOTH not in df_predictions_output.columns or \
       PREDICTED_ARCHETYPE_COL_FROM_MODEL not in df_predictions_output.columns:
        raise ValueError("Missing key columns in df_predictions_output.")

except FileNotFoundError:
    print(f"Error: 2024 clustered data file not found at {PATH_TO_2024_CLUSTERED_ARCHETYPES}")
    exit()
except ValueError as e:
    print(e)
    exit()


# --- Merge based on Player ID to align true 2024 archetypes with model predictions ---
# Select only necessary columns to avoid duplicate stat columns
# The df_predictions_output already contains identifiers from the prediction script
df_for_comparison = pd.merge(
    df_predictions_output[[PLAYER_ID_COL_IN_BOTH, 'player', 'team_name', PREDICTED_ARCHETYPE_COL_FROM_MODEL]], # Assuming 'player', 'team_name' are in df_predictions_output
    df_true_2024_clustered[[PLAYER_ID_COL_IN_BOTH, TRUE_ARCHETYPE_COL_2024]],
    on=PLAYER_ID_COL_IN_BOTH,
    how='inner' # Only compare players present in both and meeting all filters
)

if df_for_comparison.empty:
    print("Error: No common players found after merging predicted and true 2024 archetypes. Check filters and player_id consistency.")
    exit()

print(f"\nNumber of players for comparison after merge: {len(df_for_comparison)}")
print("Sample of merged data for comparison:")
print(df_for_comparison.head())

# --- Prepare labels for scikit-learn metrics ---
y_true_2024_names = df_for_comparison[TRUE_ARCHETYPE_COL_2024]
y_pred_2024_names = df_for_comparison[PREDICTED_ARCHETYPE_COL_FROM_MODEL]

# Ensure archetype names are consistent for encoding.
# The label_encoder loaded earlier was fit on the 2024 archetypes.
# If 2024 clustered archetypes use slightly different names or have a different set,
# this loaded label_encoder might fail or misinterpret.
# It's safer to fit a new one based on the actual names present in this comparison,
# or rigorously ensure your 2024 clustering process uses the exact same final archetype string names.

# Using the loaded label_encoder (assuming names are consistent from your 2024 training)
# This label_encoder was loaded at the beginning of predict_new_data.py
# label_encoder: LabelEncoder = joblib.load(LABEL_ENCODER_PATH)

# OR, for more robustness if names MIGHT differ slightly or if one set has names the other doesn't:
all_names_in_comparison = pd.concat([y_true_2024_names, y_pred_2024_names]).astype(str).unique() # astype(str) to handle potential mixed types
all_names_in_comparison.sort()
comparison_label_encoder = LabelEncoder()
comparison_label_encoder.fit(all_names_in_comparison)
print(f"Comparison Label Encoder Classes: {comparison_label_encoder.classes_}")


try:
    y_true_2024_encoded = comparison_label_encoder.transform(y_true_2024_names.astype(str))
    y_pred_2024_encoded = comparison_label_encoder.transform(y_pred_2024_names.astype(str))
except ValueError as e:
    print(f"ValueError during label encoding for comparison: {e}")
    print("This often means an archetype name in the 2024 data (either true or predicted) was not seen during fitting the comparison_label_encoder.")
    print(f"Unique True 2024 Archetypes: {y_true_2024_names.unique()}")
    print(f"Unique Predicted 2024 Archetypes by Model: {y_pred_2024_names.unique()}")
    print(f"Classes learned by comparison_label_encoder: {comparison_label_encoder.classes_}")
    exit()

# --- Calculate and Display Metrics ---
accuracy_2024_eval = accuracy_score(y_true_2024_encoded, y_pred_2024_encoded)
report_2024_eval = classification_report(y_true_2024_encoded, y_pred_2024_encoded,
                                         target_names=comparison_label_encoder.classes_, zero_division=0)
# Ensure labels for confusion matrix cover all classes in the encoder for consistent plotting
cm_labels_ordered = np.arange(len(comparison_label_encoder.classes_))
cm_2024_eval = confusion_matrix(y_true_2024_encoded, y_pred_2024_encoded, labels=cm_labels_ordered)

print(f"\n--- Evaluation: 2024 Model Predictions vs. 2024 Clustered Archetypes ---")
print(f"Overall Accuracy: {accuracy_2024_eval:.4f}")
print("\nClassification Report:")
print(report_2024_eval)
print("\nConfusion Matrix:")
plt.figure(figsize=(10, 7)) # Adjusted figure size for better label display
sns.heatmap(cm_2024_eval, annot=True, fmt='d', cmap='Blues',
            xticklabels=comparison_label_encoder.classes_,
            yticklabels=comparison_label_encoder.classes_)
plt.title('2024 Model Predictions vs. 2024 Clustered Archetypes')
plt.xlabel('Predicted Archetype (by 2024 Model)')
plt.ylabel('True Archetype (from 2024 Clustering)')
plt.tight_layout() # Adjust layout
plt.show()

# --- Show misclassified players for this comparison ---
df_for_comparison['correctly_classified'] = (y_true_2024_names == y_pred_2024_names)
misclassified_in_comparison = df_for_comparison[~df_for_comparison['correctly_classified']]

print("\n--- Players Where 2024 Model's Prediction Differs from 2024 Clustering ---")
if not misclassified_in_comparison.empty:
    print(misclassified_in_comparison[['player', 'team_name', TRUE_ARCHETYPE_COL_2024, PREDICTED_ARCHETYPE_COL_FROM_MODEL]])
else:
    print("No discrepancies found between model predictions and 2024 clustered archetypes (for common players).")
