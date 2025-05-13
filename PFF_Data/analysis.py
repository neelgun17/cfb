import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import numpy as np
from sklearn.cluster import AgglomerativeClustering


# --- 1. Preparation ---
# Load your CSV
df = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv')
df.columns = df.columns.str.rstrip('_')
df.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
print(len(df))
df_filtered = df[df["dropbacks"].astype(int) >= 125]
print(df.shape)
# Assume features_for_clustering is a list of column names decided earlier
features_for_clustering = [
    # --- Overall Passing Style/Efficiency (Core) ---
    'accuracy_percent',         # Overall accuracy
    'avg_depth_of_target',      # Passing aggressiveness/style
    'avg_time_to_throw',    # Pocket presence/style
    'btt_rate',                 # Big play ability (passing)
    'completion_percent',       # Basic efficiency
    # 'qb_rating',              # Often redundant with other efficiency metrics
    'sack_percent',             # Sack avoidance on dropbacks
    'twp_rate',                 # Risk aversion (passing)
    'ypa',                      # Overall passing efficiency
    'td_int_ratio',             # Outcome ratio
    'comp_pct_diff',            # Accuracy vs Expected (Advanced)
    'ypa_diff',                 # YPA vs Expected (Advanced)

    # --- Rushing Style/Efficiency/Impact (Enhanced) ---
    'designed_run_rate',        # Tendency: Designed runs % of QB plays
    'scramble_rate',            # Tendency: Scrambles % of dropbacks
    'elusive_rating',           # Quality: Rushing ability metric
    'ypa_rushing',              # Efficiency: Yards per rush attempt
    'breakaway_percent',        # Quality: Explosive run plays %
    'YAC_per_rush_attempt',     # CALCULATED: Quality: Yards After Contact / Rush Attempt
    # 'designed_YPA',           # CALCULATED: Efficiency on designed runs (Optional, maybe covered by ypa_rushing)
    # 'scramble_YPA',           # CALCULATED: Efficiency on scrambles (Optional)
    'pct_total_yards_rushing', # CALCULATED & ADDED: **IMPACT**: Rushing Yards / (Pass Yards + Rush Yards)
    'qb_rush_attempt_rate',    # CALCULATED & ADDED: **IMPACT**: QB Rushes / (Dropbacks + Designed Runs)

    # --- PFF Grades (Holistic Eval) ---
    'grades_offense',
    'grades_pass',
    'grades_run',

    # --- Play Type Tendencies & *Selective* Performance ---
    'pa_rate',                  # Tendency: Play-action usage rate
    'pa_ypa',                   # Performance: Play-action effectiveness (YPA)
    'screen_rate',              # Tendency: Screen usage rate
    # 'screen_ypa',             # (Optional: screen effectiveness less differentiating?)

    # --- Depth Tendencies & *Selective* Performance ---
    'deep_attempt_rate',        # Tendency: Deep passing frequency
    'deep_accuracy_percent',    # Performance: Deep ball skill/placement
    'deep_twp_rate',            # Performance: Deep ball risk
    # 'medium_attempt_rate',    # (Optional: Can be inferred if others included)
    # 'short_attempt_rate',     # (Covered by complement of deep/medium?)
    # 'behind_los_attempt_rate',# (Less critical maybe?)

    # --- Pressure Handling (Crucial Split) ---
    'pressure_rate',            # Tendency: How often pressured (Context)
    'pressure_sack_percent',    # Performance: Sack avoidance *under pressure*
    'pressure_twp_rate',        # Performance: Decision making *under pressure*
    'pressure_accuracy_percent',# Performance: Accuracy *under pressure*
    # 'pressure_ypa',           # (Optional: Maybe less critical than TWP/Sack%/Acc%)

    # --- Time in Pocket Tendency ---
    'quick_throw_rate',         # Tendency: % throws < 2.5s (Style)
    # Remove performance metrics based on time (less critical than pressure/depth)

    # --- Other ---
    # 'yprr',                   # (If deemed relevant and reliable for QBs)
]

# Calculate lengths

print(f"Revised feature count: Approx {len(features_for_clustering)}") # Should be around 35-40
# Create a copy for clustering analysis to avoid modifying the original filtered df
cluster_data = df_filtered[features_for_clustering].copy()

# Keep identifiers separate for later use
identifiers = df_filtered[['player', 'player_id', 'team_name']].reset_index(drop=True) # Reset index if needed

print(f"Starting with {cluster_data.shape[0]} QBs and {cluster_data.shape[1]} features for clustering.")

# --- 2. Preprocessing ---

# a) Handle Infinite Values (if any, from division by zero in calculated rates)
if np.isinf(cluster_data.values).any():
    print("Replacing infinite values with NaN...")
    cluster_data.replace([np.inf, -np.inf], np.nan, inplace=True)
    print(f"NaNs after replacing inf: {cluster_data.isna().sum().sum()}")

# b) Impute Missing Values (NaNs)
# Strategy: Use median imputation as it's robust to outliers common in sports stats
print("Handling missing values using Median Imputation...")
imputer = SimpleImputer(strategy='median')
# Fit on data and transform. Output is a NumPy array.
cluster_data_imputed = imputer.fit_transform(cluster_data)
# Convert back to DataFrame, preserving column names and index
cluster_data_imputed_df = pd.DataFrame(cluster_data_imputed, columns=features_for_clustering, index=cluster_data.index)
print(f"Remaining NaNs after imputation: {cluster_data_imputed_df.isna().sum().sum()}") # Should be 0

# c) Scale Features
# StandardScaler scales data to have zero mean and unit variance. Crucial for PCA and K-Means.
print("Scaling features using StandardScaler...")
scaler = StandardScaler()
# Fit on data and transform. Output is NumPy array.
cluster_data_scaled = scaler.fit_transform(cluster_data_imputed_df)
# Convert back to DataFrame, preserving column names and index
cluster_data_scaled_df = pd.DataFrame(cluster_data_scaled, columns=features_for_clustering, index=cluster_data.index)
print("Features successfully scaled.")
print(f"Scaled data shape: {cluster_data_scaled_df.shape}")


# --- ASSUME YOUR DATA IS LOADED AND PREPARED ---
# df_filtered: DataFrame with original player info and stats (filtered by dropbacks)
# cluster_data_scaled_df: DataFrame with scaled features for clustering
# cluster_data_imputed_df: DataFrame with imputed (but NOT scaled) features for profile means
# key_profile_stats: List of stats you want to see in the profile printout

# --- a) Perform Agglomerative Clustering for k=4 ---
N_CLUSTERS_HIERARCHICAL_K4 = 4 # <<< CHANGED TO 4

print(f"\nPerforming Agglomerative Clustering with k={N_CLUSTERS_HIERARCHICAL_K4}...")
agg_cluster_k4 = AgglomerativeClustering(n_clusters=N_CLUSTERS_HIERARCHICAL_K4, linkage='ward')
hierarchical_labels_k4 = agg_cluster_k4.fit_predict(cluster_data_scaled_df) # Use scaled data

# --- b) Add numeric cluster labels to your main DataFrame ---
df_filtered_hierarchical_k4 = df_filtered.copy()
df_filtered_hierarchical_k4['hierarchical_cluster_k4'] = hierarchical_labels_k4

print("\nHierarchical Cluster Distribution (k=4 - numeric labels):")
print(df_filtered_hierarchical_k4['hierarchical_cluster_k4'].value_counts().sort_index())

# --- ADDING NAMED ARCHETYPES ---
# 1. Define the mapping from cluster number to archetype name
# <<< USE THE NAMES WE DERIVED FROM THE HIERARCHICAL K=4 PROFILES >>>
archetype_map_hierarchical_k4 = {
    0: "Scrambling Survivors",
    1: "Pocket Managers",
    2: "Dynamic Dual-Threats",
    3: "Elite Efficient Passers"
}
print(f"\nUsing archetype map: {archetype_map_hierarchical_k4}")

# 2. Create the new 'archetype_name' column using the map
if 'hierarchical_cluster_k4' in df_filtered_hierarchical_k4.columns:
    df_filtered_hierarchical_k4['archetype_name'] = df_filtered_hierarchical_k4['hierarchical_cluster_k4'].map(archetype_map_hierarchical_k4)
    print("Successfully added 'archetype_name' column.")

    # 3. Verify the results with named archetypes
    print("\nFirst few rows with the new 'archetype_name' column:")
    print(df_filtered_hierarchical_k4[['player', 'team_name', 'hierarchical_cluster_k4', 'archetype_name']].head())

    print("\nDistribution of players across NAMED archetypes:")
    print(df_filtered_hierarchical_k4['archetype_name'].value_counts())
else:
    print("Error: 'hierarchical_cluster_k4' column not found for mapping to names.")
# --- END OF ADDING NAMED ARCHETYPES ---


# --- c) Generate Full Mean Statistical Profiles (FOR ME TO ANALYZE) ---

# Add the k4 hierarchical cluster labels to the imputed (but not scaled) data
cluster_data_imputed_df_k4 = cluster_data_imputed_df.copy() # Make sure this is your non-scaled data
cluster_data_imputed_df_k4['hierarchical_cluster_k4'] = hierarchical_labels_k4 # Ensure labels align

# Calculate mean stats for each of the 4 hierarchical clusters
hierarchical_profiles_k4 = cluster_data_imputed_df_k4.groupby('hierarchical_cluster_k4').mean()

print("\n--- Mean Statistical Profiles for Hierarchical Clustering (k=4) ---")

# Option 2: Save to CSV (Preferred for sharing with me if it's large)
output_profiles_csv = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_profiles_k4.csv'
hierarchical_profiles_k4.to_csv(output_profiles_csv)
print(f"\nMean profiles saved to: {output_profiles_csv}")


output_player_lists_csv = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_player_assignments_k4.csv'
# Select key columns for the player list output
player_list_cols = ['player', 'player_id', 'team_name', 'hierarchical_cluster_k4', 'archetype_name', 'dropbacks', 'grades_offense'] # Added archetype_name
# Add any other key stats you want to see alongside the player name
player_list_cols_present = [col for col in player_list_cols if col in df_filtered_hierarchical_k4.columns]

df_player_assignments_k4 = df_filtered_hierarchical_k4[player_list_cols_present].sort_values(by=['hierarchical_cluster_k4', 'dropbacks'], ascending=[True, False])
df_player_assignments_k4.to_csv(output_player_lists_csv, index=False)
print(f"\nPlayer assignments with k=4 hierarchical clustering (and names) saved to: {output_player_lists_csv}")
