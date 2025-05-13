import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.manifold import TSNE
from sklearn.cluster import KMeans
from sklearn.impute import SimpleImputer
from sklearn.metrics import silhouette_score
import numpy as np

# --- 1. Preparation ---
# Load your CSV
df = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv')
df.columns = df.columns.str.rstrip('_')
# normalize column names: replace dots with underscores
df.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
# Assume df_filtered exists (QBs meeting min dropbacks threshold)
df_filtered = df[df["dropbacks"].astype(int) >= 125]

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
# --- 3. Dimensionality Reduction (PCA) ---

# Decide on the number of components.
# Option 1: Choose a fixed number (e.g., 10-30)
# Option 2: Choose based on explained variance (e.g., 0.90 or 0.95)
N_COMPONENTS = 0.90 # Aim to retain 90% of the variance
# N_COMPONENTS = 20 # Alternatively, choose a fixed number

print(f"Applying PCA to reduce dimensions, aiming for {N_COMPONENTS} variance or components...")
pca = PCA(n_components=N_COMPONENTS, random_state=42)
cluster_data_pca = pca.fit_transform(cluster_data_scaled_df)

print(f"Shape after PCA: {cluster_data_pca.shape}")
print(f"Total explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
# Optional: Plot explained variance per component
# plt.figure(figsize=(10, 6))
# plt.plot(range(1, pca.n_components_ + 1), np.cumsum(pca.explained_variance_ratio_), marker='o')
# plt.xlabel('Number of Components')
# plt.ylabel('Cumulative Explained Variance')
# plt.title('PCA Explained Variance')
# plt.grid(True)
# plt.show()

# --- 4. Clustering (K-Means) ---

# a) Determine the Optimal Number of Clusters (k)
# Use Elbow Method (Inertia) and Silhouette Score

k_range = range(4, 10) # Explore values around your desired 5-7 archetypes
inertias = []
silhouette_scores = []

print("\nCalculating Inertia and Silhouette Scores for K-Means...")
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42, n_init=10) # n_init=10 helps stability
    kmeans.fit(cluster_data_pca)
    inertias.append(kmeans.inertia_) # Inertia: Sum of squared distances to closest centroid
    silhouette_scores.append(silhouette_score(cluster_data_pca, kmeans.labels_))
    print(f"  k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette Score={silhouette_scores[-1]:.3f}")

# # Plot Elbow Method
# plt.figure(figsize=(12, 5))
# plt.subplot(1, 2, 1)
# plt.plot(k_range, inertias, marker='o')
# plt.title('Elbow Method (Inertia)')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Inertia')
# plt.grid(True)

# # Plot Silhouette Scores
# plt.subplot(1, 2, 2)
# plt.plot(k_range, silhouette_scores, marker='o')
# plt.title('Silhouette Scores')
# plt.xlabel('Number of clusters (k)')
# plt.ylabel('Silhouette Score')
# plt.grid(True)

# plt.tight_layout()
# plt.show()

# b) Choose k and Run Final K-Means
# Look at the plots:
# - Elbow: Where does the plot "bend" like an elbow? (Rate of decrease slows)
# - Silhouette: Where is the score highest? (Higher is better, closer to 1)
# Balance these with your desired 5-7 archetypes. Let's assume 6 seems best.

chosen_k = 4 # <<< ADJUST THIS based on your analysis of the plots and preference!
print(f"\nBased on analysis, choosing k = {chosen_k}")

final_kmeans = KMeans(n_clusters=chosen_k, random_state=42, n_init=10)
final_kmeans.fit(cluster_data_pca)
cluster_labels = final_kmeans.labels_

# --- 5. Assign Labels and Analyze Results ---

# Add cluster labels back to the identifiers DataFrame
identifiers['cluster'] = cluster_labels

# Add cluster labels back to the scaled data for analysis (optional but useful)
cluster_data_scaled_df['cluster'] = cluster_labels
# Add cluster labels back to the imputed (but not scaled) data for interpretation
cluster_data_imputed_df['cluster'] = cluster_labels

print(f"\nAssigned QBs to {chosen_k} clusters.")
print("Cluster Distribution:")
print(identifiers['cluster'].value_counts().sort_index())

# Analyze Cluster Characteristics
# Calculate the mean (or median) of the *scaled* features for each cluster centroid
# This shows how clusters differ in standardized terms
cluster_centers_scaled = pd.DataFrame(scaler.inverse_transform(pca.inverse_transform(final_kmeans.cluster_centers_)),
                                     columns=features_for_clustering) # Transform centers back to original feature scale

# OR calculate mean/median of *original imputed* features per cluster (easier interpretation)
print("\nCluster Profiles (Mean Values of Imputed Features):")
cluster_profiles = cluster_data_imputed_df.groupby('cluster').mean() # Use .median() if preferred

# Display profiles - focus on key differentiating features
# Example: Check a few key stats
key_profile_stats = [
    'designed_run_rate', 'scramble_rate', 'ypa_rushing', 'elusive_rating', # Rushing
    'avg_depth_of_target', 'btt_rate', 'twp_rate', 'accuracy_percent', # Passing Style
    'pa_rate', 'pa_ypa', # Play Action
    'pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate', # Pressure Handling
    'grades_offense', 'grades_pass', 'grades_run' # PFF Grades
]
# Filter profile display for clarity - check intersection in case some names differ slightly
display_stats = [col for col in key_profile_stats if col in cluster_profiles.columns]
print(cluster_profiles[display_stats])
df_for_display = df_filtered[['player', 'player_id', 'team_name', 'dropbacks', 'grades_offense']].copy() # Add other relevant stats
df_for_display['cluster'] = cluster_labels
# Display first few players per cluster
print("\nExample Players per Cluster:")
for i in range(chosen_k):
    print(f"\n--- Cluster {i} ---")
    # print(identifiers[identifiers['cluster'] == i].head())
    cluster_members = df_for_display[df_for_display['cluster'] == i]

    sorted_members = cluster_members.sort_values(by=['dropbacks', 'grades_offense'], ascending=[False, False])
    print(sorted_members[['player', 'team_name', 'dropbacks', 'grades_offense']].head(10))

# --- Optional: Visualize Clusters (using first 2 PCA components) ---
# if pca.n_components_ >= 2:
#     plt.figure(figsize=(10, 8))
#     sns.scatterplot(x=cluster_data_pca[:, 0], y=cluster_data_pca[:, 1], hue=cluster_labels, palette='viridis', s=50, alpha=0.7)
#     # Optional: Plot cluster centers
#     # centers_pca = final_kmeans.cluster_centers_
#     # plt.scatter(centers_pca[:, 0], centers_pca[:, 1], c='red', s=200, alpha=0.75, marker='X');
#     plt.title('QB Clusters (PCA Components 1 & 2)')
#     plt.xlabel('Principal Component 1')
#     plt.ylabel('Principal Component 2')
#     plt.legend(title='Cluster')
#     plt.grid(True)
#     plt.show()

# Assume cluster_profiles is your DataFrame of mean stats per cluster
# And 'key_profile_stats' is the list we defined earlier (or a similar one)

# Ensure key_profile_stats only contains columns present in cluster_profiles
# Save cluster profiles (filtered by display_stats) to Excel
output_path = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/qb_cluster_profiles.xlsx'
with pd.ExcelWriter(output_path, engine='xlsxwriter') as writer:
    cluster_profiles[display_stats].to_excel(writer, sheet_name='Cluster_Profiles')
print(f"Cluster profiles written to Excel at {output_path}")


# print("\nPrototypical Players per Cluster (Closest to Centroid in PCA Space):")
# # cluster_data_pca is your data after PCA
# # final_kmeans.cluster_centers_ are the centroids in PCA space
# # cluster_labels are the assigned cluster for each player in cluster_data_pca

# for i in range(chosen_k): # chosen_k is your number of clusters
#     print(f"\n--- Cluster {i} ---")
#     # Get indices of players in this cluster
#     indices_in_cluster = np.where(cluster_labels == i)[0]
    
#     # Get the PCA data for players in this cluster
#     pca_data_of_cluster_members = cluster_data_pca[indices_in_cluster]
    
#     # Get the centroid for this cluster
#     centroid_pca = final_kmeans.cluster_centers_[i].reshape(1, -1) # Reshape for distance calculation
    
#     # Calculate distances
#     distances = euclidean_distances(pca_data_of_cluster_members, centroid_pca)
    
#     # Get indices of the closest N players *within this cluster's subset*
#     num_prototypes = 5
#     closest_indices_in_subset = np.argsort(distances.ravel())[:num_prototypes]
    
#     # Map these subset indices back to original indices in 'identifiers' or 'df_filtered'
#     original_indices_of_prototypes = indices_in_cluster[closest_indices_in_subset]
    
#     # Get the player names and other info
#     # Assuming 'identifiers' DataFrame index aligns with 'cluster_data_pca' rows
#     prototypical_players = identifiers.iloc[original_indices_of_prototypes]
#     print(prototypical_players[['player', 'team_name']])import pandas as pd

# Assume 'df_filtered' is your DataFrame containing the players who were clustered
# and it has a column named 'cluster' with values 0, 1, 2, 3 from the k=4 run.

# 1. Define the mapping from cluster number to archetype name
# <<< Make sure these names exactly match the ones you finalized above >>>
import pandas as pd
import numpy as np # Make sure numpy is imported

# --- ASSUMPTIONS ---
# 1. 'df_filtered' is your DataFrame containing the QBs who met the dropback threshold
#    (the same set of players, in the same order, that went into the clustering process).
# 2. 'cluster_labels' is the numpy array or list containing the cluster assignments (0, 1, 2, 3)
#    that resulted from running final_kmeans.fit() or final_kmeans.predict() on your PCA data.
#    It MUST have the same length as the number of rows in df_filtered.

# --- Step 1: Verify Lengths Match (Crucial Sanity Check!) ---
if len(cluster_labels) == len(df_filtered):
    print(f"Lengths match: cluster_labels ({len(cluster_labels)}) and df_filtered ({len(df_filtered)}). Proceeding.")

    # --- Step 2: Assign the labels to the new 'cluster' column ---
    # This directly adds the array/list as a new column.
    # It relies on the order of players in df_filtered being the same
    # as the order of data points used to generate cluster_labels.
    # If you reset the index on df_filtered just before clustering, this works fine.
    df_filtered['cluster'] = cluster_labels

    print("Successfully created the 'cluster' column.")

    # --- Step 3: Verify the column was added ---
    print("\nFirst few rows showing the new 'cluster' column:")
    print(df_filtered[['player', 'team_name', 'cluster']].head())

    print("\nData types including new column:")
    print(df_filtered.info())

else:
    print(f"Error: Length mismatch!")
    print(f"Length of cluster_labels: {len(cluster_labels)}")
    print(f"Length of df_filtered: {len(df_filtered)}")
    print("The cluster labels cannot be assigned directly.")
    print("Ensure 'df_filtered' contains exactly the same players, in the same order,")
    print("as the data used to generate the 'cluster_labels'.")

# --- Now you can run the mapping code from the previous step ---
# (Assuming the above steps worked correctly)

archetype_map = {
    0: 'Pocket Managers',
    1: 'High-Variance Dual-Threats',
    2: 'Run-Focused Scramblers',
    3: 'Efficient Dual-Threats'
}
# 2. Create the new 'archetype' column using the map
if 'cluster' in df_filtered.columns:
    df_filtered.insert(2, 'archetype', df_filtered['cluster'].map(archetype_map))
    print("\nSuccessfully added 'archetype' column.")

    # 3. Verify the results
    print("\nFirst few rows with the new 'archetype' column:")
    print(df_filtered[['player', 'team_name', 'cluster', 'archetype']].head())
    print("\nDistribution of players across archetypes:")
    print(df_filtered['archetype'].value_counts())


else:
    # This error shouldn't happen if the first part worked
    print("\nError: 'cluster' column assignment failed earlier.")


# 4. Save the updated DataFrame (Optional but Recommended)
output_filename = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/qb_data_with_archetypes_k4.csv'
df_filtered.to_csv(output_filename, index=False)
print(f"\nDataFrame with archetypes saved to {output_filename}")