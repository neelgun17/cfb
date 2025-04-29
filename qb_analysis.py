import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt

human_labels = {
    0: 'Balanced Dual-Threat',
    1: 'Run-Heavy Option',
    2: 'Pure Game-Manager',
    3: 'Situational Pocket QBs',
    4: 'Pro-Style Starters',
    5: 'High-Volume Dual-Threats',
    6: 'Pure Triple-Option'
}

# 1) Load your data (adjust path if needed)
df = pd.read_csv("/Users/neelgundlapally/Documents/Projects/cfb/top50_qb_profiles.csv")

# 2) Pick the numeric columns to feed into clustering
numeric_cols = [
    'weight','height',
    'passing_att','passing_completions','passing_int','passing_pct','passing_td','passing_yds','passing_ypa',
    'rushing_car','rushing_long','rushing_td','rushing_yds','rushing_ypc',
    'usage.overall','usage.pass','usage.rush',
    'averagePPA.all','averagePPA.pass','averagePPA.rush',
    'pct_team_pass_snaps','pct_team_run_snaps','share_team_pass_snaps'
]

# 3) Drop any rows with missing values in those features
df_clean = df.dropna(subset=numeric_cols).copy()

# 4) Standardize the features
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[numeric_cols])

# 5) Elbow method: compute SSE for k=1..10
sse = []
ks = range(1, 11)
for k in ks:
    km = KMeans(n_clusters=k, random_state=42)
    km.fit(X)
    sse.append(km.inertia_)

# 6) Plot the elbow curve
plt.figure(figsize=(6,4))
plt.plot(ks, sse, marker='o')
plt.title("Elbow Method: SSE vs. Number of Clusters")
plt.xlabel("k (number of clusters)")
plt.ylabel("SSE (inertia)")
plt.show()

# 7) Pick a k (e.g. k=4) based on the elbow “knee”
k_opt = 7
kmeans = KMeans(n_clusters=k_opt, random_state=42)
df_clean['cluster'] = kmeans.fit_predict(X)

# 8) Show how many QBs fell into each cluster
counts = df_clean['cluster'].value_counts().sort_index()
print("Cluster counts:")
print(counts)

# 9) Peek at a few assignments
print("\nSample assignments:")
print(df_clean[['id','name','qb_profile','cluster']].head(19))

# --- Inspect Cluster Centroids ---
centers = kmeans.cluster_centers_
# invert scaling to original feature space
orig_centers = scaler.inverse_transform(centers)
centroid_df = pd.DataFrame(orig_centers, columns=numeric_cols)
centroid_df['cluster'] = range(len(centroid_df))
print("\nCluster centroids (original feature values):")
print(centroid_df.round(2))
print(centroid_df.sort_values('pct_team_pass_snaps', ascending=False).round(2))

# --- Label Clusters with generic names ---
labels = {i: f"Cluster_{i}" for i in range(k_opt)}
df_clean['cluster_label'] = df_clean['cluster'].map(labels)
# print("\nAssigned cluster labels:")
# print(df_clean[['id','name','cluster','cluster_label']].head(19))
print("\nAssigned human-friendly cluster labels (sorted by cluster):")
# Sort by the numeric cluster before printing
sorted_labels = df_clean.sort_values(by='cluster')
print(sorted_labels[['id','name','cluster','cluster_label']])

# --- PCA Visualization of Clusters ---
from sklearn.decomposition import PCA
pca = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
plt.figure(figsize=(6,6))
for cl in sorted(df_clean['cluster'].unique()):
    mask = df_clean['cluster'] == cl
    plt.scatter(coords[mask,0], coords[mask,1],
                label=human_labels.get(cl, f"Cluster {cl}"), alpha=0.7)
# Annotate each point with the QB’s name
for i, txt in enumerate(df_clean['name']):
    x, y = coords[i, 0], coords[i, 1]
    plt.text(x, y, txt, fontsize=8, alpha=0.75)
plt.legend(bbox_to_anchor=(1.05,1))
plt.title("QB Clusters (PCA projection)")
plt.xlabel("PC1")
plt.ylabel("PC2")
plt.tight_layout()
plt.show()

# --- Cross-tab of Cluster Labels vs Prototype ---
ct = pd.crosstab(df_clean['cluster_label'], df_clean['qb_profile'], normalize='index')
print("\nCross-tab of cluster_label vs qb_profile (proportion by cluster):")
print(ct.round(2))