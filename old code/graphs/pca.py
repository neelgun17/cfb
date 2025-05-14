import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# 1) List the exact features you used for clustering
numeric_cols = [
    # 'weight','height',
    # 'passing_att','passing_completions','passing_int','passing_pct','passing_td','passing_yds','passing_ypa',
    # 'rushing_car','rushing_long','rushing_td','rushing_yds','rushing_ypc',
    # 'usage.overall','usage.pass','usage.rush',
    # 'averagePPA.all','averagePPA.pass','averagePPA.rush',
    # 'pct_team_pass_snaps','pct_team_run_snaps','share_team_pass_snaps'
     # Passing volume & efficiency
    'passing_att', 'passing_pct', 'passing_yds', 'passing_td',
    # Rushing volume & efficiency
    'rushing_car', 'rushing_yds', 'rushing_td', 'rushing_ypc',
    # Usage rates (team-relative play shares)
    'usage.pass', 'usage.rush', 'usage.passingDowns', 'usage.thirdDown',
    # PPA
    'averagePPA.all','averagePPA.pass','averagePPA.rush','averagePPA.passingDowns'
]
# 2) Load your final, labeled QB dataset (update path as needed)
df = pd.read_csv('team_starting_qbs_labeled.csv')

# 3) Drop any rows missing those numeric features
df_clean = df.dropna(subset=numeric_cols)

# 4) Standardize the feature matrix
scaler = StandardScaler()
X = scaler.fit_transform(df_clean[numeric_cols])

# 5) Fit a 2-component PCA
pca = PCA(n_components=2, random_state=42)
pca.fit(X)

# 6) Extract the loadings (feature weights) for PC1 and PC2
loadings = pd.DataFrame(
    pca.components_.T,
    index=numeric_cols,
    columns=['PC1', 'PC2']
)

# 7) Add absolute values to rank by magnitude
loadings['abs_PC1'] = loadings['PC1'].abs()
loadings['abs_PC2'] = loadings['PC2'].abs()

# 8) Display the full loadings (or export them)
print(loadings.round(3))

# 9) Print out the top 5 drivers of each component
top_pc1 = loadings.sort_values('abs_PC1', ascending=False).head(5).index.tolist()
top_pc2 = loadings.sort_values('abs_PC2', ascending=False).head(5).index.tolist()
print("Top 5 features for PC1:", top_pc1)
print("Top 5 features for PC2:", top_pc2)