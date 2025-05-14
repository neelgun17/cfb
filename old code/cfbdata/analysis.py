import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import adjusted_rand_score, accuracy_score, silhouette_score, confusion_matrix, classification_report

# 1. Load your labeled data
df = pd.read_csv('team_starting_qbs_labeled.csv')
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
# 2. Make sure you still have the manual labels in a column (rename them if needed)
#    For example, if you overwrote `qb_profile`, you should have saved the original as:
#    df['manual_profile'] = df['your_old_manu   al_column']
manual = df['manual_profile']
pred = df['qb_profile']  # human‐friendly labels you just mapped

# 3. Encode labels to integers
le = LabelEncoder().fit(pd.concat([manual, pred]))
y_true = le.transform(manual)
y_pred = le.transform(pred)

# 4. Compute ARI (unsupervised agreement) and accuracy
ari = adjusted_rand_score(y_true, y_pred)
accuracy = accuracy_score(y_true, y_pred)

# 5. Silhouette score on your feature matrix
features = numeric_cols  # the list you clustered on
X = df[features].values
sil = silhouette_score(X, df['cluster'])

# 6. Confusion matrix and classification report
cm = confusion_matrix(y_true, y_pred)
report = classification_report(y_true, y_pred, target_names=le.classes_)

# 7. Print out
print(f"Adjusted Rand Index: {ari:.3f}")
print(f"Label Accuracy       : {accuracy:.3%}")
print(f"Silhouette Score     : {sil:.3f}\n")
print("Confusion Matrix:")
print(cm)
print("\nClassification Report:")
print(report)