import pandas as pd

# Load CSV
df = pd.read_csv("top50_qb_profiles.csv")

# Define logical column groups
personal_info = [
    "id", "team", "name", "jersey", "position", "hometown", "height", "weight"
]
team_info = [
    "teamColor", "teamColorSecondary", "conference"
]
passing_stats = [
    "passing_att", "passing_completions", "passing_pct", "passing_td", "passing_int",
    "passing_yds", "passing_ypa"
]
rushing_stats = [
    "rushing_car", "rushing_yds", "rushing_td", "rushing_ypc", "rushing_long"
]
fumbles = [
    "fumbles_fum", "fumbles_lost", "fumbles_rec"
]
usage = [
    "usage.overall", "usage.pass", "usage.rush", "usage.firstDown",
    "usage.secondDown", "usage.thirdDown", "usage.standardDowns", "usage.passingDowns"
]
efficiency = [
    "countablePlays", "averagePPA.all", "averagePPA.pass", "averagePPA.rush",
    "averagePPA.firstDown", "averagePPA.secondDown", "averagePPA.thirdDown",
    "averagePPA.standardDowns", "averagePPA.passingDowns"
]
team_share = [
    "pct_team_pass_snaps", "pct_team_run_snaps", "share_team_pass_snaps"
]
other = [
    "qb_profile"
]

# Combine the column groups
ordered_columns = (
    personal_info + team_info + passing_stats + rushing_stats +
    fumbles + usage + efficiency + team_share + other
)

# Ensure no missing columns (in case of unexpected CSV structure)
ordered_columns = [col for col in ordered_columns if col in df.columns]

# Reorder and save
df = df[ordered_columns]
df.to_csv("top50_qb_profiles_readable.csv", index=False)

print("CSV reordered and saved as 'top50_qb_profiles_readable.csv'")

# Save grouped CSVs
df[personal_info + team_info].to_csv("top50_qb_personal_team_info.csv", index=False)
df[["name", "team"] + passing_stats + rushing_stats + fumbles].to_csv("top50_qb_basic_stats.csv", index=False)
df[["name", "team"] + usage + efficiency + team_share + other].to_csv("top50_qb_advanced_metrics.csv", index=False)

print("Split CSVs saved as:")
print("- top50_qb_personal_team_info.csv")
print("- top50_qb_basic_stats.csv")
print("- top50_qb_advanced_metrics.csv")

# Save full readable version as JSON
df.to_json("top50_qb_profiles_readable.json", orient="records", indent=2)
print("- top50_qb_profiles_readable.json (JSON format)")