import pandas as pd
SEC = [
    "ALABAMA",
    "ARKANSAS",
    "AUBURN",
    "FLORIDA",
    "GEORGIA",
    "KENTUCKY",
    "LSU",
    "MISSISSIPPI",
    "MISSISSIPPI STATE",
    "MISSOURI",
    "OKLAHOMA",
    "SOUTH CAROLINA",
    "TENNESSEE",
    "TEXAS A&M",
    "TEXAS",        
    "VANDERBILT"
]
data = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/data/analysis/2024/hierarchical_player_assignments_k4.csv')

print(data.head())
print("TESTING")
for row in data.itertuples():
    if row.team_name in SEC:
        print(row.archetype_name, row.player)
