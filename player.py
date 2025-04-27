from math import nan
import pandas as pd

try:
    total_players = pd.read_csv("/Users/neelgundlapally/Documents/Projects/cfb/College Football Data.csv")
except FileNotFoundError:
    print("Error: College Football Data.csv not found.")
    exit()

# header = list(total_players.columns)
# print(header)
# print(total_players.shape)

# total_players.info()

# remove players with Elibility of Withdrawn and players with destinations already
eligible_players = total_players[total_players["Eligibility"] != "Withdrawn"]

# only keep players with no assigned Destination
free_players = eligible_players[eligible_players["Destination"].isna()]
free_players['Stars'] = free_players['Stars'].fillna(0)
free_players.info()
print(free_players.head())


# summarize free players by Position
position_summary = free_players['Position'].value_counts().reset_index()
position_summary.columns = ['Position', 'Count']
print("\nFree players by Position:")
print(position_summary)

# summarize free players by StarCategory
star_summary = free_players['Stars'].value_counts().reset_index()
star_summary.columns = ['StarCategory', 'Count']
print("\nFree players by StarCategory:")
print(star_summary)

# cross-tabulation of free players by Position and StarCategory
crosstab = pd.crosstab(free_players['Position'], free_players['Stars'])
print("\nFree players cross-tabulated by Position and Stars:")
print(crosstab)

