import csv
import os

input_file = 'team_starting_qbs.csv'       # Original CSV
# output_file = 'insuffiecent_players.csv'  # New CSV with only players ending in many commas

# with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
#      open(output_file, 'w', newline='', encoding='utf-8') as outfile:

#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     for row in reader:
#         # Join the row into a string and check if it ends with 15 commas
#         if ','.join(row).endswith(',' * 15):
#             writer.writerow(row)

# Create new CSV with the remaining players
# filtered_output_file = 'filtered_players.csv'

# with open(input_file, 'r', newline='', encoding='utf-8') as infile, \
#      open(filtered_output_file, 'w', newline='', encoding='utf-8') as outfile:

#     reader = csv.reader(infile)
#     writer = csv.writer(outfile)

#     header = next(reader)
#     filtered_header = [h.rstrip('_') for h in (header[:6] + header[11:])]
#     writer.writerow(filtered_header)

#     for row in reader:
#         if not ','.join(row).endswith(',' * 15):
#             filtered_row = row[:6] + row[11:]
#             writer.writerow(filtered_row)



import pandas as pd

output_dir = 'profile_csvs'
os.makedirs(output_dir, exist_ok=True)

# 1. Load the labeled QB file
df = pd.read_csv('team_starting_qbs_labeled.csv')

# 2. Make sure we have the qb_profile column
if 'qb_profile' not in df.columns:
    raise KeyError("Column qb_profile not found in your CSV")

# 3. Group by profile and write each group to its own CSV
for profile, group in df.groupby('qb_profile'):
    # sanitize profile name for filename
    safe_name = profile.replace(" ", "_").replace("-", "_")
    out_file = os.path.join(output_dir, f"{safe_name}.csv")
    group.to_csv(out_file, index=False)
    print(f"→ Wrote {len(group)} rows to {out_file}")            