import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)

URL = "https://raw.githubusercontent.com/JackLich10/nfl-draft-data/main/nfl_draft_prospects.csv"

try:
    df = pd.read_csv(URL)
    print(f"Columns: {df.columns.tolist()}")
    if 'draft_year' in df.columns:
        print(f"Max Year: {df['draft_year'].max()}")
        print(f"2025 Prospects count: {len(df[df['draft_year'] == 2025])}")
    else:
        print("No 'draft_year' column found.")
        # Check standard cols
        print(df.head())
        
except Exception as e:
    print(f"Error: {e}")
