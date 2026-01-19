import pandas as pd
from pathlib import Path

# Try to read the summary file
path = Path("/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/data/raw/2025/passing_summary.csv")
if path.exists():
    df = pd.read_csv(path)
    print("Cols:", df.columns.tolist())
    print("Head:", df.head(1).to_dict())
else:
    print("File not found")
