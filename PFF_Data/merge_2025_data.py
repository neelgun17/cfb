import pandas as pd
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

DATA_DIR = Path("/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/data/raw/2025")

def load_and_standardize(file_name, prefix_map, common_cols=['player_id', 'player', 'team_name', 'position']):
    """
    Loads a CSV. If prefix_map is empty, kept as is.
    """
    path = DATA_DIR / file_name
    if not path.exists():
        logger.warning(f"File not found: {path}")
        return None
    
    df = pd.read_csv(path)
    
    # Identify non-common columns (metrics)
    metric_cols = [c for c in df.columns if c not in common_cols]
    
    if prefix_map:
        # Renaming logic
        rename_dict = {}
        for c in metric_cols:
            if c.startswith(prefix_map):
                rename_dict[c] = c # Keep as is
            else:
                rename_dict[c] = f"{prefix_map}{c}"
        df.rename(columns=rename_dict, inplace=True)
    
    return df

def merge_dfs(df_list, on_cols=['player_id', 'player', 'team_name', 'position']):
    """Merges a list of dataframes on common columns."""
    if not df_list:
        return None
    
    merged = df_list[0]
    for i in range(1, len(df_list)):
        if df_list[i] is not None:
            merged = pd.merge(merged, df_list[i], on=on_cols, how='outer')
            
    return merged

def main():
    # 0. Load Summary first
    if (DATA_DIR / "Passing Grades Summary.csv").exists():
        pd.read_csv(DATA_DIR / "Passing Grades Summary.csv").to_csv(DATA_DIR / "passing_summary.csv", index=False)
    
    # 1. Merge Concepts
    # Update: 'Passing Concept Play Action.csv' seems to be a wide file with EVERYTHING.
    # We will load it with NO prefix.
    # We will verify if it has 'pa_attempts' and 'screen_attempts'. 
    # If so, we are good.
    
    df_concept = load_and_standardize("Passing Concept Play Action.csv", "")
    
    if df_concept is not None:
        # Check if we need to merge Screen file?
        if 'screen_attempts' not in df_concept.columns and (DATA_DIR / "Passing Concept Screen.csv").exists():
            # If main file ignores screen, we load screen file
             df_screen = load_and_standardize("Passing Concept Screen.csv", "screen_")
             df_concept = merge_dfs([df_concept, df_screen])
             
        # Infer missing columns if needed (NPA)
        if 'npa_attempts' not in df_concept.columns:
            # Try to infer
             if (DATA_DIR / "passing_summary.csv").exists():
                 summary = pd.read_csv(DATA_DIR / "passing_summary.csv")
                 merged = pd.merge(df_concept, summary[['player_id', 'attempts']], on='player_id', how='left')
                 if 'pa_attempts' in merged.columns:
                     merged['npa_attempts'] = merged['attempts'] - merged['pa_attempts'].fillna(0)
                     merged['npa_attempts'] = merged['npa_attempts'].apply(lambda x: max(0, x))
                 df_concept = merged.drop(columns=['attempts'], errors='ignore')
        
        # Ensure 'no_screen_attempts' exists
        if 'no_screen_attempts' not in df_concept.columns:
              # Only if screen_attempts exists
             if 'screen_attempts' in df_concept.columns and (DATA_DIR / "passing_summary.csv").exists():
                 summary = pd.read_csv(DATA_DIR / "passing_summary.csv")
                 merged = pd.merge(df_concept, summary[['player_id', 'attempts']], on='player_id', how='left')
                 merged['no_screen_attempts'] = merged['attempts'] - merged['screen_attempts'].fillna(0)
                 merged['no_screen_attempts'] = merged['no_screen_attempts'].apply(lambda x: max(0, x))
                 df_concept = merged.drop(columns=['attempts'], errors='ignore')

        df_concept.to_csv(DATA_DIR / "passing_concept.csv", index=False)
        logger.info(f"Created passing_concept.csv")

    # 2. Merge Depth
    depth_files = [
        ("Passing Depth Short.csv", "short_"),
        ("Passing Depth Intermediate.csv", "medium_"),
        ("Passing Depth Deep.csv", "deep_"),
        ("Passing Depth Behind LOS.csv", "behind_los_")
    ]
    
    depth_dfs = []
    for fname, prefix in depth_files:
        df = load_and_standardize(fname, prefix)
        if df is not None:
            depth_dfs.append(df)
    
    merged_depth = merge_dfs(depth_dfs)
    if merged_depth is not None:
        if 'deep_completions' not in merged_depth.columns:
             merged_depth['deep_completions'] = 0 
        if 'deep_turnover_worthy_plays' not in merged_depth.columns:
             merged_depth['deep_turnover_worthy_plays'] = 0
             
        merged_depth.to_csv(DATA_DIR / "passing_depth.csv", index=False)
        
    # 3. Other files
    if (DATA_DIR / "Rushing Grades Summary.csv").exists():
        pd.read_csv(DATA_DIR / "Rushing Grades Summary.csv").to_csv(DATA_DIR / "rushing_summary.csv", index=False)

    if (DATA_DIR / "Time In Pocket.csv").exists():
        pd.read_csv(DATA_DIR / "Time In Pocket.csv").to_csv(DATA_DIR / "time_in_pocket.csv", index=False)
        
    # Pressure
    if not (DATA_DIR / "passing_pressure.csv").exists() and (DATA_DIR / "Passing Pressure pressure:clean.csv").exists():
         pd.read_csv(DATA_DIR / "Passing Pressure pressure:clean.csv").to_csv(DATA_DIR / "passing_pressure.csv", index=False)

if __name__ == "__main__":
    main()
