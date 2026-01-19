"""
System Fit Analysis & Transfer Portal Matcher
Identifies team QB archetypes and finds matching candidates from other teams.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import sys
import logging

# Add current directory to path to allow imports if running directly
current_dir = Path(__file__).parent
if str(current_dir) not in sys.path:
    sys.path.append(str(current_dir))

from config import get_analysis_files, CURRENT_YEAR

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')  # Simpler format for CLI tool
logger = logging.getLogger(__name__)

class SystemFitMatcher:
    def __init__(self, year: int = None):
        self.year = year if year is not None else CURRENT_YEAR
        self.analysis_files = get_analysis_files(self.year)
        self.df = None
        self.coaching_overrides = {}
        
    def load_data(self):
        """Load the dataset with archetype assignments and coaching overrides."""
        # 1. Load Coaching Overrides
        override_path = current_dir / "data/coaching_changes.json"
        if override_path.exists():
            try:
                with open(override_path, 'r') as f:
                    self.coaching_overrides = json.load(f)
                logger.info(f"Loaded {len(self.coaching_overrides)} coaching overrides.")
            except Exception as e:
                logger.error(f"Failed to load coaching overrides: {e}")

        # Primary path from config
        primary_path = self.analysis_files['player_assignments']
        
        if primary_path.exists():
            self.df = pd.read_csv(primary_path)
            logger.info(f"Loaded data from {primary_path}")
        else:
            # Fallback (legacy)
            potential_path = current_dir / "qb_data_with_archetypes_k4.csv"
            if potential_path.exists():
                self.df = pd.read_csv(potential_path)
                logger.warning(f"Using legacy data from {potential_path}")
            else:
                logger.error(f"Could not find data file at {primary_path} or {potential_path}")
                return False
            
        # Clean up column names if needed (standardize)
        self.df.columns = self.df.columns.str.lower().str.strip()
        
        # Standardize archetype column
        if 'archetype_name' in self.df.columns:
            self.df.rename(columns={'archetype_name': 'archetype'}, inplace=True)
            
        return True

    def get_team_archetype(self, team_name: str):
        """
        Determines the dominant archetype for a team.
        PRIORITY 1: Coaching Change Override
        PRIORITY 2: 2024 Starter / Most Played QB
        """
        # 1. Check Overrides
        # Try exact match or upper case
        override = self.coaching_overrides.get(team_name) or self.coaching_overrides.get(team_name.upper())
        
        if override:
            logger.info(f"Using coaching override for {team_name}: {override['target_archetype']}")
            # We return the mapped archetype and a dummy 'starter' object with that archetype to satisfy return sig
            dummy_starter = pd.Series({'archetype': override['target_archetype'], 'player': 'Coaching Override'})
            return override['target_archetype'], dummy_starter

        # 2. Fallback to Historical Data
        team_qbs = self.df[self.df['team_name'].str.lower() == team_name.lower()].copy()
        
        if team_qbs.empty:
            return None, None
            
        # Sort by dropbacks (interpreting 'attempts' or similar as proxy for usage if dropbacks missing)
        # Using 'dropbacks' from prior README knowledge, defaulting to usage proxy
        sort_col = 'dropbacks' if 'dropbacks' in team_qbs.columns else 'attempts'
        
        starter = team_qbs.sort_values(by=sort_col, ascending=False).iloc[0]
        
        return starter['archetype'], starter

    def find_transfer_matches(self, target_archetype: str, exclude_team: str, exclude_players: list = None, allowed_players: list = None, top_n: int = 5):
        """
        Finds best players matching the archetype from OTHER teams.
        If allowed_players is provided, ONLY recommends players in that list.
        """
        if exclude_players is None:
            exclude_players = []

        # Filter for archetype
        candidates = self.df[
            (self.df['archetype'] == target_archetype) & 
            (self.df['team_name'].str.lower() != exclude_team.lower()) &
            (~self.df['player'].isin(exclude_players))
        ].copy()
        
        # Filter for Allowed Players (Portal Targets) if provided
        if allowed_players is not None and len(allowed_players) > 0:
            # Normalize names for better matching (optional but good)
            candidates = candidates[candidates['player'].isin(allowed_players)]
        
        if candidates.empty:
            return pd.DataFrame()
            
        # Ranking Logic:
        # Default to 'grades_offense' if available, otherwise 'grades_pass'
        rank_col = 'grades_offense' if 'grades_offense' in candidates.columns else 'grades_pass'
        
        # Sort desc
        ranked = candidates.sort_values(by=rank_col, ascending=False).head(top_n)
        
        return ranked[['player', 'team_name', rank_col, 'archetype']]

def main():
    matcher = SystemFitMatcher()
    if not matcher.load_data():
        print("Failed to load data. Exiting.")
        return

    print("\n--- Transfer Portal System Fit Matcher ---")
    print("Finds replacement QBs that match your team's current offensive style.")
    print("Tip: Enter 'Team Name, Number' to see more results (e.g. 'Indiana, 20')")
    
    while True:
        user_input = input("\nEnter Team Name (or 'q' to quit): ").strip()
        if user_input.lower() == 'q':
            break
            
        # Parse input for optional count
        if ',' in user_input:
            parts = user_input.rsplit(',', 1)
            team_input = parts[0].strip()
            try:
                limit = int(parts[1].strip())
            except ValueError:
                limit = 10
        else:
            team_input = user_input
            limit = 10 # Increased default from 5
            
        archetype, starter = matcher.get_team_archetype(team_input)
        
        if not archetype:
            print(f"Team '{team_input}' not found in dataset.")
            # Optional: fuzzy search could go here
            continue
            
        print(f"\n{team_input} ({matcher.year} Style): {archetype}")
        print(f"Based on Starter: {starter['player']}")
        
        print(f"\nTop {limit} Available Matches ({archetype}) from other teams:")
        
        # Load exclusions from file
        known_draft_declarations = []
        exclusions_file = matcher.analysis_files['player_assignments'].parent.parent.parent / "manual_exclusions.txt"
        
        # Also check PFF_Data/data/manual_exclusions.txt explicitly if path navigation above is tricky
        if not exclusions_file.exists():
             exclusions_file = Path("PFF_Data/data/manual_exclusions.txt")
             
        if exclusions_file.exists():
            try:
                with open(exclusions_file, 'r') as f:
                    known_draft_declarations = [line.strip() for line in f if line.strip() and not line.startswith('#')]
                print(f"(loaded {len(known_draft_declarations)} exclusions from {exclusions_file.name})")
            except Exception as e:
                logger.error(f"Error loading exclusions: {e}") 
        
        matches = matcher.find_transfer_matches(archetype, team_input, exclude_players=known_draft_declarations, top_n=limit)
        
        if matches.empty:
            print("No matches found.")
        else:
            # Beautify output
            print(matches.to_string(index=False))

if __name__ == "__main__":
    main()
