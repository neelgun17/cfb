import logging
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

# List of known 2025 NFL Draft Declarations (QBs and notable others)
# Sources: NFL.com, CBS Sports, 247Sports (Jan 2025)
DRAFT_DECLARATIONS = [
    # Quarterbacks
    "Shedeur Sanders",
    "Cam Ward",
    "Quinn Ewers",
    "Jalen Milroe",
    "Jaxson Dart",
    "Kyle McCord",
    "Carson Beck", # Often rumored/likely
    "Dillon Gabriel",
    "Will Howard",
    "Riley Leonard",
    "Cameron Rising",
    "DJ Uiagalelei",
    "Tyler Van Dyke",
    "Graham Mertz",
    "Kurtis Rourke", # Indiana starter -> Draft
    "Will Rogers",
    
    # Notable Skill Players (if matching logic expands)
    "Travis Hunter",
    "Tetairoa McMillan",
    "Luther Burden III",
    "Ashton Jeanty",
    "Ollie Gordon II",
    "TreVeyon Henderson",
    "Quinshon Judkins",
    "Emeka Egbuka",
    "Isaiah Bond"
]

import requests
import pandas as pd
from io import StringIO

def fetch_declarations():
    """Scrapes 2025 Draft prospects from Drafttek (Rankings Page)."""
    # Drafttek lists top prospects in a table
    url = "https://www.drafttek.com/2025-NFL-Draft-Prospect-Rankings/Top-College-Quarterbacks-2025-Draft.asp"
    try:
        headers = {'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36'}
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        
        dfs = pd.read_html(StringIO(response.text))
        
        players = set()
        for df in dfs:
            cols = [str(c).lower() for c in df.columns]
            # Drafttek usually has 'Player', 'School' columns
            if 'player' in cols:
                # Found the data table
                extracted = df['Player'].dropna().astype(str).tolist()
                # Clean up names (remove trailing whitespace or odd chars)
                extracted = [n.strip() for n in extracted if len(n) > 3]
                players.update(extracted)
                logger.info(f"Scraped table with {len(extracted)} players")
                
        if len(players) > 5:
            logger.info(f"Successfully scraped {len(players)} QBs from Drafttek.")
            return list(players)
        else:
            return []

    except Exception as e:
        logger.warning(f"Drafttek scrape failed: {e}")
        return []

def update_exclusions():
    """Updates manual_exclusions.txt with the known draft list."""
    
    exclusion_file = Path("PFF_Data/data/manual_exclusions.txt")
    
    # Try dynamic fetch first
    declared_players = fetch_declarations()
    
    # If fetch failed or returned few results, merge with hardcoded backup
    if not declared_players:
        declared_players = DRAFT_DECLARATIONS
    else:
        # Merge both to be safe (hardcoded list has high-confidence QBs)
        declared_players = list(set(declared_players + DRAFT_DECLARATIONS))
    
    # Read existing to avoid duplicates
    existing = set()
    if exclusion_file.exists():
        with open(exclusion_file, 'r') as f:
            existing = {line.strip() for line in f if line.strip() and not line.startswith('#')}
    
    # Merge
    new_entries = 0
    with open(exclusion_file, 'a') as f:
        # If file was empty/new, add header
        if not existing:
             f.write("# Auto-generated NFL Draft Exclusions (Wiki + Curated)\n")
             
        for player in declared_players:
            if player not in existing:
                f.write(f"{player}\n")
                existing.add(player)
                new_entries += 1
    
    if new_entries > 0:
        logger.info(f"Successfully added {new_entries} players to exclusion list.")
    else:
        logger.info("Exclusion list is already up to date.")
        
    logger.info(f"Total exclusions: {len(existing)}")
    logger.info(f"File location: {exclusion_file.absolute()}")

if __name__ == "__main__":
    update_exclusions()
