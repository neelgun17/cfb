import pandas as pd
import json
import logging
from pathlib import Path
import sys

# Import SystemFitMatcher
from system_fit_analysis import SystemFitMatcher
# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Constants
DATA_DIR = Path("data")
RAW_DIR = DATA_DIR / "raw/2025"
OUTPUT_FILE = DATA_DIR / "dashboard_data.json"
STATUS_FILE = DATA_DIR / "status_tracker.json"
PORTAL_FILE = DATA_DIR / "portal_targets.json"
ARCHETYPE_FILE = Path("analysis/player_assignments_2025.csv") # Check this path dynamically

# Mappings
TEAM_NAME_MAP = {
    "Oklahoma State": "OKLA STATE",
    "Ole Miss": "MISSISSIPPI",
    "UConn": "CONNECTICUT",
    "UMass": "MASSACHUSETTS",
    "App State": "APP STATE",
    "Louisiana-Monroe": "UL MONROE",
    "UL Monroe": "UL MONROE",
    "Southern Miss": "SOUTHERN MISS",
    "Middle Tennessee": "MID TENNESSEE",
    "South Florida": "S FLORIDA"
}

def load_data():
    """Load raw stats and status data."""
    # 1. Load Passing Summary
    passing_path = RAW_DIR / "passing_summary.csv"
    if not passing_path.exists():
        logger.error(f"Missing passing data: {passing_path}")
        return None, None
        
    df_passing = pd.read_csv(passing_path)
    
    # 2. Load Status Tracker
    status_map = {}
    if STATUS_FILE.exists():
        with open(STATUS_FILE, 'r') as f:
            status_map = json.load(f)
    else:
        logger.warning(f"No status file found at {STATUS_FILE}")
        
    # 3. Load Archetypes (Optional but recommended)
    # Trying to find the most recent archetype file
    df_archetypes = None
    if ARCHETYPE_FILE.exists():
        df_archetypes = pd.read_csv(ARCHETYPE_FILE)
    else:
        # Fallback to looking in processed or root
        alternatives = list(Path(".").glob("qb_data_with_archetypes_*.csv"))
        if alternatives:
            df_archetypes = pd.read_csv(alternatives[0])
            
    # 4. Load Portal Targets
    portal_targets = []
    if PORTAL_FILE.exists():
        with open(PORTAL_FILE, 'r') as f:
            data = json.load(f)
            # Extract names
            portal_targets = [p['name'] for p in data]
    else:
        logger.warning("No portal targets file found.")

    return df_passing, status_map, df_archetypes, portal_targets

def generate_dashboard_json():
    logger.info("Starting Dashboard Data Generation...")
    
    df_passing, status_map, df_archetypes, portal_targets = load_data()
    if df_passing is None:
        return

    # Clean / Normalize
    df_passing['team_name'] = df_passing['team_name'].fillna('Unknown')
    
    # Filter for significant players (e.g., > 10 attempts to reduce noise?)
    # User asked for ALL players leaving, but practically we want ones with impact.
    # Let's keep all who have stats for now to be safe.
    
    # Aggregations by Team
    team_stats = df_passing.groupby('team_name').agg({
        'yards': 'sum',
        'attempts': 'sum',
        'touchdowns': 'sum',
        'dropbacks': 'sum'
    }).rename(columns={
        'yards': 'team_yards',
        'attempts': 'team_attempts',
        'touchdowns': 'team_tds',
        'dropbacks': 'team_dropbacks'
    }).to_dict('index')

    dashboard_data = {}

    teams = df_passing['team_name'].unique()
    
    # Initialize Matcher
    matcher = SystemFitMatcher()
    matcher.load_data()
    
    # Archetype Lookup Helper
    archetype_map = {}
    if df_archetypes is not None:
        # Normalize columns
        df_archetypes.columns = df_archetypes.columns.str.lower().str.strip()
        if 'player' in df_archetypes.columns and 'archetype' in df_archetypes.columns:
            archetype_map = df_archetypes.set_index('player')['archetype'].to_dict()
        elif 'player' in df_archetypes.columns and 'archetype_name' in df_archetypes.columns:
            archetype_map = df_archetypes.set_index('player')['archetype_name'].to_dict()

    for team in teams:
        # Standardize for archetype lookup
        lookup_team_name = TEAM_NAME_MAP.get(team, team)
        
        team_row = team_stats.get(team, {})
        team_total_yards = team_row.get('team_yards', 0)
        team_total_attempts = team_row.get('team_attempts', 0)
        
        # Get players for this team
        team_players_df = df_passing[df_passing['team_name'] == team]
        
        players_data = []
        team_has_departure = False
        
        for _, player in team_players_df.iterrows():
            p_name = player.get('player', 'Unknown')
            p_yards = player.get('yards', 0)
            p_attempts = player.get('attempts', 0)
            
            # Impact Calculation
            impact_yards_share = (p_yards / team_total_yards) if team_total_yards > 0 else 0
            impact_attempts_share = (p_attempts / team_total_attempts) if team_total_attempts > 0 else 0
            
            # Status
            # Check exact match first
            p_status = status_map.get(p_name, "Returning")
            
            if p_status != "Returning":
                team_has_departure = True
            
            players_data.append({
                "name": p_name,
                "position": player.get('position', 'QB'),
                "status": p_status,
                "stats": {
                    "yards": int(p_yards),
                    "attempts": int(p_attempts),
                    "touchdowns": int(player.get('touchdowns', 0))
                },
                "impact": {
                    "yards_share": round(impact_yards_share, 3),
                    "attempts_share": round(impact_attempts_share, 3)
                },
                "archetype": archetype_map.get(p_name, "Unknown")
            })
            
        # Sort players by attempt count desc
        players_data.sort(key=lambda x: x['stats']['attempts'], reverse=True)
        
        # Recommendations
        recommendations = []
        # Find team archetype
        team_archetype, starter_info = matcher.get_team_archetype(lookup_team_name)
        
        if team_archetype:
            # Get exclusion list from status map (anyone not returning)
            # Actually, we should exclude players who are NOT in the portal.
            # But simpler: exclude players from THIS team. `find_transfer_matches` handles that.
            # We also want to exclude players who have declared for draft (from status_map).
            start_exclusions = [name for name, status in status_map.items() if status == 'Draft']
            
            matches = matcher.find_transfer_matches(
                team_archetype, 
                lookup_team_name, 
                exclude_players=start_exclusions, 
                allowed_players=portal_targets,
                top_n=5
            )
            
            if not matches.empty:
                recommendations = matches.to_dict('records')

        dashboard_data[team] = {
            "team_stats": team_row,
            "players": players_data,
            "has_departures": team_has_departure,
            "team_archetype": team_archetype,
            "recommendations": recommendations
        }

    # Write Output
    with open(OUTPUT_FILE, 'w') as f:
        json.dump(dashboard_data, f, indent=2)
        
    logger.info(f"Dashboard data generated at {OUTPUT_FILE}")
    logger.info(f"Processed {len(teams)} teams.")

if __name__ == "__main__":
    generate_dashboard_json()
