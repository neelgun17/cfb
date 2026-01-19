import pandas as pd
import json
import logging
from system_fit_analysis import SystemFitMatcher
from pathlib import Path

# Setup
logging.basicConfig(level=logging.INFO, format='%(message)s')
logger = logging.getLogger(__name__)

COMMITS_FILE = Path("data/portal_commits.json")
PORTAL_FULL_LIST = Path("data/portal_targets.json")

def run_test():
    print("--- Starting Accuracy Test ---")
    
    # 1. Load Data
    with open(COMMITS_FILE, 'r') as f:
        commits = json.load(f)
        
    if PORTAL_FULL_LIST.exists():
         with open(PORTAL_FULL_LIST, 'r') as f:
             data = json.load(f)
             if isinstance(data, list) and len(data) > 0 and isinstance(data[0], dict):
                 all_portal_players = [p['name'] for p in data]
             else:
                 all_portal_players = data
    else:
        all_portal_players = [c['name'] for c in commits]
        
    matcher = SystemFitMatcher()
    if not matcher.load_data():
        print("Failed to load PFF data.")
        return

    correct_count = 0
    total_tested = 0
    results = []

    for commit in commits:
        player_name = commit['name']
        destination_team = commit['to']
        source_team = commit['from']
        
        if destination_team == "Uncommitted":
            continue
            
        total_tested += 1
        
        # 1. Find Archetype (with Mappings)
        target_archetype, _ = matcher.get_team_archetype(destination_team)
        
        if not target_archetype:
            target_archetype, _ = matcher.get_team_archetype(destination_team.upper())
            
        if not target_archetype:
            mappings = {
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
            if destination_team in mappings:
                corrected_name = mappings[destination_team]
                target_archetype, _ = matcher.get_team_archetype(corrected_name)
                # Update destination team for exclude logic so we don't accidentally exclude ourselves if names matched weirdly
                destination_team = corrected_name

        if not target_archetype:
            print(f"⚠️ Could not find archetype for {destination_team}. Skipping.")
            results.append({
                "player": player_name,
                "team": destination_team,
                "found": False,
                "rank": "N/A",
                "reason": "Team Archetype Not Found"
            })
            continue

        # 2. Find Recommendations
        matches = matcher.find_transfer_matches(
            target_archetype=target_archetype,
            exclude_team=destination_team, 
            allowed_players=all_portal_players,
            top_n=10 
        )
        
        if matches.empty:
             results.append({
                "player": player_name,
                "team": destination_team,
                "found": False,
                "rank": "No Matches",
                "reason": "No matches found"
            })
             continue
             
        matches = matches.reset_index(drop=True)
        found_rows = matches[matches['player'] == player_name]
        
        if not found_rows.empty:
            rank = found_rows.index[0] + 1
            is_top_5 = rank <= 5
            if is_top_5:
                correct_count += 1
                
            results.append({
                "player": player_name,
                "team": destination_team,
                "found": True,
                "rank": rank,
                "archetype": target_archetype,
                "grade": found_rows.iloc[0].get('grades_offense', 'N/A')
            })
        else:
            p_rows = matcher.df[matcher.df['player'] == player_name]
            p_arch = p_rows.iloc[0]['archetype'] if not p_rows.empty else "Unknown"
            
            results.append({
                "player": player_name,
                "team": destination_team,
                "found": False,
                "rank": ">10",
                "team_archetype": target_archetype,
                "player_archetype": p_arch
            })

    print("\n--- Results ---")
    print(f"Accuracy (Top 5): {correct_count}/{total_tested} ({correct_count/total_tested*100:.1f}%)")
    
    print("\nDetailed Breakdown:")
    for r in results:
        status = "✅" if r.get('found') and r['rank'] <= 5 else "❌"
        if r.get('found'):
            print(f"{status} {r['player']} -> {r['team']}: Ranked #{r['rank']} (System: {r['archetype']})")
        else:
            reason = r.get('reason', '')
            if 'player_archetype' in r:
                 reason = f"System Mismatch ({r['team_archetype']} vs {r['player_archetype']})"
            print(f"{status} {r['player']} -> {r['team']}: Not in Top 10. {reason}")

if __name__ == "__main__":
    run_test()
