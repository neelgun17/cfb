import os
import configparser
import requests
import pandas as pd
import json

# Constants
API_YEAR = 2024
BASE_URL = "https://api.collegefootballdata.com"

def get_api_key():
    """Retrieve the CFBD API key from environment, config file, or prompt if missing."""
    # 1. Try environment variable
    key = os.getenv("CFBD_API_KEY")
    # 2. Try config file
    if not key:
        config = configparser.ConfigParser()
        config.read('config.ini')
        key = config.get('cfb', 'api_key', fallback=None)
    # 3. Prompt user
    if not key:
        key = input("Enter your CollegeFootballData API key: ").strip()
    return key

def fetch_player(search_term):
    """Fetch player profile data and return as DataFrame."""
    headers = {"Authorization": f"Bearer {get_api_key()}","Accept": "application/json"}
    params = {"searchTerm": search_term}
    resp = requests.get(f"{BASE_URL}/player/search", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())    
    df = df.drop(columns=["firstName", "lastName", "player"], errors="ignore")
    # keep only the last row (most recent player)
    df = df.tail(1)
    return df

def fetch_season_stats(team,player_id, year=API_YEAR):
    """Fetch and pivot season stats for a single player."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "team": team}
    resp = requests.get(f"{BASE_URL}/stats/player/season", headers=headers, params=params)
    resp.raise_for_status()
    data = resp.json()
    df = pd.DataFrame(data)
    # Pivot to wide format
    pivot = df.pivot_table(
        index=["playerId", "player", "team", "conference"],
        columns=["category", "statType"],
        values="stat",
        aggfunc="first"
    )
    pivot.columns = [f"{cat.lower()}_{stype.lower()}" for cat, stype in pivot.columns]
    pivot = pivot.reset_index().rename(columns={"playerId": "id"})    
    # filter to only the requested player
    pivot = pivot[pivot['id'] == player_id]
    return pivot

def fetch_usage(player_id, year=API_YEAR):
    """Fetch usage metrics for a player."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "playerId": player_id}
    resp = requests.get(f"{BASE_URL}/player/usage", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.json_normalize(resp.json())
    # Remove metadata columns
    df = df.drop(columns=["season", "name", "position", "team", "conference"], errors="ignore")
    print("usage ", df.shape)
    return df

def fetch_ppa(player_id, year=API_YEAR):
    """Fetch PPA metrics for a player."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "playerId": player_id}
    resp = requests.get(f"{BASE_URL}/ppa/players/season", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.json_normalize(resp.json(), sep='.')
    # Remove metadata columns
    df = df.drop(columns=["season", "name", "position", "team", "conference"], errors="ignore")
    return df

def fetch_team(team, year = API_YEAR):
    """Fetch team profile data and return as DataFrame."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "team": team}
    resp = requests.get(f"{BASE_URL}/stats/season", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.json_normalize(resp.json(), sep='.')
    print("DF ", df.shape)
    print("DF ", df.head())
    return df

def fetch_team_usage(team, year=API_YEAR):
    """Fetch team play-type splits for usage normalization."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year}
    resp = requests.get(f"{BASE_URL}/stats/season/advanced", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.json_normalize(resp.json(), sep='.')
    team_row = df[df["team"] == team].iloc[0]
    return {
        "team_std_down_rate":     team_row["offense.standardDowns.rate"],
        "team_pass_down_rate":    team_row["offense.passingDowns.rate"],
        "team_rushing_play_rate": team_row["offense.rushingPlays.rate"],
        "team_passing_play_rate": team_row["offense.passingPlays.rate"],
        "team_offense_plays": team_row["offense.plays"]
    }

def merge_data(profile_df, stats_df, usage_df, ppa_df):
    """Merge all DataFrames into one."""
    merged = (
        profile_df
        .merge(stats_df, left_on=["id", "name", "team"], right_on=["id", "player", "team"], how="left")
        .merge(usage_df, on="id", how="left")
        .merge(ppa_df, on="id", how="left")
    )
    return merged

def select_and_order(merged):
    """Select relevant columns and order them logically."""
    # Ensure pandas prints all columns
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)

    metadata_cols = [
        "id", "team", "name", "weight", "height", "jersey",
        "position", "hometown", "teamColor", "teamColorSecondary", "conference"
    ]
    stat_cols = [
        "fumbles_fum", "fumbles_lost", "fumbles_rec",
        "passing_att", "passing_completions", "passing_int", "passing_pct", "passing_td",
        "passing_yds", "passing_ypa",
        "rushing_car", "rushing_long", "rushing_td", "rushing_yds", "rushing_ypc"
    ]
    usage_cols = [
        "usage.overall", "usage.pass", "usage.rush", "usage.firstDown", "usage.secondDown",
        "usage.thirdDown", "usage.standardDowns", "usage.passingDowns", "countablePlays"
    ]
    avg_ppa_cols = [
        "averagePPA.all", "averagePPA.pass", "averagePPA.rush", "averagePPA.firstDown",
        "averagePPA.secondDown", "averagePPA.thirdDown", "averagePPA.standardDowns",
        "averagePPA.passingDowns"
    ]
    extra_cols = ["pct_team_pass_snaps", "pct_team_run_snaps", "share_team_pass_snaps"]
    desired_cols = metadata_cols + stat_cols + usage_cols + avg_ppa_cols + extra_cols

    df = merged[desired_cols].tail(1).copy()
    return df

def main():
    # Example usage
    search_term = "Jalen Milroe"
    profile_df = fetch_player(search_term)
    # Use the most recent entry
    row = profile_df.iloc[-1]
    team, player_id = row["team"], row["id"]

    stats_df = fetch_season_stats(team, player_id)
    usage_df = fetch_usage(player_id)
    ppa_df = fetch_ppa(player_id)
    # team_stats = fetch_team(team)
    team_usage = fetch_team_usage(team)

    merged = merge_data(profile_df, stats_df, usage_df, ppa_df)

    # Compute normalized usage ratios
    merged["pct_team_pass_snaps"] = merged["usage.pass"] / team_usage["team_passing_play_rate"]
    merged["pct_team_run_snaps"]  = merged["usage.rush"] / team_usage["team_rushing_play_rate"]

    # Compute share of team pass snaps
    player_pass_snaps = merged["countablePlays"] * merged["usage.pass"]
    team_pass_snaps   = team_usage["team_offense_plays"] * team_usage["team_passing_play_rate"]
    merged["share_team_pass_snaps"] = player_pass_snaps / team_pass_snaps

    final_df = select_and_order(merged)

    # Print as pretty JSON
    print(json.dumps(final_df.iloc[0].to_dict(), indent=2))

if __name__ == "__main__":
    main()