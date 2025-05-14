import os
import configparser
import requests
import pandas as pd
import json
import numpy as np
import argparse

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

def fetch_player(search_term, year=API_YEAR):
    """Fetch player profile data and return as DataFrame."""
    headers = {"Authorization": f"Bearer {get_api_key()}","Accept": "application/json"}
    params = {"searchTerm": search_term, "year": year}
    resp = requests.get(f"{BASE_URL}/player/search", headers=headers, params=params)
    resp.raise_for_status()
    df = pd.DataFrame(resp.json())
    df = df.drop(columns=["firstName", "lastName", "player"], errors="ignore")
    # Keep only entries for the target season (if available)
    if "season" in df.columns:
        df = df[df["season"] == year]
    # keep only the last row (most recent player) after season filtering
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
    # Ensure stat column is numeric
    df['stat'] = pd.to_numeric(df['stat'], errors='coerce')
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
    # print("pivot", pivot.info())
    # print("pivot ", pivot.head())
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
    extra_cols = ["pct_team_pass_snaps", "pct_team_run_snaps", "share_team_pass_snaps", "qb_profile"]
    desired_cols = metadata_cols + stat_cols + usage_cols + avg_ppa_cols + extra_cols

    df = merged[desired_cols].tail(1).copy()
    return df

def assign_profile(row):
    if row["is_dual"]:
        return "Dual Threat"
    if row["is_gunslinger"]:
        return "Gunslinger"
    if row["is_pocket"]:
        return "Pocket Passer"
    return "Game Manager"

def enrich_player_metrics(merged, team_usage):
    """
    Given a merged DataFrame for a player-season and the team's usage dict,
    compute normalized usage ratios, share metrics, prototype flags, and profile.
    """
    # normalized usage ratios
    merged["pct_team_pass_snaps"] = merged["usage.pass"] / team_usage["team_passing_play_rate"]
    merged["pct_team_run_snaps"]  = merged["usage.rush"] / team_usage["team_rushing_play_rate"]
    # share of team pass snaps
    player_pass_snaps = merged["countablePlays"] * merged["usage.pass"]
    team_pass_snaps   = team_usage["team_offense_plays"] * team_usage["team_passing_play_rate"]
    merged["share_team_pass_snaps"] = player_pass_snaps / team_pass_snaps
    # prototype flags
    merged["is_dual"] = (merged["usage.rush"] >= 0.15) & (merged["averagePPA.rush"] >= 0.15)
    merged["is_pocket"] = (merged["usage.rush"] <= 0.05) & (merged["passing_pct"] >= 0.65)
    merged["is_gunslinger"] = (
        (merged["averagePPA.pass"] >= 0.20) &
        ((merged["passing_int"] / merged["passing_att"]) >= 0.03)
    )
    # assign profile string
    merged["qb_profile"] = merged.apply(assign_profile, axis=1)
    return merged

def main():
    search_term = "Cameron Ward"
    profile_df = fetch_player(search_term)
    if profile_df.empty:
        raise ValueError(f"No player found matching '{search_term}'")
    row = profile_df.iloc[-1]
    team, player_id = row["team"], row["id"]

    # 2) Fetch everything
    stats_df  = fetch_season_stats(team, player_id)
    usage_df  = fetch_usage(player_id)
    ppa_df    = fetch_ppa(player_id)
    team_usage= fetch_team_usage(team)

    # 3) Merge into one DF
    merged = merge_data(profile_df, stats_df, usage_df, ppa_df)

    # 4) Compute your two metrics
    merged = enrich_player_metrics(merged, team_usage)

    # 5) Select & print just that one QB’s profile
    final_df = select_and_order(merged)
    profile_json = final_df.iloc[0].to_dict()
    print(json.dumps(profile_json, indent=2))

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--top50", action="store_true", help="Process top 50 QBs and save to CSV")
    args = parser.parse_args()
    if args.top50:
        records = []
        qb_names = pd.read_csv("qb_names.csv", header=None)[0].to_list()
        for qb in qb_names:
            profile_df = fetch_player(qb)
            if profile_df.empty: 
                continue
            # Extract scalar values from the single-row DataFrame
            row = profile_df.iloc[0]
            name = row["name"]
            pid = row["id"]
            team = row["team"]
            # fetch and merge as before using pid and team
            # inline existing logic to compute merged, metrics, and assign profile
            stats_df = fetch_season_stats(team, pid)
            usage_df = fetch_usage(pid)
            ppa_df = fetch_ppa(pid)
            team_usage = fetch_team_usage(team)
            merged = merge_data(profile_df, stats_df, usage_df, ppa_df)
            merged = enrich_player_metrics(merged, team_usage)
            final_df = select_and_order(merged)
            records.append(final_df.iloc[0].to_dict())
        out_df = pd.DataFrame(records)
        out_df.to_csv("top50_qb_profiles.csv", index=False)
        print("Saved top 50 QB profiles to top50_qb_profiles.csv")
    else:
        main()
