#!/usr/bin/env python3
import os
import sys
import configparser
import time
import requests
from requests.exceptions import HTTPError
import pandas as pd
from pandas import json_normalize
import numpy as np

API_YEAR = 2024
BASE_URL = "https://api.collegefootballdata.com"

def get_api_key():
    key = os.getenv("CFBD_API_KEY")
    if not key:
        config = configparser.ConfigParser()
        config.read('config.ini')
        key = config.get('cfb','api_key',fallback=None)
    if not key:
        key = input("Enter your CollegeFootballData API key: ").strip()
    return key

def fetch_all_player_stats(year=API_YEAR):
    """Fetch season stats for every player in one call."""
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "application/json"
    }
    params = {"year": year}
    retries = 3
    backoff = 1
    for attempt in range(retries):
        try:
            resp = requests.get(f"{BASE_URL}/stats/player/season", headers=headers, params=params)
            resp.raise_for_status()
            break
        except HTTPError as e:
            if resp.status_code == 429 and attempt < retries - 1:
                print(f"Warning: rate limited, retrying after {backoff}s...")
                time.sleep(backoff)
                backoff *= 2
                continue
            else:
                raise
    else:
        print(f"Warning: fetch_all_player_stats failed after retries: {e}")
        return pd.DataFrame(columns=['player_id','player_name','team','conference'])
    try:
        df = pd.DataFrame(resp.json())
        # Ensure playerId is numeric for filtering
        df['playerId'] = pd.to_numeric(df['playerId'], errors='coerce').astype('Int64')
        # Drop aggregate "Team" entries with negative playerId
        # print("DF: ", df.info())
        df = df[df['playerId'] > 0]
        df['stat'] = pd.to_numeric(df['stat'], errors='coerce')
        wide = (
            df
            .pivot_table(
                index=['playerId','player','team','conference'],
                columns=['category','statType'],
                values='stat',
                aggfunc='first'
            )
            .reset_index()
        )
        # Rename key index columns to snake_case identifiers
        wide = wide.rename(columns={
            'playerId': 'player_id',
            'player': 'player_name',
            'team': 'team',
            'conference': 'conference'
        })
        # Flatten any remaining tuple-based stat columns, dropping trailing underscores if statType is empty
        wide.columns = [
            col if not isinstance(col, tuple)
            else (
                f"{col[0].lower()}_{col[1].lower()}"
                if col[1]
                else col[0].lower()
            )
            for col in wide.columns
        ]
        print("WIDE")
        print(wide.info())
        return wide
    except Exception as e:
        print(f"Warning: fetch_all_player_stats failed: {e}")
        return pd.DataFrame(columns=['playerId','player','team','conference'])

def select_starters(stats_df):
    """
    Given a wide stats DataFrame (from fetch_all_player_stats),
    select the QB with max passing_att per team.
    """
    if 'passing_att' not in stats_df.columns:
        print("Error: 'passing_att' column missing in stats_df. Cannot select starters.")
        sys.exit(1)
    # Drop rows missing passing_att
    df = stats_df.dropna(subset=['passing_att']).copy()
    df['passing_att'] = df['passing_att'].astype(int)
    # For each team, pick the QB with highest passing_att
    print(df.info())
    starters = (
        df
        .sort_values(['team','passing_att'], ascending=[True, False])
        .groupby('team', as_index=False)
        .first()
    )
    return starters

def fetch_usage(player_id, year=API_YEAR):
    """Fetch usage metrics for a player and return a one-row DataFrame."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "playerId": player_id}
    try:
        resp = requests.get(f"{BASE_URL}/player/usage", headers=headers, params=params)
        resp.raise_for_status()
        usage = pd.json_normalize(resp.json())
        # drop metadata columns
        usage = usage.drop(columns=["season", "name", "position", "team", "conference"], errors="ignore")
        usage["id"] = player_id
        return usage
    except Exception as e:
        print(f"Warning: fetch_usage({player_id}) failed: {e}")
        return pd.DataFrame()

def fetch_ppa(player_id, year=API_YEAR):
    """Fetch PPA metrics for a player and return a one-row DataFrame."""
    headers = {"Authorization": f"Bearer {get_api_key()}", "Accept": "application/json"}
    params = {"year": year, "playerId": player_id}
    try:
        resp = requests.get(f"{BASE_URL}/ppa/players/season", headers=headers, params=params)
        resp.raise_for_status()
        ppa = pd.json_normalize(resp.json(), sep='.')
        ppa = ppa.drop(columns=["season", "name", "position", "team", "conference"], errors="ignore")
        ppa["id"] = player_id
        return ppa
    except Exception as e:
        print(f"Warning: fetch_ppa({player_id}) failed: {e}")
        return pd.DataFrame()

# --- Extracted: fetch all teams' advanced stats ---
def fetch_team_advanced_stats(teams, year=API_YEAR):
    """Fetch offense counts per team from /stats/season and compute rates."""
    headers = {
        "Authorization": f"Bearer {get_api_key()}",
        "Accept": "application/json"
    }
    session = requests.Session()
    resp = session.get(f"{BASE_URL}/stats/season", headers=headers, params={"year": year})
    raw = pd.DataFrame(resp.json())
    raw = raw.pivot(index='team', columns='statName', values='statValue')
    raw = raw.fillna(0)
    # Compute rates vectorized
    raw['team_overall'] = raw['passAttempts'] + raw['rushingAttempts']
    raw['team_pass'] = raw['passAttempts'] / raw['team_overall']
    raw['team_rush'] = raw['rushingAttempts'] / raw['team_overall']
    raw['team_standardDowns'] = raw['firstDowns'] / raw['team_overall']
    raw['team_passingDowns'] = raw['thirdDownConversions'] / raw['thirdDowns']
    return raw[['team_overall','team_pass','team_rush','team_standardDowns','team_passingDowns']].to_dict(orient='index')

def main():
    print("Fetching all player stats…")
    stats = fetch_all_player_stats()

    if stats.empty or 'passing_att' not in stats.columns:
        print("Error: No player stats available (fetch may have failed). Exiting.")
        sys.exit(1)

    print("Selecting starters…")
    starters = select_starters(stats)

    # Enrich each starter with usage and PPA metrics
    usage_records = []
    ppa_records = []
    starters= starters.head(5)
    print("THE NUM OF STARTERS ", len(starters))
    for pid in starters["player_id"]:
        start = time.time()
        print("pid: ",pid)
        # Fetch usage and handle missing data
        u_df = fetch_usage(pid)
        if not u_df.empty:
            # rename the fetched 'id' column to 'player_id'
            u_df = u_df.rename(columns={'id':'player_id'})
            usage_records.append(u_df.iloc[0].to_dict())
        else:
            usage_records.append({'player_id': pid})
        # Fetch PPA and handle missing data
        p_df = fetch_ppa(pid)
        if not p_df.empty:
            # rename the fetched 'id' column to 'player_id'
            p_df = p_df.rename(columns={'id':'player_id'})
            ppa_records.append(p_df.iloc[0].to_dict())
        else:
            ppa_records.append({'player_id': pid})
    usage_df = pd.DataFrame(usage_records)
    ppa_df = pd.DataFrame(ppa_records)
    starters = starters.merge(usage_df, on="player_id", how="left")
    starters = starters.merge(ppa_df,   on="player_id", how="left")

    # Drop players with no usage or PPA metrics
    drop_cols = [
        'usage.pass','usage.rush','usage.thirdDown',
        'usage.standardDowns','usage.passingDowns',
        'averagePPA.all','averagePPA.pass','averagePPA.rush',
        'averagePPA.firstDown','averagePPA.secondDown','averagePPA.thirdDown',
        'averagePPA.standardDowns','averagePPA.passingDowns'
    ]
    starters = starters.dropna(subset=drop_cols, how='all')

    # Fetch team advanced stats and map to starters
    teams = starters['team'].unique()
    team_adv = fetch_team_advanced_stats(teams)
    for key in ['team_overall','team_pass','team_rush','team_standardDowns','team_passingDowns']:
        starters[key] = starters['team'].map(lambda t: team_adv.get(t, {}).get(key))

    # ensure usage and team rate columns are numeric
    metrics = ['overall','pass','rush','firstDown','secondDown','standardDowns','passingDowns']
    share_columns = []
    for m in metrics:
        share_columns.extend([f'usage.{m}', f'team_{m}'])
    for col in share_columns:
        if col in starters.columns:
            starters[col] = pd.to_numeric(starters[col], errors='coerce')

    # Vectorized usage-share calculations
    for m in metrics:
        use_col = f'usage.{m}'
        team_col = f'team_{m}'
        share_col = f'usage_share_{m}'
        if use_col in starters.columns and team_col in starters.columns:
            starters[share_col] = (starters[use_col] / starters[team_col]).replace([np.inf, -np.inf], np.nan)
        else:
            starters[share_col] = np.nan

    # Only keep prevalent QB metrics
    desired_cols = [
        'team','player_id','player_name','conference',
        'fumbles_fum', 'fumbles_lost',
        'passing_att', 'passing_completions', 'passing_int', 'passing_pct', 'passing_td', 'passing_yds', 'passing_ypa',
                'rushing_car', 'rushing_long', 'rushing_td', 'rushing_yds', 'rushing_ypc',
        # Usage metrics
        'usage.pass', 'usage.rush',
        'usage.thirdDown',
        'usage.standardDowns', 'usage.passingDowns',
        # PPA metrics
        'averagePPA.all', 'averagePPA.pass', 'averagePPA.rush',
        'averagePPA.firstDown', 'averagePPA.secondDown', 'averagePPA.thirdDown',
        'averagePPA.standardDowns', 'averagePPA.passingDowns'
    ]
    
    starters = starters[desired_cols]

    out_file = "team_starting_qbs.csv"
    starters.to_csv(out_file, index=False)
    print(f"Saved {len(starters)} starters to {out_file}")
    end = time.time()
    print("TIME TAKEN: ", end-start)

if __name__=="__main__":
    main()