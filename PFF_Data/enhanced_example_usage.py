"""
Enhanced Example Usage for QB Archetype Prediction API
Demonstrates both player lookup and stats-based prediction functionality
"""

import requests
import pandas as pd
import json

# API base URL
BASE_URL = "http://localhost:5001"

def predict_qb_archetype(qb_stats, model_type="recall_optimized"):
    """
    Predict QB archetype from stats.
    
    Args:
        qb_stats (dict): Dictionary containing QB statistics
        model_type (str): Type of model to use ('recall_optimized' or 'standard')
    
    Returns:
        dict: Prediction result with archetype, confidence, and probabilities
    """
    try:
        payload = {
            "model_type": model_type,
            **qb_stats
        }
        
        response = requests.post(f"{BASE_URL}/predict", json=payload)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making prediction: {e}")
        return None

def predict_multiple_qbs(qbs_stats, model_type="recall_optimized"):
    """
    Predict archetypes for multiple QBs.
    
    Args:
        qbs_stats (list): List of dictionaries containing QB statistics
        model_type (str): Type of model to use
    
    Returns:
        dict: Batch prediction results
    """
    try:
        payload = {
            "model_type": model_type,
            "qbs": qbs_stats
        }
        
        response = requests.post(f"{BASE_URL}/predict/batch", json=payload)
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error making batch prediction: {e}")
        return None



def search_players(query, limit=10):
    """
    Search for players by name.
    
    Args:
        query (str): Search query
        limit (int): Maximum number of results
    
    Returns:
        dict: Search results
    """
    try:
        response = requests.get(f"{BASE_URL}/search?q={query}&limit={limit}")
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error searching players: {e}")
        return None

def get_archetype_distribution():
    """
    Get archetype distribution for the current year.
    
    Returns:
        dict: Archetype distribution data
    """
    try:
        response = requests.get(f"{BASE_URL}/archetypes/distribution")
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting archetype distribution: {e}")
        return None

def get_top_players_by_archetype(limit=5):
    """
    Get top players for each archetype.
    
    Args:
        limit (int): Number of top players per archetype
    
    Returns:
        dict: Top players by archetype
    """
    try:
        response = requests.get(f"{BASE_URL}/archetypes/top-players?limit={limit}")
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting top players: {e}")
        return None

def get_required_features():
    """
    Get the list of features required for prediction.
    
    Returns:
        dict: Required features information
    """
    try:
        response = requests.get(f"{BASE_URL}/features")
        response.raise_for_status()
        
        return response.json()
    except requests.exceptions.RequestException as e:
        print(f"Error getting features: {e}")
        return None

def main():
    """Demonstrate enhanced API usage."""
    print("🚀 Enhanced QB Archetype Prediction API - Example Usage")
    print("="*70)
    
    # 1. Get required features
    print("\n1. Getting Required Features")
    print("-" * 40)
    features_info = get_required_features()
    if features_info:
        print(f"Number of required features: {features_info['n_features']}")
        print("First 10 features:")
        for i, feature in enumerate(features_info['required_features'][:10], 1):
            print(f"  {i}. {feature}")
    

    
    # 2. Player Search Examples
    print("\n2. Player Search Examples")
    print("-" * 40)
    
    search_queries = ["Gabriel", "Beck", "Sanders"]
    for query in search_queries:
        results = search_players(query, limit=3)
        if results:
            print(f"\n🔍 Search for '{query}':")
            print(f"  Found {results['count']} players:")
            for player in results['results']:
                print(f"    - {player['player_name']}: {player['archetype']} ({player['team_name']})")
    
    # 3. Archetype Distribution
    print("\n3. Archetype Distribution")
    print("-" * 40)
    
    distribution = get_archetype_distribution()
    if distribution:
        print(f"Year: {distribution['year']}")
        print(f"Total players: {distribution['total_players']}")
        print("\nArchetype breakdown:")
        for archetype, count in distribution['archetype_counts'].items():
            percentage = distribution['archetype_percentages'][archetype]
            print(f"  {archetype}: {count} players ({percentage:.1f}%)")
    
    # 4. Top Players by Archetype
    print("\n4. Top Players by Archetype")
    print("-" * 40)
    
    top_players = get_top_players_by_archetype(limit=3)
    if top_players:
        print(f"Top {top_players['limit']} players by archetype:")
        for archetype, players in top_players['top_players'].items():
            print(f"\n  {archetype}:")
            for i, player in enumerate(players, 1):
                print(f"    {i}. {player}")
    
    # 5. Stats-based Prediction Examples
    print("\n5. Stats-based Prediction Examples")
    print("-" * 40)
    
    # Example 1: Pocket Manager type QB
    pocket_manager_stats = {
        "accuracy_percent": 78.0,
        "avg_depth_of_target": 9.2,
        "avg_time_to_throw": 2.9,
        "btt_rate": 6.1,
        "completion_percent": 68.0,
        "sack_percent": 3.8,
        "twp_rate": 2.8,
        "ypa": 8.2,
        "td_int_ratio": 5.0,
        "designed_run_rate": 12.0,
        "scramble_rate": 8.0,
        "elusive_rating": 70.0,
        "ypa_rushing": 6.7,
        "breakaway_percent": 18.0,
        "qb_rush_attempt_rate": 12.0,
        "grades_offense": 80.0,
        "grades_pass": 82.0,
        "grades_run": 70.0,
        "pa_rate": 90.0,
        "pa_ypa": 8.5,
        "screen_rate": 12.0,
        "deep_attempt_rate": 25.0,
        "pressure_rate": 15.0,
        "pressure_sack_percent": 20.0,
        "pressure_twp_rate": 10.0,
        "pressure_accuracy_percent": 65.0,
        "quick_throw_rate": 30.0
    }
    
    result = predict_qb_archetype(pocket_manager_stats)
    if result:
        print(f"✅ Pocket Manager QB: {result['archetype']} (Confidence: {result['confidence']:.3f})")
    
    # Example 2: Dynamic Dual-Threat type QB
    dual_threat_stats = {
        "accuracy_percent": 65.0,
        "avg_depth_of_target": 7.8,
        "avg_time_to_throw": 3.2,
        "btt_rate": 4.2,
        "completion_percent": 62.0,
        "sack_percent": 6.8,
        "twp_rate": 4.1,
        "ypa": 6.8,
        "td_int_ratio": 1.25,
        "designed_run_rate": 25.0,
        "scramble_rate": 20.0,
        "elusive_rating": 85.0,
        "ypa_rushing": 7.5,
        "breakaway_percent": 25.0,
        "qb_rush_attempt_rate": 44.4,
        "grades_offense": 70.0,
        "grades_pass": 72.0,
        "grades_run": 85.0,
        "pa_rate": 80.0,
        "pa_ypa": 7.2,
        "screen_rate": 8.0,
        "deep_attempt_rate": 15.0,
        "pressure_rate": 25.0,
        "pressure_sack_percent": 35.0,
        "pressure_twp_rate": 18.0,
        "pressure_accuracy_percent": 55.0,
        "quick_throw_rate": 20.0
    }
    
    result = predict_qb_archetype(dual_threat_stats)
    if result:
        print(f"✅ Dual-Threat QB: {result['archetype']} (Confidence: {result['confidence']:.3f})")
    
    # 6. Batch Prediction Example
    print("\n6. Batch Prediction Example")
    print("-" * 40)
    
    # Create a DataFrame with QB stats
    qb_data = [
        {
            "name": "QB A",
            "accuracy_percent": 75.0,
            "avg_depth_of_target": 8.5,
            "avg_time_to_throw": 2.8,
            "btt_rate": 5.0,
            "completion_percent": 67.0,
            "sack_percent": 4.5,
            "twp_rate": 3.0,
            "ypa": 7.5,
            "td_int_ratio": 2.8,
            "designed_run_rate": 10.0,
            "scramble_rate": 6.0,
            "elusive_rating": 68.0,
            "ypa_rushing": 5.0,
            "breakaway_percent": 16.0,
            "qb_rush_attempt_rate": 15.0,
            "grades_offense": 76.0,
            "grades_pass": 79.0,
            "grades_run": 68.0,
            "pa_rate": 85.0,
            "pa_ypa": 8.0,
            "screen_rate": 15.0,
            "deep_attempt_rate": 20.0,
            "pressure_rate": 18.0,
            "pressure_sack_percent": 22.0,
            "pressure_twp_rate": 11.0,
            "pressure_accuracy_percent": 63.0,
            "quick_throw_rate": 25.0
        },
        {
            "name": "QB B",
            "accuracy_percent": 70.0,
            "avg_depth_of_target": 7.5,
            "avg_time_to_throw": 3.0,
            "btt_rate": 4.0,
            "completion_percent": 64.0,
            "sack_percent": 5.5,
            "twp_rate": 3.5,
            "ypa": 7.0,
            "td_int_ratio": 2.0,
            "designed_run_rate": 18.0,
            "scramble_rate": 15.0,
            "elusive_rating": 78.0,
            "ypa_rushing": 6.5,
            "breakaway_percent": 20.0,
            "qb_rush_attempt_rate": 25.0,
            "grades_offense": 72.0,
            "grades_pass": 75.0,
            "grades_run": 78.0,
            "pa_rate": 82.0,
            "pa_ypa": 7.8,
            "screen_rate": 12.0,
            "deep_attempt_rate": 18.0,
            "pressure_rate": 20.0,
            "pressure_sack_percent": 28.0,
            "pressure_twp_rate": 14.0,
            "pressure_accuracy_percent": 58.0,
            "quick_throw_rate": 22.0
        }
    ]
    
    # Remove 'name' field for prediction
    qbs_stats = [{k: v for k, v in qb.items() if k != 'name'} for qb in qb_data]
    
    batch_result = predict_multiple_qbs(qbs_stats)
    if batch_result:
        print(f"✅ Batch prediction successful for {len(batch_result['predictions'])} QBs:")
        for i, pred in enumerate(batch_result['predictions']):
            if 'error' not in pred:
                print(f"  QB {i+1}: {pred['archetype']} (Confidence: {pred['confidence']:.3f})")
            else:
                print(f"  QB {i+1}: Error - {pred['error']}")
    
    # 7. QB Comparison Example
    print("\n7. QB Comparison Example")
    print("-" * 40)
    
    # Compare two QBs
    compare_qbs("Kyle McCord", "Dillon Gabriel")
    
    # Compare QBs with different archetypes
    print("\n8. QB Comparison (Different Archetypes)")
    print("-" * 40)
    compare_qbs("Garrett Nussmeier", "Chandler Morris", ["archetype", "matchup"])
    
    print("\n" + "="*70)
    print("✅ Enhanced API example usage completed!")
    print("="*70)

def compare_qbs(qb1_name, qb2_name, analysis_types=None):
    """Compare two QBs and get comprehensive insights."""
    if analysis_types is None:
        analysis_types = ['archetype', 'statistical', 'matchup']
    
    try:
        data = {
            "qb1": qb1_name,
            "qb2": qb2_name,
            "analysis_types": analysis_types
        }
        
        response = requests.post(f"{BASE_URL}/compare", json=data)
        if response.status_code == 200:
            result = response.json()
            comparison = result['comparison']
            
            print(f"✅ QB Comparison: {qb1_name} vs {qb2_name}")
            print(f"Year: {result['year']}")
            
            # Display QB info
            print(f"\n📊 QB Information:")
            print(f"  {comparison['qb1']['name']} ({comparison['qb1']['team']}): {comparison['qb1']['archetype']}")
            print(f"  {comparison['qb2']['name']} ({comparison['qb2']['team']}): {comparison['qb2']['archetype']}")
            
            # Display archetype analysis
            if 'archetype_analysis' in comparison:
                archetype = comparison['archetype_analysis']
                print(f"\n🎯 Archetype Analysis:")
                print(f"  Similarity: {archetype['archetype_similarity']}")
                print(f"  Expected Game Flow: {archetype['expected_game_flow']}")
                print(f"  QB1 Description: {archetype['qb1_description']}")
                print(f"  QB2 Description: {archetype['qb2_description']}")
            
            # Display key differences
            if comparison.get('key_differences'):
                print(f"\n🔍 Key Differences:")
                for diff in comparison['key_differences'][:5]:  # Top 5
                    print(f"  • {diff}")
            
            # Display matchup insights
            if 'matchup_insights' in comparison:
                insights = comparison['matchup_insights']
                print(f"\n🏈 Matchup Insights:")
                if insights.get('key_advantages'):
                    print(f"  Key Advantages:")
                    for advantage in insights['key_advantages']:
                        print(f"    • {advantage}")
                
                if insights.get('defensive_considerations'):
                    print(f"  Defensive Considerations:")
                    for consideration in insights['defensive_considerations']:
                        print(f"    • {consideration}")
                
                print(f"  Predicted Game Style: {insights['predicted_game_style']}")
            
        else:
            print(f"❌ Failed to compare QBs: {response.status_code}")
            print(f"Error: {response.text}")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    main()
