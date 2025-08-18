# Enhanced QB Archetype Prediction API Documentation

## Overview

The Enhanced QB Archetype Prediction API provides comprehensive functionality for QB archetype analysis, including both **player search** and **stats-based prediction** capabilities. This API serves multiple use cases:

- **Player Search**: Find archetypes for existing players in the dataset
- **Stats-based Prediction**: Predict archetypes for new/unknown QBs or hypothetical scenarios
- **QB Comparison**: Compare two QBs with comprehensive insights (archetype, statistical, matchup analysis)
- **Analytics**: Get archetype distributions and top players by archetype
- **Search**: Search for players by name with partial matching

## API Endpoints

### 1. Health Check

**Endpoint:** `GET /health`

**Description:** Check API status and available models.

**Response:**
```json
{
  "status": "healthy",
  "available_models": ["recall_optimized"],
  "year": 2023,
  "player_data_loaded": true
}
```

### 2. Get Required Features

**Endpoint:** `GET /features`

**Description:** Get the list of features required for stats-based prediction.

**Response:**
```json
{
  "required_features": [
    "accuracy_percent",
    "avg_depth_of_target",
    "avg_time_to_throw",
    // ... 27 total features
  ],
  "n_features": 27
}
```

### 3. Get Model Information

**Endpoint:** `GET /models`

**Description:** Get detailed information about available models.

**Response:**
```json
{
  "available_models": {
    "recall_optimized": {
      "model_type": "recall_optimized",
      "model_class": "RandomForestClassifier",
      "n_features": 27,
      "classes": [
        "Dynamic Dual-Threats",
        "Mobile Pocket Passer",
        "Pocket Managers",
        "Scrambling Survivors"
      ],
      "year": 2023
    }
  },
  "year": 2023
}
```

### 4. Player Search

**Endpoint:** `GET /search`

**Description:** Search for players by name with partial matching.

**Parameters:**
- `q` (query): Search query (required)
- `limit` (query): Maximum number of results (default: 10)

**Example:** `GET /search?q=Gabriel&limit=5`

**Response:**
```json
{
  "query": "Gabriel",
  "results": [
    {
      "player_name": "Dillon Gabriel",
      "team_name": "Oklahoma",
      "archetype": "Mobile Pocket Passer",
      "cluster": 1
    }
  ],
  "count": 1,
  "year": 2023
}
```

### 5. Archetype Distribution

**Endpoint:** `GET /archetypes/distribution`

**Description:** Get archetype distribution for the current year.

**Response:**
```json
{
  "year": 2023,
  "total_players": 39,
  "archetype_counts": {
    "Pocket Managers": 18,
    "Mobile Pocket Passer": 7,
    "Dynamic Dual-Threats": 5,
    "Scrambling Survivors": 9
  },
  "archetype_percentages": {
    "Pocket Managers": 46.2,
    "Mobile Pocket Passer": 17.9,
    "Dynamic Dual-Threats": 12.8,
    "Scrambling Survivors": 23.1
  }
}
```

### 6. Top Players by Archetype

**Endpoint:** `GET /archetypes/top-players`

**Description:** Get top players for each archetype.

**Parameters:**
- `limit` (query): Number of top players per archetype (default: 5)

**Example:** `GET /archetypes/top-players?limit=3`

**Response:**
```json
{
  "top_players": {
    "Pocket Managers": [
      "Kyle McCord",
      "Cade Klubnik",
      "Shedeur Sanders"
    ],
    "Mobile Pocket Passer": [
      "Dillon Gabriel",
      "Carson Beck"
    ],
    "Dynamic Dual-Threats": [
      "Jayden Daniels",
      "Caleb Williams"
    ],
    "Scrambling Survivors": [
      "Drake Maye",
      "Bo Nix"
    ]
  },
  "limit": 3,
  "year": 2023
}
```

### 7. Single QB Prediction (Stats-based)

**Endpoint:** `POST /predict`

**Description:** Predict QB archetype from statistics.

**Request Body:**
```json
{
  "model_type": "recall_optimized",
  "accuracy_percent": 75.5,
  "avg_depth_of_target": 8.5,
  "avg_time_to_throw": 2.8,
  "btt_rate": 5.2,
  "completion_percent": 68.0,
  "sack_percent": 4.5,
  "twp_rate": 3.2,
  "ypa": 7.8,
  "td_int_ratio": 2.5,
  "designed_run_rate": 8.0,
  "scramble_rate": 5.0,
  "elusive_rating": 65.2,
  "ypa_rushing": 4.2,
  "breakaway_percent": 15.0,
  "qb_rush_attempt_rate": 12.5,
  "grades_offense": 75.5,
  "grades_pass": 78.2,
  "grades_run": 65.1,
  "pa_rate": 88.9,
  "pa_ypa": 8.2,
  "screen_rate": 15.0,
  "deep_attempt_rate": 20.0,
  "pressure_rate": 17.5,
  "pressure_sack_percent": 25.0,
  "pressure_twp_rate": 12.0,
  "pressure_accuracy_percent": 62.0,
  "quick_throw_rate": 25.0
}
```

**Response:**
```json
{
  "archetype": "Pocket Managers",
  "confidence": 0.823,
  "probabilities": {
    "Dynamic Dual-Threats": 0.045,
    "Mobile Pocket Passer": 0.132,
    "Pocket Managers": 0.823,
    "Scrambling Survivors": 0.000
  },
  "model_type": "recall_optimized",
  "year": 2023
}
```

### 8. Batch QB Prediction

**Endpoint:** `POST /predict/batch`

**Description:** Predict archetypes for multiple QBs.

**Request Body:**
```json
{
  "model_type": "recall_optimized",
  "qbs": [
    {
      "accuracy_percent": 75.5,
      "avg_depth_of_target": 8.5,
      // ... all required features for QB 1
    },
    {
      "accuracy_percent": 65.0,
      "avg_depth_of_target": 7.8,
      // ... all required features for QB 2
    }
  ]
}
```

**Response:**
```json
{
  "predictions": [
    {
      "qb_index": 0,
      "archetype": "Pocket Managers",
      "confidence": 0.823,
      "probabilities": {
        "Dynamic Dual-Threats": 0.045,
        "Mobile Pocket Passer": 0.132,
        "Pocket Managers": 0.823,
        "Scrambling Survivors": 0.000
      },
      "model_type": "recall_optimized",
      "year": 2023
    },
    {
      "qb_index": 1,
      "archetype": "Dynamic Dual-Threats",
      "confidence": 0.756,
      "probabilities": {
        "Dynamic Dual-Threats": 0.756,
        "Mobile Pocket Passer": 0.123,
        "Pocket Managers": 0.089,
        "Scrambling Survivors": 0.032
      },
      "model_type": "recall_optimized",
      "year": 2023
    }
  ],
  "model_type": "recall_optimized",
  "year": 2023
}
```

## QB Archetypes

The API classifies QBs into four archetypes:

1. **Pocket Managers**: Traditional pocket passers with high accuracy and decision-making
2. **Mobile Pocket Passer**: QBs who can extend plays but primarily pass from the pocket
3. **Dynamic Dual-Threats**: QBs who are equally effective as passers and runners
4. **Scrambling Survivors**: QBs who rely heavily on mobility to compensate for passing limitations

## Getting Started

### 1. Start the API Server

```bash
python3 enhanced_api.py
```

The API will start on `http://localhost:5001`

### 2. Test the API

```bash
python3 test_enhanced_api.py
```

### 3. Use the API

```bash
python3 enhanced_example_usage.py
```

## Usage Examples

### Python Examples

#### Player Search
```python
# Search for players
response = requests.get("http://localhost:5001/search?q=Gabriel&limit=5")
search_results = response.json()
for player in search_results['results']:
    print(f"{player['player_name']}: {player['archetype']}")
```

#### Stats-based Prediction
```python
# Predict archetype from stats
qb_stats = {
    "accuracy_percent": 75.5,
    "avg_depth_of_target": 8.5,
    # ... all required features
}

response = requests.post("http://localhost:5001/predict", json={
    "model_type": "recall_optimized",
    **qb_stats
})
prediction = response.json()
print(f"Predicted archetype: {prediction['archetype']}")
```

#### Get Archetype Distribution
```python
# Get distribution
response = requests.get("http://localhost:5001/archetypes/distribution")
distribution = response.json()
for archetype, count in distribution['archetype_counts'].items():
    percentage = distribution['archetype_percentages'][archetype]
    print(f"{archetype}: {count} players ({percentage:.1f}%)")
```

### cURL Examples

#### Player Search
```bash
curl -X GET "http://localhost:5001/search?q=Gabriel&limit=5"
```

#### Stats-based Prediction
```bash
curl -X POST "http://localhost:5001/predict" \
  -H "Content-Type: application/json" \
  -d '{
    "model_type": "recall_optimized",
    "accuracy_percent": 75.5,
    "avg_depth_of_target": 8.5,
    "avg_time_to_throw": 2.8,
    "btt_rate": 5.2,
    "completion_percent": 68.0,
    "sack_percent": 4.5,
    "twp_rate": 3.2,
    "ypa": 7.8,
    "td_int_ratio": 2.5,
    "designed_run_rate": 8.0,
    "scramble_rate": 5.0,
    "elusive_rating": 65.2,
    "ypa_rushing": 4.2,
    "breakaway_percent": 15.0,
    "qb_rush_attempt_rate": 12.5,
    "grades_offense": 75.5,
    "grades_pass": 78.2,
    "grades_run": 65.1,
    "pa_rate": 88.9,
    "pa_ypa": 8.2,
    "screen_rate": 15.0,
    "deep_attempt_rate": 20.0,
    "pressure_rate": 17.5,
    "pressure_sack_percent": 25.0,
    "pressure_twp_rate": 12.0,
    "pressure_accuracy_percent": 62.0,
    "quick_throw_rate": 25.0
  }'
```

#### Get Archetype Distribution
```bash
curl -X GET "http://localhost:5001/archetypes/distribution"
```

#### QB Comparison
```bash
curl -X POST "http://localhost:5001/compare" \
  -H "Content-Type: application/json" \
  -d '{
    "qb1": "Kyle McCord",
    "qb2": "Dillon Gabriel",
    "analysis_types": ["archetype", "statistical", "matchup"]
  }'
```

**Response includes:**
- **Archetype Analysis**: Playing style comparison and expected game flow
- **Statistical Comparison**: Detailed stat-by-stat breakdown with significance levels  
- **Matchup Insights**: Key advantages and defensive considerations
- **Key Differences**: Top 5 most significant statistical differences

**Perfect for:**
- 🏈 Pre-game QB analysis
- 📊 Strategic defensive planning
- 🎯 Scouting report generation
- 📈 Performance comparison studies

## Error Handling

The API returns appropriate HTTP status codes and error messages:

- `200`: Success
- `400`: Bad Request (missing required parameters)
- `404`: Not Found (player not found)
- `500`: Internal Server Error

Error responses include a descriptive message:
```json
{
  "error": "Player \"Unknown Player\" not found"
}
```

## Configuration

The API automatically loads:
- Trained models from the `models/` directory
- Player data from the analysis results
- Configuration from `config.py`

## Performance

- **Player Lookup**: O(1) average case for exact matches
- **Player Search**: O(n) for partial matches with indexing
- **Stats-based Prediction**: O(1) for single predictions
- **Batch Prediction**: O(n) where n is the number of QBs

## Security Notes

- The API runs in debug mode by default
- No authentication is implemented
- Input validation is performed on all endpoints
- Consider implementing rate limiting for production use

## Troubleshooting

### Common Issues

1. **Port Already in Use**: Change the port in `enhanced_api.py` or stop the conflicting service
2. **Model Not Found**: Ensure models are trained and saved in the correct directory
3. **Player Not Found**: Check spelling or use the search endpoint for partial matches
4. **Missing Features**: Use the `/features` endpoint to get the complete list of required features

### Debug Mode

The API runs in debug mode by default. Check the console output for detailed error messages and logs.

## Integration with Pandas

The enhanced example usage script demonstrates how to integrate the API with Pandas DataFrames for batch processing and analysis.
