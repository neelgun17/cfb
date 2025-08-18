"""
Enhanced QB Archetype Prediction API
Supports both player name lookup and stats-based prediction
"""

from flask import Flask, request, jsonify
from flask_swagger_ui import get_swaggerui_blueprint
import pandas as pd
import numpy as np
import joblib
import logging
from pathlib import Path
from typing import Dict, List, Optional
import traceback

from config import get_model_files, get_analysis_files, CURRENT_YEAR, AI_ANALYZER_TYPE
from ai_analysis_service import QBArchetypeAIAnalyzer
from lightweight_ai_analyzer import LightweightQBAnalyzer

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

app = Flask(__name__)

# Swagger configuration
SWAGGER_URL = '/swagger'
API_URL = '/static/swagger.json'
swaggerui_blueprint = get_swaggerui_blueprint(
    SWAGGER_URL,
    API_URL,
    config={
        'app_name': "QB Archetype Prediction API"
    }
)
app.register_blueprint(swaggerui_blueprint, url_prefix=SWAGGER_URL)

class EnhancedQBArchetypePredictor:
    """Enhanced QB archetype predictor with player lookup capabilities."""
    
    def __init__(self, year: int = CURRENT_YEAR):
        self.year = year
        self.models = {}
        self.preprocessors = {}
        self.feature_names = None
        self.player_data = None
        self.load_models()
        self.load_player_data()
    
    def load_models(self):
        """Load all available trained models."""
        logger.info(f"Loading models for year {self.year}")
        
        model_types = ['recall_optimized', 'standard']
        
        for model_type in model_types:
            try:
                model_files = get_model_files(self.year, model_type)
                
                # Check if model exists
                if model_files['model'].exists():
                    # Load model and preprocessors
                    model = joblib.load(model_files['model'])
                    label_encoder = joblib.load(model_files['label_encoder'])
                    scaler = joblib.load(model_files['scaler'])
                    imputer = joblib.load(model_files['imputer'])
                    
                    self.models[model_type] = {
                        'model': model,
                        'label_encoder': label_encoder,
                        'scaler': scaler,
                        'imputer': imputer
                    }
                    
                    # Set feature names from the first loaded model
                    if self.feature_names is None:
                        self.feature_names = list(model.feature_names_in_)
                    
                    logger.info(f"Loaded {model_type} model successfully")
                else:
                    logger.warning(f"No {model_type} model found for year {self.year}")
                    
            except Exception as e:
                logger.error(f"Error loading {model_type} model: {e}")
    
    def load_player_data(self):
        """Load player data for lookup functionality."""
        try:
            analysis_files = get_analysis_files(self.year)
            if analysis_files['player_assignments'].exists():
                self.player_data = pd.read_csv(analysis_files['player_assignments'])
                logger.info(f"Loaded player data for {len(self.player_data)} players")
            else:
                logger.warning(f"No player data found for year {self.year}")
                self.player_data = None
        except Exception as e:
            logger.error(f"Error loading player data: {e}")
            self.player_data = None
    

    
    def search_players(self, query: str, limit: int = 10) -> List[Dict]:
        """Search for players by name (case-insensitive partial match)."""
        if self.player_data is None:
            return []
        
        # Case-insensitive partial match
        matches = self.player_data[
            self.player_data['player'].str.lower().str.contains(
                query.lower(), na=False
            )
        ]
        
        # Limit results
        matches = matches.head(limit)
        
        return [
            {
                'player_name': row['player'],
                'team_name': row['team_name'],
                'archetype': row['archetype_name'],
                'cluster': row['hierarchical_cluster']
            }
            for _, row in matches.iterrows()
        ]
    
    def get_archetype_distribution(self) -> Dict:
        """Get archetype distribution for the current year."""
        if self.player_data is None:
            return {}
        
        distribution = self.player_data['archetype_name'].value_counts().to_dict()
        total_players = len(self.player_data)
        
        return {
            'year': self.year,
            'total_players': total_players,
            'archetype_counts': distribution,
            'archetype_percentages': {
                archetype: (count / total_players) * 100
                for archetype, count in distribution.items()
            }
        }
    
    def get_top_players_by_archetype(self, limit: int = 5) -> Dict:
        """Get top players for each archetype."""
        if self.player_data is None:
            return {}
        
        result = {}
        for archetype in self.player_data['archetype_name'].unique():
            archetype_players = self.player_data[
                self.player_data['archetype_name'] == archetype
            ]
            result[archetype] = archetype_players['player'].tolist()[:limit]
        
        return result
    
    def preprocess_input(self, qb_data: Dict, model_type: str = 'recall_optimized') -> np.ndarray:
        """Preprocess QB data for prediction."""
        try:
            # Create DataFrame from input data
            df = pd.DataFrame([qb_data])
            
            # Ensure all required features are present
            missing_features = [f for f in self.feature_names if f not in df.columns]
            if missing_features:
                # Fill missing features with 0 or appropriate defaults
                for feature in missing_features:
                    df[feature] = 0
            
            # Select only the features used in training
            X = df[self.feature_names].copy()
            
            # Handle infinite values
            X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Get preprocessors
            preprocessors = self.models[model_type]
            imputer = preprocessors['imputer']
            scaler = preprocessors['scaler']
            
            # Apply preprocessing
            X_imputed = imputer.transform(X)
            X_scaled = scaler.transform(X_imputed)
            
            return X_scaled
            
        except Exception as e:
            logger.error(f"Error preprocessing input: {e}")
            raise
    
    def predict_archetype(self, qb_data: Dict, model_type: str = 'recall_optimized') -> Dict:
        """Predict QB archetype from stats."""
        try:
            if model_type not in self.models:
                raise ValueError(f"Model type '{model_type}' not available")
            
            # Preprocess input
            X_processed = self.preprocess_input(qb_data, model_type)
            
            # Get model and make prediction
            model = self.models[model_type]['model']
            label_encoder = self.models[model_type]['label_encoder']
            
            # Get prediction and probabilities
            prediction = model.predict(X_processed)[0]
            probabilities = model.predict_proba(X_processed)[0]
            
            # Convert prediction to archetype name
            archetype = label_encoder.inverse_transform([prediction])[0]
            
            # Get confidence (probability of predicted class)
            confidence = probabilities[prediction]
            
            # Get all archetype probabilities
            archetype_probs = {}
            for i, archetype_name in enumerate(label_encoder.classes_):
                archetype_probs[archetype_name] = float(probabilities[i])
            
            return {
                'archetype': archetype,
                'confidence': float(confidence),
                'probabilities': archetype_probs,
                'model_type': model_type,
                'year': self.year
            }
            
        except Exception as e:
            logger.error(f"Error making prediction: {e}")
            raise
    
    def get_available_models(self) -> List[str]:
        """Get list of available model types."""
        return list(self.models.keys())
    
    def get_qb_data(self, qb_name: str) -> Optional[Dict]:
        """Get complete QB data by name."""
        if self.player_data is None:
            return None
        
        # Find QB in player data
        matches = self.player_data[
            self.player_data['player'].str.lower().str.contains(
                qb_name.lower(), na=False
            )
        ]
        
        if len(matches) == 0:
            return None
        
        qb_row = matches.iloc[0]
        
        # Get QB stats from the final merged data
        try:
            analysis_files = get_analysis_files(self.year)
            if analysis_files['final_merged'].exists():
                merged_data = pd.read_csv(analysis_files['final_merged'])
                qb_stats = merged_data[merged_data['player'] == qb_row['player']]
                
                if len(qb_stats) > 0:
                    stats_row = qb_stats.iloc[0]
                    return {
                        'name': qb_row['player'],
                        'team': qb_row['team_name'],
                        'archetype': qb_row['archetype_name'],
                        'cluster': int(qb_row['hierarchical_cluster']),
                        'key_stats': {
                            'accuracy_percent': float(stats_row.get('accuracy_percent', 0)),
                            'completion_percent': float(stats_row.get('completion_percent', 0)),
                            'ypa': float(stats_row.get('ypa', 0)),
                            'td_int_ratio': float(stats_row.get('td_int_ratio', 0)),
                            'scramble_rate': float(stats_row.get('scramble_rate', 0)),
                            'pressure_accuracy_percent': float(stats_row.get('pressure_accuracy_percent', 0)),
                            'pressure_twp_rate': float(stats_row.get('pressure_twp_rate', 0)),
                            'elusive_rating': float(stats_row.get('elusive_rating', 0)),
                            'grades_offense': float(stats_row.get('grades_offense', 0)),
                            'grades_pass': float(stats_row.get('grades_pass', 0)),
                            'grades_run': float(stats_row.get('grades_run', 0)),
                            'avg_depth_of_target': float(stats_row.get('avg_depth_of_target', 0)),
                            'avg_time_to_throw': float(stats_row.get('avg_time_to_throw', 0)),
                            'sack_percent': float(stats_row.get('sack_percent', 0)),
                            'twp_rate': float(stats_row.get('twp_rate', 0)),
                            'designed_run_rate': float(stats_row.get('designed_run_rate', 0)),
                            'ypa_rushing': float(stats_row.get('ypa_rushing', 0)),
                            'breakaway_percent': float(stats_row.get('breakaway_percent', 0)),
                            'qb_rush_attempt_rate': float(stats_row.get('qb_rush_attempt_rate', 0))
                        }
                    }
        except Exception as e:
            logger.error(f"Error getting QB stats for {qb_name}: {e}")
        
        # Fallback to basic data
        return {
            'name': qb_row['player'],
            'team': qb_row['team_name'],
            'archetype': qb_row['archetype_name'],
            'cluster': int(qb_row['hierarchical_cluster']),
            'key_stats': {}
        }
    
    def _analyze_archetypes(self, qb1_data: Dict, qb2_data: Dict) -> Dict:
        """Analyze archetype differences between two QBs."""
        qb1_archetype = qb1_data['archetype']
        qb2_archetype = qb2_data['archetype']
        
        # Archetype descriptions
        archetype_descriptions = {
            'Pocket Managers': 'Traditional pocket passer with high accuracy and decision-making',
            'Dynamic Dual-Threats': 'Equally effective as passer and runner with explosive playmaking',
            'Scrambling Survivors': 'Relies heavily on mobility to compensate for passing limitations',
            'Mobile Pocket Passer': 'Can extend plays but primarily passes from the pocket'
        }
        
        # Determine similarity and differences
        if qb1_archetype == qb2_archetype:
            archetype_similarity = f"Both are {qb1_archetype}"
            style_differences = "Similar playing styles, focus on statistical differences"
        else:
            archetype_similarity = f"Different archetypes: {qb1_archetype} vs {qb2_archetype}"
            style_differences = f"{qb1_data['name']} is a {qb1_archetype}, {qb2_data['name']} is a {qb2_archetype}"
        
        # Generate expected game flow based on archetypes
        if qb1_archetype == qb2_archetype:
            if qb1_archetype == 'Pocket Managers':
                expected_flow = "Both QBs will primarily stay in pocket, efficient passing game"
            elif qb1_archetype == 'Dynamic Dual-Threats':
                expected_flow = "Both QBs will mix passing and running, explosive plays expected"
            elif qb1_archetype == 'Scrambling Survivors':
                expected_flow = "Both QBs will rely heavily on mobility and extending plays"
            else:  # Mobile Pocket Passer
                expected_flow = "Both QBs can extend plays but prefer passing from pocket"
        else:
            expected_flow = f"Contrasting styles: {qb1_archetype} vs {qb2_archetype} will create interesting matchup"
        
        return {
            'qb1_archetype': qb1_archetype,
            'qb2_archetype': qb2_archetype,
            'archetype_similarity': archetype_similarity,
            'style_differences': style_differences,
            'expected_game_flow': expected_flow,
            'qb1_description': archetype_descriptions.get(qb1_archetype, ''),
            'qb2_description': archetype_descriptions.get(qb2_archetype, '')
        }
    
    def _compare_statistics(self, qb1_data: Dict, qb2_data: Dict) -> Dict:
        """Compare key statistics between two QBs."""
        qb1_stats = qb1_data['key_stats']
        qb2_stats = qb2_data['key_stats']
        
        # Key metrics to compare
        comparison_metrics = [
            'accuracy_percent',
            'completion_percent',
            'ypa',
            'td_int_ratio',
            'scramble_rate',
            'pressure_accuracy_percent',
            'pressure_twp_rate',
            'elusive_rating',
            'grades_offense',
            'grades_pass',
            'grades_run',
            'avg_depth_of_target',
            'avg_time_to_throw',
            'sack_percent',
            'twp_rate',
            'designed_run_rate',
            'ypa_rushing',
            'breakaway_percent',
            'qb_rush_attempt_rate'
        ]
        
        comparison = {}
        
        for metric in comparison_metrics:
            qb1_value = qb1_stats.get(metric, 0)
            qb2_value = qb2_stats.get(metric, 0)
            
            if qb1_value != 0 or qb2_value != 0:  # Only include if we have data
                difference = qb1_value - qb2_value
                
                # Determine advantage
                if difference > 0:
                    advantage = f"{qb1_data['name']} (+{abs(difference):.1f})"
                elif difference < 0:
                    advantage = f"{qb2_data['name']} (+{abs(difference):.1f})"
                else:
                    advantage = "Equal"
                
                # Determine significance
                if abs(difference) > 10:
                    significance = "high"
                elif abs(difference) > 5:
                    significance = "moderate"
                elif abs(difference) > 2:
                    significance = "low"
                else:
                    significance = "minimal"
                
                comparison[metric] = {
                    'qb1_value': qb1_value,
                    'qb2_value': qb2_value,
                    'advantage': advantage,
                    'significance': significance
                }
        
        return comparison
    
    def _generate_matchup_insights(self, qb1_data: Dict, qb2_data: Dict) -> Dict:
        """Generate matchup insights and predictions."""
        qb1_stats = qb1_data['key_stats']
        qb2_stats = qb2_data['key_stats']
        
        # Analyze key advantages
        advantages = []
        
        # Accuracy comparison
        if qb1_stats.get('accuracy_percent', 0) > qb2_stats.get('accuracy_percent', 0):
            diff = qb1_stats.get('accuracy_percent', 0) - qb2_stats.get('accuracy_percent', 0)
            advantages.append(f"{qb1_data['name']}: Better accuracy (+{diff:.1f}%)")
        elif qb2_stats.get('accuracy_percent', 0) > qb1_stats.get('accuracy_percent', 0):
            diff = qb2_stats.get('accuracy_percent', 0) - qb1_stats.get('accuracy_percent', 0)
            advantages.append(f"{qb2_data['name']}: Better accuracy (+{diff:.1f}%)")
        
        # Mobility comparison
        qb1_scramble = qb1_stats.get('scramble_rate', 0)
        qb2_scramble = qb2_stats.get('scramble_rate', 0)
        
        if qb1_scramble > qb2_scramble:
            diff = qb1_scramble - qb2_scramble
            advantages.append(f"{qb1_data['name']}: More mobile ({qb1_scramble:.1f}% vs {qb2_scramble:.1f}% scramble rate)")
        elif qb2_scramble > qb1_scramble:
            diff = qb2_scramble - qb1_scramble
            advantages.append(f"{qb2_data['name']}: More mobile ({qb2_scramble:.1f}% vs {qb1_scramble:.1f}% scramble rate)")
        else:
            advantages.append(f"Both QBs have similar mobility ({qb1_scramble:.1f}% scramble rate)")
        
        # Pressure handling
        if qb1_stats.get('pressure_accuracy_percent', 0) > qb2_stats.get('pressure_accuracy_percent', 0):
            diff = qb1_stats.get('pressure_accuracy_percent', 0) - qb2_stats.get('pressure_accuracy_percent', 0)
            advantages.append(f"{qb1_data['name']}: Better under pressure (+{diff:.1f}%)")
        elif qb2_stats.get('pressure_accuracy_percent', 0) > qb1_stats.get('pressure_accuracy_percent', 0):
            diff = qb2_stats.get('pressure_accuracy_percent', 0) - qb1_stats.get('pressure_accuracy_percent', 0)
            advantages.append(f"{qb2_data['name']}: Better under pressure (+{diff:.1f}%)")
        
        # Generate defensive considerations
        defensive_considerations = []
        
        # Based on archetypes
        if qb1_data['archetype'] == 'Pocket Managers' and qb2_data['archetype'] == 'Pocket Managers':
            defensive_considerations.append("Both QBs are pocket passers - pressure the pocket")
        elif 'Dynamic Dual-Threats' in [qb1_data['archetype'], qb2_data['archetype']]:
            mobile_qbs = []
            if qb1_data['archetype'] == 'Dynamic Dual-Threats':
                mobile_qbs.append(qb1_data['name'])
            if qb2_data['archetype'] == 'Dynamic Dual-Threats':
                mobile_qbs.append(qb2_data['name'])
            defensive_considerations.append(f"{', '.join(mobile_qbs)} are mobile QBs - contain and spy")
        
        # Based on mobility
        high_scramble_qbs = []
        if qb1_stats.get('scramble_rate', 0) > 15:
            high_scramble_qbs.append(f"{qb1_data['name']} ({qb1_stats.get('scramble_rate', 0):.1f}%)")
        if qb2_stats.get('scramble_rate', 0) > 15:
            high_scramble_qbs.append(f"{qb2_data['name']} ({qb2_stats.get('scramble_rate', 0):.1f}%)")
        
        if high_scramble_qbs:
            defensive_considerations.append(f"High scramble rates from {', '.join(high_scramble_qbs)} - maintain contain")
        
        # Based on pressure handling
        if qb1_stats.get('pressure_accuracy_percent', 0) < 60:
            defensive_considerations.append(f"{qb1_data['name']}: Poor pressure handling ({qb1_stats.get('pressure_accuracy_percent', 0):.1f}%) - blitz frequently")
        if qb2_stats.get('pressure_accuracy_percent', 0) < 60:
            defensive_considerations.append(f"{qb2_data['name']}: Poor pressure handling ({qb2_stats.get('pressure_accuracy_percent', 0):.1f}%) - blitz frequently")
        
        # Predict game style
        if qb1_data['archetype'] == qb2_data['archetype']:
            if qb1_data['archetype'] == 'Pocket Managers':
                game_style = "Efficient passing game from both QBs, limited rushing"
            elif qb1_data['archetype'] == 'Dynamic Dual-Threats':
                game_style = "Explosive plays expected from both QBs, high rushing totals"
            elif qb1_data['archetype'] == 'Scrambling Survivors':
                game_style = "Both QBs will rely heavily on mobility and extending plays"
            else:
                game_style = "Balanced approach from both QBs, moderate rushing"
        else:
            game_style = f"Contrasting styles: {qb1_data['archetype']} vs {qb2_data['archetype']} will create dynamic matchup"
        
        return {
            'key_advantages': advantages,
            'defensive_considerations': defensive_considerations,
            'predicted_game_style': game_style
        }
    
    def compare_qbs(self, qb1_name: str, qb2_name: str, analysis_types: List[str] = None) -> Dict:
        """Compare two QBs and provide comprehensive insights."""
        if analysis_types is None:
            analysis_types = ['archetype', 'statistical', 'matchup']
        
        # Get QB data
        qb1_data = self.get_qb_data(qb1_name)
        qb2_data = self.get_qb_data(qb2_name)
        
        if qb1_data is None:
            raise ValueError(f"QB '{qb1_name}' not found in database")
        if qb2_data is None:
            raise ValueError(f"QB '{qb2_name}' not found in database")
        
        # Build comparison results
        comparison = {
            'qb1': qb1_data,
            'qb2': qb2_data
        }
        
        # Perform different types of analysis
        if 'archetype' in analysis_types:
            comparison['archetype_analysis'] = self._analyze_archetypes(qb1_data, qb2_data)
        
        if 'statistical' in analysis_types:
            comparison['statistical_comparison'] = self._compare_statistics(qb1_data, qb2_data)
        
        if 'matchup' in analysis_types:
            comparison['matchup_insights'] = self._generate_matchup_insights(qb1_data, qb2_data)
        
        # Generate key differences summary
        key_differences = []
        if 'statistical' in analysis_types:
            stats_comparison = comparison['statistical_comparison']
            for metric, data in stats_comparison.items():
                if data['significance'] in ['moderate', 'high']:
                    key_differences.append(f"{data['advantage']} in {metric.replace('_', ' ')}")
        
        comparison['key_differences'] = key_differences[:5]  # Top 5 differences
        
        return comparison

# Initialize predictor and AI analyzer
predictor = EnhancedQBArchetypePredictor()

# Global variable to hold the current analyzer
current_ai_analyzer = None

def get_ai_analyzer():
    """Get the current AI analyzer based on configuration"""
    global current_ai_analyzer
    if AI_ANALYZER_TYPE == "qwen":
        if current_ai_analyzer is None or not isinstance(current_ai_analyzer, QBArchetypeAIAnalyzer):
            current_ai_analyzer = QBArchetypeAIAnalyzer()
            logger.info(f"Switched to Qwen3:8B AI analyzer (detailed analysis, slower)")
    else:
        if current_ai_analyzer is None or not isinstance(current_ai_analyzer, LightweightQBAnalyzer):
            current_ai_analyzer = LightweightQBAnalyzer()
            logger.info(f"Switched to Lightweight AI analyzer (fast analysis, rule-based)")
    return current_ai_analyzer

# Initialize with default analyzer
ai_analyzer = get_ai_analyzer()

@app.route('/static/swagger.json')
def create_swagger_spec():
    """Generate Swagger/OpenAPI specification."""
    swagger_spec = {
        "openapi": "3.0.0",
        "info": {
            "title": "QB Archetype Prediction API",
            "description": "A comprehensive API for QB archetype analysis with player search and stats-based prediction capabilities.",
            "version": "1.0.0",
            "contact": {
                "name": "API Support",
                "email": "support@example.com"
            }
        },
        "servers": [
            {
                "url": "http://localhost:5001",
                "description": "Development server"
            }
        ],
        "paths": {
            "/health": {
                "get": {
                    "summary": "Health Check",
                    "description": "Check API status and available models",
                    "responses": {
                        "200": {
                            "description": "API is healthy",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "available_models": {"type": "array", "items": {"type": "string"}},
                                            "year": {"type": "integer"},
                                            "player_data_loaded": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/features": {
                "get": {
                    "summary": "Get Required Features",
                    "description": "Get the list of features required for stats-based prediction",
                    "responses": {
                        "200": {
                            "description": "List of required features",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "required_features": {"type": "array", "items": {"type": "string"}},
                                            "n_features": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/models": {
                "get": {
                    "summary": "Get Model Information",
                    "description": "Get detailed information about available models",
                    "responses": {
                        "200": {
                            "description": "Model information",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "available_models": {"type": "object"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/search": {
                "get": {
                    "summary": "Search Players",
                    "description": "Search for players by name with partial matching",
                    "parameters": [
                        {
                            "name": "q",
                            "in": "query",
                            "required": True,
                            "description": "Search query",
                            "schema": {"type": "string"}
                        },
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "description": "Maximum number of results",
                            "schema": {"type": "integer", "default": 10}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Search results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "query": {"type": "string"},
                                            "results": {"type": "array"},
                                            "count": {"type": "integer"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - missing query parameter"
                        }
                    }
                }
            },
            "/archetypes/distribution": {
                "get": {
                    "summary": "Get Archetype Distribution",
                    "description": "Get archetype distribution for the current year",
                    "responses": {
                        "200": {
                            "description": "Archetype distribution data",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "year": {"type": "integer"},
                                            "total_players": {"type": "integer"},
                                            "archetype_counts": {"type": "object"},
                                            "archetype_percentages": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/archetypes/top-players": {
                "get": {
                    "summary": "Get Top Players by Archetype",
                    "description": "Get top players for each archetype",
                    "parameters": [
                        {
                            "name": "limit",
                            "in": "query",
                            "required": False,
                            "description": "Number of top players per archetype",
                            "schema": {"type": "integer", "default": 5}
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Top players by archetype",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "top_players": {"type": "object"},
                                            "limit": {"type": "integer"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/predict": {
                "post": {
                    "summary": "Predict QB Archetype",
                    "description": "Predict QB archetype from statistics",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_type": {
                                            "type": "string",
                                            "description": "Type of model to use",
                                            "default": "recall_optimized",
                                            "enum": ["recall_optimized", "standard"]
                                        },
                                        "accuracy_percent": {"type": "number"},
                                        "avg_depth_of_target": {"type": "number"},
                                        "avg_time_to_throw": {"type": "number"},
                                        "btt_rate": {"type": "number"},
                                        "completion_percent": {"type": "number"},
                                        "sack_percent": {"type": "number"},
                                        "twp_rate": {"type": "number"},
                                        "ypa": {"type": "number"},
                                        "td_int_ratio": {"type": "number"},
                                        "designed_run_rate": {"type": "number"},
                                        "scramble_rate": {"type": "number"},
                                        "elusive_rating": {"type": "number"},
                                        "ypa_rushing": {"type": "number"},
                                        "breakaway_percent": {"type": "number"},
                                        "qb_rush_attempt_rate": {"type": "number"},
                                        "grades_offense": {"type": "number"},
                                        "grades_pass": {"type": "number"},
                                        "grades_run": {"type": "number"},
                                        "pa_rate": {"type": "number"},
                                        "pa_ypa": {"type": "number"},
                                        "screen_rate": {"type": "number"},
                                        "deep_attempt_rate": {"type": "number"},
                                        "pressure_rate": {"type": "number"},
                                        "pressure_sack_percent": {"type": "number"},
                                        "pressure_twp_rate": {"type": "number"},
                                        "pressure_accuracy_percent": {"type": "number"},
                                        "quick_throw_rate": {"type": "number"}
                                    },
                                    "required": ["accuracy_percent", "avg_depth_of_target", "avg_time_to_throw", "btt_rate", "completion_percent", "sack_percent", "twp_rate", "ypa", "td_int_ratio", "designed_run_rate", "scramble_rate", "elusive_rating", "ypa_rushing", "breakaway_percent", "qb_rush_attempt_rate", "grades_offense", "grades_pass", "grades_run", "pa_rate", "pa_ypa", "screen_rate", "deep_attempt_rate", "pressure_rate", "pressure_sack_percent", "pressure_twp_rate", "pressure_accuracy_percent", "quick_throw_rate"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Prediction result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "archetype": {"type": "string"},
                                            "confidence": {"type": "number"},
                                            "probabilities": {"type": "object"},
                                            "model_type": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/predict/batch": {
                "post": {
                    "summary": "Batch QB Prediction",
                    "description": "Predict archetypes for multiple QBs",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "model_type": {
                                            "type": "string",
                                            "description": "Type of model to use",
                                            "default": "recall_optimized",
                                            "enum": ["recall_optimized", "standard"]
                                        },
                                        "qbs": {
                                            "type": "array",
                                            "items": {
                                                "type": "object",
                                                "properties": {
                                                    "accuracy_percent": {"type": "number"},
                                                    "avg_depth_of_target": {"type": "number"},
                                                    "avg_time_to_throw": {"type": "number"},
                                                    "btt_rate": {"type": "number"},
                                                    "completion_percent": {"type": "number"},
                                                    "sack_percent": {"type": "number"},
                                                    "twp_rate": {"type": "number"},
                                                    "ypa": {"type": "number"},
                                                    "td_int_ratio": {"type": "number"},
                                                    "designed_run_rate": {"type": "number"},
                                                    "scramble_rate": {"type": "number"},
                                                    "elusive_rating": {"type": "number"},
                                                    "ypa_rushing": {"type": "number"},
                                                    "breakaway_percent": {"type": "number"},
                                                    "qb_rush_attempt_rate": {"type": "number"},
                                                    "grades_offense": {"type": "number"},
                                                    "grades_pass": {"type": "number"},
                                                    "grades_run": {"type": "number"},
                                                    "pa_rate": {"type": "number"},
                                                    "pa_ypa": {"type": "number"},
                                                    "screen_rate": {"type": "number"},
                                                    "deep_attempt_rate": {"type": "number"},
                                                    "pressure_rate": {"type": "number"},
                                                    "pressure_sack_percent": {"type": "number"},
                                                    "pressure_twp_rate": {"type": "number"},
                                                    "pressure_accuracy_percent": {"type": "number"},
                                                    "quick_throw_rate": {"type": "number"}
                                                }
                                            }
                                        }
                                    },
                                    "required": ["qbs"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Batch prediction results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "predictions": {"type": "array"},
                                            "model_type": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/compare": {
                "post": {
                    "summary": "Compare Two QBs",
                    "description": "Compare two QBs and provide comprehensive insights including archetype analysis, statistical comparison, and matchup insights",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb1": {
                                            "type": "string",
                                            "description": "Name of first QB to compare",
                                            "example": "Kyle McCord"
                                        },
                                        "qb2": {
                                            "type": "string",
                                            "description": "Name of second QB to compare",
                                            "example": "Dillon Gabriel"
                                        },
                                        "analysis_types": {
                                            "type": "array",
                                            "items": {
                                                "type": "string",
                                                "enum": ["archetype", "statistical", "matchup"]
                                            },
                                            "description": "Types of analysis to perform",
                                            "default": ["archetype", "statistical", "matchup"]
                                        },
                                        "include_ai": {
                                            "type": "boolean",
                                            "description": "Include AI-powered analysis in the comparison",
                                            "default": False
                                        }
                                    },
                                    "required": ["qb1", "qb2"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "QB comparison results",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "comparison": {
                                                "type": "object",
                                                "properties": {
                                                    "qb1": {"$ref": "#/components/schemas/QBData"},
                                                    "qb2": {"$ref": "#/components/schemas/QBData"},
                                                    "archetype_analysis": {"type": "object"},
                                                    "statistical_comparison": {"type": "object"},
                                                    "matchup_insights": {"type": "object"},
                                                    "key_differences": {"type": "array", "items": {"type": "string"}}
                                                }
                                            },
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error"
                        }
                    }
                }
            },
            "/ai/config": {
                "get": {
                    "summary": "Get AI Analyzer Configuration",
                    "description": "Get the current AI analyzer configuration and available options.",
                    "responses": {
                        "200": {
                            "description": "AI analyzer configuration",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "current_analyzer": {"type": "string"},
                                            "available_analyzers": {"type": "array", "items": {"type": "string"}},
                                            "lightweight": {"type": "object"},
                                            "qwen": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                },
                "post": {
                    "summary": "Set AI Analyzer Configuration",
                    "description": "Set the AI analyzer configuration (requires API restart).",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "analyzer_type": {
                                            "type": "string",
                                            "description": "Type of analyzer to set ('lightweight' or 'qwen')",
                                            "enum": ["lightweight", "qwen"]
                                        }
                                    },
                                    "required": ["analyzer_type"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "AI analyzer configuration updated",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "message": {"type": "string"},
                                            "current_analyzer": {"type": "string"},
                                            "requested_analyzer": {"type": "string"},
                                            "restart_required": {"type": "boolean"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid analyzer type"
                        }
                    }
                }
            },
            "/analyze/ai/qb": {
                "post": {
                    "summary": "AI-Powered QB Analysis",
                    "description": "Generate intelligent analysis for a single QB using AI",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb_name": {
                                            "type": "string",
                                            "description": "Name of QB to analyze",
                                            "example": "Dillon Gabriel"
                                        }
                                    },
                                    "required": ["qb_name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "AI analysis result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb_data": {"$ref": "#/components/schemas/QBData"},
                                            "ai_analysis": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            },
            "/compare/ai": {
                "post": {
                    "summary": "AI-Powered QB Comparison",
                    "description": "Generate intelligent comparison between two QBs using AI",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb1": {
                                            "type": "string",
                                            "description": "Name of first QB to compare",
                                            "example": "Dillon Gabriel"
                                        },
                                        "qb2": {
                                            "type": "string",
                                            "description": "Name of second QB to compare",
                                            "example": "Will Howard"
                                        }
                                    },
                                    "required": ["qb1", "qb2"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "AI comparison result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb1": {"$ref": "#/components/schemas/QBData"},
                                            "qb2": {"$ref": "#/components/schemas/QBData"},
                                            "ai_comparison": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            },
            "/analyze/ai/strategy": {
                "post": {
                    "summary": "AI-Powered Strategic Insights",
                    "description": "Generate strategic insights for a QB in specific scenarios",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb_name": {
                                            "type": "string",
                                            "description": "Name of QB to analyze",
                                            "example": "Dillon Gabriel"
                                        },
                                        "context": {
                                            "type": "string",
                                            "description": "Strategic context or scenario",
                                            "example": "upcoming_game_against_strong_pass_rush"
                                        }
                                    },
                                    "required": ["qb_name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Strategic insights result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb_data": {"$ref": "#/components/schemas/QBData"},
                                            "strategic_insights": {"type": "string"},
                                            "context": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            }
        },
        "components": {
            "schemas": {
                "Player": {
                    "type": "object",
                    "properties": {
                        "player_name": {"type": "string"},
                        "team_name": {"type": "string"},
                        "archetype": {"type": "string"},
                        "cluster": {"type": "integer"}
                    }
                },
                "QBData": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "team": {"type": "string"},
                        "archetype": {"type": "string"},
                        "cluster": {"type": "integer"},
                        "key_stats": {
                            "type": "object",
                            "properties": {
                                "accuracy_percent": {"type": "number"},
                                "completion_percent": {"type": "number"},
                                "ypa": {"type": "number"},
                                "td_int_ratio": {"type": "number"},
                                "scramble_rate": {"type": "number"},
                                "pressure_accuracy_percent": {"type": "number"},
                                "pressure_twp_rate": {"type": "number"},
                                "elusive_rating": {"type": "number"},
                                "grades_offense": {"type": "number"},
                                "grades_pass": {"type": "number"},
                                "grades_run": {"type": "number"}
                            }
                        }
                    }
                },
                "Archetype": {
                    "type": "string",
                    "enum": ["Pocket Managers", "Dynamic Dual-Threats", "Scrambling Survivors", "Mobile Pocket Passer"]
                }
            },
            "/analyze/ai/qb": {
                "post": {
                    "summary": "AI-Powered QB Analysis",
                    "description": "Generate intelligent analysis for a single QB using AI",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb_name": {
                                            "type": "string",
                                            "description": "Name of QB to analyze",
                                            "example": "Dillon Gabriel"
                                        }
                                    },
                                    "required": ["qb_name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "AI analysis result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb_data": {"$ref": "#/components/schemas/QBData"},
                                            "ai_analysis": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            },
            "/compare/ai": {
                "post": {
                    "summary": "AI-Powered QB Comparison",
                    "description": "Generate intelligent comparison between two QBs using AI",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb1": {
                                            "type": "string",
                                            "description": "Name of first QB to compare",
                                            "example": "Dillon Gabriel"
                                        },
                                        "qb2": {
                                            "type": "string",
                                            "description": "Name of second QB to compare",
                                            "example": "Will Howard"
                                        }
                                    },
                                    "required": ["qb1", "qb2"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "AI comparison result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb1": {"$ref": "#/components/schemas/QBData"},
                                            "qb2": {"$ref": "#/components/schemas/QBData"},
                                            "ai_comparison": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            },
            "/analyze/ai/strategy": {
                "post": {
                    "summary": "AI-Powered Strategic Insights",
                    "description": "Generate strategic insights for a QB in specific scenarios",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "qb_name": {
                                            "type": "string",
                                            "description": "Name of QB to analyze",
                                            "example": "Dillon Gabriel"
                                        },
                                        "context": {
                                            "type": "string",
                                            "description": "Strategic context or scenario",
                                            "example": "upcoming_game_against_strong_pass_rush"
                                        }
                                    },
                                    "required": ["qb_name"]
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Strategic insights result",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "qb_data": {"$ref": "#/components/schemas/QBData"},
                                            "strategic_insights": {"type": "string"},
                                            "context": {"type": "string"},
                                            "year": {"type": "integer"}
                                        }
                                    }
                                }
                            }
                        },
                        "400": {
                            "description": "Bad request - invalid data"
                        },
                        "404": {
                            "description": "QB not found in database"
                        },
                        "500": {
                            "description": "Internal server error or AI service unavailable"
                        }
                    }
                }
            }
        }
    }
    return jsonify(swagger_spec)

@app.route('/health', methods=['GET'])
def health_check():
    """Health check endpoint."""
    return jsonify({
        'status': 'healthy',
        'available_models': predictor.get_available_models(),
        'year': predictor.year,
        'player_data_loaded': predictor.player_data is not None
    })

@app.route('/features', methods=['GET'])
def get_features():
    """Get required features for prediction."""
    return jsonify({
        'required_features': predictor.feature_names,
        'n_features': len(predictor.feature_names) if predictor.feature_names else 0
    })

@app.route('/models', methods=['GET'])
def get_models():
    """Get available models information."""
    try:
        models_info = {}
        for model_type in predictor.get_available_models():
            model = predictor.models[model_type]['model']
            label_encoder = predictor.models[model_type]['label_encoder']
            
            models_info[model_type] = {
                'model_type': model_type,
                'model_class': type(model).__name__,
                'n_features': model.n_features_in_,
                'classes': label_encoder.classes_.tolist(),
                'year': predictor.year
            }
        
        return jsonify({
            'available_models': models_info,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"Error getting models info: {e}")
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict', methods=['POST'])
def predict_archetype():
    """Predict QB archetype from stats."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        # Get model type (optional, defaults to recall_optimized)
        model_type = data.get('model_type', 'recall_optimized')
        
        # Remove model_type from qb_data
        qb_data = {k: v for k, v in data.items() if k != 'model_type'}
        
        # Make prediction
        result = predictor.predict_archetype(qb_data, model_type)
        
        return jsonify(result)
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 400
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/predict/batch', methods=['POST'])
def predict_batch():
    """Predict QB archetypes for multiple QBs."""
    try:
        # Get request data
        data = request.get_json()
        
        if not data or 'qbs' not in data:
            return jsonify({'error': 'No QBs data provided'}), 400
        
        qbs_data = data['qbs']
        model_type = data.get('model_type', 'recall_optimized')
        
        if not isinstance(qbs_data, list):
            return jsonify({'error': 'QBs data must be a list'}), 400
        
        results = []
        for i, qb_data in enumerate(qbs_data):
            try:
                result = predictor.predict_archetype(qb_data, model_type)
                result['qb_index'] = i
                results.append(result)
            except Exception as e:
                results.append({
                    'qb_index': i,
                    'error': str(e),
                    'archetype': None,
                    'confidence': 0.0
                })
        
        return jsonify({
            'predictions': results,
            'model_type': model_type,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500



@app.route('/search', methods=['GET'])
def search_players():
    """Search for players by name."""
    try:
        query = request.args.get('q', '')
        limit = int(request.args.get('limit', 10))
        
        if not query:
            return jsonify({'error': 'Query parameter "q" is required'}), 400
        
        results = predictor.search_players(query, limit)
        
        return jsonify({
            'query': query,
            'results': results,
            'count': len(results),
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"Player search error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/archetypes/distribution', methods=['GET'])
def get_archetype_distribution():
    """Get archetype distribution for the current year."""
    try:
        distribution = predictor.get_archetype_distribution()
        return jsonify(distribution)
        
    except Exception as e:
        logger.error(f"Distribution error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/archetypes/top-players', methods=['GET'])
def get_top_players_by_archetype():
    """Get top players for each archetype."""
    try:
        limit = int(request.args.get('limit', 5))
        top_players = predictor.get_top_players_by_archetype(limit)
        
        return jsonify({
            'top_players': top_players,
            'limit': limit,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"Top players error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/compare', methods=['POST'])
def compare_qbs():
    """Compare two QBs and provide comprehensive insights."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        qb1 = data.get('qb1')
        qb2 = data.get('qb2')
        analysis_types = data.get('analysis_types', ['archetype', 'statistical', 'matchup'])
        include_ai = data.get('include_ai', False)
        
        if not qb1 or not qb2:
            return jsonify({'error': 'Both qb1 and qb2 are required'}), 400
        
        # Validate analysis types
        valid_types = ['archetype', 'statistical', 'matchup']
        if not all(atype in valid_types for atype in analysis_types):
            return jsonify({'error': f'Invalid analysis type. Must be one of: {valid_types}'}), 400
        
        # Perform comparison
        comparison = predictor.compare_qbs(qb1, qb2, analysis_types)
        
        # Add AI analysis if requested
        if include_ai:
            try:
                ai_analyzer = get_ai_analyzer()
                if ai_analyzer.test_connection():
                    qb1_data = predictor.get_qb_data(qb1)
                    qb2_data = predictor.get_qb_data(qb2)
                    if qb1_data and qb2_data:
                        ai_comparison = ai_analyzer.compare_qbs_ai(qb1_data, qb2_data)
                        if ai_comparison:
                            comparison['ai_analysis'] = ai_comparison
                        else:
                            comparison['ai_analysis'] = "AI analysis failed to generate"
                    else:
                        comparison['ai_analysis'] = "Could not retrieve QB data for AI analysis"
                else:
                    if AI_ANALYZER_TYPE == "qwen":
                        comparison['ai_analysis'] = "AI service unavailable. Please ensure Ollama is running with qwen3:8b model."
                    else:
                        comparison['ai_analysis'] = "AI service unavailable. Please ensure archetype profiles are loaded."
            except Exception as e:
                logger.error(f"Error adding AI analysis to comparison: {e}")
                comparison['ai_analysis'] = "AI analysis failed due to technical error"
        
        return jsonify({
            'comparison': comparison,
            'year': predictor.year
        })
        
    except ValueError as e:
        return jsonify({'error': str(e)}), 404
    except Exception as e:
        logger.error(f"QB comparison error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/ai/config', methods=['GET'])
def get_ai_config():
    """Get current AI analyzer configuration."""
    return jsonify({
        'current_analyzer': AI_ANALYZER_TYPE,
        'available_analyzers': ['lightweight', 'qwen'],
        'lightweight': {
            'description': 'Fast, rule-based analysis',
            'speed': 'Instant (milliseconds)',
            'quality': 'Consistent, statistical insights'
        },
        'qwen': {
            'description': 'Detailed, LLM-based analysis',
            'speed': 'Slow (60-120 seconds)',
            'quality': 'Nuanced, natural language insights'
        }
    })

@app.route('/ai/config', methods=['POST'])
def set_ai_config():
    """Set AI analyzer configuration (dynamic switching)."""
    global AI_ANALYZER_TYPE, current_ai_analyzer
    
    data = request.get_json()
    analyzer_type = data.get('analyzer_type', 'lightweight')
    
    if analyzer_type not in ['lightweight', 'qwen']:
        return jsonify({'error': 'Invalid analyzer type. Use "lightweight" or "qwen".'}), 400
    
    # Update the configuration
    AI_ANALYZER_TYPE = analyzer_type
    
    # Reset the current analyzer to force reload
    current_ai_analyzer = None
    
    # Get the new analyzer
    new_analyzer = get_ai_analyzer()
    
    return jsonify({
        'message': f'AI analyzer successfully switched to {analyzer_type}',
        'previous_analyzer': 'unknown',  # We don't track previous
        'current_analyzer': AI_ANALYZER_TYPE,
        'analyzer_loaded': new_analyzer is not None,
        'restart_required': False
    })

@app.route('/analyze/ai/qb', methods=['POST'])
def analyze_qb_ai():
    """Generate AI-powered analysis for a single QB."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        qb_name = data.get('qb_name')
        if not qb_name:
            return jsonify({'error': 'qb_name is required'}), 400
        
        # Get QB data
        qb_data = predictor.get_qb_data(qb_name)
        if qb_data is None:
            return jsonify({'error': f"QB '{qb_name}' not found in database"}), 404
        
        # Test AI connection
        ai_analyzer = get_ai_analyzer()
        if not ai_analyzer.test_connection():
            if AI_ANALYZER_TYPE == "qwen":
                return jsonify({'error': 'AI service unavailable. Please ensure Ollama is running with qwen3:8b model.'}), 500
            else:
                return jsonify({'error': 'AI service unavailable. Please ensure archetype profiles are loaded.'}), 500
        
        # Generate AI analysis
        ai_analysis = ai_analyzer.analyze_qb_performance(qb_data)
        if not ai_analysis or ai_analysis == "Error generating analysis":
            return jsonify({'error': 'Failed to generate AI analysis'}), 500
        
        return jsonify({
            'qb_data': qb_data,
            'ai_analysis': ai_analysis,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"AI QB analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/compare/ai', methods=['POST'])
def compare_qbs_ai():
    """Generate AI-powered comparison between two QBs."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        qb1_name = data.get('qb1')
        qb2_name = data.get('qb2')
        
        if not qb1_name or not qb2_name:
            return jsonify({'error': 'Both qb1 and qb2 are required'}), 400
        
        # Get QB data
        qb1_data = predictor.get_qb_data(qb1_name)
        qb2_data = predictor.get_qb_data(qb2_name)
        
        if qb1_data is None:
            return jsonify({'error': f"QB '{qb1_name}' not found in database"}), 404
        if qb2_data is None:
            return jsonify({'error': f"QB '{qb2_name}' not found in database"}), 404
        
        # Test AI connection
        ai_analyzer = get_ai_analyzer()
        if not ai_analyzer.test_connection():
            if AI_ANALYZER_TYPE == "qwen":
                return jsonify({'error': 'AI service unavailable. Please ensure Ollama is running with qwen3:8b model.'}), 500
            else:
                return jsonify({'error': 'AI service unavailable. Please ensure archetype profiles are loaded.'}), 500
        
        # Generate AI comparison
        ai_comparison = ai_analyzer.compare_qbs_ai(qb1_data, qb2_data)
        if not ai_comparison or ai_comparison == "Error generating comparison":
            return jsonify({'error': 'Failed to generate AI comparison'}), 500
        
        return jsonify({
            'qb1': qb1_data,
            'qb2': qb2_data,
            'ai_comparison': ai_comparison,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"AI QB comparison error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

@app.route('/analyze/ai/strategy', methods=['POST'])
def analyze_strategy_ai():
    """Generate AI-powered strategic insights for a QB."""
    try:
        data = request.get_json()
        
        if not data:
            return jsonify({'error': 'No data provided'}), 400
        
        qb_name = data.get('qb_name')
        context = data.get('context', '')
        
        if not qb_name:
            return jsonify({'error': 'qb_name is required'}), 400
        
        # Get QB data
        qb_data = predictor.get_qb_data(qb_name)
        if qb_data is None:
            return jsonify({'error': f"QB '{qb_name}' not found in database"}), 404
        
        # Test AI connection
        ai_analyzer = get_ai_analyzer()
        if not ai_analyzer.test_connection():
            if AI_ANALYZER_TYPE == "qwen":
                return jsonify({'error': 'AI service unavailable. Please ensure Ollama is running with qwen3:8b model.'}), 500
            else:
                return jsonify({'error': 'AI service unavailable. Please ensure archetype profiles are loaded.'}), 500
        
        # Generate strategic insights
        strategic_insights = ai_analyzer.generate_strategic_insights(qb_data, context)
        if not strategic_insights or strategic_insights == "Error generating strategic insights":
            return jsonify({'error': 'Failed to generate strategic insights'}), 500
        
        return jsonify({
            'qb_data': qb_data,
            'strategic_insights': strategic_insights,
            'context': context,
            'year': predictor.year
        })
        
    except Exception as e:
        logger.error(f"AI strategy analysis error: {e}")
        logger.error(traceback.format_exc())
        return jsonify({'error': 'Internal server error'}), 500

if __name__ == '__main__':
    logger.info("Starting Enhanced QB Archetype Prediction API...")
    logger.info(f"Available models: {predictor.get_available_models()}")
    logger.info(f"Player data loaded: {predictor.player_data is not None}")
    app.run(debug=True, host='0.0.0.0', port=5001)
