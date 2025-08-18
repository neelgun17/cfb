"""
AI-Powered QB Analysis Service
Integrates with Ollama and Qwen2.5:8B for intelligent quarterback analysis
"""

import requests
import json
import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path
from config import QWEN_TIMEOUT, QWEN_MODEL, QWEN_URL

logger = logging.getLogger(__name__)

class QBArchetypeAIAnalyzer:
    """AI-powered QB analysis using Qwen3:8B via Ollama."""
    
    def __init__(self, ollama_url: str = QWEN_URL, model: str = QWEN_MODEL):
        self.ollama_url = ollama_url
        self.model = model
        self.archetype_profiles = None
        self.load_archetype_profiles()
    
    def load_archetype_profiles(self):
        """Load archetype profiles for context in AI analysis."""
        try:
            profiles_path = Path("data/analysis/2024/hierarchical_profiles_k4.csv")
            if profiles_path.exists():
                self.archetype_profiles = pd.read_csv(profiles_path)
                logger.info(f"Loaded archetype profiles for {len(self.archetype_profiles)} clusters")
            else:
                logger.warning("Archetype profiles not found, using default profiles")
                self.archetype_profiles = None
        except Exception as e:
            logger.error(f"Error loading archetype profiles: {e}")
            self.archetype_profiles = None
    
    def _get_archetype_context(self) -> str:
        """Generate archetype context for AI prompts."""
        if self.archetype_profiles is not None:
            context = "QUARTERBACK ARCHETYPE DEFINITIONS:\n"
            context += "Based on statistical analysis of college football QBs, there are 4 distinct archetypes:\n\n"
            
            archetype_names = [
                "SCRAMBLING SURVIVORS (Cluster 0)",
                "POCKET MANAGERS (Cluster 1)", 
                "DYNAMIC DUAL-THREATS (Cluster 2)",
                "MOBILE POCKET PASSER (Cluster 3)"
            ]
            
            descriptions = [
                "High pressure situations, limited mobility, risky passing style",
                "Low mobility, efficient passing, low risk, high accuracy",
                "High designed run rate, explosive rushing, aggressive passing",
                "Balanced mobility, high passing efficiency, low turnover risk"
            ]
            
            for i, (name, desc) in enumerate(zip(archetype_names, descriptions)):
                if i < len(self.archetype_profiles):
                    row = self.archetype_profiles.iloc[i]
                    context += f"{i+1}. {name}:\n"
                    context += f"- Average Accuracy: {row['accuracy_percent']:.1f}% | "
                    context += f"Scramble Rate: {row['scramble_rate']*100:.1f}% | "
                    context += f"Pressure Accuracy: {row['pressure_accuracy_percent']:.1f}%\n"
                    context += f"- Characteristics: {desc}\n"
                    context += f"- PFF Grades: Offense {row['grades_offense']:.1f} | "
                    context += f"Pass {row['grades_pass']:.1f} | Run {row['grades_run']:.1f}\n"
                    context += f"- Style: {desc}\n\n"
        else:
            # Fallback to default profiles
            context = """QUARTERBACK ARCHETYPE DEFINITIONS:
Based on statistical analysis of college football QBs, there are 4 distinct archetypes:

1. SCRAMBLING SURVIVORS (Cluster 0):
- Average Accuracy: 67.0% | Scramble Rate: 4.8% | Pressure Accuracy: 53.2%
- Characteristics: High pressure situations, limited mobility, risky passing style
- PFF Grades: Offense 63.3 | Pass 61.8 | Run 58.9
- Style: Relies heavily on mobility to compensate for passing limitations

2. POCKET MANAGERS (Cluster 1):  
- Average Accuracy: 75.0% | Scramble Rate: 6.6% | Pressure Accuracy: 62.4%
- Characteristics: Low mobility, efficient passing, low risk, high accuracy
- PFF Grades: Offense 80.9 | Pass 78.5 | Run 66.3
- Style: Traditional pocket passer with high accuracy and decision-making

3. DYNAMIC DUAL-THREATS (Cluster 2):
- Average Accuracy: 72.0% | Scramble Rate: 8.7% | Pressure Accuracy: 60.6%
- Characteristics: High designed run rate, explosive rushing, aggressive passing
- PFF Grades: Offense 70.0 | Pass 67.0 | Run 64.7
- Style: Equally effective as passer and runner with explosive playmaking

4. MOBILE POCKET PASSER (Cluster 3):
- Average Accuracy: 68.5% | Scramble Rate: 7.2% | Pressure Accuracy: 54.3%
- Characteristics: Balanced mobility, high passing efficiency, low turnover risk
- PFF Grades: Offense 75.7 | Pass 70.4 | Run 71.9
- Style: Can extend plays but primarily passes from the pocket

"""
        return context
    
    def _call_ollama(self, prompt: str) -> Optional[str]:
        """Make API call to Ollama."""
        try:
            payload = {
                "model": self.model,
                "prompt": prompt,
                "stream": False
            }
            
            response = requests.post(
                f"{self.ollama_url}/api/generate",
                json=payload,
                timeout=QWEN_TIMEOUT  # Use configured timeout
            )
            
            if response.status_code == 200:
                result = response.json()
                return result.get('response', '')
            else:
                logger.error(f"Ollama API error: {response.status_code} - {response.text}")
                return None
                
        except requests.exceptions.RequestException as e:
            logger.error(f"Error calling Ollama API: {e}")
            return None
        except Exception as e:
            logger.error(f"Unexpected error in Ollama call: {e}")
            return None
    
    def analyze_qb_performance(self, qb_data: Dict[str, Any]) -> Optional[str]:
        """Generate AI analysis for individual QB performance."""
        try:
            archetype_context = self._get_archetype_context()
            
            # Format QB stats for comparison with archetype
            qb_stats = qb_data.get('key_stats', {})
            archetype = qb_data.get('archetype', 'Unknown')
            
            # Find archetype average for comparison
            archetype_avg = {}
            if self.archetype_profiles is not None:
                archetype_map = {
                    'Scrambling Survivors': 0,
                    'Pocket Managers': 1,
                    'Dynamic Dual-Threats': 2,
                    'Mobile Pocket Passer': 3
                }
                if archetype in archetype_map:
                    cluster_idx = archetype_map[archetype]
                    if cluster_idx < len(self.archetype_profiles):
                        row = self.archetype_profiles.iloc[cluster_idx]
                        archetype_avg = {
                            'accuracy_percent': row['accuracy_percent'],
                            'scramble_rate': row['scramble_rate'] * 100,
                            'pressure_accuracy_percent': row['pressure_accuracy_percent'],
                            'grades_offense': row['grades_offense'],
                            'grades_pass': row['grades_pass'],
                            'grades_run': row['grades_run']
                        }
            
            prompt = f"""Analyze this QB's performance:

QB: {qb_data.get('name', 'Unknown')} ({qb_data.get('team', 'Unknown')})
Archetype: {archetype}

Key Stats:
Accuracy: {qb_stats.get('accuracy_percent', 0):.1f}% (avg: {archetype_avg.get('accuracy_percent', 0):.1f}%)
Pressure Accuracy: {qb_stats.get('pressure_accuracy_percent', 0):.1f}% (avg: {archetype_avg.get('pressure_accuracy_percent', 0):.1f}%)
PFF Pass Grade: {qb_stats.get('grades_pass', 0):.1f} (avg: {archetype_avg.get('grades_pass', 0):.1f})
Scramble Rate: {qb_stats.get('scramble_rate', 0)*100:.1f}% (avg: {archetype_avg.get('scramble_rate', 0):.1f}%)
TD/INT Ratio: {qb_stats.get('td_int_ratio', 0):.1f}
Sack %: {qb_stats.get('sack_percent', 0):.1f}%

Provide brief analysis:
1. Archetype Fit: How well do they match their archetype?
2. Strengths: What do they do well?
3. Weaknesses: What are their limitations?
4. Strategy: How should defenses approach them?

Keep it concise and actionable."""
            
            return self._call_ollama(prompt)
            
        except Exception as e:
            logger.error(f"Error generating QB performance analysis: {e}")
            return None
    
    def compare_qbs_ai(self, qb1_data: Dict[str, Any], qb2_data: Dict[str, Any]) -> Optional[str]:
        """Generate AI analysis for QB comparison."""
        try:
            archetype_context = self._get_archetype_context()
            
            # Format QB stats for comparison
            qb1_stats = qb1_data.get('key_stats', {})
            qb2_stats = qb2_data.get('key_stats', {})
            
            # Get archetype averages for comparison
            archetype_avgs = {}
            if self.archetype_profiles is not None:
                archetype_map = {
                    'Scrambling Survivors': 0,
                    'Pocket Managers': 1,
                    'Dynamic Dual-Threats': 2,
                    'Mobile Pocket Passer': 3
                }
                
                for qb_data in [qb1_data, qb2_data]:
                    archetype = qb_data.get('archetype', 'Unknown')
                    if archetype in archetype_map:
                        cluster_idx = archetype_map[archetype]
                        if cluster_idx < len(self.archetype_profiles):
                            row = self.archetype_profiles.iloc[cluster_idx]
                            archetype_avgs[archetype] = {
                                'accuracy_percent': row['accuracy_percent'],
                                'scramble_rate': row['scramble_rate'] * 100,
                                'pressure_accuracy_percent': row['pressure_accuracy_percent'],
                                'grades_offense': row['grades_offense'],
                                'grades_pass': row['grades_pass'],
                                'grades_run': row['grades_run']
                            }
            
            prompt = f"""Compare these QBs:

{qb1_data.get('name', 'QB1')} ({qb1_data.get('archetype', 'Unknown')}):
Accuracy: {qb1_stats.get('accuracy_percent', 0):.1f}% | Pressure: {qb1_stats.get('pressure_accuracy_percent', 0):.1f}% | PFF Pass: {qb1_stats.get('grades_pass', 0):.1f} | Scramble: {qb1_stats.get('scramble_rate', 0)*100:.1f}% | TD/INT: {qb1_stats.get('td_int_ratio', 0):.1f}

{qb2_data.get('name', 'QB2')} ({qb2_data.get('archetype', 'Unknown')}):
Accuracy: {qb2_stats.get('accuracy_percent', 0):.1f}% | Pressure: {qb2_stats.get('pressure_accuracy_percent', 0):.1f}% | PFF Pass: {qb2_stats.get('grades_pass', 0):.1f} | Scramble: {qb2_stats.get('scramble_rate', 0)*100:.1f}% | TD/INT: {qb2_stats.get('td_int_ratio', 0):.1f}

Brief comparison:
1. Key Differences: What are the biggest statistical gaps?
2. Strategic Advantages: What does each QB do better?
3. Defensive Approach: How should defenses handle each QB?
4. Game Prediction: What type of game should we expect?

Keep it concise and actionable."""
            
            return self._call_ollama(prompt)
            
        except Exception as e:
            logger.error(f"Error generating QB comparison analysis: {e}")
            return None
    
    def generate_strategic_insights(self, qb_data: Dict[str, Any], context: str = "") -> Optional[str]:
        """Generate strategic insights for specific game scenarios."""
        try:
            archetype_context = self._get_archetype_context()
            
            qb_stats = qb_data.get('key_stats', {})
            archetype = qb_data.get('archetype', 'Unknown')
            
            prompt = f"""Strategic analysis for {qb_data.get('name', 'Unknown')} ({qb_data.get('archetype', 'Unknown')}):

Context: {context if context else "General analysis"}

Key stats: Accuracy {qb_stats.get('accuracy_percent', 0):.1f}% | Pressure {qb_stats.get('pressure_accuracy_percent', 0):.1f}% | Scramble {qb_stats.get('scramble_rate', 0)*100:.1f}% | Sack % {qb_stats.get('sack_percent', 0):.1f}%

Provide brief strategic insights:
1. Offensive Strategy: How should their team use them?
2. Defensive Approach: How should opponents defend them?
3. Key Situations: What adjustments are needed?
4. Risk Factors: What are the biggest concerns?

Keep it concise and actionable."""
            
            return self._call_ollama(prompt)
            
        except Exception as e:
            logger.error(f"Error generating strategic insights: {e}")
            return None
    
    def test_connection(self) -> bool:
        """Test connection to Ollama API."""
        try:
            response = requests.get(f"{self.ollama_url}/api/tags", timeout=10)
            return response.status_code == 200
        except Exception as e:
            logger.error(f"Error testing Ollama connection: {e}")
            return False
