"""
Lightweight AI-Powered QB Analysis Service
Provides intelligent quarterback analysis using rule-based systems and statistical insights
"""

import logging
from typing import Dict, List, Optional, Any
import pandas as pd
from pathlib import Path

logger = logging.getLogger(__name__)

class LightweightQBAnalyzer:
    """Lightweight AI-powered QB analysis using rule-based systems."""
    
    def __init__(self):
        self.archetype_profiles = None
        self.load_archetype_profiles()
        
        # Define archetype characteristics
        self.archetype_definitions = {
            'Scrambling Survivors': {
                'description': 'High pressure situations, limited mobility, risky passing style',
                'strengths': ['mobility', 'play_extension'],
                'weaknesses': ['accuracy', 'decision_making', 'pressure_handling']
            },
            'Pocket Managers': {
                'description': 'Low mobility, efficient passing, low risk, high accuracy',
                'strengths': ['accuracy', 'decision_making', 'efficiency'],
                'weaknesses': ['mobility', 'play_extension']
            },
            'Dynamic Dual-Threats': {
                'description': 'High designed run rate, explosive rushing, aggressive passing',
                'strengths': ['rushing', 'explosive_plays', 'versatility'],
                'weaknesses': ['consistency', 'pocket_passing']
            },
            'Mobile Pocket Passer': {
                'description': 'Balanced mobility, high passing efficiency, low turnover risk',
                'strengths': ['balance', 'efficiency', 'mobility'],
                'weaknesses': ['explosiveness', 'deep_passing']
            }
        }
    
    def load_archetype_profiles(self):
        """Load archetype profiles for context in analysis."""
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
    
    def _get_archetype_averages(self, archetype: str) -> Dict[str, float]:
        """Get average stats for an archetype."""
        if self.archetype_profiles is None:
            return {}
        
        archetype_map = {
            'Scrambling Survivors': 0,
            'Pocket Managers': 1,
            'Dynamic Dual-Threats': 2,
            'Mobile Pocket Passer': 3
        }
        
        if archetype in archetype_map:
            cluster_idx = archetype_map[archetype]
            if cluster_idx < len(self.archetype_profiles):
                try:
                    row = self.archetype_profiles.iloc[cluster_idx]
                    return {
                        'accuracy_percent': float(row['accuracy_percent']),
                        'scramble_rate': float(row['scramble_rate']) * 100,
                        'pressure_accuracy_percent': float(row['pressure_accuracy_percent']),
                        'grades_offense': float(row['grades_offense']),
                        'grades_pass': float(row['grades_pass']),
                        'grades_run': float(row['grades_run'])
                    }
                except (KeyError, ValueError, IndexError) as e:
                    logger.error(f"Error getting archetype averages for {archetype}: {e}")
                    return {}
        return {}
    
    def _analyze_statistical_deviations(self, qb_stats: Dict, archetype_avg: Dict) -> Dict[str, Any]:
        """Analyze how QB stats deviate from archetype averages."""
        deviations = {}
        
        for stat, qb_value in qb_stats.items():
            if stat in archetype_avg:
                try:
                    qb_value = float(qb_value)
                    avg_value = float(archetype_avg[stat])
                    if avg_value != 0:
                        deviation = ((qb_value - avg_value) / avg_value) * 100
                        deviations[stat] = {
                            'qb_value': qb_value,
                            'avg_value': avg_value,
                            'deviation_percent': deviation,
                            'significance': 'high' if abs(deviation) > 15 else 'medium' if abs(deviation) > 8 else 'low'
                        }
                except (ValueError, TypeError) as e:
                    logger.error(f"Error analyzing deviation for {stat}: {e}")
                    continue
        
        return deviations
    
    def _generate_archetype_fit_analysis(self, qb_stats: Dict, archetype: str, deviations: Dict) -> str:
        """Generate archetype fit analysis."""
        archetype_def = self.archetype_definitions.get(archetype, {})
        
        # Calculate fit score
        fit_score = 0
        total_checks = 0
        
        if 'accuracy_percent' in deviations:
            acc_dev = deviations['accuracy_percent']['deviation_percent']
            if abs(acc_dev) < 10:
                fit_score += 1
            total_checks += 1
        
        if 'pressure_accuracy_percent' in deviations:
            press_dev = deviations['pressure_accuracy_percent']['deviation_percent']
            if abs(press_dev) < 15:
                fit_score += 1
            total_checks += 1
        
        if 'scramble_rate' in deviations:
            scramble_dev = deviations['scramble_rate']['deviation_percent']
            if abs(scramble_dev) < 20:
                fit_score += 1
            total_checks += 1
        
        fit_percentage = (fit_score / total_checks * 100) if total_checks > 0 else 0
        
        if fit_percentage >= 80:
            fit_level = "Excellent"
        elif fit_percentage >= 60:
            fit_level = "Good"
        elif fit_percentage >= 40:
            fit_level = "Fair"
        else:
            fit_level = "Poor"
        
        analysis = f"**Archetype Fit Analysis**\n"
        analysis += f"{qb_stats.get('name', 'QB')} shows **{fit_level}** alignment with the {archetype} archetype ({fit_percentage:.0f}% fit).\n\n"
        
        # Key deviations
        significant_deviations = [(stat, dev) for stat, dev in deviations.items() if dev['significance'] in ['high', 'medium']]
        if significant_deviations:
            analysis += "**Key Deviations:**\n"
            for stat, dev in significant_deviations[:3]:  # Top 3 deviations
                direction = "above" if dev['deviation_percent'] > 0 else "below"
                analysis += f"• {stat.replace('_', ' ').title()}: {dev['qb_value']:.1f} ({direction} archetype avg by {abs(dev['deviation_percent']):.1f}%)\n"
        
        return analysis
    
    def _generate_strengths_analysis(self, qb_stats: Dict, archetype: str, deviations: Dict) -> str:
        """Generate strengths analysis."""
        analysis = "**Performance Strengths**\n"
        
        strengths = []
        
        # Check accuracy
        if 'accuracy_percent' in deviations:
            acc_dev = deviations['accuracy_percent']['deviation_percent']
            if acc_dev > 5:
                strengths.append(f"**Superior Accuracy**: {qb_stats['accuracy_percent']:.1f}% (above archetype average)")
        
        # Check pressure handling
        if 'pressure_accuracy_percent' in deviations:
            press_dev = deviations['pressure_accuracy_percent']['deviation_percent']
            if press_dev > 8:
                strengths.append(f"**Strong Pressure Handling**: {qb_stats['pressure_accuracy_percent']:.1f}% accuracy under pressure")
        
        # Check PFF grades
        if 'grades_pass' in deviations:
            pass_dev = deviations['grades_pass']['deviation_percent']
            if pass_dev > 5:
                strengths.append(f"**Elite Passing Grade**: PFF Pass Grade of {qb_stats['grades_pass']:.1f}")
        
        # Check TD/INT ratio
        if qb_stats.get('td_int_ratio', 0) > 2.5:
            strengths.append(f"**Efficient Decision Making**: TD/INT ratio of {qb_stats['td_int_ratio']:.1f}")
        
        # Check scramble rate for mobile QBs
        if archetype in ['Dynamic Dual-Threats', 'Mobile Pocket Passer']:
            if 'scramble_rate' in deviations:
                scramble_dev = deviations['scramble_rate']['deviation_percent']
                if scramble_dev > 10:
                    strengths.append(f"**Effective Mobility**: {qb_stats['scramble_rate']*100:.1f}% scramble rate")
        
        if strengths:
            for strength in strengths[:4]:  # Top 4 strengths
                analysis += f"• {strength}\n"
        else:
            analysis += "• Shows solid baseline performance for their archetype\n"
        
        return analysis
    
    def _generate_weaknesses_analysis(self, qb_stats: Dict, archetype: str, deviations: Dict) -> str:
        """Generate weaknesses analysis."""
        analysis = "**Performance Weaknesses**\n"
        
        weaknesses = []
        
        # Check accuracy issues
        if 'accuracy_percent' in deviations:
            acc_dev = deviations['accuracy_percent']['deviation_percent']
            if acc_dev < -8:
                weaknesses.append(f"**Accuracy Concerns**: {qb_stats['accuracy_percent']:.1f}% (below archetype average)")
        
        # Check pressure handling issues
        if 'pressure_accuracy_percent' in deviations:
            press_dev = deviations['pressure_accuracy_percent']['deviation_percent']
            if press_dev < -10:
                weaknesses.append(f"**Pressure Struggles**: {qb_stats['pressure_accuracy_percent']:.1f}% accuracy under pressure")
        
        # Check sack percentage
        if qb_stats.get('sack_percent', 0) > 7.0:
            weaknesses.append(f"**Sack Vulnerability**: {qb_stats['sack_percent']:.1f}% sack rate")
        
        # Check turnover issues
        if qb_stats.get('twp_rate', 0) > 4.0:
            weaknesses.append(f"**Turnover Risk**: {qb_stats['twp_rate']:.1f}% turnover-worthy play rate")
        
        # Check mobility for pocket QBs
        if archetype in ['Pocket Managers']:
            if 'scramble_rate' in deviations:
                scramble_dev = deviations['scramble_rate']['deviation_percent']
                if scramble_dev > 20:
                    weaknesses.append(f"**Over-reliance on Mobility**: {qb_stats['scramble_rate']*100:.1f}% scramble rate (unusual for pocket passer)")
        
        if weaknesses:
            for weakness in weaknesses[:4]:  # Top 4 weaknesses
                analysis += f"• {weakness}\n"
        else:
            analysis += "• No significant weaknesses identified\n"
        
        return analysis
    
    def _generate_strategic_insights(self, qb_stats: Dict, archetype: str, deviations: Dict) -> str:
        """Generate strategic insights."""
        analysis = "**Strategic Insights**\n"
        
        insights = []
        
        # Defensive strategies
        if qb_stats.get('pressure_accuracy_percent', 0) < 55:
            insights.append("**Defensive Strategy**: Apply consistent pressure - QB struggles under duress")
        elif qb_stats.get('accuracy_percent', 0) > 75:
            insights.append("**Defensive Strategy**: Focus on coverage over pressure - QB is highly accurate")
        
        # Offensive strategies
        if qb_stats.get('scramble_rate', 0) > 0.1:
            insights.append("**Offensive Strategy**: Incorporate designed runs and rollouts to maximize mobility")
        elif qb_stats.get('accuracy_percent', 0) > 70:
            insights.append("**Offensive Strategy**: Emphasize quick, accurate passing game")
        
        # Situational considerations
        if qb_stats.get('sack_percent', 0) > 6:
            insights.append("**Situational**: Protect QB with extra blockers in obvious passing situations")
        
        if qb_stats.get('td_int_ratio', 0) < 2.0:
            insights.append("**Situational**: Conservative play-calling in critical situations")
        
        if insights:
            for insight in insights[:3]:  # Top 3 insights
                analysis += f"• {insight}\n"
        else:
            analysis += "• Standard defensive and offensive approaches appropriate\n"
        
        return analysis
    
    def analyze_qb_performance(self, qb_data: Dict[str, Any]) -> str:
        """Generate intelligent analysis for individual QB performance."""
        try:
            qb_stats = qb_data.get('key_stats', {})
            archetype = qb_data.get('archetype', 'Unknown')
            
            logger.info(f"Analyzing QB: {qb_data.get('name', 'Unknown')}, Archetype: {archetype}")
            logger.info(f"QB stats keys: {list(qb_stats.keys())}")
            
            # Get archetype averages
            archetype_avg = self._get_archetype_averages(archetype)
            logger.info(f"Archetype averages: {list(archetype_avg.keys())}")
            
            # Analyze deviations
            deviations = self._analyze_statistical_deviations(qb_stats, archetype_avg)
            logger.info(f"Deviations calculated: {len(deviations)}")
            
            # Generate analysis sections
            analysis = f"# AI Analysis: {qb_data.get('name', 'Unknown')} ({qb_data.get('team', 'Unknown')})\n\n"
            analysis += f"**Archetype**: {archetype}\n\n"
            
            analysis += self._generate_archetype_fit_analysis(qb_stats, archetype, deviations)
            analysis += "\n"
            
            analysis += self._generate_strengths_analysis(qb_stats, archetype, deviations)
            analysis += "\n"
            
            analysis += self._generate_weaknesses_analysis(qb_stats, archetype, deviations)
            analysis += "\n"
            
            analysis += self._generate_strategic_insights(qb_stats, archetype, deviations)
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating QB performance analysis: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
            return "Error generating analysis"
    
    def compare_qbs_ai(self, qb1_data: Dict[str, Any], qb2_data: Dict[str, Any]) -> str:
        """Generate intelligent comparison between two QBs."""
        try:
            qb1_stats = qb1_data.get('key_stats', {})
            qb2_stats = qb2_data.get('key_stats', {})
            
            analysis = f"# AI Comparison: {qb1_data.get('name', 'QB1')} vs {qb2_data.get('name', 'QB2')}\n\n"
            
            # Key statistical differences
            analysis += "**Key Statistical Differences**\n"
            
            differences = []
            
            # Accuracy comparison
            acc_diff = qb1_stats.get('accuracy_percent', 0) - qb2_stats.get('accuracy_percent', 0)
            if abs(acc_diff) > 3:
                winner = qb1_data.get('name', 'QB1') if acc_diff > 0 else qb2_data.get('name', 'QB2')
                differences.append(f"**Accuracy**: {winner} (+{abs(acc_diff):.1f}%)")
            
            # Pressure accuracy comparison
            press_diff = qb1_stats.get('pressure_accuracy_percent', 0) - qb2_stats.get('pressure_accuracy_percent', 0)
            if abs(press_diff) > 5:
                winner = qb1_data.get('name', 'QB1') if press_diff > 0 else qb2_data.get('name', 'QB2')
                differences.append(f"**Pressure Handling**: {winner} (+{abs(press_diff):.1f}%)")
            
            # PFF Pass Grade comparison
            pass_diff = qb1_stats.get('grades_pass', 0) - qb2_stats.get('grades_pass', 0)
            if abs(pass_diff) > 5:
                winner = qb1_data.get('name', 'QB1') if pass_diff > 0 else qb2_data.get('name', 'QB2')
                differences.append(f"**Passing Grade**: {winner} (+{abs(pass_diff):.1f})")
            
            # TD/INT ratio comparison
            tdint_diff = qb1_stats.get('td_int_ratio', 0) - qb2_stats.get('td_int_ratio', 0)
            if abs(tdint_diff) > 0.5:
                winner = qb1_data.get('name', 'QB1') if tdint_diff > 0 else qb2_data.get('name', 'QB2')
                differences.append(f"**Efficiency**: {winner} (+{abs(tdint_diff):.1f} TD/INT ratio)")
            
            for diff in differences[:4]:  # Top 4 differences
                analysis += f"• {diff}\n"
            
            analysis += "\n**Strategic Implications**\n"
            
            # Defensive approach
            if press_diff > 5:
                analysis += f"• Defenses should pressure {qb2_data.get('name', 'QB2')} more aggressively\n"
            elif press_diff < -5:
                analysis += f"• Defenses should pressure {qb1_data.get('name', 'QB1')} more aggressively\n"
            
            # Game prediction
            if abs(acc_diff) > 5:
                analysis += f"• Expect {qb1_data.get('name', 'QB1') if acc_diff > 0 else qb2_data.get('name', 'QB2')} to control the game through accuracy\n"
            
            if abs(pass_diff) > 8:
                analysis += f"• {qb1_data.get('name', 'QB1') if pass_diff > 0 else qb2_data.get('name', 'QB2')} likely to make more explosive plays\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating QB comparison analysis: {e}")
            return "Error generating comparison"
    
    def generate_strategic_insights(self, qb_data: Dict[str, Any], context: str = "") -> str:
        """Generate strategic insights for specific game scenarios."""
        try:
            qb_stats = qb_data.get('key_stats', {})
            archetype = qb_data.get('archetype', 'Unknown')
            
            analysis = f"# Strategic Analysis: {qb_data.get('name', 'Unknown')}\n\n"
            analysis += f"**Context**: {context if context else 'General strategic analysis'}\n\n"
            
            analysis += "**Offensive Strategy**\n"
            
            # Offensive recommendations
            if qb_stats.get('accuracy_percent', 0) > 70:
                analysis += "• Emphasize quick, accurate passing game\n"
                analysis += "• Use timing routes and precision throws\n"
            else:
                analysis += "• Focus on high-percentage throws and screen game\n"
                analysis += "• Minimize complex route combinations\n"
            
            if qb_stats.get('scramble_rate', 0) > 0.08:
                analysis += "• Incorporate designed runs and rollouts\n"
                analysis += "• Use QB as a running threat\n"
            
            analysis += "\n**Defensive Strategy**\n"
            
            # Defensive recommendations
            if qb_stats.get('pressure_accuracy_percent', 0) < 55:
                analysis += "• Apply consistent pressure and blitz frequently\n"
                analysis += "• Force QB to make quick decisions\n"
            else:
                analysis += "• Focus on coverage over pressure\n"
                analysis += "• Use disguised coverages to confuse QB\n"
            
            if qb_stats.get('sack_percent', 0) > 6:
                analysis += "• Target protection schemes with stunts and blitzes\n"
            
            analysis += "\n**Key Situations**\n"
            
            # Situational considerations
            if qb_stats.get('td_int_ratio', 0) < 2.0:
                analysis += "• Conservative play-calling in critical situations\n"
                analysis += "• Avoid risky throws in red zone\n"
            
            if qb_stats.get('accuracy_percent', 0) > 75:
                analysis += "• QB can be trusted in clutch situations\n"
                analysis += "• Aggressive play-calling when trailing\n"
            
            return analysis
            
        except Exception as e:
            logger.error(f"Error generating strategic insights: {e}")
            return "Error generating strategic insights"
    
    def test_connection(self) -> bool:
        """Test if the analyzer is ready."""
        return self.archetype_profiles is not None
