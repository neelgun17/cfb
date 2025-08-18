"""
Pipeline Summary Module
Handles pipeline result summaries and reporting
"""

import logging
from typing import Dict

logger = logging.getLogger(__name__)

class PipelineSummary:
    """Handles pipeline result summaries and reporting."""
    
    @staticmethod
    def print_pipeline_summary(pipeline_results: Dict):
        """Print a summary of the pipeline results."""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        for step, result in pipeline_results.items():
            logger.info(f"\n{step.upper()}:")
            if result['status'] == 'success':
                logger.info(f"  Status: ✓ Success")
                if 'elapsed_time' in result:
                    logger.info(f"  Time: {result['elapsed_time']:.2f} seconds")
                
                # Print additional details for clustering
                if step == 'clustering' and 'summary' in result:
                    summary = result['summary']
                    logger.info(f"  Total players: {summary.get('total_players', 'N/A')}")
                    logger.info(f"  Clusters: {summary.get('n_clusters', 'N/A')}")
                    logger.info(f"  Features: {summary.get('features_used', 'N/A')}")
                
                # Print additional details for model training
                if step == 'model_training' and 'models' in result:
                    for model_type, model_result in result['models'].items():
                        if model_result['status'] == 'success':
                            logger.info(f"  {model_type}: ✓ Success ({model_result['elapsed_time']:.2f}s)")
                        else:
                            logger.info(f"  {model_type}: ✗ Failed")
                
                # Print additional details for final merged CSV
                if step == 'final_merged_csv' and 'summary' in result:
                    summary = result['summary']
                    logger.info(f"  Total players: {summary.get('total_players', 'N/A')}")
                    logger.info(f"  Total features: {summary.get('total_features', 'N/A')}")
                    logger.info(f"  Archetype distribution: {summary.get('archetype_distribution', 'N/A')}")
            
            elif result['status'] == 'failed':
                logger.info(f"  Status: ✗ Failed")
                if 'elapsed_time' in result:
                    logger.info(f"  Time: {result['elapsed_time']:.2f} seconds")
            
            elif result['status'] == 'error':
                logger.info(f"  Status: ✗ Error")
                if 'error' in result:
                    logger.info(f"  Error: {result['error']}")
    
    @staticmethod
    def print_completion_message(year: int, total_time: float):
        """Print pipeline completion message."""
        logger.info("="*80)
        logger.info("PIPELINE COMPLETED SUCCESSFULLY")
        logger.info(f"Year: {year}")
        logger.info(f"Total elapsed time: {total_time:.2f} seconds")
        logger.info("="*80)
