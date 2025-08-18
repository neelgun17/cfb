"""
Main Pipeline for QB Archetype Analysis
Orchestrates the complete workflow from data processing to model training
"""

import logging
from pathlib import Path
from typing import Dict, List
import time

from config import create_directories, validate_paths, get_data_paths, CURRENT_YEAR
from pipeline_steps import DataProcessingStep, ClusteringStep, ModelTrainingStep, FinalMergedCSVStep
from pipeline_summary import PipelineSummary

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class QBPipeline:
    """Main pipeline class that orchestrates the entire QB archetype analysis workflow."""
    
    def __init__(self, year=None):
        self.year = year if year is not None else CURRENT_YEAR
        self.pipeline_results = {}
        
    def setup_environment(self) -> bool:
        """Set up the environment and validate paths."""
        logger.info(f"Setting up environment for year {self.year}...")
        
        try:
            create_directories(self.year)
            if not validate_paths(self.year):
                logger.warning(f"Some required paths are missing for year {self.year}. Pipeline may fail.")
            
            logger.info(f"Environment setup completed for year {self.year}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False
    
    def run_full_pipeline(self, model_types: List[str] = None, force_retrain: bool = False) -> bool:
        """Run the complete pipeline."""
        logger.info("="*80)
        logger.info(f"STARTING QB ARCHETYPE ANALYSIS PIPELINE (Year {self.year})")
        logger.info("="*80)
        
        pipeline_start_time = time.time()
        
        try:
            # Step 1: Setup environment
            if not self.setup_environment():
                logger.error("Environment setup failed")
                return False
            
            # Step 2: Data processing
            data_step = DataProcessingStep(self.year)
            result = data_step.run()
            self.pipeline_results['data_processing'] = result
            if result['status'] != 'success':
                logger.error("Data processing failed")
                return False
            
            # Step 3: Clustering
            clustering_step = ClusteringStep(self.year)
            result = clustering_step.run()
            self.pipeline_results['clustering'] = result
            if result['status'] != 'success':
                logger.error("Clustering failed")
                return False
            
            # Step 4: Model training
            model_step = ModelTrainingStep(self.year)
            result = model_step.run(model_types, force_retrain)
            self.pipeline_results['model_training'] = result
            if result['status'] not in ['success', 'completed']:
                logger.error("Model training failed")
                return False
            
            # Step 5: Create final merged CSV
            csv_step = FinalMergedCSVStep(self.year)
            result = csv_step.run()
            self.pipeline_results['final_merged_csv'] = result
            if result['status'] != 'success':
                logger.error("Final merged CSV creation failed")
                return False
            
            # Pipeline completed successfully
            total_elapsed_time = time.time() - pipeline_start_time
            PipelineSummary.print_completion_message(self.year, total_elapsed_time)
            PipelineSummary.print_pipeline_summary(self.pipeline_results)
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False
    
    def run_individual_step(self, step: str, **kwargs) -> bool:
        """Run an individual pipeline step."""
        logger.info(f"Running individual step: {step} (Year {self.year})")
        
        try:
            if step == 'data_processing':
                step_obj = DataProcessingStep(self.year)
                result = step_obj.run()
                self.pipeline_results['data_processing'] = result
                return result['status'] == 'success'
                
            elif step == 'clustering':
                step_obj = ClusteringStep(self.year)
                result = step_obj.run()
                self.pipeline_results['clustering'] = result
                return result['status'] == 'success'
                
            elif step == 'model_training':
                model_types = kwargs.get('model_types', ['recall_optimized'])
                force_retrain = kwargs.get('force_retrain', False)
                step_obj = ModelTrainingStep(self.year)
                result = step_obj.run(model_types, force_retrain)
                self.pipeline_results['model_training'] = result
                return result['status'] in ['success', 'completed']
                
            else:
                logger.error(f"Unknown step: {step}")
                return False
                
        except Exception as e:
            logger.error(f"Error running step {step}: {e}")
            return False

def main():
    """Main function to run the pipeline."""
    import argparse
    
    parser = argparse.ArgumentParser(description='QB Archetype Analysis Pipeline')
    parser.add_argument('--step', choices=['data_processing', 'clustering', 'model_training', 'full'], 
                       default='full', help='Pipeline step to run')
    parser.add_argument('--year', type=int, default=CURRENT_YEAR, 
                       help=f'Year to process (default: {CURRENT_YEAR})')
    parser.add_argument('--models', nargs='+', choices=['recall_optimized', 'standard'],
                       default=['recall_optimized'], help='Model types to train')
    parser.add_argument('--force-retrain', action='store_true',
                       help='Force retraining of models even if they already exist')
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run pipeline
    pipeline = QBPipeline(year=args.year)
    
    if args.step == 'full':
        success = pipeline.run_full_pipeline(args.models, args.force_retrain)
    else:
        success = pipeline.run_individual_step(args.step, model_types=args.models, force_retrain=args.force_retrain)
    
    if success:
        print(f"\nPipeline completed successfully for year {args.year}!")
        return 0
    else:
        print(f"\nPipeline failed for year {args.year}!")
        return 1

if __name__ == "__main__":
    exit(main()) 