"""
Main Pipeline for QB Archetype Analysis
Orchestrates the complete workflow from data processing to model training
"""

import logging
from pathlib import Path
from typing import Dict, List
import time

from config import create_directories, validate_paths, get_data_paths, CURRENT_YEAR
from data_processing import DataProcessor
from clustering import QBClustering
from model_training import QBModelTrainer

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
        self.data_processor = None
        self.clustering = None
        self.model_trainer = None
        self.pipeline_results = {}
        
    def setup_environment(self) -> bool:
        """Set up the environment and validate paths."""
        logger.info(f"Setting up environment for year {self.year}...")
        
        try:
            # Create necessary directories for the specific year
            create_directories(self.year)
            
            # Validate paths for the specific year
            if not validate_paths(self.year):
                logger.warning(f"Some required paths are missing for year {self.year}. Pipeline may fail.")
            
            logger.info(f"Environment setup completed for year {self.year}")
            return True
            
        except Exception as e:
            logger.error(f"Error setting up environment: {e}")
            return False
    
    def run_data_processing(self) -> bool:
        """Run the data processing pipeline."""
        logger.info("="*60)
        logger.info(f"STEP 1: DATA PROCESSING (Year {self.year})")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            self.data_processor = DataProcessor(year=self.year)
            success = self.data_processor.run_full_pipeline()
            
            if success:
                elapsed_time = time.time() - start_time
                logger.info(f"Data processing completed successfully in {elapsed_time:.2f} seconds")
                self.pipeline_results['data_processing'] = {
                    'status': 'success',
                    'elapsed_time': elapsed_time,
                    'year': self.year
                }
                return True
            else:
                logger.error("Data processing failed")
                self.pipeline_results['data_processing'] = {
                    'status': 'failed',
                    'elapsed_time': time.time() - start_time,
                    'year': self.year
                }
                return False
                
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            self.pipeline_results['data_processing'] = {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }
            return False
    
    def run_clustering(self) -> bool:
        """Run the clustering pipeline."""
        logger.info("="*60)
        logger.info(f"STEP 2: CLUSTERING ANALYSIS (Year {self.year})")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            self.clustering = QBClustering(year=self.year)
            success = self.clustering.run_full_clustering_pipeline()
            
            if success:
                elapsed_time = time.time() - start_time
                logger.info(f"Clustering analysis completed successfully in {elapsed_time:.2f} seconds")
                
                # Get clustering summary
                summary = self.clustering.get_cluster_summary()
                self.pipeline_results['clustering'] = {
                    'status': 'success',
                    'elapsed_time': elapsed_time,
                    'summary': summary,
                    'year': self.year
                }
                return True
            else:
                logger.error("Clustering analysis failed")
                self.pipeline_results['clustering'] = {
                    'status': 'failed',
                    'elapsed_time': time.time() - start_time,
                    'year': self.year
                }
                return False
                
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            self.pipeline_results['clustering'] = {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }
            return False
    
    def run_model_training(self, model_types: List[str] = None) -> bool:
        """Run the model training pipeline."""
        logger.info("="*60)
        logger.info(f"STEP 3: MODEL TRAINING (Year {self.year})")
        logger.info("="*60)
        
        if model_types is None:
            model_types = ['recall_optimized', 'standard']
        
        try:
            start_time = time.time()
            training_results = {}
            
            for model_type in model_types:
                logger.info(f"Training {model_type} model...")
                model_start_time = time.time()
                
                trainer = QBModelTrainer(model_type=model_type, year=self.year)
                success = trainer.run_full_training_pipeline()
                
                if success:
                    model_elapsed_time = time.time() - model_start_time
                    logger.info(f"{model_type} model training completed in {model_elapsed_time:.2f} seconds")
                    
                    # Get training summary
                    summary = trainer.get_training_summary()
                    training_results[model_type] = {
                        'status': 'success',
                        'elapsed_time': model_elapsed_time,
                        'summary': summary
                    }
                else:
                    logger.error(f"{model_type} model training failed")
                    training_results[model_type] = {
                        'status': 'failed',
                        'elapsed_time': time.time() - model_start_time
                    }
            
            total_elapsed_time = time.time() - start_time
            self.pipeline_results['model_training'] = {
                'status': 'completed',
                'elapsed_time': total_elapsed_time,
                'models': training_results,
                'year': self.year
            }
            
            # Check if at least one model was successful
            successful_models = [m for m, r in training_results.items() if r['status'] == 'success']
            if successful_models:
                logger.info(f"Model training completed. Successful models: {successful_models}")
                return True
            else:
                logger.error("All model training attempts failed")
                return False
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            self.pipeline_results['model_training'] = {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }
            return False
    
    def run_full_pipeline(self, model_types: List[str] = None) -> bool:
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
            if not self.run_data_processing():
                logger.error("Data processing failed")
                return False
            
            # Step 3: Clustering
            if not self.run_clustering():
                logger.error("Clustering failed")
                return False
            
            # Step 4: Model training
            if not self.run_model_training(model_types):
                logger.error("Model training failed")
                return False
            
            # Pipeline completed successfully
            total_elapsed_time = time.time() - pipeline_start_time
            logger.info("="*80)
            logger.info("PIPELINE COMPLETED SUCCESSFULLY")
            logger.info(f"Year: {self.year}")
            logger.info(f"Total elapsed time: {total_elapsed_time:.2f} seconds")
            logger.info("="*80)
            
            # Print summary
            self.print_pipeline_summary()
            
            return True
            
        except Exception as e:
            logger.error(f"Pipeline failed with error: {e}")
            return False
    
    def print_pipeline_summary(self):
        """Print a summary of the pipeline results."""
        logger.info("\n" + "="*60)
        logger.info("PIPELINE SUMMARY")
        logger.info("="*60)
        
        for step, result in self.pipeline_results.items():
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
            
            elif result['status'] == 'failed':
                logger.info(f"  Status: ✗ Failed")
                if 'elapsed_time' in result:
                    logger.info(f"  Time: {result['elapsed_time']:.2f} seconds")
            
            elif result['status'] == 'error':
                logger.info(f"  Status: ✗ Error")
                if 'error' in result:
                    logger.info(f"  Error: {result['error']}")
    
    def run_individual_step(self, step: str, **kwargs) -> bool:
        """Run an individual pipeline step."""
        logger.info(f"Running individual step: {step} (Year {self.year})")
        
        if step == 'data_processing':
            return self.run_data_processing()
        elif step == 'clustering':
            return self.run_clustering()
        elif step == 'model_training':
            model_types = kwargs.get('model_types', ['recall_optimized'])
            return self.run_model_training(model_types)
        else:
            logger.error(f"Unknown step: {step}")
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
    parser.add_argument('--log-level', choices=['DEBUG', 'INFO', 'WARNING', 'ERROR'],
                       default='INFO', help='Logging level')
    
    args = parser.parse_args()
    
    # Set log level
    logging.getLogger().setLevel(getattr(logging, args.log_level))
    
    # Create and run pipeline
    pipeline = QBPipeline(year=args.year)
    
    if args.step == 'full':
        success = pipeline.run_full_pipeline(args.models)
    else:
        success = pipeline.run_individual_step(args.step, model_types=args.models)
    
    if success:
        print(f"\nPipeline completed successfully for year {args.year}!")
        return 0
    else:
        print(f"\nPipeline failed for year {args.year}!")
        return 1

if __name__ == "__main__":
    exit(main()) 