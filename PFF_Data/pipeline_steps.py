"""
Pipeline Steps Module
Contains individual pipeline step classes for better modularity
"""

import logging
from pathlib import Path
from typing import Dict, List
import time
import pandas as pd

from config import create_directories, validate_paths, get_data_paths, CURRENT_YEAR
from data_processing import DataProcessor
from clustering import QBClustering
from model_training import QBModelTrainer

logger = logging.getLogger(__name__)

class DataProcessingStep:
    """Handles the data processing pipeline step."""
    
    def __init__(self, year: int):
        self.year = year
        self.data_processor = None
        
    def run(self) -> Dict:
        """Run the data processing step."""
        logger.info("="*60)
        logger.info(f"STEP 1: DATA PROCESSING (Year {self.year})")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            self.data_processor = DataProcessor(year=self.year)
            success = self.data_processor.run_full_pipeline()
            
            elapsed_time = time.time() - start_time
            
            if success:
                logger.info(f"Data processing completed successfully in {elapsed_time:.2f} seconds")
                return {
                    'status': 'success',
                    'elapsed_time': elapsed_time,
                    'year': self.year
                }
            else:
                logger.error("Data processing failed")
                return {
                    'status': 'failed',
                    'elapsed_time': elapsed_time,
                    'year': self.year
                }
                
        except Exception as e:
            logger.error(f"Error in data processing: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }

class ClusteringStep:
    """Handles the clustering analysis pipeline step."""
    
    def __init__(self, year: int):
        self.year = year
        self.clustering = None
        
    def run(self) -> Dict:
        """Run the clustering step."""
        logger.info("="*60)
        logger.info(f"STEP 2: CLUSTERING ANALYSIS (Year {self.year})")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            self.clustering = QBClustering(year=self.year)
            success = self.clustering.run_full_clustering_pipeline()
            
            elapsed_time = time.time() - start_time
            
            if success:
                logger.info(f"Clustering analysis completed successfully in {elapsed_time:.2f} seconds")
                
                # Get clustering summary
                summary = self.clustering.get_cluster_summary()
                return {
                    'status': 'success',
                    'elapsed_time': elapsed_time,
                    'summary': summary,
                    'year': self.year
                }
            else:
                logger.error("Clustering analysis failed")
                return {
                    'status': 'failed',
                    'elapsed_time': elapsed_time,
                    'year': self.year
                }
                
        except Exception as e:
            logger.error(f"Error in clustering: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }

class ModelTrainingStep:
    """Handles the model training pipeline step."""
    
    def __init__(self, year: int):
        self.year = year
        
    def run(self, model_types: List[str] = None, force_retrain: bool = False) -> Dict:
        """Run the model training step."""
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
                
                trainer = QBModelTrainer(model_type=model_type, year=self.year, force_retrain=force_retrain)
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
            
            # Check if at least one model was successful
            successful_models = [m for m, r in training_results.items() if r['status'] == 'success']
            if successful_models:
                logger.info(f"Model training completed. Successful models: {successful_models}")
                return {
                    'status': 'completed',
                    'elapsed_time': total_elapsed_time,
                    'models': training_results,
                    'year': self.year
                }
            else:
                logger.error("All model training attempts failed")
                return {
                    'status': 'failed',
                    'elapsed_time': total_elapsed_time,
                    'year': self.year
                }
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }

class FinalMergedCSVStep:
    """Handles the final merged CSV creation step."""
    
    def __init__(self, year: int):
        self.year = year
        
    def run(self) -> Dict:
        """Run the final merged CSV creation step."""
        logger.info("="*60)
        logger.info(f"STEP 4: CREATING FINAL MERGED CSV (Year {self.year})")
        logger.info("="*60)
        
        try:
            start_time = time.time()
            
            # Define paths
            data_dir = Path("data")
            processed_file = data_dir / "processed" / str(self.year) / "qb_player_merged_summary.csv"
            archetype_file = data_dir / "analysis" / str(self.year) / "hierarchical_player_assignments_k4.csv"
            output_file = data_dir / "analysis" / str(self.year) / "final_merged_qb_data_with_archetypes.csv"
            
            logger.info(f"Creating final merged CSV for year {self.year}")
            logger.info(f"Processed data file: {processed_file}")
            logger.info(f"Archetype assignments file: {archetype_file}")
            logger.info(f"Output file: {output_file}")
            
            # Load processed data
            logger.info("Loading processed QB data...")
            df_processed = pd.read_csv(processed_file)
            df_processed.columns = df_processed.columns.str.rstrip('_')
            df_processed.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
            logger.info(f"Loaded processed data: {df_processed.shape}")
            
            # Load archetype assignments
            logger.info("Loading archetype assignments...")
            df_archetypes = pd.read_csv(archetype_file)
            logger.info(f"Loaded archetype assignments: {df_archetypes.shape}")
            
            # Filter processed data to only include players with >= 150 dropbacks (matching clustering criteria)
            logger.info("Filtering data to players with >= 150 dropbacks...")
            df_filtered = df_processed[df_processed["dropbacks"].astype(int) >= 150].copy()
            logger.info(f"After filtering by dropbacks >= 150: {df_filtered.shape}")
            
            # Merge the data
            logger.info("Merging processed data with archetype assignments...")
            df_final = pd.merge(
                df_filtered,
                df_archetypes[['player_id', 'hierarchical_cluster', 'archetype_name']],
                on='player_id',
                how='inner'
            )
            
            logger.info(f"Final merged dataset: {df_final.shape}")
            
            # Reorder columns to put archetype information near the front
            logger.info("Reordering columns...")
            column_order = [
                'player', 'player_id', 'team_name', 
                'hierarchical_cluster', 'archetype_name',
                'dropbacks', 'player_game_count'
            ]
            
            # Add all other columns
            other_columns = [col for col in df_final.columns if col not in column_order]
            final_column_order = column_order + other_columns
            
            df_final = df_final[final_column_order]
            
            # Save the final merged CSV
            logger.info(f"Saving final merged CSV to: {output_file}")
            df_final.to_csv(output_file, index=False)
            
            # Calculate summary statistics
            archetype_counts = df_final['archetype_name'].value_counts()
            
            elapsed_time = time.time() - start_time
            logger.info(f"Final merged CSV created successfully in {elapsed_time:.2f} seconds")
            
            # Print summary
            logger.info("\n" + "="*60)
            logger.info("FINAL MERGED CSV SUMMARY")
            logger.info("="*60)
            logger.info(f"Total players: {len(df_final)}")
            logger.info(f"Total features: {len(df_final.columns)}")
            logger.info(f"Archetype distribution:")
            for archetype, count in archetype_counts.items():
                logger.info(f"  {archetype}: {count} players")
            
            return {
                'status': 'success',
                'elapsed_time': elapsed_time,
                'year': self.year,
                'summary': {
                    'total_players': len(df_final),
                    'total_features': len(df_final.columns),
                    'archetype_distribution': archetype_counts.to_dict()
                }
            }
            
        except Exception as e:
            logger.error(f"Error creating final merged CSV: {e}")
            return {
                'status': 'error',
                'error': str(e),
                'year': self.year
            }
