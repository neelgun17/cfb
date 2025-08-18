"""
Data Processing Module for QB Archetype Analysis
Handles data cleaning, filtering, and feature engineering
"""

import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Dict, List, Optional
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler

from config import (
    get_raw_data_files, get_processed_data_files, get_data_paths,
    DATA_PARAMS, COLUMNS_TO_DROP, CURRENT_YEAR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class DataProcessor:
    """Handles all data processing operations for QB archetype analysis."""
    
    def __init__(self, year: int = None):
        """Initialize the data processor for a specific year."""
        self.year = year if year is not None else CURRENT_YEAR
        self.raw_data_files = get_raw_data_files(self.year)
        self.processed_data_files = get_processed_data_files(self.year)
        self.data_paths = get_data_paths(self.year)
        
        logger.info(f"DataProcessor initialized for year {self.year}")
        logger.info(f"Raw data directory: {self.data_paths['raw']}")
        logger.info(f"Processed data directory: {self.data_paths['processed']}")
    
    def filter_csv_by_position(self, file_path: Path, position: str = 'QB') -> pd.DataFrame:
        """Filter CSV data by position."""
        try:
            df = pd.read_csv(file_path)
            if 'position' in df.columns:
                filtered_df = df[df['position'] == position].copy()
                logger.info(f"Filtered {file_path.name}: {len(df)} -> {len(filtered_df)} rows")
                return filtered_df
            else:
                logger.warning(f"No 'position' column found in {file_path.name}")
                return df
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return pd.DataFrame()
    
    def process_all_raw_data(self) -> Dict[str, pd.DataFrame]:
        """Process all raw data files and return filtered DataFrames."""
        logger.info("Processing all raw data files...")
        
        processed_data = {}
        
        # Process each raw data file
        for data_type, file_path in self.raw_data_files.items():
            if file_path.exists():
                logger.info(f"Processing {data_type} data...")
                df = self.filter_csv_by_position(file_path, DATA_PARAMS['position_filter'])
                if not df.empty:
                    processed_data[data_type] = df
                else:
                    logger.warning(f"No data found for {data_type}")
            else:
                logger.warning(f"File not found: {file_path}")
        
        logger.info(f"Processed {len(processed_data)} data files")
        return processed_data
    
    def extract_player_passing(self, passing_summary: pd.DataFrame) -> pd.DataFrame:
        """Extract and process player passing data."""
        logger.info("Extracting player passing data...")
        
        # Drop unnecessary columns
        columns_to_drop = COLUMNS_TO_DROP['passing']
        available_columns = [col for col in columns_to_drop if col in passing_summary.columns]
        passing_clean = passing_summary.drop(columns=available_columns, errors='ignore')
        
        # Handle infinite values
        numeric_columns = passing_clean.select_dtypes(include=[np.number]).columns
        passing_clean[numeric_columns] = passing_clean[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Extracted passing data: {passing_clean.shape}")
        return passing_clean
    
    def extract_player_rushing(self, rushing_summary: pd.DataFrame) -> pd.DataFrame:
        """Extract and process player rushing data."""
        logger.info("Extracting player rushing data...")
        
        # Drop unnecessary columns
        columns_to_drop = COLUMNS_TO_DROP['rushing']
        available_columns = [col for col in columns_to_drop if col in rushing_summary.columns]
        rushing_clean = rushing_summary.drop(columns=available_columns, errors='ignore')
        
        # Handle infinite values
        numeric_columns = rushing_clean.select_dtypes(include=[np.number]).columns
        rushing_clean[numeric_columns] = rushing_clean[numeric_columns].replace([np.inf, -np.inf], np.nan)
        
        logger.info(f"Extracted rushing data: {rushing_clean.shape}")
        return rushing_clean
    
    def extract_concept_fields(self, concept_data: pd.DataFrame) -> pd.DataFrame:
        """Extract concept-related fields from concept data."""
        logger.info("Extracting concept fields...")
        
        # Calculate rates based on available columns
        concept_fields = concept_data.groupby('player_id').agg({
            'pa_attempts': 'sum',
            'pa_ypa': 'mean',
            'screen_attempts': 'sum',
            'npa_attempts': 'sum',  # Non-play action attempts
            'no_screen_attempts': 'sum'  # No screen attempts
        }).reset_index()
        
        # Calculate rates
        total_attempts = concept_fields['pa_attempts'] + concept_fields['npa_attempts']
        concept_fields['pa_rate'] = np.where(total_attempts > 0, 
                                           concept_fields['pa_attempts'] / total_attempts, 0)
        concept_fields['screen_rate'] = np.where(total_attempts > 0, 
                                               concept_fields['screen_attempts'] / total_attempts, 0)
        
        # Drop the raw attempt columns and keep only the calculated rates
        concept_fields = concept_fields[['player_id', 'pa_rate', 'pa_ypa', 'screen_rate']]
        
        logger.info(f"Extracted concept fields: {concept_fields.shape}")
        return concept_fields
    
    def extract_depth_fields(self, depth_data: pd.DataFrame) -> pd.DataFrame:
        """Extract depth-related fields from depth data."""
        logger.info("Extracting depth fields...")
        
        # Calculate rates based on available columns
        depth_fields = depth_data.groupby('player_id').agg({
            'short_attempts': 'sum',
            'medium_attempts': 'sum',
            'deep_attempts': 'sum',
            'behind_los_attempts': 'sum',
            'deep_accuracy_percent': 'mean',
            'deep_twp_rate': 'mean',
            'deep_completions': 'sum',
            'deep_turnover_worthy_plays': 'sum'
        }).reset_index()
        
        # Calculate total attempts
        total_attempts = (depth_fields['short_attempts'] + depth_fields['medium_attempts'] + 
                         depth_fields['deep_attempts'] + depth_fields['behind_los_attempts'])
        
        # Calculate rates
        depth_fields['short_attempt_rate'] = np.where(total_attempts > 0, 
                                                    depth_fields['short_attempts'] / total_attempts, 0)
        depth_fields['medium_attempt_rate'] = np.where(total_attempts > 0, 
                                                     depth_fields['medium_attempts'] / total_attempts, 0)
        depth_fields['deep_attempt_rate'] = np.where(total_attempts > 0, 
                                                   depth_fields['deep_attempts'] / total_attempts, 0)
        depth_fields['behind_los_attempt_rate'] = np.where(total_attempts > 0, 
                                                         depth_fields['behind_los_attempts'] / total_attempts, 0)
        
        # Keep the calculated rates and deep data
        depth_fields = depth_fields[['player_id', 'short_attempt_rate', 'medium_attempt_rate', 
                                   'deep_attempt_rate', 'behind_los_attempt_rate',
                                   'deep_accuracy_percent', 'deep_twp_rate', 'deep_completions', 'deep_turnover_worthy_plays']]
        
        logger.info(f"Extracted depth fields: {depth_fields.shape}")
        return depth_fields
    
    def extract_pressure_fields(self, pressure_data: pd.DataFrame) -> pd.DataFrame:
        """Extract pressure-related fields from pressure data."""
        logger.info("Extracting pressure fields...")
        
        # Calculate rates based on available columns
        pressure_fields = pressure_data.groupby('player_id').agg({
            'pressure_attempts': 'sum',
            'pressure_sack_percent': 'mean',
            'pressure_twp_rate': 'mean',
            'pressure_accuracy_percent': 'mean',
            'blitz_attempts': 'sum',
            'no_blitz_attempts': 'sum'
        }).reset_index()
        
        # Calculate pressure rate (pressure attempts / total attempts)
        total_attempts = pressure_fields['pressure_attempts'] + pressure_fields['no_blitz_attempts']
        pressure_fields['pressure_rate'] = np.where(total_attempts > 0, 
                                                  pressure_fields['pressure_attempts'] / total_attempts, 0)
        
        # Calculate blitz rate
        pressure_fields['blitz_rate'] = np.where(total_attempts > 0, 
                                               pressure_fields['blitz_attempts'] / total_attempts, 0)
        
        # Keep only the calculated fields
        pressure_fields = pressure_fields[['player_id', 'pressure_rate', 'pressure_sack_percent', 
                                         'pressure_twp_rate', 'pressure_accuracy_percent', 'blitz_rate']]
        
        logger.info(f"Extracted pressure fields: {pressure_fields.shape}")
        return pressure_fields
    
    def extract_pocket_time_fields(self, pocket_data: pd.DataFrame) -> pd.DataFrame:
        """Extract pocket time-related fields from pocket data."""
        logger.info("Extracting pocket time fields...")
        
        # Calculate rates based on available columns
        pocket_fields = pocket_data.groupby('player_id').agg({
            'avg_time_to_throw': 'mean',
            'less_attempts': 'sum',
            'more_attempts': 'sum'
        }).reset_index()
        
        # Calculate quick and long throw rates
        total_attempts = pocket_fields['less_attempts'] + pocket_fields['more_attempts']
        pocket_fields['quick_throw_rate'] = np.where(total_attempts > 0, 
                                                   pocket_fields['less_attempts'] / total_attempts, 0)
        pocket_fields['long_throw_rate'] = np.where(total_attempts > 0, 
                                                  pocket_fields['more_attempts'] / total_attempts, 0)
        
        # Keep only the calculated fields
        pocket_fields = pocket_fields[['player_id', 'avg_time_to_throw', 
                                     'quick_throw_rate', 'long_throw_rate']]
        
        logger.info(f"Extracted pocket time fields: {pocket_fields.shape}")
        return pocket_fields
    
    def merge_all_summaries(self, processed_data: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """Merge all processed summaries into a single DataFrame."""
        logger.info("Merging all summaries...")
        
        # Start with passing summary
        if 'passing_summary' not in processed_data:
            logger.error("Passing summary not found in processed data")
            return pd.DataFrame()
        
        merged_df = processed_data['passing_summary'].copy()
        
        # Merge with other summaries
        summaries_to_merge = [
            ('rushing_summary', 'rushing'),
            ('concept_summary', 'concept'),
            ('depth_summary', 'depth'),
            ('pressure_summary', 'pressure'),
            ('pocket_time_summary', 'pocket_time')
        ]
        
        for summary_name, name in summaries_to_merge:
            if summary_name in processed_data:
                logger.info(f"Merging {summary_name}...")
                merged_df = pd.merge(
                    merged_df, 
                    processed_data[summary_name], 
                    on='player_id', 
                    how='left',
                    suffixes=('', f'_{name}')
                )
        
        logger.info(f"Merged summary shape: {merged_df.shape}")
        return merged_df
    
    def add_calculated_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Add calculated features to the DataFrame."""
        logger.info("Adding calculated features...")
        
        # Create a copy to avoid modifying the original
        df_with_features = df.copy()
        
        # Calculate new features
        for idx, row in df_with_features.iterrows():
            try:
                # TD to INT ratio
                if row.get('interceptions', 0) > 0:
                    df_with_features.at[idx, 'td_int_ratio'] = row.get('touchdowns', 0) / row.get('interceptions', 0)
                else:
                    df_with_features.at[idx, 'td_int_ratio'] = row.get('touchdowns', 0) if row.get('touchdowns', 0) > 0 else np.nan
                
                # Designed run rate
                total_plays = row.get('dropbacks', 0) + row.get('designed_rush_attempts', 0)
                if total_plays > 0:
                    df_with_features.at[idx, 'designed_run_rate'] = row.get('designed_rush_attempts', 0) / total_plays
                else:
                    df_with_features.at[idx, 'designed_run_rate'] = np.nan
                
                # Scramble rate
                if row.get('dropbacks', 0) > 0:
                    df_with_features.at[idx, 'scramble_rate'] = row.get('scrambles', 0) / row.get('dropbacks', 0)
                else:
                    df_with_features.at[idx, 'scramble_rate'] = np.nan
                
                # YAC per rush attempt
                if row.get('rush_attempts', 0) > 0:
                    df_with_features.at[idx, 'YAC_per_rush_attempt'] = row.get('yac_rushing', 0) / row.get('rush_attempts', 0)
                else:
                    df_with_features.at[idx, 'YAC_per_rush_attempt'] = np.nan
                
                # Percentage of total yards from rushing
                total_yards = row.get('passing_yards', 0) + row.get('rushing_yards', 0)
                if total_yards > 0:
                    df_with_features.at[idx, 'pct_total_yards_rushing'] = row.get('rushing_yards', 0) / total_yards
                else:
                    df_with_features.at[idx, 'pct_total_yards_rushing'] = np.nan
                
                # QB rush attempt rate
                total_attempts = row.get('attempts', 0) + row.get('rush_attempts', 0)
                if total_attempts > 0:
                    df_with_features.at[idx, 'qb_rush_attempt_rate'] = row.get('rush_attempts', 0) / total_attempts
                else:
                    df_with_features.at[idx, 'qb_rush_attempt_rate'] = np.nan
                
                # Designed YPA
                if row.get('designed_rush_attempts', 0) > 0:
                    df_with_features.at[idx, 'designed_YPA'] = row.get('designed_rush_yards', 0) / row.get('designed_rush_attempts', 0)
                else:
                    df_with_features.at[idx, 'designed_YPA'] = np.nan
                
                # Total attempts
                df_with_features.at[idx, 'tot_attempts'] = row.get('attempts', 0) + row.get('rush_attempts', 0)
                
                # Total rushing attempts
                df_with_features.at[idx, 'tot_rushing_attempts'] = row.get('designed_rush_attempts', 0) + row.get('scrambles', 0)
                
                # QB designed run rate of all plays
                total_plays = row.get('dropbacks', 0) + row.get('designed_rush_attempts', 0)
                if total_plays > 0:
                    df_with_features.at[idx, 'qb_designed_run_rate_of_all_plays'] = row.get('designed_rush_attempts', 0) / total_plays
                else:
                    df_with_features.at[idx, 'qb_designed_run_rate_of_all_plays'] = np.nan
                
                # Rush attempts per game
                if row.get('player_game_count', 0) > 0:
                    df_with_features.at[idx, 'rush_attempts_per_game'] = row.get('rush_attempts', 0) / row.get('player_game_count', 0)
                else:
                    df_with_features.at[idx, 'rush_attempts_per_game'] = np.nan
                
                # Completion percentage difference (vs expected)
                # This is a derived metric - for now using a simple calculation
                expected_comp_pct = 65.0  # Baseline expectation
                actual_comp_pct = row.get('completion_percent', 0)
                df_with_features.at[idx, 'comp_pct_diff'] = actual_comp_pct - expected_comp_pct
                
                # Yards per attempt difference (vs expected)
                expected_ypa = 7.0  # Baseline expectation
                actual_ypa = row.get('ypa', 0)
                df_with_features.at[idx, 'ypa_diff'] = actual_ypa - expected_ypa
                
                # Deep accuracy percent (from depth data) - use existing value if available
                if pd.isna(row.get('deep_accuracy_percent', np.nan)):
                    deep_attempts = row.get('deep_attempts', 0)
                    deep_completions = row.get('deep_completions', 0)
                    if deep_attempts > 0:
                        df_with_features.at[idx, 'deep_accuracy_percent'] = (deep_completions / deep_attempts) * 100
                    else:
                        df_with_features.at[idx, 'deep_accuracy_percent'] = np.nan
                
                # Deep turnover worthy play rate - use existing value if available
                if pd.isna(row.get('deep_twp_rate', np.nan)):
                    deep_attempts = row.get('deep_attempts', 0)
                    if deep_attempts > 0:
                        df_with_features.at[idx, 'deep_twp_rate'] = (row.get('deep_turnover_worthy_plays', 0) / deep_attempts) * 100
                    else:
                        df_with_features.at[idx, 'deep_twp_rate'] = np.nan
                
            except Exception as e:
                logger.warning(f"Error calculating features for row {idx}: {e}")
                continue
        
        logger.info(f"Added calculated features. Final shape: {df_with_features.shape}")
        return df_with_features
    
    def save_processed_data(self, df: pd.DataFrame, filename: str) -> bool:
        """Save processed data to CSV file."""
        try:
            output_path = self.data_paths['processed'] / filename
            df.to_csv(output_path, index=False)
            logger.info(f"Saved {filename} to {output_path}")
            return True
        except Exception as e:
            logger.error(f"Error saving {filename}: {e}")
            return False
    
    def run_full_pipeline(self) -> bool:
        """Run the complete data processing pipeline."""
        logger.info("Starting data processing pipeline...")
        
        try:
            # Step 1: Process all raw data
            processed_data = self.process_all_raw_data()
            if not processed_data:
                logger.error("No data processed from raw files")
                return False
            
            # Step 2: Extract specific data types
            if 'passing_summary' in processed_data:
                processed_data['passing_summary'] = self.extract_player_passing(processed_data['passing_summary'])
            
            if 'rushing_summary' in processed_data:
                processed_data['rushing_summary'] = self.extract_player_rushing(processed_data['rushing_summary'])
            
            # Step 3: Extract concept fields
            if 'passing_concept' in processed_data:
                processed_data['concept_summary'] = self.extract_concept_fields(processed_data['passing_concept'])
            
            # Step 4: Extract depth fields
            if 'passing_depth' in processed_data:
                processed_data['depth_summary'] = self.extract_depth_fields(processed_data['passing_depth'])
            
            # Step 5: Extract pressure fields
            if 'passing_pressure' in processed_data:
                processed_data['pressure_summary'] = self.extract_pressure_fields(processed_data['passing_pressure'])
            
            # Step 6: Extract pocket time fields
            if 'time_in_pocket' in processed_data:
                processed_data['pocket_time_summary'] = self.extract_pocket_time_fields(processed_data['time_in_pocket'])
            
            # Step 7: Merge all summaries
            merged_df = self.merge_all_summaries(processed_data)
            if merged_df.empty:
                logger.error("Failed to merge summaries")
                return False
            
            # Step 8: Add calculated features
            final_df = self.add_calculated_features(merged_df)
            
            # Step 9: Save individual summaries
            for name, df in processed_data.items():
                if name in ['passing_summary', 'rushing_summary', 'concept_summary', 
                           'depth_summary', 'pressure_summary', 'pocket_time_summary']:
                    filename = f"qb_player_{name.replace('_summary', '')}_summary.csv"
                    self.save_processed_data(df, filename)
            
            # Step 10: Save merged summary
            self.save_processed_data(final_df, "qb_player_merged_summary.csv")
            
            logger.info("Data processing pipeline completed successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error in data processing pipeline: {e}")
            return False

def main():
    """Main function to run data processing."""
    processor = DataProcessor()
    success = processor.run_full_pipeline()
    
    if success:
        print("Data processing completed successfully!")
    else:
        print("Data processing failed. Check logs for details.")

if __name__ == "__main__":
    main() 