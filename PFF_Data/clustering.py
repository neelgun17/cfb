"""
Clustering Module for QB Archetype Analysis
Handles clustering, archetype discovery, and profile generation
"""

import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.metrics import euclidean_distances, silhouette_score
from sklearn.cluster import AgglomerativeClustering, KMeans
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns

from config import (
    get_processed_data_files, get_analysis_files, get_data_paths,
    CLUSTERING_PARAMS, FEATURES_FOR_CLUSTERING, ARCHETYPE_MAP, CURRENT_YEAR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QBClustering:
    """Handles clustering analysis for QB archetype discovery."""
    
    def __init__(self, year: int = None):
        """Initialize the clustering module for a specific year."""
        self.year = year if year is not None else CURRENT_YEAR
        self.processed_data_files = get_processed_data_files(self.year)
        self.analysis_files = get_analysis_files(self.year)
        self.data_paths = get_data_paths(self.year)
        
        self.df_filtered = None
        self.cluster_data = None
        self.cluster_data_imputed = None
        self.cluster_data_scaled = None
        self.cluster_data_pca = None
        self.cluster_labels = None
        self.cluster_profiles = None
        self.identifiers = None
        
        logger.info(f"QBClustering initialized for year {self.year}")
        logger.info(f"Processed data directory: {self.data_paths['processed']}")
        logger.info(f"Analysis directory: {self.data_paths['analysis']}")
        
    def load_and_prepare_data(self) -> bool:
        """Load and prepare data for clustering."""
        logger.info("Loading and preparing data for clustering...")
        
        try:
            # Load merged data
            df = pd.read_csv(self.processed_data_files['merged_summary'])
            df.columns = df.columns.str.rstrip('_')
            df.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
            
            logger.info(f"Loaded dataset with {len(df)} total players")
            
            # Filter by dropbacks
            self.df_filtered = df[df["dropbacks"].astype(int) >= CLUSTERING_PARAMS['min_dropbacks']].copy()
            logger.info(f"After filtering by dropbacks >= {CLUSTERING_PARAMS['min_dropbacks']}: {len(self.df_filtered)} players")
            
            # Prepare identifiers
            identifier_columns = ['player', 'player_id', 'team_name']
            self.identifiers = self.df_filtered[identifier_columns].copy()
            
            # Prepare features for clustering
            available_features = [f for f in FEATURES_FOR_CLUSTERING if f in self.df_filtered.columns]
            missing_features = [f for f in FEATURES_FOR_CLUSTERING if f not in self.df_filtered.columns]
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            self.cluster_data = self.df_filtered[available_features].copy()
            logger.info(f"Prepared {len(available_features)} features for clustering")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            return False
    
    def preprocess_features(self) -> bool:
        """Preprocess features for clustering."""
        logger.info("Preprocessing features...")
        
        try:
            # Handle infinite values
            if np.isinf(self.cluster_data.values).any():
                logger.info("Replacing infinite values with NaN...")
                self.cluster_data.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Check for features with no observed values
            features_with_data = []
            for col in self.cluster_data.columns:
                if self.cluster_data[col].notna().sum() > 0:
                    features_with_data.append(col)
                else:
                    logger.warning(f"Feature '{col}' has no observed values, dropping it")
            
            # Keep only features with data
            self.cluster_data = self.cluster_data[features_with_data].copy()
            logger.info(f"Kept {len(features_with_data)} features with observed values")
            
            # Impute missing values
            logger.info("Imputing missing values...")
            imputer = SimpleImputer(strategy='median')
            self.cluster_data_imputed = imputer.fit_transform(self.cluster_data)
            self.cluster_data_imputed = pd.DataFrame(
                self.cluster_data_imputed, 
                columns=self.cluster_data.columns, 
                index=self.cluster_data.index
            )
            
            # Scale features
            logger.info("Scaling features...")
            scaler = StandardScaler()
            self.cluster_data_scaled = scaler.fit_transform(self.cluster_data_imputed)
            self.cluster_data_scaled = pd.DataFrame(
                self.cluster_data_scaled,
                columns=self.cluster_data_imputed.columns,
                index=self.cluster_data_imputed.index
            )
            
            # Save scaled data
            self.cluster_data_scaled.to_csv(self.analysis_files['cluster_data_scaled'], index=False)
            logger.info(f"Scaled data saved to: {self.analysis_files['cluster_data_scaled']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return False
    
    def apply_pca(self) -> bool:
        """Apply PCA for dimensionality reduction."""
        logger.info("Applying PCA...")
        
        try:
            pca = PCA(n_components=CLUSTERING_PARAMS['pca_variance'], random_state=CLUSTERING_PARAMS['random_state'])
            self.cluster_data_pca = pca.fit_transform(self.cluster_data_scaled)
            
            logger.info(f"Shape after PCA: {self.cluster_data_pca.shape}")
            logger.info(f"Explained variance ratio: {pca.explained_variance_ratio_.sum():.3f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error applying PCA: {e}")
            return False
    
    def find_optimal_clusters(self, k_range: range = range(4, 10)) -> Tuple[int, Dict]:
        """Find optimal number of clusters using elbow method and silhouette analysis."""
        logger.info("Finding optimal number of clusters...")
        
        inertias = []
        silhouette_scores = []
        
        for k in k_range:
            kmeans = KMeans(n_clusters=k, random_state=CLUSTERING_PARAMS['random_state'], n_init=10)
            kmeans.fit(self.cluster_data_pca)
            inertias.append(kmeans.inertia_)
            silhouette_scores.append(silhouette_score(self.cluster_data_pca, kmeans.labels_))
            logger.info(f"k={k}: Inertia={kmeans.inertia_:.0f}, Silhouette={silhouette_scores[-1]:.3f}")
        
        # Find optimal k (you can implement more sophisticated logic here)
        optimal_k = CLUSTERING_PARAMS['n_clusters']  # Default to configured value
        
        results = {
            'k_range': list(k_range),
            'inertias': inertias,
            'silhouette_scores': silhouette_scores,
            'optimal_k': optimal_k
        }
        
        return optimal_k, results
    
    def perform_hierarchical_clustering(self) -> bool:
        """Perform hierarchical clustering with the optimal number of clusters."""
        logger.info("Performing hierarchical clustering...")
        
        try:
            agg_cluster = AgglomerativeClustering(
                n_clusters=CLUSTERING_PARAMS['n_clusters'], 
                linkage='ward'
            )
            self.cluster_labels = agg_cluster.fit_predict(self.cluster_data_scaled)
            
            # Add cluster labels to main DataFrame
            self.df_filtered['hierarchical_cluster'] = self.cluster_labels
            self.df_filtered['archetype_name'] = self.df_filtered['hierarchical_cluster'].map(ARCHETYPE_MAP)
            
            logger.info("Cluster distribution:")
            cluster_counts = self.df_filtered['archetype_name'].value_counts()
            for archetype, count in cluster_counts.items():
                logger.info(f"  {archetype}: {count}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error performing hierarchical clustering: {e}")
            return False
    
    def generate_cluster_profiles(self) -> bool:
        """Generate statistical profiles for each cluster."""
        logger.info("Generating cluster profiles...")
        
        try:
            # Add cluster labels to imputed data
            self.cluster_data_imputed['hierarchical_cluster'] = self.cluster_labels
            
            # Calculate mean profiles
            self.cluster_profiles = self.cluster_data_imputed.groupby('hierarchical_cluster').mean()
            
            # Save profiles
            self.cluster_profiles.to_csv(self.analysis_files['cluster_profiles'])
            logger.info(f"Cluster profiles saved to: {self.analysis_files['cluster_profiles']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error generating cluster profiles: {e}")
            return False
    
    def find_prototype_players(self, n_prototypes: int = 3) -> Dict[str, List[str]]:
        """Find prototype players for each cluster."""
        logger.info("Finding prototype players...")
        
        try:
            prototypes = {}
            
            for cluster_id in range(CLUSTERING_PARAMS['n_clusters']):
                # Get players in this cluster
                cluster_mask = self.cluster_labels == cluster_id
                cluster_indices = np.where(cluster_mask)[0]
                
                if len(cluster_indices) == 0:
                    continue
                
                # Get cluster centroid (exclude the cluster label column)
                cluster_centroid = self.cluster_profiles.loc[cluster_id].drop('hierarchical_cluster', errors='ignore').values
                
                # Calculate distances to centroid
                cluster_data = self.cluster_data_scaled.iloc[cluster_indices].values
                
                # Ensure dimensions match
                if cluster_centroid.shape[0] != cluster_data.shape[1]:
                    logger.warning(f"Dimension mismatch for cluster {cluster_id}: centroid has {cluster_centroid.shape[0]} features, data has {cluster_data.shape[1]} features")
                    # Use the first n features from centroid to match data
                    min_features = min(cluster_centroid.shape[0], cluster_data.shape[1])
                    cluster_centroid = cluster_centroid[:min_features]
                    cluster_data = cluster_data[:, :min_features]
                
                distances = euclidean_distances(cluster_data, [cluster_centroid]).ravel()
                
                # Find closest players
                closest_indices = np.argsort(distances)[:n_prototypes]
                prototype_indices = cluster_indices[closest_indices]
                
                # Get player names
                archetype_name = ARCHETYPE_MAP[cluster_id]
                prototype_players = []
                
                for idx in prototype_indices:
                    player_name = self.identifiers.iloc[idx]['player']
                    team_name = self.identifiers.iloc[idx]['team_name']
                    distance = distances[np.where(cluster_indices == idx)[0][0]]
                    prototype_players.append(f"{player_name} ({team_name}) - Distance: {distance:.4f}")
                
                prototypes[archetype_name] = prototype_players
                logger.info(f"\n{archetype_name} prototypes:")
                for player in prototype_players:
                    logger.info(f"  - {player}")
            
            return prototypes
            
        except Exception as e:
            logger.error(f"Error finding prototype players: {e}")
            return {}
    
    def save_player_assignments(self) -> bool:
        """Save player assignments to CSV."""
        logger.info("Saving player assignments...")
        
        try:
            # Select key columns for output (including usage/grade stats for system fit analysis)
            output_columns = ['player', 'player_id', 'team_name', 'hierarchical_cluster', 'archetype_name', 
                              'dropbacks', 'attempts', 'grades_offense', 'grades_pass']
            
            # Ensure columns exist before selecting
            valid_cols = [c for c in output_columns if c in self.df_filtered.columns]
            output_df = self.df_filtered[valid_cols].copy()
            
            # Save to CSV
            output_df.to_csv(self.analysis_files['player_assignments'], index=False)
            logger.info(f"Player assignments saved to: {self.analysis_files['player_assignments']}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving player assignments: {e}")
            return False
    
    def create_visualizations(self) -> bool:
        """Create visualizations for clustering results."""
        logger.info("Creating visualizations...")
        
        try:
            # Create cluster distribution plot
            plt.figure(figsize=(10, 6))
            cluster_counts = self.df_filtered['archetype_name'].value_counts()
            plt.bar(cluster_counts.index, cluster_counts.values)
            plt.title('QB Archetype Distribution')
            plt.xlabel('Archetype')
            plt.ylabel('Number of Players')
            plt.xticks(rotation=45)
            plt.tight_layout()
            plt.savefig(self.analysis_files['cluster_profiles'].parent / 'archetype_distribution.png')
            plt.close()
            
            # Create PCA visualization if we have 2+ components
            if self.cluster_data_pca.shape[1] >= 2:
                plt.figure(figsize=(10, 8))
                scatter = plt.scatter(
                    self.cluster_data_pca[:, 0], 
                    self.cluster_data_pca[:, 1], 
                    c=self.cluster_labels, 
                    cmap='viridis', 
                    alpha=0.7
                )
                plt.title('QB Clusters (PCA Components 1 & 2)')
                plt.xlabel('Principal Component 1')
                plt.ylabel('Principal Component 2')
                plt.colorbar(scatter)
                plt.tight_layout()
                plt.savefig(self.analysis_files['cluster_profiles'].parent / 'pca_clusters.png')
                plt.close()
            
            logger.info("Visualizations created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {e}")
            return False
    
    def run_full_clustering_pipeline(self) -> bool:
        """Run the complete clustering pipeline."""
        logger.info("Starting full clustering pipeline...")
        
        try:
            # Step 1: Load and prepare data
            if not self.load_and_prepare_data():
                return False
            
            # Step 2: Preprocess features
            if not self.preprocess_features():
                return False
            
            # Step 3: Apply PCA
            if not self.apply_pca():
                return False
            
            # Step 4: Find optimal clusters (optional)
            optimal_k, results = self.find_optimal_clusters()
            logger.info(f"Using {optimal_k} clusters")
            
            # Step 5: Perform hierarchical clustering
            if not self.perform_hierarchical_clustering():
                return False
            
            # Step 6: Generate cluster profiles
            if not self.generate_cluster_profiles():
                return False
            
            # Step 7: Find prototype players
            prototypes = self.find_prototype_players()
            
            # Step 8: Save results
            if not self.save_player_assignments():
                return False
            
            # Step 9: Create visualizations
            if not self.create_visualizations():
                return False
            
            logger.info("Clustering pipeline completed successfully!")
            return True
            
        except Exception as e:
            logger.error(f"Error in clustering pipeline: {e}")
            return False
    
    def get_cluster_summary(self) -> Dict:
        """Get a summary of clustering results."""
        if self.df_filtered is None:
            return {}
        
        summary = {
            'total_players': len(self.df_filtered),
            'n_clusters': CLUSTERING_PARAMS['n_clusters'],
            'archetype_distribution': self.df_filtered['archetype_name'].value_counts().to_dict(),
            'features_used': len(FEATURES_FOR_CLUSTERING),
            'pca_components': self.cluster_data_pca.shape[1] if self.cluster_data_pca is not None else None
        }
        
        return summary

def main():
    """Main function to run clustering analysis."""
    clustering = QBClustering()
    success = clustering.run_full_clustering_pipeline()
    
    if success:
        print("Clustering analysis completed successfully!")
        summary = clustering.get_cluster_summary()
        print(f"\nClustering Summary:")
        print(f"Total players: {summary['total_players']}")
        print(f"Number of clusters: {summary['n_clusters']}")
        print(f"Archetype distribution: {summary['archetype_distribution']}")
    else:
        print("Clustering analysis failed. Check logs for details.")

if __name__ == "__main__":
    main() 