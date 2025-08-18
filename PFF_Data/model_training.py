"""
Model Training Module for QB Archetype Analysis
Handles classification model training and evaluation
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, recall_score
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from config import (
    get_analysis_files, get_model_files, get_processed_data_files, get_data_paths,
    MODEL_PARAMS, FEATURES_FOR_CLUSTERING, CURRENT_YEAR
)

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class QBModelTrainer:
    """Handles training and evaluation of QB archetype classification models."""
    
    def __init__(self, model_type: str = 'recall_optimized', year: int = None, force_retrain: bool = False):
        """Initialize the model trainer for a specific year and model type."""
        self.model_type = model_type
        self.year = year if year is not None else CURRENT_YEAR
        self.force_retrain = force_retrain
        self.analysis_files = get_analysis_files(self.year)
        self.model_files = get_model_files(self.year, model_type)
        self.processed_data_files = get_processed_data_files(self.year)
        self.data_paths = get_data_paths(self.year)
        
        self.df_merged = None
        self.X = None
        self.y = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.identifiers_train = None
        self.identifiers_test = None
        self.model = None
        self.label_encoder = None
        self.scaler = None
        self.imputer = None
        self.feature_importance = None
        self.model_loaded = False
        
        logger.info(f"QBModelTrainer initialized for year {self.year}, model type: {self.model_type}")
        logger.info(f"Analysis directory: {self.data_paths['analysis']}")
        logger.info(f"Models directory: {self.data_paths['models']}")
        logger.info(f"Force retrain: {self.force_retrain}")
        
    def model_exists(self) -> bool:
        """Check if a trained model already exists."""
        try:
            model_path = self.model_files['model']
            label_encoder_path = self.model_files['label_encoder']
            scaler_path = self.model_files['scaler']
            imputer_path = self.model_files['imputer']
            
            # Check if all required files exist
            all_exist = all([
                model_path.exists(),
                label_encoder_path.exists(),
                scaler_path.exists(),
                imputer_path.exists()
            ])
            
            if all_exist:
                logger.info(f"Existing model found: {model_path}")
            else:
                logger.info(f"No existing model found. Missing files:")
                if not model_path.exists():
                    logger.info(f"  - Model: {model_path}")
                if not label_encoder_path.exists():
                    logger.info(f"  - Label encoder: {label_encoder_path}")
                if not scaler_path.exists():
                    logger.info(f"  - Scaler: {scaler_path}")
                if not imputer_path.exists():
                    logger.info(f"  - Imputer: {imputer_path}")
            
            return all_exist
            
        except Exception as e:
            logger.error(f"Error checking if model exists: {e}")
            return False
    
    def load_existing_model(self) -> bool:
        """Load an existing trained model and preprocessors."""
        logger.info("Loading existing model and preprocessors...")
        
        try:
            # Load model and preprocessors
            self.model = joblib.load(self.model_files['model'])
            self.label_encoder = joblib.load(self.model_files['label_encoder'])
            self.scaler = joblib.load(self.model_files['scaler'])
            self.imputer = joblib.load(self.model_files['imputer'])
            
            logger.info(f"Successfully loaded existing model: {self.model}")
            logger.info(f"Model type: {type(self.model).__name__}")
            logger.info(f"Number of features: {self.model.n_features_in_}")
            logger.info(f"Classes: {self.label_encoder.classes_}")
            
            self.model_loaded = True
            return True
            
        except Exception as e:
            logger.error(f"Error loading existing model: {e}")
            return False
    
    def should_retrain(self) -> bool:
        """Determine if model should be retrained."""
        if self.force_retrain:
            logger.info("Force retrain requested - will retrain model")
            return True
        
        if not self.model_exists():
            logger.info("No existing model found - will train new model")
            return True
        
        logger.info("Existing model found and no force retrain requested - will load existing model")
        return False
        
    def load_and_prepare_data(self) -> bool:
        """Load and prepare data for model training."""
        logger.info("Loading and preparing data for model training...")
        
        try:
            # Load player assignments
            df_assignments = pd.read_csv(self.analysis_files['player_assignments'])
            logger.info(f"Loaded player assignments: {df_assignments.shape}")
            
            # Load merged data
            df_merged = pd.read_csv(self.processed_data_files['merged_summary'])
            df_merged.columns = df_merged.columns.str.rstrip('_')
            df_merged.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
            logger.info(f"Loaded merged data: {df_merged.shape}")
            
            # Filter by dropbacks
            df_filtered = df_merged[df_merged["dropbacks"].astype(int) >= 150].copy()
            logger.info(f"After filtering by dropbacks >= 150: {df_filtered.shape}")
            
            # Merge to get features and labels
            self.df_merged = pd.merge(
                df_filtered,
                df_assignments[['player_id', 'archetype_name']],
                on='player_id',
                how='inner'
            )
            
            if self.df_merged.empty:
                logger.error("Merged DataFrame is empty")
                return False
            
            logger.info(f"Final merged dataset: {self.df_merged.shape}")
            
            # Prepare features and targets
            available_features = [f for f in FEATURES_FOR_CLUSTERING if f in self.df_merged.columns]
            missing_features = [f for f in FEATURES_FOR_CLUSTERING if f not in self.df_merged.columns]
            
            if missing_features:
                logger.warning(f"Missing features: {missing_features}")
            
            self.X = self.df_merged[available_features].copy()
            self.y = self.df_merged['archetype_name'].copy()
            
            logger.info(f"Features shape: {self.X.shape}")
            logger.info(f"Target shape: {self.y.shape}")
            logger.info(f"Unique archetypes: {self.y.unique()}")
            logger.info(f"Archetype distribution:\n{self.y.value_counts()}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error loading and preparing data: {e}")
            return False
    
    def preprocess_features(self) -> bool:
        """Preprocess features for model training."""
        logger.info("Preprocessing features...")
        
        try:
            # Handle infinite values
            if np.isinf(self.X.values).any():
                logger.info("Replacing infinite values with NaN...")
                self.X.replace([np.inf, -np.inf], np.nan, inplace=True)
            
            # Check for features with no observed values
            features_with_data = []
            for col in self.X.columns:
                if self.X[col].notna().sum() > 0:
                    features_with_data.append(col)
                else:
                    logger.warning(f"Feature '{col}' has no observed values, dropping it")
            
            # Keep only features with data
            self.X = self.X[features_with_data].copy()
            logger.info(f"Kept {len(features_with_data)} features with observed values")
            
            # Impute missing values
            logger.info("Imputing missing values...")
            imputer = SimpleImputer(strategy='median')
            X_imputed = imputer.fit_transform(self.X)
            self.X = pd.DataFrame(X_imputed, columns=self.X.columns, index=self.X.index)
            
            # Scale features
            logger.info("Scaling features...")
            scaler = StandardScaler()
            X_scaled = scaler.fit_transform(self.X)
            self.X = pd.DataFrame(X_scaled, columns=self.X.columns, index=self.X.index)
            
            # Store preprocessors for later use
            self.imputer = imputer
            self.scaler = scaler
            
            logger.info(f"Preprocessed features shape: {self.X.shape}")
            return True
            
        except Exception as e:
            logger.error(f"Error preprocessing features: {e}")
            return False
    
    def encode_labels(self) -> bool:
        """Encode target labels."""
        logger.info("Encoding target labels...")
        
        try:
            self.label_encoder = LabelEncoder()
            y_encoded = self.label_encoder.fit_transform(self.y)
            
            # Convert back to Series
            self.y = pd.Series(y_encoded, index=self.y.index)
            
            logger.info("Label encoding:")
            for i, class_name in enumerate(self.label_encoder.classes_):
                logger.info(f"  {i} -> {class_name}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error encoding labels: {e}")
            return False
    
    def split_data(self) -> bool:
        """Split data into training and test sets."""
        logger.info("Splitting data...")
        
        try:
            # Prepare identifiers for later use
            identifiers = self.df_merged[['player', 'team_name', 'player_id']].copy()
            
            # Split data
            (self.X_train, self.X_test, self.y_train, self.y_test, 
             self.identifiers_train, self.identifiers_test) = train_test_split(
                self.X, self.y, identifiers,
                test_size=MODEL_PARAMS['test_size'],
                random_state=MODEL_PARAMS['random_state'],
                stratify=self.y
            )
            
            logger.info(f"Training set: {self.X_train.shape}")
            logger.info(f"Test set: {self.X_test.shape}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error splitting data: {e}")
            return False
    
    def train_recall_optimized_model(self) -> bool:
        """Train a Random Forest model optimized for recall."""
        logger.info("Training recall-optimized model...")
        
        try:
            # Define parameter grid for recall optimization
            param_grid = {
                'n_estimators': [200, 300, 400],
                'max_depth': [None, 10, 15, 20],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced', 'balanced_subsample'],
                'max_features': ['sqrt', 'log2', None]
            }
            
            # Set up cross-validation
            cv = StratifiedKFold(
                n_splits=MODEL_PARAMS['cv_folds'], 
                shuffle=True, 
                random_state=MODEL_PARAMS['random_state']
            )
            
            # Grid search with recall optimization
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=MODEL_PARAMS['random_state']),
                param_grid=param_grid,
                scoring=MODEL_PARAMS['scoring'],
                cv=cv,
                verbose=1,
                n_jobs=-1
            )
            
            # Fit the model
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            logger.info("Best parameters:")
            for param, value in grid_search.best_params_.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best cross-validated score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training recall-optimized model: {e}")
            return False
    
    def train_standard_model(self) -> bool:
        """Train a standard Random Forest model."""
        logger.info("Training standard model...")
        
        try:
            # Define parameter grid for standard optimization
            param_grid = {
                'n_estimators': [100, 200, 300],
                'max_depth': [None, 5, 10, 15],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4],
                'class_weight': ['balanced']
            }
            
            # Set up cross-validation
            cv = StratifiedKFold(
                n_splits=MODEL_PARAMS['cv_folds'], 
                shuffle=True, 
                random_state=MODEL_PARAMS['random_state']
            )
            
            # Grid search with F1 optimization
            grid_search = GridSearchCV(
                estimator=RandomForestClassifier(random_state=MODEL_PARAMS['random_state']),
                param_grid=param_grid,
                scoring='f1_weighted',
                cv=cv,
                verbose=1,
                n_jobs=-1
            )
            
            # Fit the model
            grid_search.fit(self.X_train, self.y_train)
            
            # Get best model
            self.model = grid_search.best_estimator_
            
            logger.info("Best parameters:")
            for param, value in grid_search.best_params_.items():
                logger.info(f"  {param}: {value}")
            logger.info(f"Best cross-validated score: {grid_search.best_score_:.4f}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error training standard model: {e}")
            return False
    
    def evaluate_model(self) -> Dict:
        """Evaluate the trained model."""
        logger.info("Evaluating model...")
        
        try:
            # Make predictions
            y_pred = self.model.predict(self.X_test)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            recall_macro = recall_score(self.y_test, y_pred, average='macro')
            precision_macro = recall_score(self.y_test, y_pred, average='macro')
            
            # Classification report
            report = classification_report(
                self.y_test, y_pred, 
                target_names=self.label_encoder.classes_, 
                zero_division=0
            )
            
            # Confusion matrix
            cm = confusion_matrix(self.y_test, y_pred)
            
            # Per-class recall
            recall_per_class = recall_score(self.y_test, y_pred, average=None)
            
            # Results dictionary
            results = {
                'accuracy': accuracy,
                'recall_macro': recall_macro,
                'precision_macro': precision_macro,
                'classification_report': report,
                'confusion_matrix': cm,
                'recall_per_class': recall_per_class,
                'predictions': y_pred
            }
            
            # Log results
            logger.info(f"Test Accuracy: {accuracy:.4f}")
            logger.info(f"Test Macro Recall: {recall_macro:.4f}")
            logger.info(f"Test Macro Precision: {precision_macro:.4f}")
            logger.info("\nClassification Report:")
            logger.info(report)
            
            # Per-class recall analysis
            logger.info("\nPer-Class Recall Analysis:")
            for i, (class_name, recall_val) in enumerate(zip(self.label_encoder.classes_, recall_per_class)):
                logger.info(f"  {class_name:<25}: {recall_val:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"Error evaluating model: {e}")
            return {}
    
    def analyze_feature_importance(self) -> bool:
        """Analyze and save feature importance."""
        logger.info("Analyzing feature importance...")
        
        try:
            # Get feature importances
            importances = self.model.feature_importances_
            feature_names = self.X.columns
            
            # Create DataFrame
            self.feature_importance = pd.DataFrame({
                'feature': feature_names,
                'importance': importances
            }).sort_values('importance', ascending=False)
            
            # Save feature importance
            importance_path = self.data_paths['analysis'] / 'feature_importance.csv'
            self.feature_importance.to_csv(importance_path, index=False)
            
            # Create visualization
            plt.figure(figsize=(12, 8))
            top_15 = self.feature_importance.head(15)
            sns.barplot(x='importance', y='feature', data=top_15, palette='viridis')
            plt.title('Top 15 Feature Importances for QB Archetype Classification')
            plt.xlabel('Mean Decrease in Impurity')
            plt.ylabel('Feature')
            plt.tight_layout()
            
            # Save plot
            plot_path = self.data_paths['analysis'] / 'feature_importance.png'
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info("Feature importance analysis completed")
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing feature importance: {e}")
            return False
    
    def analyze_misclassifications(self, results: Dict) -> bool:
        """Analyze misclassifications."""
        logger.info("Analyzing misclassifications...")
        
        try:
            y_pred = results['predictions']
            misclassified_indices = np.where(self.y_test != y_pred)[0]
            
            if len(misclassified_indices) == 0:
                logger.info("No misclassifications found!")
                return True
            
            # Get misclassified players
            misclassified_df = self.identifiers_test.iloc[misclassified_indices].copy()
            misclassified_df['true_archetype'] = self.label_encoder.inverse_transform(self.y_test.iloc[misclassified_indices])
            misclassified_df['predicted_archetype'] = self.label_encoder.inverse_transform(y_pred[misclassified_indices])
            
            logger.info(f"\nMisclassified players ({len(misclassified_df)}):")
            for _, row in misclassified_df.iterrows():
                logger.info(f"  {row['player']} ({row['team_name']}): {row['true_archetype']} -> {row['predicted_archetype']}")
            
            # Save misclassifications
            misclass_path = self.data_paths['analysis'] / 'misclassifications.csv'
            misclassified_df.to_csv(misclass_path, index=False)
            
            return True
            
        except Exception as e:
            logger.error(f"Error analyzing misclassifications: {e}")
            return False
    
    def calculate_model_score(self, metrics: Dict) -> float:
        """Calculate a weighted score for model performance comparison."""
        # Weighted score based on business priorities (recall > precision)
        recall_weight = 0.4      # Most important (you said recall > precision)
        accuracy_weight = 0.3    # Overall performance
        f1_weight = 0.2         # Balance between precision/recall
        precision_weight = 0.1   # Least important
        
        score = (
            metrics.get('macro_recall', 0) * recall_weight +
            metrics.get('accuracy', 0) * accuracy_weight +
            metrics.get('macro_f1', 0) * f1_weight +
            metrics.get('macro_precision', 0) * precision_weight
        )
        return score
    
    def evaluate_existing_model(self) -> Dict:
        """Evaluate the existing model on current test data."""
        logger.info("Evaluating existing model for comparison...")
        
        try:
            # Load existing model
            existing_model = joblib.load(self.model_files['model'])
            existing_label_encoder = joblib.load(self.model_files['label_encoder'])
            existing_scaler = joblib.load(self.model_files['scaler'])
            existing_imputer = joblib.load(self.model_files['imputer'])
            
            # Preprocess test data with existing preprocessors
            X_test_processed = existing_imputer.transform(self.X_test)
            X_test_scaled = existing_scaler.transform(X_test_processed)
            
            # Make predictions
            y_pred = existing_model.predict(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(self.y_test, y_pred)
            macro_recall = recall_score(self.y_test, y_pred, average='macro')
            macro_precision = recall_score(self.y_test, y_pred, average='macro')
            macro_f1 = 2 * (macro_precision * macro_recall) / (macro_precision + macro_recall) if (macro_precision + macro_recall) > 0 else 0
            
            metrics = {
                'accuracy': accuracy,
                'macro_recall': macro_recall,
                'macro_precision': macro_precision,
                'macro_f1': macro_f1,
                'predictions': y_pred
            }
            
            logger.info(f"Existing model metrics - Accuracy: {accuracy:.4f}, Recall: {macro_recall:.4f}")
            return metrics
            
        except Exception as e:
            logger.error(f"Error evaluating existing model: {e}")
            return {}
    
    def should_overwrite_model(self, new_metrics: Dict) -> bool:
        """Determine if new model should overwrite existing model."""
        if not self.model_exists():
            logger.info("No existing model found - will save new model")
            return True
        
        # Evaluate existing model
        existing_metrics = self.evaluate_existing_model()
        if not existing_metrics:
            logger.warning("Could not evaluate existing model - will save new model")
            return True
        
        # Calculate scores
        new_score = self.calculate_model_score(new_metrics)
        existing_score = self.calculate_model_score(existing_metrics)
        
        logger.info(f"Model comparison:")
        logger.info(f"  Existing model score: {existing_score:.4f}")
        logger.info(f"  New model score: {new_score:.4f}")
        logger.info(f"  Improvement: {new_score - existing_score:.4f}")
        
        # Only overwrite if new model is better (with small threshold to avoid noise)
        threshold = 0.001  # 0.1% improvement threshold
        should_overwrite = new_score > (existing_score + threshold)
        
        if should_overwrite:
            logger.info(f"New model performs better - will overwrite existing model")
        else:
            logger.info(f"Existing model performs better - keeping existing model")
        
        return should_overwrite
    
    def save_model(self) -> bool:
        """Save the trained model and preprocessors with smart overwriting."""
        logger.info("Saving model and preprocessors...")
        
        try:
            # Get current model performance
            current_metrics = self.evaluate_model()
            if not current_metrics:
                logger.error("Could not evaluate current model")
                return False
            
            # Check if we should overwrite existing model
            if not self.should_overwrite_model(current_metrics):
                logger.info("Keeping existing model - new model not saved")
                return True  # Return True since this is successful behavior
            
            # Get model files for this type
            model_files = self.model_files
            
            # Create directory if it doesn't exist
            model_files['model'].parent.mkdir(parents=True, exist_ok=True)
            
            # Save model and preprocessors
            joblib.dump(self.model, model_files['model'])
            joblib.dump(self.label_encoder, model_files['label_encoder'])
            joblib.dump(self.scaler, model_files['scaler'])
            joblib.dump(self.imputer, model_files['imputer'])
            
            logger.info(f"Model and preprocessors saved to: {model_files['model'].parent}")
            return True
            
        except Exception as e:
            logger.error(f"Error saving model: {e}")
            return False
    
    def run_full_training_pipeline(self) -> bool:
        """Run the complete model training pipeline with smart loading."""
        logger.info(f"Starting full model training pipeline ({self.model_type})...")
        
        try:
            # Check if we should retrain or load existing model
            if self.should_retrain():
                # Full training pipeline
                logger.info("="*60)
                logger.info("TRAINING NEW MODEL")
                logger.info("="*60)
                
                # Step 1: Load and prepare data
                if not self.load_and_prepare_data():
                    return False
                
                # Step 2: Preprocess features
                if not self.preprocess_features():
                    return False
                
                # Step 3: Encode labels
                if not self.encode_labels():
                    return False
                
                # Step 4: Split data
                if not self.split_data():
                    return False
                
                # Step 5: Train model
                if self.model_type == 'recall_optimized':
                    if not self.train_recall_optimized_model():
                        return False
                else:
                    if not self.train_standard_model():
                        return False
                
                # Step 6: Evaluate model
                results = self.evaluate_model()
                if not results:
                    return False
                
                # Step 7: Analyze feature importance
                if not self.analyze_feature_importance():
                    return False
                
                # Step 8: Analyze misclassifications
                if not self.analyze_misclassifications(results):
                    return False
                
                # Step 9: Save model
                if not self.save_model():
                    return False
                
                logger.info("Model training pipeline completed successfully!")
                
            else:
                # Load existing model
                logger.info("="*60)
                logger.info("LOADING EXISTING MODEL")
                logger.info("="*60)
                
                # Load existing model
                if not self.load_existing_model():
                    return False
                
                # Load and prepare data for evaluation (needed for feature importance analysis)
                if not self.load_and_prepare_data():
                    return False
                
                if not self.preprocess_features():
                    return False
                
                if not self.encode_labels():
                    return False
                
                if not self.split_data():
                    return False
                
                # Evaluate loaded model
                results = self.evaluate_model()
                if not results:
                    return False
                
                # Analyze feature importance
                if not self.analyze_feature_importance():
                    return False
                
                # Analyze misclassifications
                if not self.analyze_misclassifications(results):
                    return False
                
                logger.info("Model loading and evaluation completed successfully!")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in model training pipeline: {e}")
            return False
    
    def get_training_summary(self) -> Dict:
        """Get a summary of training results."""
        if self.model is None:
            return {}
        
        summary = {
            'model_type': self.model_type,
            'training_samples': len(self.X_train),
            'test_samples': len(self.X_test),
            'n_features': self.X.shape[1],
            'n_classes': len(self.label_encoder.classes_),
            'class_distribution': self.y.value_counts().to_dict()
        }
        
        return summary

def main():
    """Main function to run model training."""
    # Train recall-optimized model
    trainer_recall = QBModelTrainer('recall_optimized')
    success_recall = trainer_recall.run_full_training_pipeline()
    
    if success_recall:
        print("Recall-optimized model training completed successfully!")
        summary = trainer_recall.get_training_summary()
        print(f"\nTraining Summary:")
        print(f"Model type: {summary['model_type']}")
        print(f"Training samples: {summary['training_samples']}")
        print(f"Test samples: {summary['test_samples']}")
        print(f"Features: {summary['n_features']}")
        print(f"Classes: {summary['n_classes']}")
    else:
        print("Recall-optimized model training failed. Check logs for details.")
    
    # Train standard model
    trainer_standard = QBModelTrainer('standard')
    success_standard = trainer_standard.run_full_training_pipeline()
    
    if success_standard:
        print("Standard model training completed successfully!")
    else:
        print("Standard model training failed. Check logs for details.")

if __name__ == "__main__":
    main() 