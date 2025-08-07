import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.impute import SimpleImputer

# --- Configuration & Constants ---
MODEL_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/best_qb_archetype_rf_model.joblib'
LABEL_ENCODER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_label_encoder.joblib'
SCALER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_feature_scaler.joblib'
IMPUTER_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/best_rf_model/archetype_feature_imputer.joblib'

# Paths for validation data (you'll need to create this separate dataset)
VALIDATION_DATA_PATH = '/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/validation_data/qb_player_merged_summary_validation.csv'

FEATURES_FOR_CLUSTERING = [
    'accuracy_percent', 'avg_depth_of_target', 'avg_time_to_throw', 'btt_rate',
    'completion_percent', 'sack_percent', 'twp_rate', 'ypa', 'td_int_ratio',
    'comp_pct_diff', 'ypa_diff', 'designed_run_rate', 'scramble_rate',
    'elusive_rating', 'ypa_rushing', 'breakaway_percent', 'YAC_per_rush_attempt',
    'pct_total_yards_rushing', 'qb_rush_attempt_rate', 'grades_offense',
    'grades_pass', 'grades_run', 'pa_rate', 'pa_ypa', 'screen_rate',
    'deep_attempt_rate', 'deep_accuracy_percent', 'deep_twp_rate',
    'pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate',
    'pressure_accuracy_percent', 'quick_throw_rate'
]

def load_model_and_preprocessors():
    """Load the trained model and preprocessing components."""
    try:
        model = joblib.load(MODEL_PATH)
        label_encoder = joblib.load(LABEL_ENCODER_PATH)
        scaler = joblib.load(SCALER_PATH)
        imputer = joblib.load(IMPUTER_PATH)
        print("Model and preprocessors loaded successfully.")
        return model, label_encoder, scaler, imputer
    except FileNotFoundError as e:
        print(f"Error loading saved objects: {e}")
        return None, None, None, None

def analyze_feature_importance(model, feature_names):
    """Analyze and display feature importance from the Random Forest model."""
    print("\n" + "="*60)
    print("FEATURE IMPORTANCE ANALYSIS")
    print("="*60)
    
    # Get feature importances
    importances = model.feature_importances_
    feature_importance_df = pd.DataFrame({
        'feature': feature_names,
        'importance': importances
    }).sort_values('importance', ascending=False)
    
    # Display top 15 features
    print("\nTop 15 Most Important Features:")
    print("-" * 50)
    for i, (_, row) in enumerate(feature_importance_df.head(15).iterrows(), 1):
        print(f"{i:2d}. {row['feature']:<25} {row['importance']:.4f}")
    
    # Create visualization
    plt.figure(figsize=(12, 8))
    top_15 = feature_importance_df.head(15)
    sns.barplot(x='importance', y='feature', data=top_15, palette='viridis')
    plt.title('Top 15 Feature Importances for QB Archetype Classification', fontsize=14, fontweight='bold')
    plt.xlabel('Mean Decrease in Impurity', fontsize=12)
    plt.ylabel('Feature', fontsize=12)
    plt.tight_layout()
    plt.show()
    
    # Analyze feature categories
    print("\nFeature Importance by Category:")
    print("-" * 40)
    
    # Define feature categories
    categories = {
        'Passing Efficiency': ['accuracy_percent', 'completion_percent', 'ypa', 'comp_pct_diff', 'ypa_diff'],
        'Passing Style': ['avg_depth_of_target', 'btt_rate', 'twp_rate', 'avg_time_to_throw'],
        'Rushing/Mobility': ['designed_run_rate', 'scramble_rate', 'ypa_rushing', 'elusive_rating', 'qb_rush_attempt_rate'],
        'Pressure Handling': ['pressure_rate', 'pressure_sack_percent', 'pressure_twp_rate', 'pressure_accuracy_percent'],
        'PFF Grades': ['grades_offense', 'grades_pass', 'grades_run'],
        'Play Types': ['pa_rate', 'pa_ypa', 'screen_rate', 'deep_attempt_rate'],
        'Other': ['sack_percent', 'td_int_ratio', 'breakaway_percent', 'YAC_per_rush_attempt', 
                 'pct_total_yards_rushing', 'deep_accuracy_percent', 'deep_twp_rate', 'quick_throw_rate']
    }
    
    for category, features in categories.items():
        category_importance = feature_importance_df[feature_importance_df['feature'].isin(features)]['importance'].sum()
        print(f"{category:<20}: {category_importance:.4f}")
    
    return feature_importance_df

def create_validation_dataset():
    """Create a separate validation dataset from your existing data."""
    print("\n" + "="*60)
    print("CREATING VALIDATION DATASET")
    print("="*60)
    
    # Load the full dataset
    try:
        df = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/processed_data/qb_player_merged_summary.csv')
        df.columns = df.columns.str.rstrip('_')
        df.rename(columns=lambda c: c.replace('.', '_'), inplace=True)
        print(f"Loaded dataset with {len(df)} total players")
    except FileNotFoundError:
        print("Error: Could not find the main dataset file")
        return None
    
    # Filter by dropbacks
    df_filtered = df[df["dropbacks"].astype(int) >= 150].copy()
    print(f"After filtering by dropbacks >= 150: {len(df_filtered)} players")
    
    # Load the archetype assignments
    try:
        archetype_assignments = pd.read_csv('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/analysis/hierarchical_player_assignments_k4.csv')
        print(f"Loaded archetype assignments for {len(archetype_assignments)} players")
    except FileNotFoundError:
        print("Error: Could not find archetype assignments file")
        return None
    
    # Merge to get players with archetypes
    df_with_archetypes = pd.merge(
        df_filtered,
        archetype_assignments[['player_id', 'archetype_name']],
        on='player_id',
        how='inner'
    )
    print(f"Players with archetype assignments: {len(df_with_archetypes)}")
    
    # Create validation dataset (e.g., take 20% of players for validation)
    # Use a different random seed to ensure different split than training
    validation_df = df_with_archetypes.sample(frac=0.2, random_state=123).copy()
    training_df = df_with_archetypes.drop(validation_df.index).copy()
    
    print(f"Training dataset size: {len(training_df)}")
    print(f"Validation dataset size: {len(validation_df)}")
    
    # Save validation dataset
    import os
    os.makedirs('/Users/neelgundlapally/Documents/Projects/cfb/PFF_Data/validation_data', exist_ok=True)
    validation_df.to_csv(VALIDATION_DATA_PATH, index=False)
    print(f"Validation dataset saved to: {VALIDATION_DATA_PATH}")
    
    # Display archetype distribution in validation set
    print("\nArchetype distribution in validation set:")
    print(validation_df['archetype_name'].value_counts())
    
    return validation_df, training_df

def validate_on_separate_dataset(model, label_encoder, scaler, imputer, validation_df):
    """Validate the model on a completely separate dataset."""
    print("\n" + "="*60)
    print("VALIDATION ON SEPARATE DATASET")
    print("="*60)
    
    if validation_df is None:
        print("No validation dataset available. Run create_validation_dataset() first.")
        return
    
    # Prepare features
    X_val = validation_df[FEATURES_FOR_CLUSTERING].copy()
    y_val_str = validation_df['archetype_name'].copy()
    
    # Preprocess validation data
    X_val.replace([np.inf, -np.inf], np.nan, inplace=True)
    X_val_imputed = imputer.transform(X_val)
    X_val_scaled = scaler.transform(X_val_imputed)
    
    # Encode labels
    y_val_encoded = label_encoder.transform(y_val_str)
    
    # Make predictions
    y_val_pred = model.predict(X_val_scaled)
    y_val_pred_str = label_encoder.inverse_transform(y_val_pred)
    
    # Calculate metrics
    from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
    
    accuracy = accuracy_score(y_val_encoded, y_val_pred)
    recall_macro = recall_score(y_val_encoded, y_val_pred, average='macro')
    precision_macro = precision_score(y_val_encoded, y_val_pred, average='macro')
    f1_macro = f1_score(y_val_encoded, y_val_pred, average='macro')
    
    print(f"Validation Accuracy: {accuracy:.4f}")
    print(f"Validation Macro Recall: {recall_macro:.4f}")
    print(f"Validation Macro Precision: {precision_macro:.4f}")
    print(f"Validation Macro F1: {f1_macro:.4f}")
    
    # Detailed classification report
    print("\nDetailed Classification Report:")
    print(classification_report(y_val_encoded, y_val_pred, target_names=label_encoder.classes_))
    
    # Confusion matrix
    cm = confusion_matrix(y_val_encoded, y_val_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
    plt.title('Confusion Matrix - Validation Set')
    plt.xlabel('Predicted Label')
    plt.ylabel('True Label')
    plt.show()
    
    # Analyze misclassifications
    misclassified = validation_df[y_val_str != y_val_pred_str].copy()
    misclassified['true_archetype'] = y_val_str[y_val_str != y_val_pred_str]
    misclassified['predicted_archetype'] = y_val_pred_str[y_val_str != y_val_pred_str]
    
    print(f"\nMisclassified players ({len(misclassified)}):")
    print(misclassified[['player', 'team_name', 'true_archetype', 'predicted_archetype']])
    
    return accuracy, recall_macro, precision_macro, f1_macro

def main():
    """Main function to run the analysis."""
    # Load model and preprocessors
    model, label_encoder, scaler, imputer = load_model_and_preprocessors()
    if model is None:
        return
    
    # Analyze feature importance
    feature_importance_df = analyze_feature_importance(model, FEATURES_FOR_CLUSTERING)
    
    # Create validation dataset
    validation_df, training_df = create_validation_dataset()
    
    # Validate on separate dataset
    if validation_df is not None:
        validate_on_separate_dataset(model, label_encoder, scaler, imputer, validation_df)

if __name__ == "__main__":
    main() 