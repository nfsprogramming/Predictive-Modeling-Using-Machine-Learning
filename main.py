"""
Predictive Modeling Using Machine Learning

This script demonstrates a complete supervised machine learning workflow using
the breast cancer dataset with Random Forest classification, proper evaluation,
and model persistence.
"""

import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
import logging
import joblib
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, 
    classification_report, 
    confusion_matrix, 
    ConfusionMatrixDisplay, 
    roc_auc_score
)

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_paths():
    """Setup project paths using pathlib for cross-platform compatibility."""
    BASE_DIR = Path(__file__).resolve().parent
    DATA_DIR = BASE_DIR / "data"
    OUTPUT_DIR = BASE_DIR / "outputs"
    MODEL_DIR = BASE_DIR / "models"
    
    # Create directories if they don't exist
    DATA_DIR.mkdir(exist_ok=True)
    OUTPUT_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)
    
    return BASE_DIR, DATA_DIR, OUTPUT_DIR, MODEL_DIR


def load_data(data_dir: Path) -> pd.DataFrame:
    """Load the breast cancer dataset with validation."""
    data_path = data_dir / "dataset.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Dataset not found at {data_path}")
    
    logger.info(f"Loading dataset from {data_path}")
    df = pd.read_csv(data_path)
    
    logger.info(f"Dataset shape: {df.shape}")
    logger.info(f"Columns: {list(df.columns)}")
    
    return df


def validate_data(df: pd.DataFrame) -> None:
    """Validate dataset structure and contents."""
    logger.info("Validating dataset...")
    
    if df.empty:
        raise ValueError("Dataset is empty")
    
    # Check for target column
    if 'target' not in df.columns:
        raise ValueError("Target column 'target' not found in dataset")
    
    # Check for sufficient data
    if len(df) < 50:
        logger.warning(f"Dataset has only {len(df)} rows, which may be insufficient for reliable modeling")
    
    # Check for target class meanings
    unique_classes = sorted(df['target'].unique())
    logger.info(f"Target classes detected: {unique_classes}")
    
    # Note: In this dataset, the majority class (1) appears to be the positive class
    # For breast cancer, we interpret based on the data distribution
    logger.info("Class interpretation:")
    logger.info("  0 = Class 0 (minority class)")
    logger.info("  1 = Class 1 (majority class)")
    logger.info("Note: Specific medical interpretation requires domain knowledge")
    
    # Check class balance
    target_counts = df['target'].value_counts()
    logger.info(f"Target class distribution:\n{target_counts}")
    
    if len(target_counts) < 2:
        raise ValueError("Dataset must have at least 2 classes for classification")
    
    # Check for sufficient samples per class
    min_class_samples = target_counts.min()
    if min_class_samples < 10:
        logger.warning(f"Smallest class has only {min_class_samples} samples")
    
    logger.info(f"Dataset validation passed. Rows: {len(df)}, Columns: {len(df.columns)}")


def analyze_dataset(df: pd.DataFrame) -> dict:
    """Perform initial dataset analysis."""
    logger.info("Analyzing dataset characteristics...")
    
    analysis = {
        'shape': df.shape,
        'target_distribution': df['target'].value_counts().to_dict(),
        'target_percentage': (df['target'].value_counts(normalize=True) * 100).to_dict(),
        'missing_values': df.isnull().sum().sum(),
        'feature_count': len(df.columns) - 1  # Exclude target
    }
    
    logger.info(f"Dataset Analysis:")
    logger.info(f"  Shape: {analysis['shape']}")
    logger.info(f"  Target distribution: {analysis['target_distribution']}")
    logger.info(f"  Target percentage: {analysis['target_percentage']}")
    logger.info(f"  Missing values: {analysis['missing_values']}")
    logger.info(f"  Feature count: {analysis['feature_count']}")
    
    return analysis


def preprocess_data(df: pd.DataFrame) -> tuple:
    """Preprocess data for modeling."""
    logger.info("Preprocessing data...")
    
    # Separate features and target
    X = df.drop("target", axis=1).copy()
    y = df["target"].copy()
    
    logger.info(f"Features shape: {X.shape}")
    logger.info(f"Target shape: {y.shape}")
    
    return X, y


def split_data(X: pd.DataFrame, y: pd.Series, random_state: int = 42) -> tuple:
    """Split data into train and test sets with stratification."""
    logger.info("Splitting data into train and test sets...")
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=random_state,
        stratify=y  # Stratified split to maintain class balance
    )
    
    logger.info(f"Train set size: {X_train.shape[0]} samples")
    logger.info(f"Test set size: {X_test.shape[0]} samples")
    logger.info(f"Train target distribution: {y_train.value_counts().to_dict()}")
    logger.info(f"Test target distribution: {y_test.value_counts().to_dict()}")
    
    return X_train, X_test, y_train, y_test


def train_model(X_train: pd.DataFrame, y_train: pd.Series, random_state: int = 42) -> RandomForestClassifier:
    """Train Random Forest classifier."""
    logger.info("Training Random Forest classifier...")
    
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=random_state,
        n_jobs=-1  # Use all available cores
    )
    
    model.fit(X_train, y_train)
    
    logger.info("Model training completed")
    
    return model


def evaluate_model(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> dict:
    """Evaluate model performance and generate metrics."""
    logger.info("Evaluating model performance...")
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_pred_proba = model.predict_proba(X_test)[:, 1]  # Probability of positive class
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    
    # ROC-AUC (only for binary classification)
    try:
        roc_auc = roc_auc_score(y_test, y_pred_proba)
    except:
        roc_auc = None
        logger.warning("Could not calculate ROC-AUC score")
    
    # Classification report
    class_report = classification_report(y_test, y_pred, output_dict=True)
    
    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred)
    
    logger.info(f"Model Accuracy: {accuracy:.4f}")
    if roc_auc:
        logger.info(f"ROC-AUC Score: {roc_auc:.4f}")
    
    logger.info(f"\nClassification Report:\n{classification_report(y_test, y_pred)}")
    
    # Save metrics
    metrics = {
        'accuracy': accuracy,
        'roc_auc': roc_auc,
        'confusion_matrix': cm.tolist(),
        'classification_report': class_report
    }
    
    return metrics, y_pred, y_pred_proba


def create_confusion_matrix_visualization(y_test: pd.Series, y_pred: np.ndarray, output_dir: Path) -> None:
    """Create and save confusion matrix visualization."""
    logger.info("Creating confusion matrix visualization...")
    
    cm = confusion_matrix(y_test, y_pred)
    
    plt.figure(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Class 0', 'Class 1'])
    disp.plot(cmap='Blues', values_format='d')
    plt.title('Confusion Matrix - Binary Classification', fontweight='bold', fontsize=14)
    plt.tight_layout()
    
    cm_path = output_dir / "confusion_matrix.png"
    plt.savefig(cm_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Confusion matrix saved to {cm_path}")


def create_roc_curve_visualization(model: RandomForestClassifier, X_test: pd.DataFrame, y_test: pd.Series, output_dir: Path) -> None:
    """Create and save ROC curve visualization."""
    logger.info("Creating ROC curve visualization...")
    
    from sklearn.metrics import roc_curve, auc
    
    # Get predicted probabilities
    y_pred_proba = model.predict_proba(X_test)[:, 1]
    
    # Calculate ROC curve
    fpr, tpr, thresholds = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    # Plot ROC curve
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Binary Classification', fontweight='bold', fontsize=14)
    plt.legend(loc="lower right")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    
    roc_path = output_dir / "roc_curve.png"
    plt.savefig(roc_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"ROC curve saved to {roc_path}")


def create_feature_importance_visualization(model: RandomForestClassifier, feature_names: list, output_dir: Path) -> None:
    """Create and save feature importance visualization."""
    logger.info("Creating feature importance visualization...")
    
    # Get feature importances
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Select top 15 features for readability
    top_n = min(15, len(feature_names))
    top_indices = indices[:top_n]
    
    plt.figure(figsize=(10, 6))
    plt.title('Top 15 Feature Importances - Random Forest', fontweight='bold', fontsize=14)
    plt.bar(range(top_n), importances[top_indices], align='center', color='steelblue', edgecolor='black')
    plt.xticks(range(top_n), [feature_names[i] for i in top_indices], rotation=45, ha='right')
    plt.xlabel('Features', fontsize=12)
    plt.ylabel('Importance', fontsize=12)
    plt.tight_layout()
    
    importance_path = output_dir / "feature_importance.png"
    plt.savefig(importance_path, dpi=150, bbox_inches='tight')
    plt.close()
    
    logger.info(f"Feature importance saved to {importance_path}")


def save_model(model: RandomForestClassifier, model_dir: Path, model_name: str = "random_forest_classifier.joblib") -> Path:
    """Save trained model to disk."""
    logger.info("Saving trained model...")
    
    model_path = model_dir / model_name
    joblib.dump(model, model_path)
    
    logger.info(f"Model saved to {model_path}")
    
    return model_path


def save_evaluation_metrics(metrics: dict, output_dir: Path) -> None:
    """Save evaluation metrics to file."""
    logger.info("Saving evaluation metrics...")
    
    # Create a summary DataFrame
    metrics_summary = pd.DataFrame({
        'Metric': ['Accuracy', 'ROC-AUC'],
        'Value': [metrics['accuracy'], metrics['roc_auc'] if metrics['roc_auc'] else 'N/A']
    })
    
    metrics_path = output_dir / "model_metrics.csv"
    metrics_summary.to_csv(metrics_path, index=False)
    
    logger.info(f"Metrics saved to {metrics_path}")


def print_target_explanation() -> None:
    """Print explanation of target classes."""
    logger.info("\n" + "=" * 60)
    logger.info("TARGET CLASS EXPLANATION")
    logger.info("=" * 60)
    logger.info("Dataset Classes:")
    logger.info("0 = Class 0 (212 samples - 37.3%)")
    logger.info("1 = Class 1 (357 samples - 62.7%)")
    logger.info("Note: Specific class interpretation requires domain knowledge")
    logger.info("=" * 60 + "\n")


def main():
    """Main execution function."""
    try:
        logger.info("=" * 60)
        logger.info("Predictive Modeling Using Machine Learning Started")
        logger.info("=" * 60)
        
        # Setup paths
        BASE_DIR, DATA_DIR, OUTPUT_DIR, MODEL_DIR = setup_paths()
        
        # Load data
        df = load_data(DATA_DIR)
        
        # Validate data
        validate_data(df)
        
        # Analyze dataset
        analysis = analyze_dataset(df)
        
        # Explain target classes
        print_target_explanation()
        
        # Preprocess data
        X, y = preprocess_data(df)
        
        # Split data (stratified)
        RANDOM_SEED = 42
        X_train, X_test, y_train, y_test = split_data(X, y, random_state=RANDOM_SEED)
        
        # Train model
        model = train_model(X_train, y_train, random_state=RANDOM_SEED)
        
        # Evaluate model
        metrics, y_pred, y_pred_proba = evaluate_model(model, X_test, y_test, OUTPUT_DIR)
        
        # Create visualizations
        create_confusion_matrix_visualization(y_test, y_pred, OUTPUT_DIR)
        create_roc_curve_visualization(model, X_test, y_test, OUTPUT_DIR)
        create_feature_importance_visualization(model, X.columns.tolist(), OUTPUT_DIR)
        
        # Save model
        model_path = save_model(model, MODEL_DIR)
        
        # Save metrics
        save_evaluation_metrics(metrics, OUTPUT_DIR)
        
        logger.info("=" * 60)
        logger.info("Predictive Modeling Project Completed Successfully")
        logger.info("=" * 60)
        logger.info(f"Model saved to: {model_path}")
        logger.info(f"Outputs saved to: {OUTPUT_DIR}")
        logger.info(f"Random Seed Used: {RANDOM_SEED}")
        
        # Performance disclaimer
        logger.info("\n" + "!" * 60)
        logger.info("PERFORMANCE DISCLAIMER")
        logger.info("!" * 60)
        logger.info("Model performance metrics are based on a single")
        logger.info("train/test split and represent an estimate of")
        logger.info("generalization performance. For production use,")
        logger.info("consider cross-validation and external validation.")
        logger.info("This model is not production-ready.")
        logger.info("!" * 60)
        
    except Exception as e:
        logger.error(f"Error during execution: {str(e)}")
        raise


if __name__ == "__main__":
    main()