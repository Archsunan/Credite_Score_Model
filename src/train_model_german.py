import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from model import CreditScoreModel
from load_german_credit import GermanCreditDataLoader

def plot_feature_importance(model, save_path='models/feature_importance.png'):
    """Plot and save feature importance chart."""
    importance_df = model.get_feature_importance()
    
    plt.figure(figsize=(10, 8))
    sns.barplot(data=importance_df.head(15), x='importance', y='feature')
    plt.title('Top 15 Feature Importances')
    plt.xlabel('Importance Score')
    plt.tight_layout()
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    plt.savefig(save_path)
    print(f"Feature importance plot saved to {save_path}")
    plt.close()

def plot_confusion_matrix(cm, classes, save_path='models/confusion_matrix.png'):
    """Plot and save confusion matrix."""
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    
    plt.savefig(save_path)
    print(f"Confusion matrix plot saved to {save_path}")
    plt.close()

def train_with_german_credit():
    """Training pipeline using German Credit Dataset."""
    print("=" * 60)
    print("Credit Score Model Training - German Credit Dataset")
    print("=" * 60)
    
    # Step 1: Load German Credit Dataset
    print("\n[1/7] Loading German Credit Dataset...")
    loader = GermanCreditDataLoader()
    data = loader.load_and_prepare(as_multi_class=True)
    print(f"Loaded {len(data)} samples")
    print(f"Class distribution:\n{data['credit_score'].value_counts()}")
    
    # Step 2: Prepare data
    print("\n[2/7] Preprocessing and feature engineering...")
    preprocessor = DataPreprocessor()
    X, y = preprocessor.prepare_data(data, fit_scaler=True)
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 3: Split data
    print("\n[3/7] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Apply SMOTE for class balancing
    print("\n[4/7] Applying SMOTE for class balancing...")
    from imblearn.over_sampling import SMOTE
    from sklearn.preprocessing import LabelEncoder
    
    # Encode labels for SMOTE
    le = LabelEncoder()
    y_train_encoded = le.fit_transform(y_train)
    
    smote = SMOTE(random_state=42, k_neighbors=3)
    X_train_balanced, y_train_balanced = smote.fit_resample(X_train, y_train_encoded)
    
    # Decode labels back
    y_train_balanced = le.inverse_transform(y_train_balanced)
    
    print(f"After SMOTE: {len(X_train_balanced)} samples")
    print(f"Balanced class distribution:")
    import pandas as pd
    print(pd.Series(y_train_balanced).value_counts())
    
    # Step 5: Train model
    print("\n[5/7] Training XGBoost model...")
    model = CreditScoreModel()
    model.train(X_train_balanced, y_train_balanced, X_test, y_test)
    
    # Step 6: Evaluate model
    print("\n[6/7] Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    # Step 7: Save model and visualizations
    print("\n[7/7] Saving model and visualizations...")
    model.save('models/credit_model.pkl')
    
    # Save preprocessor
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Create visualizations
    plot_feature_importance(model)
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        model.classes
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nModel Performance Summary:")
    print(f"Accuracy: {metrics['accuracy']:.2%}")
    if metrics['roc_auc'] is not None:
        print(f"ROC AUC: {metrics['roc_auc']:.2%}")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server: python src/api.py")
    print("2. Open web/index.html in your browser")
    print("=" * 60)

def train_with_synthetic_data():
    """Original training pipeline using synthetic data."""
    print("=" * 60)
    print("Credit Score Model Training - Synthetic Data")
    print("=" * 60)
    
    # Step 1: Generate synthetic data
    print("\n[1/6] Generating synthetic data...")
    preprocessor = DataPreprocessor()
    data = preprocessor.generate_synthetic_data(n_samples=10000)
    print(f"Generated {len(data)} samples")
    print(f"Class distribution:\n{data['credit_score'].value_counts()}")
    
    # Step 2: Prepare data
    print("\n[2/6] Preprocessing and feature engineering...")
    X, y = preprocessor.prepare_data(data, fit_scaler=True)
    print(f"Feature matrix shape: {X.shape}")
    
    # Step 3: Split data
    print("\n[3/6] Splitting data into train/test sets...")
    X_train, X_test, y_train, y_test = preprocessor.split_data(X, y, test_size=0.2)
    print(f"Training set: {len(X_train)} samples")
    print(f"Test set: {len(X_test)} samples")
    
    # Step 4: Train model
    print("\n[4/6] Training XGBoost model...")
    model = CreditScoreModel()
    model.train(X_train, y_train, X_test, y_test)
    
    # Step 5: Evaluate model
    print("\n[5/6] Evaluating model performance...")
    metrics = model.evaluate(X_test, y_test)
    
    # Step 6: Save model and visualizations
    print("\n[6/6] Saving model and visualizations...")
    model.save('models/credit_model.pkl')
    
    # Save preprocessor
    import joblib
    os.makedirs('models', exist_ok=True)
    joblib.dump(preprocessor, 'models/preprocessor.pkl')
    print("Preprocessor saved to models/preprocessor.pkl")
    
    # Create visualizations
    plot_feature_importance(model)
    plot_confusion_matrix(
        metrics['confusion_matrix'], 
        model.classes
    )
    
    print("\n" + "=" * 60)
    print("Training completed successfully!")
    print("=" * 60)
    print("\nNext steps:")
    print("1. Start the API server: python src/api.py")
    print("2. Open web/index.html in your browser")
    print("=" * 60)

def main():
    """Main training pipeline with dataset selection."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Train Credit Score Model')
    parser.add_argument(
        '--dataset',
        type=str,
        choices=['german', 'synthetic'],
        default='german',
        help='Dataset to use for training (default: german)'
    )
    
    args = parser.parse_args()
    
    if args.dataset == 'german':
        train_with_german_credit()
    else:
        train_with_synthetic_data()

if __name__ == "__main__":
    main()
