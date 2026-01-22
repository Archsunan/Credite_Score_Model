import os
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from data_preprocessing import DataPreprocessor
from model import CreditScoreModel

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

def main():
    """Main training pipeline."""
    print("=" * 60)
    print("Credit Score Model Training Pipeline")
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

if __name__ == "__main__":
    main()
