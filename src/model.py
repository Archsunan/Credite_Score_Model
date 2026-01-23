import numpy as np
import pandas as pd
from xgboost import XGBClassifier
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score
from sklearn.preprocessing import LabelEncoder
import joblib
import os

class CreditScoreModel:
    """Credit score prediction model using XGBoost."""
    
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=500,
            max_depth=7,
            learning_rate=0.03,
            subsample=0.85,
            colsample_bytree=0.85,
            min_child_weight=3,
            gamma=0.2,
            reg_alpha=0.05,
            reg_lambda=1.5,
            random_state=42,
            eval_metric='mlogloss',
            tree_method='hist',
            scale_pos_weight=1
        )
        self.label_encoder = LabelEncoder()
        self.classes = ['Excellent', 'Good', 'Fair', 'Poor']
        self.risk_mapping = {
            'Excellent': 'Very Low',
            'Good': 'Low',
            'Fair': 'Moderate',
            'Poor': 'High'
        }
    
    def train(self, X_train, y_train, X_val=None, y_val=None):
        """
        Train the credit score model.
        
        Args:
            X_train: Training features
            y_train: Training target
            X_val: Validation features (optional)
            y_val: Validation target (optional)
        """
        # Encode labels
        y_train_encoded = self.label_encoder.fit_transform(y_train)
        
        if X_val is not None and y_val is not None:
            y_val_encoded = self.label_encoder.transform(y_val)
            eval_set = [(X_train, y_train_encoded), (X_val, y_val_encoded)]
            self.model.fit(
                X_train, y_train_encoded,
                eval_set=eval_set,
                verbose=False
            )
        else:
            self.model.fit(X_train, y_train_encoded)
        
        print("Model training completed!")
    
    def predict(self, X):
        """
        Predict credit score categories.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of predicted categories
        """
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
    
    def predict_proba(self, X):
        """
        Predict probability distributions for each class.
        
        Args:
            X: Feature matrix
            
        Returns:
            Array of probability distributions
        """
        return self.model.predict_proba(X)
    
    def predict_single(self, features_dict):
        """
        Predict credit score for a single applicant.
        
        Args:
            features_dict: Dictionary of feature values
            
        Returns:
            Dictionary with prediction, probability, and risk level
        """
        # Convert to DataFrame
        df = pd.DataFrame([features_dict])
        
        # Get prediction and probabilities
        prediction = self.predict(df)[0]
        probabilities = self.predict_proba(df)[0]
        
        # Get class names from label encoder
        class_names = self.label_encoder.classes_
        
        # Get the probability of the predicted class
        pred_idx = list(class_names).index(prediction)
        confidence = probabilities[pred_idx]
        
        return {
            'credit_score': prediction,
            'probability': float(confidence),
            'risk_level': self.risk_mapping[prediction],
            'all_probabilities': {
                cls: float(prob) 
                for cls, prob in zip(class_names, probabilities)
            }
        }
    
    def evaluate(self, X_test, y_test):
        """
        Evaluate model performance.
        
        Args:
            X_test: Test features
            y_test: Test target
            
        Returns:
            Dictionary of evaluation metrics
        """
        y_pred = self.predict(X_test)
        y_proba = self.predict_proba(X_test)
        
        # Classification report
        print("\nClassification Report:")
        print(classification_report(y_test, y_pred))
        
        # Confusion matrix
        print("\nConfusion Matrix:")
        cm = confusion_matrix(y_test, y_pred, labels=self.classes)
        print(cm)
        
        # ROC AUC (one-vs-rest)
        try:
            roc_auc = roc_auc_score(y_test, y_proba, multi_class='ovr', average='weighted')
            print(f"\nWeighted ROC AUC: {roc_auc:.4f}")
        except Exception as e:
            print(f"Could not calculate ROC AUC: {e}")
            roc_auc = None
        
        # Accuracy
        accuracy = (y_pred == y_test).mean()
        print(f"Accuracy: {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'roc_auc': roc_auc,
            'confusion_matrix': cm
        }
    
    def get_feature_importance(self):
        """
        Get feature importance scores.
        
        Returns:
            DataFrame with feature names and importance scores
        """
        importance = self.model.feature_importances_
        feature_names = self.model.get_booster().feature_names
        
        df = pd.DataFrame({
            'feature': feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)
        
        return df
    
    def save(self, filepath='models/credit_model.pkl'):
        """
        Save the trained model to disk.
        
        Args:
            filepath: Path to save the model
        """
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(self, filepath)
        print(f"Model saved to {filepath}")
    
    @staticmethod
    def load(filepath='models/credit_model.pkl'):
        """
        Load a trained model from disk.
        
        Args:
            filepath: Path to the saved model
            
        Returns:
            Loaded CreditScoreModel instance
        """
        model = joblib.load(filepath)
        print(f"Model loaded from {filepath}")
        return model
