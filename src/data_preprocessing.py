import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

class DataPreprocessor:
    """Handles data preprocessing and feature engineering for credit scoring."""
    
    def __init__(self):
        self.scaler = StandardScaler()
        self.feature_names = [
            'age', 'income', 'employment_length', 'loan_amount', 
            'loan_term', 'credit_history_length', 'num_credit_lines',
            'debt_to_income', 'num_delinquencies', 'num_inquiries'
        ]
    
    def generate_synthetic_data(self, n_samples=10000):
        """
        Generate synthetic credit data for training.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic credit data
        """
        np.random.seed(42)
        
        # Generate features
        age = np.random.normal(40, 12, n_samples).clip(18, 75)
        income = np.random.lognormal(10.8, 0.6, n_samples).clip(20000, 200000)
        employment_length = np.random.exponential(5, n_samples).clip(0, 40)
        loan_amount = np.random.lognormal(10, 0.8, n_samples).clip(5000, 100000)
        loan_term = np.random.choice([12, 24, 36, 48, 60], n_samples, p=[0.1, 0.2, 0.4, 0.2, 0.1])
        credit_history_length = np.random.exponential(8, n_samples).clip(0, 50)
        num_credit_lines = np.random.poisson(4, n_samples).clip(0, 20)
        debt_to_income = np.random.beta(2, 5, n_samples).clip(0, 1)
        num_delinquencies = np.random.poisson(0.5, n_samples).clip(0, 10)
        num_inquiries = np.random.poisson(1.5, n_samples).clip(0, 15)
        
        # Create DataFrame
        data = pd.DataFrame({
            'age': age,
            'income': income,
            'employment_length': employment_length,
            'loan_amount': loan_amount,
            'loan_term': loan_term,
            'credit_history_length': credit_history_length,
            'num_credit_lines': num_credit_lines,
            'debt_to_income': debt_to_income,
            'num_delinquencies': num_delinquencies,
            'num_inquiries': num_inquiries
        })
        
        # Generate target based on features (credit score category)
        # Calculate a risk score based on features
        risk_score = (
            -0.01 * data['age'] +
            -0.00002 * data['income'] +
            -0.05 * data['employment_length'] +
            0.00003 * data['loan_amount'] +
            0.01 * data['loan_term'] +
            -0.1 * data['credit_history_length'] +
            -0.05 * data['num_credit_lines'] +
            2.0 * data['debt_to_income'] +
            0.4 * data['num_delinquencies'] +
            0.15 * data['num_inquiries']
        )
        
        # Add some noise
        risk_score += np.random.normal(0, 0.3, n_samples)
        
        # Convert to categories
        credit_score = pd.cut(
            risk_score, 
            bins=[-np.inf, -0.3, 0.3, 0.8, np.inf],
            labels=['Excellent', 'Good', 'Fair', 'Poor']
        )
        
        data['credit_score'] = credit_score
        
        return data
    
    def engineer_features(self, df):
        """
        Create additional features from existing ones.
        
        Args:
            df: Input DataFrame
            
        Returns:
            DataFrame with engineered features
        """
        df = df.copy()
        
        # Ratio features
        df['loan_to_income'] = df['loan_amount'] / df['income']
        df['credit_lines_per_year'] = df['num_credit_lines'] / (df['credit_history_length'] + 1)
        df['age_income_interaction'] = df['age'] * np.log1p(df['income'])
        
        # Risk indicators
        df['high_debt'] = (df['debt_to_income'] > 0.4).astype(int)
        df['has_delinquencies'] = (df['num_delinquencies'] > 0).astype(int)
        df['many_inquiries'] = (df['num_inquiries'] > 3).astype(int)
        
        return df
    
    def prepare_data(self, df, fit_scaler=False):
        """
        Prepare data for modeling by engineering features and scaling.
        
        Args:
            df: Input DataFrame
            fit_scaler: Whether to fit the scaler (True for training data)
            
        Returns:
            Tuple of (X, y) where X is feature matrix and y is target
        """
        df = self.engineer_features(df)
        
        # Separate features and target
        if 'credit_score' in df.columns:
            X = df.drop('credit_score', axis=1)
            y = df['credit_score']
        else:
            X = df
            y = None
        
        # Scale features
        if fit_scaler:
            X_scaled = self.scaler.fit_transform(X)
        else:
            X_scaled = self.scaler.transform(X)
        
        X_scaled = pd.DataFrame(X_scaled, columns=X.columns, index=X.index)
        
        return X_scaled, y
    
    def split_data(self, X, y, test_size=0.2, random_state=42):
        """
        Split data into training and testing sets.
        
        Args:
            X: Feature matrix
            y: Target variable
            test_size: Proportion of data for testing
            random_state: Random seed
            
        Returns:
            Tuple of (X_train, X_test, y_train, y_test)
        """
        return train_test_split(X, y, test_size=test_size, 
                                random_state=random_state, stratify=y)
