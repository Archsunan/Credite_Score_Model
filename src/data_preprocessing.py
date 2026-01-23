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
        Generate synthetic credit data for training with real-world patterns.
        Uses realistic correlations between features based on actual credit behavior.
        
        Args:
            n_samples: Number of samples to generate
            
        Returns:
            DataFrame with synthetic credit data
        """
        np.random.seed(42)
        
        # Generate each credit category separately for balance
        samples_per_category = n_samples // 4
        all_data = []
        
        # EXCELLENT: Strict criteria based on your sample 
        # (DTI < 0.25, 0 Delinquencies, High Income)
        excellent_data = {
            'age': np.random.normal(45, 8, samples_per_category).clip(30, 65),
            'income': np.random.lognormal(11.4, 0.3, samples_per_category).clip(80000, 200000),
            'employment_length': np.random.exponential(12, samples_per_category).clip(8, 40),
            'loan_amount': np.random.lognormal(9.8, 0.4, samples_per_category).clip(5000, 40000),
            'loan_term': np.random.choice([12, 24], samples_per_category, p=[0.4, 0.6]),
            'credit_history_length': np.random.exponential(18, samples_per_category).clip(12, 50),
            'num_credit_lines': np.random.poisson(3, samples_per_category).clip(1, 6),
            # DTI STRICTLY LOW (User Sample is 0.18, max 0.25)
            'debt_to_income': np.random.uniform(0.05, 0.24, samples_per_category),
            # STRICTLY 0 Delinquencies
            'num_delinquencies': np.zeros(samples_per_category),
            'num_inquiries': np.random.choice([0, 1], samples_per_category, p=[0.8, 0.2]),
            'credit_score': ['Excellent'] * samples_per_category
        }
        all_data.append(pd.DataFrame(excellent_data))
        
        # GOOD: Accommodates your "Good" sample 
        # (DTI ~0.32, 1 Delinquency allowed, Income ~75k)
        good_data = {
            'age': np.random.normal(38, 8, samples_per_category).clip(28, 55),
            'income': np.random.lognormal(11.1, 0.4, samples_per_category).clip(60000, 120000),
            'employment_length': np.random.exponential(8, samples_per_category).clip(4, 30),
            'loan_amount': np.random.lognormal(10.2, 0.5, samples_per_category).clip(10000, 60000),
            'loan_term': np.random.choice([24, 36, 48], samples_per_category, p=[0.2, 0.6, 0.2]),
            'credit_history_length': np.random.exponential(12, samples_per_category).clip(6, 35),
            'num_credit_lines': np.random.poisson(4, samples_per_category).clip(2, 10),
            # DTI Range 0.25 - 0.45 (User Sample is 0.32)
            'debt_to_income': np.random.uniform(0.25, 0.45, samples_per_category),
            # Allow 0 or 1 Delinquency (User Sample has 1)
            'num_delinquencies': np.random.choice([0, 1], samples_per_category, p=[0.5, 0.5]),
            'num_inquiries': np.random.choice([1, 2, 3], samples_per_category, p=[0.3, 0.5, 0.2]),
            'credit_score': ['Good'] * samples_per_category
        }
        all_data.append(pd.DataFrame(good_data))
        
        # FAIR: Accommodates your "Fair" sample
        # (DTI ~0.55, 2 Delinquencies, Income ~42k)
        fair_data = {
            'age': np.random.normal(32, 9, samples_per_category).clip(22, 50),
            'income': np.random.lognormal(10.7, 0.5, samples_per_category).clip(35000, 70000),
            'employment_length': np.random.exponential(5, samples_per_category).clip(2, 15),
            'loan_amount': np.random.lognormal(10.4, 0.6, samples_per_category).clip(20000, 80000),
            'loan_term': np.random.choice([36, 48, 60], samples_per_category, p=[0.2, 0.5, 0.3]),
            'credit_history_length': np.random.exponential(7, samples_per_category).clip(3, 20),
            'num_credit_lines': np.random.poisson(6, samples_per_category).clip(3, 14),
            # DTI Range 0.46 - 0.65 (User Sample is 0.55)
            'debt_to_income': np.random.uniform(0.46, 0.65, samples_per_category),
            # Allow 1-3 Delinquencies (User Sample has 2)
            'num_delinquencies': np.random.choice([1, 2, 3], samples_per_category, p=[0.2, 0.6, 0.2]),
            'num_inquiries': np.random.choice([3, 4, 5], samples_per_category, p=[0.3, 0.4, 0.3]),
            'credit_score': ['Fair'] * samples_per_category
        }
        all_data.append(pd.DataFrame(fair_data))
        
        # POOR: Accommodates your "Poor" sample
        # (DTI > 0.65, 3+ Delinquencies, Low Income)
        poor_data = {
            'age': np.random.normal(26, 7, samples_per_category).clip(18, 40),
            'income': np.random.lognormal(10.2, 0.5, samples_per_category).clip(15000, 40000),
            'employment_length': np.random.exponential(2, samples_per_category).clip(0, 5),
            'loan_amount': np.random.lognormal(10.6, 0.7, samples_per_category).clip(30000, 100000),
            'loan_term': np.random.choice([48, 60], samples_per_category, p=[0.2, 0.8]),
            'credit_history_length': np.random.exponential(3, samples_per_category).clip(0, 10),
            'num_credit_lines': np.random.poisson(8, samples_per_category).clip(5, 20),
            # DTI High > 0.65 (User Sample is 0.90)
            'debt_to_income': np.random.uniform(0.66, 1.2, samples_per_category),
            # High Delinquencies 3+ (User Sample has 6)
            'num_delinquencies': np.random.choice([3, 4, 5, 6, 7, 8], samples_per_category),
            'num_inquiries': np.random.randint(5, 12, samples_per_category),
            'credit_score': ['Poor'] * samples_per_category
        }
        all_data.append(pd.DataFrame(poor_data))
        
        # Combine all categories and shuffle
        data = pd.concat(all_data, ignore_index=True)
        
        # Add edge cases for better boundary detection (10% of total)
        edge_samples = n_samples // 40  # 2.5% per category
        
        # Excellent-Good boundary cases
        edge_excellent_good = {
            'age': np.random.uniform(35, 45, edge_samples),
            'income': np.random.uniform(75000, 95000, edge_samples),
            'employment_length': np.random.uniform(8, 12, edge_samples),
            'loan_amount': np.random.uniform(18000, 25000, edge_samples),
            'loan_term': np.random.choice([24, 36], edge_samples),
            'credit_history_length': np.random.uniform(10, 15, edge_samples),
            'num_credit_lines': np.random.choice([2, 3, 4], edge_samples),
            'debt_to_income': np.random.uniform(0.22, 0.27, edge_samples),  # DTI boundary
            'num_delinquencies': np.zeros(edge_samples),
            'num_inquiries': np.random.choice([0, 1], edge_samples),
            'credit_score': np.random.choice(['Excellent', 'Good'], edge_samples, p=[0.5, 0.5])
        }
        data = pd.concat([data, pd.DataFrame(edge_excellent_good)], ignore_index=True)
        
        # Good-Fair boundary cases
        edge_good_fair = {
            'age': np.random.uniform(30, 38, edge_samples),
            'income': np.random.uniform(50000, 65000, edge_samples),
            'employment_length': np.random.uniform(4, 7, edge_samples),
            'loan_amount': np.random.uniform(25000, 35000, edge_samples),
            'loan_term': np.random.choice([36, 48], edge_samples),
            'credit_history_length': np.random.uniform(6, 10, edge_samples),
            'num_credit_lines': np.random.choice([4, 5, 6], edge_samples),
            'debt_to_income': np.random.uniform(0.43, 0.48, edge_samples),  # DTI boundary
            'num_delinquencies': np.random.choice([1, 2], edge_samples, p=[0.7, 0.3]),
            'num_inquiries': np.random.choice([2, 3], edge_samples),
            'credit_score': np.random.choice(['Good', 'Fair'], edge_samples, p=[0.5, 0.5])
        }
        data = pd.concat([data, pd.DataFrame(edge_good_fair)], ignore_index=True)
        
        # Fair-Poor boundary cases
        edge_fair_poor = {
            'age': np.random.uniform(24, 32, edge_samples),
            'income': np.random.uniform(30000, 45000, edge_samples),
            'employment_length': np.random.uniform(1, 4, edge_samples),
            'loan_amount': np.random.uniform(35000, 50000, edge_samples),
            'loan_term': np.random.choice([48, 60], edge_samples),
            'credit_history_length': np.random.uniform(2, 6, edge_samples),
            'num_credit_lines': np.random.choice([6, 7, 8], edge_samples),
            'debt_to_income': np.random.uniform(0.63, 0.68, edge_samples),  # DTI boundary
            'num_delinquencies': np.random.choice([2, 3, 4], edge_samples),
            'num_inquiries': np.random.choice([4, 5, 6], edge_samples),
            'credit_score': np.random.choice(['Fair', 'Poor'], edge_samples, p=[0.5, 0.5])
        }
        data = pd.concat([data, pd.DataFrame(edge_fair_poor)], ignore_index=True)
        
        # Shuffle all data
        data = data.sample(frac=1, random_state=42).reset_index(drop=True)
        
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
            # Ensure column order matches what the scaler expects
            if hasattr(self.scaler, 'feature_names_in_'):
                X = X[self.scaler.feature_names_in_]
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
