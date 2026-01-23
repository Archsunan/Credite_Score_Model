import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
import os

class GermanCreditDataLoader:
    """
    Loads and preprocesses the German Credit Dataset.
    Dataset: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)
    
    The dataset contains 1000 instances with 20 attributes (7 numerical, 13 categorical)
    and a binary target variable (Good=1, Bad=2).
    """
    
    def __init__(self):
        self.label_encoders = {}
        self.feature_mapping = self._create_feature_mapping()
        
    def _create_feature_mapping(self):
        """Define the feature names and types for the German Credit Dataset."""
        return {
            'status': 'categorical',  # Status of existing checking account
            'duration': 'numerical',  # Duration in months
            'credit_history': 'categorical',  # Credit history
            'purpose': 'categorical',  # Purpose of credit
            'amount': 'numerical',  # Credit amount
            'savings': 'categorical',  # Savings account/bonds
            'employment_since': 'categorical',  # Present employment since
            'installment_rate': 'numerical',  # Installment rate in percentage of disposable income
            'personal_status_sex': 'categorical',  # Personal status and sex
            'other_debtors': 'categorical',  # Other debtors / guarantors
            'residence_since': 'numerical',  # Present residence since
            'property': 'categorical',  # Property
            'age': 'numerical',  # Age in years
            'other_installment_plans': 'categorical',  # Other installment plans
            'housing': 'categorical',  # Housing
            'num_existing_credits': 'numerical',  # Number of existing credits at this bank
            'job': 'categorical',  # Job
            'num_people_liable': 'numerical',  # Number of people being liable to provide maintenance
            'telephone': 'categorical',  # Telephone
            'foreign_worker': 'categorical',  # Foreign worker
        }
    
    def download_data(self, force_download=False):
        """
        Download the German Credit Dataset from UCI repository.
        
        Args:
            force_download: If True, re-download even if file exists
            
        Returns:
            Path to the downloaded data file
        """
        import urllib.request
        
        data_dir = 'data'
        os.makedirs(data_dir, exist_ok=True)
        
        data_path = os.path.join(data_dir, 'german.data')
        
        if os.path.exists(data_path) and not force_download:
            print(f"Dataset already exists at {data_path}")
            return data_path
        
        url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/statlog/german/german.data'
        
        print(f"Downloading German Credit Dataset from {url}...")
        try:
            urllib.request.urlretrieve(url, data_path)
            print(f"Dataset downloaded successfully to {data_path}")
        except Exception as e:
            print(f"Error downloading dataset: {e}")
            print("You can manually download from: https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data)")
            raise
        
        return data_path
    
    def load_data(self, data_path=None):
        """
        Load the German Credit Dataset from file.
        
        Args:
            data_path: Path to the data file. If None, will attempt to download
            
        Returns:
            DataFrame with the loaded data
        """
        if data_path is None:
            data_path = self.download_data()
        
        # Define column names
        columns = list(self.feature_mapping.keys()) + ['target']
        
        # Load the data (space-separated)
        df = pd.read_csv(data_path, sep=' ', header=None, names=columns)
        
        # Convert target: 1=Good, 2=Bad -> 1=Good, 0=Bad
        df['target'] = (df['target'] == 1).astype(int)
        
        print(f"Loaded {len(df)} samples with {len(columns)-1} features")
        print(f"Target distribution:\n{df['target'].value_counts()}")
        
        return df
    
    def preprocess_data(self, df):
        """
        Preprocess the German Credit Dataset.
        
        Args:
            df: Raw DataFrame
            
        Returns:
            Preprocessed DataFrame
        """
        df = df.copy()
        
        # Encode categorical features
        categorical_features = [col for col, type_ in self.feature_mapping.items() 
                              if type_ == 'categorical']
        
        for col in categorical_features:
            if col in df.columns:
                le = LabelEncoder()
                df[col] = le.fit_transform(df[col].astype(str))
                self.label_encoders[col] = le
        
        return df
    
    def map_to_credit_score_categories(self, df):
        """
        Map the binary target to multi-class credit score categories.
        Creates balanced and realistic credit scores using percentile-based mapping.
        
        Args:
            df: DataFrame with 'target' column (0=Bad, 1=Good)
            
        Returns:
            DataFrame with 'credit_score' column
        """
        df = df.copy()
        
        # Calculate comprehensive risk score based on multiple factors
        amount_norm = (df['amount'] - df['amount'].min()) / (df['amount'].max() - df['amount'].min())
        duration_norm = (df['duration'] - df['duration'].min()) / (df['duration'].max() - df['duration'].min())
        age_norm = (df['age'] - df['age'].min()) / (df['age'].max() - df['age'].min())
        installment_norm = (df['installment_rate'] / 4.0).clip(upper=1)
        
        # Comprehensive risk score with balanced weights
        risk_score = (
            0.30 * amount_norm +              # Loan amount impact
            0.25 * duration_norm +            # Duration impact  
            -0.20 * age_norm +                # Age benefit (older = more stable)
            0.15 * (1 - df['target']) * 2 +   # Credit history (amplified)
            0.10 * installment_norm           # Payment burden
        )
        
        # Add controlled randomness
        np.random.seed(42)
        noise = np.random.normal(0, 0.08, len(df))
        risk_score = (risk_score + noise).clip(0, 1)
        
        # Separate good and bad credit applicants
        good_credits = df['target'] == 1
        bad_credits = df['target'] == 0
        
        # Initialize credit_score column
        df['credit_score'] = 'Fair'
        
        # For GOOD credit history (target=1) - 700 samples
        # Distribute as: ~30% Excellent, ~45% Good, ~25% Fair
        good_risk = risk_score[good_credits]
        good_idx = df[good_credits].index
        
        excellent_thresh = good_risk.quantile(0.30)
        good_thresh = good_risk.quantile(0.75)
        
        df.loc[good_idx[good_risk <= excellent_thresh], 'credit_score'] = 'Excellent'
        df.loc[good_idx[(good_risk > excellent_thresh) & (good_risk <= good_thresh)], 'credit_score'] = 'Good'
        df.loc[good_idx[good_risk > good_thresh], 'credit_score'] = 'Fair'
        
        # For BAD credit history (target=0) - 300 samples  
        # Distribute as: ~10% Fair, ~50% Poor
        bad_risk = risk_score[bad_credits]
        bad_idx = df[bad_credits].index
        
        fair_thresh = bad_risk.quantile(0.20)  # Top 20% of bad credits get Fair
        
        df.loc[bad_idx[bad_risk <= fair_thresh], 'credit_score'] = 'Fair'
        df.loc[bad_idx[bad_risk > fair_thresh], 'credit_score'] = 'Poor'
        
        # Drop the binary target
        df = df.drop('target', axis=1)
        
        return df
    
    def prepare_for_model(self, df):
        """
        Prepare the German Credit Dataset for your existing model pipeline.
        Maps German Credit features to your model's expected features.
        
        Args:
            df: Preprocessed German Credit DataFrame
            
        Returns:
            DataFrame with features matching your model's expectations
        """
        # Extract and map features to your model's expected format
        model_df = pd.DataFrame()
        
        # Direct mappings
        model_df['age'] = df['age']
        model_df['loan_amount'] = df['amount']
        model_df['loan_term'] = df['duration']
        
        # Derived features (approximations based on available data)
        # Income: estimate based on credit amount and installment rate
        model_df['income'] = df['amount'] / (df['installment_rate'] / 100) / df['duration'] * 12
        
        # Employment length: map from employment_since categories
        employment_mapping = {0: 0, 1: 1, 2: 4, 3: 7, 4: 10}
        model_df['employment_length'] = df['employment_since'].map(employment_mapping).fillna(5)
        
        # Credit history length: approximate from age (assume credit started at age 18-25)
        model_df['credit_history_length'] = (df['age'] - 20).clip(lower=0)
        
        # Number of credit lines
        model_df['num_credit_lines'] = df['num_existing_credits']
        
        # Debt to income ratio: approximate
        model_df['debt_to_income'] = (df['amount'] / model_df['income']).clip(upper=1.0)
        
        # Delinquencies: map from credit_history (higher values = worse history)
        delinquency_map = {0: 4, 1: 3, 2: 1, 3: 0, 4: 0}
        model_df['num_delinquencies'] = df['credit_history'].map(
            lambda x: delinquency_map.get(x, 0) if x in delinquency_map else 0
        )
        
        # Number of inquiries: approximate from purpose and other_installment_plans
        model_df['num_inquiries'] = df['other_installment_plans'].clip(upper=5)
        
        # Add the target
        if 'credit_score' in df.columns:
            model_df['credit_score'] = df['credit_score']
        
        return model_df
    
    def load_and_prepare(self, data_path=None, as_multi_class=True):
        """
        Complete pipeline: download, load, preprocess, and prepare data.
        
        Args:
            data_path: Path to data file (optional)
            as_multi_class: If True, convert to multi-class credit scores
            
        Returns:
            DataFrame ready for your model
        """
        print("="*60)
        print("German Credit Dataset Loader")
        print("="*60)
        
        # Load raw data
        df = self.load_data(data_path)
        
        # Preprocess
        print("\nPreprocessing data...")
        df = self.preprocess_data(df)
        
        # Convert to multi-class if needed
        if as_multi_class:
            print("Converting to multi-class credit scores...")
            df = self.map_to_credit_score_categories(df)
            print(f"Credit score distribution:\n{df['credit_score'].value_counts()}")
        
        # Map to model features
        print("\nMapping to model features...")
        model_df = self.prepare_for_model(df)
        
        print(f"\nFinal dataset shape: {model_df.shape}")
        print(f"Features: {list(model_df.columns)}")
        
        print("="*60)
        return model_df


if __name__ == "__main__":
    # Example usage
    loader = GermanCreditDataLoader()
    data = loader.load_and_prepare()
    
    print("\nSample data:")
    print(data.head())
    
    print("\nData statistics:")
    print(data.describe())
