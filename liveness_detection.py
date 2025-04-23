import pandas as pd
import numpy as np
from scipy.io.arff import loadarff 
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.ensemble import IsolationForest

def load_keystroke_data(filepath):
    """Load dataset with correct column handling"""
    data, meta = loadarff(filepath)
    df = pd.DataFrame(data)
    
    # Convert byte strings to regular strings
    str_cols = df.select_dtypes([object]).columns
    df[str_cols] = df[str_cols].apply(lambda x: x.str.decode('utf-8'))
    
    return df

def prepare_features(df):
    """Create features matching paper's methodology"""
    # Get all HT and FT columns
    ht_cols = [col for col in df.columns if col.startswith('HT_')]
    ft_cols = [col for col in df.columns if col.startswith('FT_')]
    
    # Calculate summary statistics
    features = pd.DataFrame()
    features['hold_time_mean'] = df[ht_cols].mean(axis=1)
    features['flight_time_var'] = df[ft_cols].var(axis=1)
    
    # Calculate CDF-based features (simplified implementation)
    for seq_type in ['HT', 'FT']:
        cols = [col for col in df.columns if col.startswith(f'{seq_type}_')]
        for i in range(len(cols)):
            # Empirical CDF calculation
            sorted_vals = np.sort(df[cols[i]].dropna())
            cdf = np.searchsorted(sorted_vals, df[cols[i]], side='right') / len(sorted_vals)
            
            # Distance metrics
            features[f'{seq_type}_cdf_manhattan_{i}'] = np.abs(cdf - cdf.mean())
            features[f'{seq_type}_cdf_euclidean_{i}'] = (cdf - cdf.mean())**2
            features[f'{seq_type}_cdf_canberra_{i}'] = np.abs(cdf - cdf.mean()) / (np.abs(cdf) + np.abs(cdf.mean()))
    
    return features

def train_detector(X, y):
    """Modified training pipeline with class checks"""
    # Check for class balance
    unique_classes, counts = np.unique(y, return_counts=True)
    
    if len(unique_classes) < 2:
        print("Insufficient class variety - using anomaly detection")
        # Use Isolation Forest for single-class scenario
        model = make_pipeline(
            StandardScaler(),
            IsolationForest(contamination=0.1, random_state=42)
        )
        X_train, X_test = train_test_split(X, test_size=0.3, random_state=42)
        model.fit(X_train)
        return model, X_test, pd.Series([0]*len(X_test)), True
    else:
        # Ensure stratified split maintains class balance
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=0.3, 
            random_state=42,
            stratify=y,
            shuffle=True
        )
        
        # Use paper's recommended SVM configuration
        model = make_pipeline(
            StandardScaler(),
            SVC(
                kernel='poly', 
                degree=2,
                C=1.0,
                class_weight='balanced',
                probability=True,
                random_state=42
            )
        )
        
        model.fit(X_train, y_train)
        return model, X_test, y_test, False


# Updated usage example
if __name__ == "__main__":
    # Load dataset
    df = load_keystroke_data("")
    
    # Filter outliers on individual HT/FT columns
    ht_cols = [col for col in df.columns if col.startswith('HT_')]
    ft_cols = [col for col in df.columns if col.startswith('FT_')]
    df = df[(df[ht_cols] <= 1500).all(axis=1) & (df[ft_cols] <= 1500).all(axis=1)]
    
    print(df['RESULT'].value_counts())

    
    # Prepare features and labels
    X = prepare_features(df)
    y = df['RESULT'].map({'legitimate': 0, 'impostor': 1})
    
    # Train model
    model, X_test, y_test, is_anamoly = train_detector(X, y)
    
    # Evaluate
    y_pred = model.predict(X_test)
    print(classification_report(y_test, y_pred))
