import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


def handle_missing_values(df):
    """
    Handle missing values in the dataset.
    For numerical columns: fill with median
    For categorical columns: fill with mode
    """
    df = df.copy()
    
    # Separate numerical and categorical columns
    numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    # Remove target variable from numerical columns if present
    if 'is_churned' in numerical_cols:
        numerical_cols.remove('is_churned')
    
    # # Fill numerical missing values with median
    # for col in numerical_cols:
    #     if df[col].isnull().sum() > 0:
    #         median_val = df[col].median()
    #         df[col] = df[col].fillna(median_val)
    
    # # Fill categorical missing values with mode
    # for col in categorical_cols:
    #     if df[col].isnull().sum() > 0:
    #         mode_val = df[col].mode()[0] if not df[col].mode().empty else 'Unknown'
    #         df[col] = df[col].fillna(mode_val)

    df[numerical_cols] = df[numerical_cols].fillna(df[numerical_cols].median())
    df[categorical_cols] = df[categorical_cols].fillna(df[categorical_cols].mode().iloc[0])
    
    return df


def handle_outliers(df, numerical_cols, method='iqr'):
    """
    Handle outliers using IQR method.
    Replace outliers with upper/lower bounds instead of removing them.
    """
    df = df.copy()
    
    for col in numerical_cols:
        if col not in df.columns:
            continue
            
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        # Cap outliers instead of removing
        df[col] = df[col].clip(lower=lower_bound, upper=upper_bound)
    
    return df


def one_hot_encode_categorical(df, categorical_cols):
    """
    Convert categorical features to numerical using one-hot encoding.
    """
    df = df.copy()
    
    # Perform one-hot encoding
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=categorical_cols, drop_first=False)
    
    return df_encoded


def standardize_numerical(df, numerical_cols, scaler=None, fit=True):
    """
    Standardize numerical features using StandardScaler.
    """
    df = df.copy()
    
    if scaler is None:
        scaler = StandardScaler()
    
    if fit:
        df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
    else:
        df[numerical_cols] = scaler.transform(df[numerical_cols])
    
    return df, scaler


def preprocess_churn_data(data_path, test_size=0.2, random_state=42):
    """
    Complete preprocessing pipeline for churn prediction dataset.
    
    Steps:
    1. Load data
    2. Handle missing values
    3. Handle outliers
    4. One-hot encode categorical features
    5. Standardize numerical features
    6. Split into train and test sets
    
    Returns:
        X_train, X_test, y_train, y_test, scaler, feature_names
    """
    # Load data
    print("Loading data...")
    df = pd.read_csv(data_path)
    print(f"Original data shape: {df.shape}")
    
    # Drop user_id if present (not useful for prediction)
    if 'user_id' in df.columns:
        df = df.drop(columns=['user_id'])
    
    # Separate target variable
    if 'is_churned' not in df.columns:
        raise ValueError("Target variable 'is_churned' not found in dataset")
    
    y = df['is_churned'].copy()
    X = df.drop(columns=['is_churned'])
    
    # Identify column types
    numerical_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = X.select_dtypes(include=['object']).columns.tolist()
    
    print(f"\nNumerical columns: {numerical_cols}")
    print(f"Categorical columns: {categorical_cols}")
    
    # Step 1: Handle missing values
    print("\n1. Handling missing values...")
    missing_before = X.isnull().sum().sum()
    X = handle_missing_values(X)
    missing_after = X.isnull().sum().sum()
    print(f"   Missing values before: {missing_before}, after: {missing_after}")
    
    # Step 2: Handle outliers (only on numerical columns)
    print("\n2. Handling outliers...")
    X = handle_outliers(X, numerical_cols)
    print("   Outliers capped using IQR method")
    
    # Step 3: One-hot encode categorical features
    print("\n3. One-hot encoding categorical features...")
    X_encoded = one_hot_encode_categorical(X, categorical_cols)
    print(f"   Shape after encoding: {X_encoded.shape}")
    print(f"   New feature names: {list(X_encoded.columns)}")
    
    # Step 4: Split data before standardization (to avoid data leakage)
    print("\n4. Splitting data into train and test sets...")
    X_train, X_test, y_train, y_test = train_test_split(
        X_encoded, y, test_size=test_size, random_state=random_state, stratify=y
    )
    print(f"   Train set: {X_train.shape}, Test set: {X_test.shape}")
    
    # Step 5: Standardize numerical features (fit on train, transform on test)
    print("\n5. Standardizing numerical features...")
    # Get all numerical column names after one-hot encoding
    # After one-hot encoding, we need to standardize the original numerical columns
    # that still exist in the encoded dataframe
    numerical_cols_after_encoding = [col for col in numerical_cols if col in X_train.columns]
    
    X_train_scaled, scaler = standardize_numerical(X_train, numerical_cols_after_encoding, fit=True)
    X_test_scaled, _ = standardize_numerical(X_test, numerical_cols_after_encoding, scaler=scaler, fit=False)
    
    print(f"   Standardized {len(numerical_cols_after_encoding)} numerical features")
    
    print("\n✓ Preprocessing complete!")
    print(f"Final train set shape: {X_train_scaled.shape}")
    print(f"Final test set shape: {X_test_scaled.shape}")
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, list(X_train_scaled.columns)

