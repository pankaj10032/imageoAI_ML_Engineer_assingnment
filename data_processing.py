# data_processing.py
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from typing import Tuple

def load_data(file_path: str) -> pd.DataFrame:
    """
    Load the dataset from a CSV file.
    
    Args:
        file_path (str): Path to the CSV file.
    
    Returns:
        pd.DataFrame: Loaded dataset.
    """
    return pd.read_csv(file_path)

def preprocess_data(data: pd.DataFrame) -> Tuple[pd.DataFrame, pd.Series]:
    """
    Preprocess the dataset by separating features and target, and normalizing features.
    
    Args:
        data (pd.DataFrame): The dataset.
    
    Returns:
        Tuple[pd.DataFrame, pd.Series]: Features (X) and target (y).
    """
    X = data.drop(columns=["hsi_id", "vomitoxin_ppb"])  # Drop non-feature columns
    y = data["vomitoxin_ppb"]
    
    # Normalize features
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    
    return X, y

def split_data(X, y, test_size: float = 0.2, random_state: int = 42) -> Tuple:
    """
    Split the dataset into training and testing sets.
    
    Args:
        X (pd.DataFrame): Features.
        y (pd.Series): Target.
        test_size (float): Proportion of the dataset to include in the test split.
        random_state (int): Random seed for reproducibility.
    
    Returns:
        Tuple: X_train, X_test, y_train, y_test.
    """
    return train_test_split(X, y, test_size=test_size, random_state=random_state)