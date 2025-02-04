import pytest
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

# Import your ML functions here
# from src.concrete_ml_analysis import train_model, predict_strength

def test_data_loading():
    """Test if the concrete dataset can be loaded correctly"""
    try:
        df = pd.read_csv("data/concrete.csv")
        assert isinstance(df, pd.DataFrame)
        assert not df.empty
        assert "strength" in df.columns
    except Exception as e:
        pytest.fail(f"Failed to load dataset: {str(e)}")

def test_data_preprocessing():
    """Test data preprocessing steps"""
    df = pd.read_csv("data/concrete.csv")
    
    # Check for missing values
    assert df.isnull().sum().sum() == 0, "Dataset contains missing values"
    
    # Check data types
    numeric_columns = df.select_dtypes(include=[np.number]).columns
    assert len(numeric_columns) == len(df.columns), "Non-numeric columns found"
    
    # Check value ranges
    assert df["strength"].min() >= 0, "Negative strength values found"
    assert df["age"].min() >= 0, "Negative age values found"

def test_train_test_split():
    """Test train-test splitting"""
    df = pd.read_csv("data/concrete.csv")
    X = df.drop("strength", axis=1)
    y = df["strength"]
    
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    assert len(X_train) == len(y_train)
    assert len(X_test) == len(y_test)
    assert len(X_train) > len(X_test)

# Add more tests for your ML models and predictions
"""
def test_model_training():
    df = pd.read_csv("data/concrete.csv")
    X = df.drop("strength", axis=1)
    y = df["strength"]
    
    model = train_model(X, y)
    assert model is not None
    
    # Test prediction shape
    test_input = X.iloc[0:1]
    prediction = predict_strength(model, test_input)
    assert isinstance(prediction, (np.ndarray, float))
"""

if __name__ == "__main__":
    pytest.main([__file__])
