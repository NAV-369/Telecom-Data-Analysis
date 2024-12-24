import pytest
import pandas as pd
import numpy as np
from src.data_preparation import DataPreparation

@pytest.fixture
def data_prep():
    return DataPreparation()

@pytest.fixture
def sample_df():
    return pd.DataFrame({
        'A': [1, 2, np.nan, 4, 5],
        'B': [10, 20, 30, 40, 50]
    })

def test_handle_missing_values(data_prep, sample_df):
    result = data_prep.handle_missing_values(sample_df.copy())
    assert not result['A'].isnull().any()
    assert result['A'].mean() == sample_df['A'].mean()

def test_handle_outliers(data_prep):
    df = pd.DataFrame({
        'A': [1, 2, 3, 100, 4, 5]  # 100 is an outlier
    })
    result = data_prep.handle_outliers(df.copy(), ['A'])
    assert result['A'].max() < 100
