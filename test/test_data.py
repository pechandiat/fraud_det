"""tests for src/data.py preprocessing utilities"""

import pandas as pd  # Type: ignore
from src.data import clean_features, create_null_indicator, to_numeric


def test_to_numeric(sample_df: pd.DataFrame):
    """Test that to_numeric converts specified columns to numeric types."""
    result = to_numeric(sample_df.copy(), columns_to_number=["Monto", "Q", "R"])
    assert pd.api.types.is_numeric_dtype(result["Monto"])
    assert result["Monto"].iloc[0] == 37.51

def test_to_numeric_leaves_other_columns_untouched(sample_df: pd.DataFrame):
    """Test that to_numeric only converts specified columns and leaves others unchanged."""
    result = to_numeric(sample_df.copy(), columns_to_number=["Monto"])
    assert result["J"].tolist() == ["CO", "MX", "US"]

def test_clean_features(sample_df: pd.DataFrame):
    """Test that clean_features removes specified columns."""
    result = clean_features(sample_df.copy(), columns_to_drop=["K"])
    assert "K" not in result.columns
    assert "A" in result.columns

def test_create_null_indicator(sample_df: pd.DataFrame):
    """Test that create_null_indicator creates a binary indicator for null values."""
    result = create_null_indicator(sample_df.copy(), columns_to_flag = ["C"])
    assert result["C_is_null"].to_list() == [0, 1, 0]

def test_create_null_indicator_does_not_modify_original_column(sample_df: pd.DataFrame):
    """Test that create_null_indicator preserves the original column with null values."""
    result = create_null_indicator(sample_df.copy(), columns_to_flag=["C"])
    assert result["C"].isna().sum() == 1
