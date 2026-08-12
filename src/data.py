"""Module for data loading and preprocessing utilities."""

import pandas as pd  # type: ignore


def load_data(path: str) -> pd.DataFrame:
    """
    Load data from a CSV file.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data.
    """
    df = pd.read_csv(path)
    return df

def to_numeric(df: pd.DataFrame, columns_to_number: list) -> pd.DataFrame:
    """
    Convert specified columns to numeric type.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_number (list): List of columns to convert to numeric.

    Returns:
        pd.DataFrame: DataFrame with converted columns.
    """
    for column in columns_to_number:
        df[column] = pd.to_numeric(df[column], errors='coerce')
    return df

def clean_features(df: pd.DataFrame, columns_to_drop: list) -> pd.DataFrame:
    """
    Clean the DataFrame by dropping unnecessary columns.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_drop (list): List of columns to drop.
    """
    df = df.drop(columns=columns_to_drop)

    return df

def create_null_indicator(df: pd.DataFrame, columns_to_flag: list) -> pd.DataFrame:
    """
    Create a new column indicating whether the specified column has null values.

    Args:
        df (pd.DataFrame): Input DataFrame.
        columns_to_flag (list): List of columns to check for null values.

    Returns:
        pd.DataFrame: DataFrame with the new indicator column.
    """
    for column in columns_to_flag:
        df[f'{column}_is_null'] = df[column].isna().astype(int)
    return df
