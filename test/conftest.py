"""Shared fixtures for the test suite"""

import pandas as pd  # type: ignore
import pytest


@pytest.fixture
def sample_df():
    """A small, hand-crafted DataFrame mimicking Fraud_Dataset.csv's shape"""
    return pd.DataFrame(
        {
            "A": [1.0, 2.0, 3.0],
            "B": [5.0, 6.0, 7.0],
            "C": [10.0, None, 30.0],  # includes null intentionally
            "D": [1.0, 2.0, 3.0],
            "J": ["CO", "MX", "US"],
            "K": [0.5, 0.6, 0.7],  # added to test if script drops it
            "Monto": ["37.51", "8.18", "13.96"],  # as text as the real csv to test to_number script
            "Q": ["0.00", "600.17", "28.65"],
            "R": ["0.00", "361.94", "29.29"],
            "Fraude": [0, 1, 0],
        }
    )
