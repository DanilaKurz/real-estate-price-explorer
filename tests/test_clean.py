"""Тесты очистки данных."""

import pandas as pd
import numpy as np
import pytest
from src.data.clean import clean_dataframe, remove_duplicates, remove_price_outliers


@pytest.fixture
def sample_raw_data():
    """Сырые данные для тестирования."""
    return pd.DataFrame({
        "price": [1_000_000, 2_000_000, 2_000_000, -500, 100, 50_000_000_000, 3_000_000],
        "region": [77, 78, 78, 77, 78, 50, 77],
        "building_type": [2, 3, 3, 4, 1, 2, 0],
        "object_type": [0, 2, 2, 0, 0, 2, 0],
        "rooms": [2, 3, 3, 1, -1, 4, 2],
        "area": [50.0, 70.0, 70.0, -10.0, 25.0, 200.0, 60.0],
        "kitchen_area": [10.0, 15.0, 15.0, np.nan, 8.0, 40.0, 12.0],
        "geo_lat": [55.75, 59.93, 59.93, np.nan, 59.93, 55.0, 55.75],
        "geo_lon": [37.62, 30.32, 30.32, np.nan, 30.32, 37.0, 37.62],
    })


def test_remove_duplicates(sample_raw_data):
    result = remove_duplicates(sample_raw_data)
    assert len(result) < len(sample_raw_data)


def test_remove_price_outliers(sample_raw_data):
    result = remove_price_outliers(sample_raw_data)
    assert (result["price"] > 0).all()
    assert (result["price"] < 10_000_000_000).all()


def test_clean_dataframe(sample_raw_data):
    result = clean_dataframe(sample_raw_data)
    assert (result["price"] > 0).all()
    assert (result["area"] > 0).all()
    assert len(result) > 0
