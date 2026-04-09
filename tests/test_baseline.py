"""Тесты Baseline модели."""

import pandas as pd
import numpy as np
import pytest
from src.models.baseline import MedianBaseline


@pytest.fixture
def train_data():
    np.random.seed(42)
    return pd.DataFrame({
        "region": [77] * 50 + [78] * 50,
        "rooms": [2] * 25 + [3] * 25 + [1] * 25 + [2] * 25,
        "building_type": [2] * 100,
        "price": np.random.normal(5_000_000, 1_000_000, 100).clip(500_000),
    })


@pytest.fixture
def test_data():
    return pd.DataFrame({
        "region": [77, 78, 99],
        "rooms": [2, 1, 2],
        "building_type": [2, 2, 2],
    })


def test_baseline_fit_predict(train_data, test_data):
    model = MedianBaseline(group_cols=["region", "rooms"])
    model.fit(train_data)
    predictions = model.predict(test_data)
    assert len(predictions) == len(test_data)
    assert all(p > 0 for p in predictions)


def test_baseline_unknown_group_returns_global_median(train_data, test_data):
    model = MedianBaseline(group_cols=["region", "rooms"])
    model.fit(train_data)
    predictions = model.predict(test_data)
    assert predictions.iloc[2] == pytest.approx(train_data["price"].median(), rel=0.01)


def test_baseline_mape(train_data):
    model = MedianBaseline(group_cols=["region", "rooms"])
    model.fit(train_data)
    predictions = model.predict(train_data)
    mape = np.mean(np.abs(predictions - train_data["price"]) / train_data["price"])
    assert mape < 1.0
