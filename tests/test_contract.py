"""Тесты Data Contract."""

import pandas as pd
import numpy as np
import pytest
from src.data.contract import validate_schema, DataContractViolation


@pytest.fixture
def valid_data():
    return pd.DataFrame({
        "price": [1_000_000.0, 2_000_000.0],
        "region": [77, 78],
        "building_type": [2, 3],
        "object_type": [0, 2],
        "rooms": [2, 3],
        "area": [50.0, 70.0],
        "geo_lat": [55.75, 59.93],
        "geo_lon": [37.62, 30.32],
    })


@pytest.fixture
def invalid_data():
    return pd.DataFrame({
        "price": [-100.0, 2_000_000.0],
        "region": [77, 9999],
        "building_type": [2, 10],
        "object_type": [0, 5],
        "rooms": [2, 3],
        "area": [50.0, -5.0],
        "geo_lat": [55.75, 200.0],
        "geo_lon": [37.62, 300.0],
    })


def test_valid_schema_passes(valid_data):
    violations = validate_schema(valid_data)
    assert len(violations) == 0


def test_invalid_schema_catches_errors(invalid_data):
    violations = validate_schema(invalid_data)
    assert len(violations) > 0


def test_missing_required_columns():
    df = pd.DataFrame({"price": [100]})
    violations = validate_schema(df)
    assert any(v.field == "region" for v in violations)
