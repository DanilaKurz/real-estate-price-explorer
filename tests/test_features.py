"""Tests for feature engineering utilities."""

import pandas as pd

from src.data.features import build_model_features, feature_registry_dataframe


def test_build_model_features_adds_expected_columns():
    df = pd.DataFrame(
        {
            "region": [77, 77],
            "object_type": [0, 2],
            "building_type": [2, 3],
            "rooms": [-1, 2],
            "area": [30.0, 60.0],
            "kitchen_area": [None, 12.0],
            "floor": [1, 10],
            "total_floors": [10, 10],
        }
    )

    features = build_model_features(df)

    expected_columns = {
        "is_studio",
        "log_area",
        "has_kitchen_area",
        "kitchen_to_area_ratio",
        "floor_ratio",
        "is_top_floor",
        "is_first_floor",
        "region_listing_count",
        "region_median_area",
    }
    assert expected_columns.issubset(features.columns)
    assert features.loc[0, "is_studio"] == 1
    assert features.loc[1, "is_top_floor"] == 1


def test_feature_registry_contains_excluded_location_identifiers():
    registry = feature_registry_dataframe()
    excluded = registry[registry["availability"] == "excluded"]["name"].tolist()

    assert "house_id" in excluded
    assert "street_id" in excluded
    assert "postal_code" in excluded
