"""Feature engineering utilities for the real-estate pipeline."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Literal

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class FeatureDefinition:
    """Describes a feature used by the project."""

    name: str
    source: str
    dtype: str
    availability: Literal["offline", "online", "offline+online", "excluded"]
    update_frequency: str
    description: str
    leakage_risk: str = "low"


FEATURE_REGISTRY: list[FeatureDefinition] = [
    FeatureDefinition(
        name="region",
        source="raw",
        dtype="categorical[int]",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Region code of the property.",
    ),
    FeatureDefinition(
        name="object_type",
        source="raw",
        dtype="categorical[int]",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="0 = secondary market, 2 = new building.",
    ),
    FeatureDefinition(
        name="building_type",
        source="raw",
        dtype="categorical[int]",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Building material/type code.",
    ),
    FeatureDefinition(
        name="rooms",
        source="raw",
        dtype="integer",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Number of rooms, where -1 means studio.",
    ),
    FeatureDefinition(
        name="area",
        source="raw",
        dtype="float",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Apartment area in square meters.",
    ),
    FeatureDefinition(
        name="kitchen_area",
        source="raw",
        dtype="float",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Kitchen area in square meters.",
    ),
    FeatureDefinition(
        name="geo_lat",
        source="raw",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Latitude from geocoding.",
        leakage_risk="medium",
    ),
    FeatureDefinition(
        name="geo_lon",
        source="raw",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Longitude from geocoding.",
        leakage_risk="medium",
    ),
    FeatureDefinition(
        name="floor",
        source="raw",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Floor number after raw-column normalization.",
    ),
    FeatureDefinition(
        name="total_floors",
        source="raw",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Total number of floors in the building.",
    ),
    FeatureDefinition(
        name="is_studio",
        source="derived",
        dtype="int",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Binary flag for studio apartments.",
    ),
    FeatureDefinition(
        name="log_area",
        source="derived",
        dtype="float",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="log1p(area) to reduce right skew.",
    ),
    FeatureDefinition(
        name="has_kitchen_area",
        source="derived",
        dtype="int",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Binary flag showing whether kitchen area is present.",
    ),
    FeatureDefinition(
        name="kitchen_to_area_ratio",
        source="derived",
        dtype="float",
        availability="offline+online",
        update_frequency="on pipeline run / per request",
        description="Kitchen area divided by apartment area.",
    ),
    FeatureDefinition(
        name="floor_ratio",
        source="derived",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Relative floor position within the building.",
    ),
    FeatureDefinition(
        name="is_top_floor",
        source="derived",
        dtype="int",
        availability="offline",
        update_frequency="on pipeline run",
        description="Binary flag for apartments located on the top floor.",
    ),
    FeatureDefinition(
        name="is_first_floor",
        source="derived",
        dtype="int",
        availability="offline",
        update_frequency="on pipeline run",
        description="Binary flag for apartments located on the first floor.",
    ),
    FeatureDefinition(
        name="region_listing_count",
        source="aggregate",
        dtype="int",
        availability="offline",
        update_frequency="on pipeline run",
        description="Number of listings in the region within the training snapshot.",
    ),
    FeatureDefinition(
        name="region_median_area",
        source="aggregate",
        dtype="float",
        availability="offline",
        update_frequency="on pipeline run",
        description="Median apartment area for the region, computed on the training snapshot.",
    ),
    FeatureDefinition(
        name="house_id",
        source="raw",
        dtype="identifier",
        availability="excluded",
        update_frequency="never",
        description="Excluded from modeling due to location memorization risk.",
        leakage_risk="high",
    ),
    FeatureDefinition(
        name="street_id",
        source="raw",
        dtype="identifier",
        availability="excluded",
        update_frequency="never",
        description="Excluded from modeling due to address-level leakage risk.",
        leakage_risk="high",
    ),
    FeatureDefinition(
        name="postal_code",
        source="raw",
        dtype="identifier",
        availability="excluded",
        update_frequency="never",
        description="Excluded from modeling due to overly precise location signal.",
        leakage_risk="high",
    ),
]


def feature_registry_dataframe() -> pd.DataFrame:
    """Returns the feature registry as a dataframe."""

    return pd.DataFrame(asdict(feature) for feature in FEATURE_REGISTRY)


def build_model_features(df: pd.DataFrame) -> pd.DataFrame:
    """Builds a modeling dataframe from cleaned source data."""

    features = df.copy()

    features["is_studio"] = (features["rooms"] == -1).astype(int)
    features["log_area"] = np.log1p(features["area"].clip(lower=0))
    features["has_kitchen_area"] = features["kitchen_area"].notna().astype(int)
    kitchen_area = features["kitchen_area"].astype(float).fillna(0.0)
    safe_area = features["area"].astype(float).replace(0, np.nan)
    features["kitchen_to_area_ratio"] = (kitchen_area / safe_area).fillna(0.0)

    if {"floor", "total_floors"}.issubset(features.columns):
        safe_total_floors = features["total_floors"].replace(0, np.nan)
        features["floor_ratio"] = (features["floor"] / safe_total_floors).fillna(0)
        features["is_top_floor"] = (
            (features["floor"].notna())
            & (features["total_floors"].notna())
            & (features["floor"] == features["total_floors"])
        ).astype(int)
        features["is_first_floor"] = (features["floor"] == 1).fillna(False).astype(int)
    else:
        features["floor_ratio"] = 0.0
        features["is_top_floor"] = 0
        features["is_first_floor"] = 0

    region_listing_count = features.groupby("region")["region"].transform("count")
    region_median_area = features.groupby("region")["area"].transform("median")
    features["region_listing_count"] = region_listing_count
    features["region_median_area"] = region_median_area

    return features
