"""Reusable ETL pipeline for the real-estate project."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

from src.data.clean import clean_dataframe
from src.data.contract import validate_schema
from src.data.features import build_model_features, feature_registry_dataframe


@dataclass(frozen=True)
class PipelineArtifacts:
    """Paths to generated data artifacts."""

    clean_data: Path
    features_data: Path
    train_data: Path
    validation_data: Path
    test_data: Path
    feature_registry: Path
    raw_validation_report: Path
    clean_validation_report: Path


def load_raw_data(input_path: str | Path) -> pd.DataFrame:
    """Loads raw Kaggle CSV and normalizes column names."""

    df = pd.read_csv(input_path, sep=";")
    return df.rename(
        columns={
            "id_region": "region",
            "level": "floor",
            "levels": "total_floors",
        }
    )


def validate_raw_data(df: pd.DataFrame) -> list:
    """Validates the input dataframe against the current data contract."""

    return validate_schema(df)


def save_validation_report(violations: list, output_path: str | Path) -> Path:
    """Saves a compact validation report as parquet for auditability."""

    output_path = Path(output_path)
    report_df = pd.DataFrame(
        [
            {
                "field": violation.field,
                "rule": violation.rule,
                "details": violation.details,
                "severity": violation.severity,
            }
            for violation in violations
        ]
    )
    if report_df.empty:
        report_df = pd.DataFrame(
            [{"field": "ALL", "rule": "ok", "details": "No violations", "severity": "info"}]
        )
    report_df.to_parquet(output_path, index=False)
    return output_path


def stratified_region_sample(
    df: pd.DataFrame,
    sample_size: int | None = 1_000_000,
    min_records_per_region: int = 2_000,
    random_state: int = 42,
) -> pd.DataFrame:
    """Creates a region-aware sample for efficient modeling."""

    if sample_size is None or sample_size >= len(df):
        return df.reset_index(drop=True)

    total_rows = len(df)
    sampled_parts = []

    for _, group in df.groupby("region", sort=False):
        proportional_size = int(round(len(group) / total_rows * sample_size))
        target_size = min(len(group), max(min_records_per_region, proportional_size))
        sampled_parts.append(group.sample(n=target_size, random_state=random_state))

    sampled = pd.concat(sampled_parts, ignore_index=True)
    if len(sampled) > sample_size:
        sampled = sampled.sample(n=sample_size, random_state=random_state)

    return sampled.reset_index(drop=True)


def split_dataset(
    df: pd.DataFrame,
    test_size: float = 0.2,
    validation_size: float = 0.1,
    random_state: int = 42,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Splits the modeling dataset into train/validation/test."""

    stratify = df["region"] if "region" in df.columns and df["region"].nunique() > 1 else None
    train_df, test_df = train_test_split(
        df,
        test_size=test_size,
        random_state=random_state,
        stratify=stratify,
    )

    train_stratify = (
        train_df["region"] if "region" in train_df.columns and train_df["region"].nunique() > 1 else None
    )
    adjusted_validation_size = validation_size / (1 - test_size)
    train_df, validation_df = train_test_split(
        train_df,
        test_size=adjusted_validation_size,
        random_state=random_state,
        stratify=train_stratify,
    )

    return (
        train_df.reset_index(drop=True),
        validation_df.reset_index(drop=True),
        test_df.reset_index(drop=True),
    )


def run_pipeline(
    input_path: str | Path,
    output_dir: str | Path = "data/processed",
    sample_size: int | None = 1_000_000,
    random_state: int = 42,
) -> PipelineArtifacts:
    """Runs the full ETL pipeline and saves the generated artifacts."""

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_df = load_raw_data(input_path)
    raw_violations = validate_raw_data(raw_df)

    clean_df = clean_dataframe(raw_df)
    clean_violations = validate_schema(clean_df)
    if any(v.severity == "error" for v in clean_violations):
        raise ValueError("Cleaned dataset still violates the data contract.")

    sampled_df = stratified_region_sample(
        clean_df,
        sample_size=sample_size,
        random_state=random_state,
    )
    features_df = build_model_features(sampled_df)
    train_df, validation_df, test_df = split_dataset(
        features_df,
        random_state=random_state,
    )

    clean_path = output_dir / "clean.parquet"
    features_path = output_dir / "features_offline.parquet"
    train_path = output_dir / "train.parquet"
    validation_path = output_dir / "val.parquet"
    test_path = output_dir / "test.parquet"
    registry_path = output_dir / "feature_registry.parquet"
    raw_validation_path = output_dir / "raw_validation_report.parquet"
    clean_validation_path = output_dir / "clean_validation_report.parquet"

    clean_df.to_parquet(clean_path, index=False)
    features_df.to_parquet(features_path, index=False)
    train_df.to_parquet(train_path, index=False)
    validation_df.to_parquet(validation_path, index=False)
    test_df.to_parquet(test_path, index=False)
    feature_registry_dataframe().to_parquet(registry_path, index=False)
    save_validation_report(raw_violations, raw_validation_path)
    save_validation_report(clean_violations, clean_validation_path)

    return PipelineArtifacts(
        clean_data=clean_path,
        features_data=features_path,
        train_data=train_path,
        validation_data=validation_path,
        test_data=test_path,
        feature_registry=registry_path,
        raw_validation_report=raw_validation_path,
        clean_validation_report=clean_validation_path,
    )


if __name__ == "__main__":
    import sys

    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/all_v2.csv"
    artifacts = run_pipeline(input_file)
    print("Pipeline completed successfully.")
    print(artifacts)
