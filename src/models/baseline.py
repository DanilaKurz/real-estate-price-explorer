"""Baseline модель - медиана по группе."""

import pandas as pd
import numpy as np


class MedianBaseline:
    """Baseline: предсказание = медианная цена по группе признаков."""

    def __init__(self, group_cols: list[str] = None, target_col: str = "price"):
        self.group_cols = group_cols or ["region", "rooms", "building_type"]
        self.target_col = target_col
        self.group_medians = None
        self.global_median = None

    def fit(self, df: pd.DataFrame) -> "MedianBaseline":
        """Вычисляет медианы по группам."""
        self.global_median = df[self.target_col].median()
        self.group_medians = (
            df.groupby(self.group_cols)[self.target_col]
            .median()
            .reset_index()
            .rename(columns={self.target_col: "prediction"})
        )
        return self

    def predict(self, df: pd.DataFrame) -> pd.Series:
        """Предсказывает цену по медиане группы."""
        merged = df.merge(self.group_medians, on=self.group_cols, how="left")
        predictions = merged["prediction"].fillna(self.global_median)
        return predictions

    def evaluate(self, df: pd.DataFrame) -> dict:
        """Оценивает baseline на данных с целевой переменной."""
        predictions = self.predict(df)
        actual = df[self.target_col]

        mape = np.mean(np.abs(predictions - actual) / actual) * 100
        mae = np.mean(np.abs(predictions - actual))
        median_ae = np.median(np.abs(predictions - actual))

        return {
            "mape": round(mape, 2),
            "mae": round(mae, 0),
            "median_ae": round(median_ae, 0),
            "n_samples": len(df),
        }
