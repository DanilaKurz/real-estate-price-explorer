"""Очистка и предобработка данных о недвижимости."""

import pandas as pd
import numpy as np
from pathlib import Path


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Удаляет дубликаты по всем столбцам."""
    initial_len = len(df)
    df = df.drop_duplicates()
    removed = initial_len - len(df)
    print(f"Удалено дубликатов: {removed} ({removed/initial_len*100:.1f}%)")
    return df


def remove_price_outliers(df: pd.DataFrame, lower: float = 100_000, upper: float = 1_000_000_000) -> pd.DataFrame:
    """Удаляет записи с аномальными ценами."""
    initial_len = len(df)
    df = df[(df["price"] > lower) & (df["price"] < upper)]
    removed = initial_len - len(df)
    print(f"Удалено выбросов по цене: {removed} ({removed/initial_len*100:.1f}%)")
    return df


def remove_area_outliers(df: pd.DataFrame, lower: float = 5.0, upper: float = 500.0) -> pd.DataFrame:
    """Удаляет записи с аномальной площадью."""
    initial_len = len(df)
    df = df[(df["area"] > lower) & (df["area"] < upper)]
    removed = initial_len - len(df)
    print(f"Удалено выбросов по площади: {removed} ({removed/initial_len*100:.1f}%)")
    return df


def fill_missing_values(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропущенные значения."""
    if "kitchen_area" in df.columns:
        df["kitchen_area"] = df["kitchen_area"].fillna(df["kitchen_area"].median())
    if "geo_lat" in df.columns:
        df["geo_lat"] = df.groupby("region")["geo_lat"].transform(
            lambda x: x.fillna(x.median())
        )
    if "geo_lon" in df.columns:
        df["geo_lon"] = df.groupby("region")["geo_lon"].transform(
            lambda x: x.fillna(x.median())
        )
    return df


def clean_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """Полный пайплайн очистки данных."""
    print(f"Исходный размер: {len(df)} записей")
    df = remove_duplicates(df)
    df = remove_price_outliers(df)
    df = remove_area_outliers(df)
    df = fill_missing_values(df)
    df = df.dropna(subset=["price", "region", "rooms", "area"])
    print(f"Итоговый размер: {len(df)} записей")
    return df.reset_index(drop=True)


def load_and_clean(input_path: str, output_path: str = "data/processed/clean.parquet") -> pd.DataFrame:
    """Загружает сырой CSV, очищает, сохраняет в Parquet."""
    print(f"Загрузка данных из {input_path}...")
    df = pd.read_csv(input_path, sep=';')
    df = df.rename(columns={"id_region": "region", "level": "floor", "levels": "total_floors"})
    df = clean_dataframe(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Очищенные данные сохранены в {output_path}")
    return df


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/all_v2.csv"
    load_and_clean(input_file)
