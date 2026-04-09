"""Скрипт загрузки датасета Russia Real Estate 2021 с Kaggle."""

import os
import zipfile
import subprocess
import sys
from pathlib import Path


def download_dataset(output_dir: str = "data/raw") -> str:
    """Загружает датасет с Kaggle и распаковывает.

    Args:
        output_dir: Директория для сохранения данных.

    Returns:
        Путь к распакованному CSV файлу.
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    dataset = "mrdaniilak/russia-real-estate-2021"
    print(f"Загрузка датасета {dataset}...")

    subprocess.run(
        [sys.executable, "-m", "kaggle", "datasets", "download", "-d", dataset, "-p", str(output_path)],
        check=True,
    )

    zip_files = list(output_path.glob("*.zip"))
    for zf in zip_files:
        print(f"Распаковка {zf}...")
        with zipfile.ZipFile(zf, "r") as z:
            z.extractall(output_path)
        zf.unlink()

    csv_files = list(output_path.glob("*.csv"))
    if csv_files:
        print(f"Датасет загружен: {csv_files[0]}")
        return str(csv_files[0])

    raise FileNotFoundError("CSV файл не найден после распаковки")


if __name__ == "__main__":
    download_dataset()
