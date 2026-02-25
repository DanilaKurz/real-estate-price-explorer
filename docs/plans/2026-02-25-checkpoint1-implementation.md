# Checkpoint 1: Постановка задачи и первичное проектирование - Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Реализовать Чекпоинт 1 проекта RealEstate Price Explorer 2021 - постановка задачи, EDA, Data Contract, baseline, архитектура, риски.

**Architecture:** FastAPI backend (ML-сервис) + Streamlit frontend (UI). На чекпоинте 1 фокус на EDA, Data Contracts и baseline-решение. Модель-сервис появится в следующих чекпоинтах, но структура проекта уже закладывается.

**Tech Stack:** Python 3.11+, pandas, numpy, matplotlib, seaborn, plotly, scikit-learn, jupyter, kaggle CLI, streamlit, fastapi

---

## Структура проекта (целевая)

```
real-estate-price-explorer/
|-- README.md
|-- requirements.txt
|-- .gitignore
|-- data/
|   |-- raw/              # Сырые данные с Kaggle (в .gitignore)
|   |-- processed/        # Очищенные данные (в .gitignore)
|-- notebooks/
|   |-- 01_eda.ipynb      # Exploratory Data Analysis
|-- src/
|   |-- data/
|   |   |-- __init__.py
|   |   |-- download.py   # Скрипт загрузки данных
|   |   |-- clean.py      # Очистка и предобработка
|   |   |-- contract.py   # Data Contract - валидация схемы
|   |-- models/
|   |   |-- __init__.py
|   |   |-- baseline.py   # Baseline (медиана по группе)
|   |-- api/
|   |   |-- __init__.py
|   |   |-- main.py       # FastAPI app (заглушка)
|   |-- app/
|   |   |-- __init__.py
|   |   |-- streamlit_app.py  # Streamlit UI (заглушка)
|-- tests/
|   |-- __init__.py
|   |-- test_contract.py  # Тесты Data Contract
|   |-- test_baseline.py  # Тесты Baseline
|   |-- test_clean.py     # Тесты очистки
|-- docs/
|   |-- ML_System_Design_Doc.md
|   |-- plans/
|       |-- 2026-02-25-checkpoint1-design.md
|       |-- 2026-02-25-checkpoint1-implementation.md
```

---

### Task 1: Инициализация проекта и Git-репозитория

**Files:**
- Create: `.gitignore`
- Create: `requirements.txt`
- Create: `README.md`
- Create: все `__init__.py` файлы для пакетов
- Create: директории `data/raw/`, `data/processed/`, `notebooks/`, `src/`, `tests/`

**Step 1: Создать .gitignore**

```gitignore
# Data
data/raw/
data/processed/
*.csv
*.parquet
*.zip

# Python
__pycache__/
*.py[cod]
*$py.class
.venv/
venv/
env/

# Jupyter
.ipynb_checkpoints/

# IDE
.idea/
.vscode/
*.swp

# OS
.DS_Store
Thumbs.db

# Models
*.joblib
*.pkl
*.pickle

# Environment
.env
```

**Step 2: Создать requirements.txt**

```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
plotly>=5.14.0
scikit-learn>=1.3.0
jupyter>=1.0.0
kaggle>=1.5.0
fastapi>=0.100.0
uvicorn>=0.22.0
streamlit>=1.28.0
requests>=2.31.0
pyarrow>=12.0.0
great-expectations>=0.17.0
```

**Step 3: Создать README.md**

```markdown
# RealEstate Price Explorer 2021

ML-система для оценки стоимости квартиры в России по параметрам на основе данных рынка недвижимости 2021 года.

## Описание

Сервис позволяет пользователям оценить стоимость квартиры, вводя параметры: регион, тип здания, количество комнат, площадь. Предсказание основано на ML-модели, обученной на 11+ миллионах реальных объявлений с российских площадок недвижимости.

## Архитектура

- **Backend:** FastAPI (ML-сервис, REST API)
- **Frontend:** Streamlit (веб-интерфейс)
- **ML:** scikit-learn / CatBoost
- **Данные:** Kaggle Russia Real Estate 2021

## Установка

```bash
python -m venv .venv
.venv\Scripts\activate  # Windows
pip install -r requirements.txt
```

## Данные

Датасет загружается с Kaggle:
```bash
python src/data/download.py
```

Или вручную: https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021

## Структура проекта

```
real-estate-price-explorer/
|-- src/           # Исходный код
|-- notebooks/     # Jupyter notebooks (EDA)
|-- tests/         # Тесты
|-- data/          # Данные (не в git)
|-- docs/          # Документация
```

## Чекпоинты

- [x] Чекпоинт 1: Постановка задачи и первичное проектирование
- [ ] Чекпоинт 2: Data Engineering и пайплайн данных
- [ ] Чекпоинт 3: Моделирование и эксперименты
- [ ] Чекпоинт 4: Деплой, мониторинг и эксплуатация
```

**Step 4: Создать структуру директорий и __init__.py**

```bash
mkdir -p data/raw data/processed notebooks src/data src/models src/api src/app tests
touch src/__init__.py src/data/__init__.py src/models/__init__.py src/api/__init__.py src/app/__init__.py tests/__init__.py
```

**Step 5: Инициализировать git и сделать первый коммит**

```bash
git init
git add .
git commit -m "feat: initialize project structure for RealEstate Price Explorer 2021"
```

---

### Task 2: Загрузка данных и виртуальное окружение

**Files:**
- Create: `src/data/download.py`

**Step 1: Создать виртуальное окружение и установить зависимости**

```bash
cd "c:/Users/kurzo/PycharmProjects/ITMO/Архитектура ИИ/real-estate-price-explorer"
python -m venv .venv
.venv/Scripts/activate
pip install -r requirements.txt
```

**Step 2: Написать скрипт загрузки данных**

```python
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
```

**Step 3: Загрузить данные**

Если kaggle CLI не настроен, скачать вручную с https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021 и положить CSV в `data/raw/`.

```bash
# Вариант 1: через kaggle CLI
pip install kaggle
python src/data/download.py

# Вариант 2: вручную - скачать ZIP, распаковать в data/raw/
```

**Step 4: Коммит**

```bash
git add src/data/download.py
git commit -m "feat: add dataset download script"
```

---

### Task 3: Очистка данных

**Files:**
- Create: `src/data/clean.py`
- Create: `tests/test_clean.py`

**Step 1: Написать тест очистки**

```python
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
```

**Step 2: Запустить тест, убедиться что падает**

```bash
pytest tests/test_clean.py -v
# Expected: FAIL - ModuleNotFoundError
```

**Step 3: Написать реализацию очистки**

```python
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
    df = pd.read_csv(input_path)
    df = clean_dataframe(df)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Очищенные данные сохранены в {output_path}")
    return df


if __name__ == "__main__":
    import sys
    input_file = sys.argv[1] if len(sys.argv) > 1 else "data/raw/all_v2.csv"
    load_and_clean(input_file)
```

**Step 4: Запустить тесты**

```bash
pytest tests/test_clean.py -v
# Expected: PASS
```

**Step 5: Коммит**

```bash
git add src/data/clean.py tests/test_clean.py
git commit -m "feat: add data cleaning pipeline with tests"
```

---

### Task 4: Data Contract

**Files:**
- Create: `src/data/contract.py`
- Create: `tests/test_contract.py`

**Step 1: Написать тест Data Contract**

```python
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
        "region": [77, 999],
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
```

**Step 2: Запустить тест, убедиться что падает**

```bash
pytest tests/test_contract.py -v
# Expected: FAIL
```

**Step 3: Написать реализацию Data Contract**

```python
"""Data Contract - валидация схемы данных о недвижимости."""

from dataclasses import dataclass
from typing import Optional
import pandas as pd
import numpy as np


@dataclass
class DataContractViolation:
    """Нарушение Data Contract."""
    field: str
    rule: str
    details: str
    severity: str = "error"  # error | warning


# Определение контракта
DATA_CONTRACT = {
    "price": {
        "type": "float64",
        "required": True,
        "min": 0,
        "max": 10_000_000_000,
        "max_null_pct": 0.0,
    },
    "region": {
        "type": "int64",
        "required": True,
        "min": 1,
        "max": 99,
        "max_null_pct": 0.0,
    },
    "object_type": {
        "type": "int64",
        "required": True,
        "allowed_values": [0, 2],
        "max_null_pct": 0.0,
    },
    "building_type": {
        "type": "int64",
        "required": True,
        "allowed_values": [0, 1, 2, 3, 4, 5, 6],
        "max_null_pct": 0.05,
    },
    "rooms": {
        "type": "int64",
        "required": True,
        "min": -1,
        "max": 20,
        "max_null_pct": 0.0,
    },
    "area": {
        "type": "float64",
        "required": True,
        "min": 0,
        "max": 1000,
        "max_null_pct": 0.02,
    },
    "kitchen_area": {
        "type": "float64",
        "required": False,
        "min": 0,
        "max": 500,
        "max_null_pct": 0.20,
    },
    "geo_lat": {
        "type": "float64",
        "required": False,
        "min": 41.0,
        "max": 82.0,
        "max_null_pct": 0.10,
    },
    "geo_lon": {
        "type": "float64",
        "required": False,
        "min": 19.0,
        "max": 180.0,
        "max_null_pct": 0.10,
    },
}


def validate_schema(df: pd.DataFrame) -> list[DataContractViolation]:
    """Валидирует DataFrame по Data Contract.

    Args:
        df: DataFrame для проверки.

    Returns:
        Список нарушений контракта.
    """
    violations = []

    for field, rules in DATA_CONTRACT.items():
        # Проверка наличия обязательного поля
        if field not in df.columns:
            if rules.get("required", False):
                violations.append(DataContractViolation(
                    field=field,
                    rule="required_field",
                    details=f"Обязательное поле '{field}' отсутствует",
                ))
            continue

        col = df[field]

        # Проверка доли пропусков
        null_pct = col.isna().mean()
        max_null = rules.get("max_null_pct", 1.0)
        if null_pct > max_null:
            violations.append(DataContractViolation(
                field=field,
                rule="null_percentage",
                details=f"Доля пропусков {null_pct:.2%} > допустимых {max_null:.2%}",
                severity="warning" if null_pct < max_null * 2 else "error",
            ))

        # Проверка диапазона значений
        non_null = col.dropna()
        if "min" in rules and len(non_null) > 0:
            below_min = (non_null < rules["min"]).sum()
            if below_min > 0:
                violations.append(DataContractViolation(
                    field=field,
                    rule="range_min",
                    details=f"{below_min} значений ниже минимума ({rules['min']})",
                ))

        if "max" in rules and len(non_null) > 0:
            above_max = (non_null > rules["max"]).sum()
            if above_max > 0:
                violations.append(DataContractViolation(
                    field=field,
                    rule="range_max",
                    details=f"{above_max} значений выше максимума ({rules['max']})",
                ))

        # Проверка допустимых значений
        if "allowed_values" in rules and len(non_null) > 0:
            invalid = ~non_null.isin(rules["allowed_values"])
            if invalid.sum() > 0:
                violations.append(DataContractViolation(
                    field=field,
                    rule="allowed_values",
                    details=f"{invalid.sum()} значений вне допустимого набора {rules['allowed_values']}",
                ))

    return violations


def print_validation_report(violations: list[DataContractViolation]) -> None:
    """Выводит отчет о валидации."""
    if not violations:
        print("Data Contract: OK - все проверки пройдены")
        return

    errors = [v for v in violations if v.severity == "error"]
    warnings = [v for v in violations if v.severity == "warning"]

    print(f"Data Contract: {len(errors)} ошибок, {len(warnings)} предупреждений")
    print("-" * 60)
    for v in violations:
        marker = "[ERROR]" if v.severity == "error" else "[WARN]"
        print(f"  {marker} {v.field}: {v.details} (правило: {v.rule})")
```

**Step 4: Запустить тесты**

```bash
pytest tests/test_contract.py -v
# Expected: PASS
```

**Step 5: Коммит**

```bash
git add src/data/contract.py tests/test_contract.py
git commit -m "feat: add Data Contract validation with tests"
```

---

### Task 5: Baseline модель

**Files:**
- Create: `src/models/baseline.py`
- Create: `tests/test_baseline.py`

**Step 1: Написать тест baseline**

```python
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
    # region=99 не встречалась в train -> должна вернуть глобальную медиану
    assert predictions.iloc[2] == pytest.approx(train_data["price"].median(), rel=0.01)


def test_baseline_mape(train_data):
    model = MedianBaseline(group_cols=["region", "rooms"])
    model.fit(train_data)
    predictions = model.predict(train_data)
    mape = np.mean(np.abs(predictions - train_data["price"]) / train_data["price"])
    assert mape < 1.0  # MAPE < 100% - baseline хоть что-то предсказывает
```

**Step 2: Запустить тест, убедиться что падает**

```bash
pytest tests/test_baseline.py -v
# Expected: FAIL
```

**Step 3: Написать baseline модель**

```python
"""Baseline модель - медиана по группе."""

import pandas as pd
import numpy as np


class MedianBaseline:
    """Baseline: предсказание = медианная цена по группе признаков.

    Если группа не встречалась в обучающих данных, возвращает глобальную медиану.
    """

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
```

**Step 4: Запустить тесты**

```bash
pytest tests/test_baseline.py -v
# Expected: PASS
```

**Step 5: Коммит**

```bash
git add src/models/baseline.py tests/test_baseline.py
git commit -m "feat: add median baseline model with tests"
```

---

### Task 6: EDA Notebook

**Files:**
- Create: `notebooks/01_eda.ipynb`

**Step 1: Создать EDA notebook**

Notebook должен содержать следующие секции (каждая - отдельная ячейка):

1. **Загрузка данных** - чтение CSV, df.info(), df.shape
2. **Общая статистика** - df.describe(), типы данных, пропуски
3. **Распределение цен** - гистограмма, boxplot, log-scale
4. **Распределение по регионам** - barplot топ-20 регионов
5. **Распределение по типам зданий** - countplot
6. **Распределение по комнатам** - countplot, median price by rooms
7. **Корреляционная матрица** - heatmap числовых признаков
8. **Географическое распределение** - scatter plot lat/lon, цветом цена
9. **Анализ выбросов** - boxplot цен по регионам, area vs price
10. **Анализ пропусков** - msno.matrix или manual heatmap
11. **Data Leakage анализ** - проверка: не кодирует ли адрес/ID напрямую цену
12. **Валидация Data Contract** - запуск contract.validate_schema()
13. **Baseline оценка** - обучение и оценка MedianBaseline
14. **Выводы** - markdown-ячейка с ключевыми выводами EDA

**Step 2: Запустить notebook и убедиться что все ячейки выполняются**

```bash
jupyter nbconvert --execute notebooks/01_eda.ipynb --to notebook --inplace
```

**Step 3: Коммит**

```bash
git add notebooks/01_eda.ipynb
git commit -m "feat: add EDA notebook with data analysis and baseline evaluation"
```

---

### Task 7: FastAPI и Streamlit заглушки

**Files:**
- Create: `src/api/main.py`
- Create: `src/app/streamlit_app.py`

**Step 1: Создать FastAPI заглушку**

```python
"""FastAPI сервис для предсказания цен на недвижимость."""

from fastapi import FastAPI
from pydantic import BaseModel

app = FastAPI(
    title="RealEstate Price Explorer API",
    description="API для оценки стоимости квартиры в России (данные 2021)",
    version="0.1.0",
)


class PredictionRequest(BaseModel):
    region: int
    building_type: int
    object_type: int = 0
    rooms: int
    area: float


class PredictionResponse(BaseModel):
    predicted_price: float
    price_range_low: float
    price_range_high: float
    model_type: str


@app.get("/health")
def health():
    return {"status": "ok", "version": "0.1.0"}


@app.post("/predict", response_model=PredictionResponse)
def predict(request: PredictionRequest):
    # Заглушка - будет заменена в чекпоинте 3
    return PredictionResponse(
        predicted_price=0.0,
        price_range_low=0.0,
        price_range_high=0.0,
        model_type="placeholder",
    )
```

**Step 2: Создать Streamlit заглушку**

```python
"""Streamlit UI для RealEstate Price Explorer."""

import streamlit as st

st.set_page_config(
    page_title="RealEstate Price Explorer 2021",
    page_icon="🏠",
    layout="wide",
)

st.title("RealEstate Price Explorer 2021")
st.markdown("ML-система для оценки стоимости квартиры в России по данным 2021 года.")

st.sidebar.header("Параметры квартиры")

region = st.sidebar.selectbox("Регион", options=[77, 78, 50, 47, 23], format_func=lambda x: {
    77: "Москва", 78: "Санкт-Петербург", 50: "Московская обл.",
    47: "Ленинградская обл.", 23: "Краснодарский край",
}.get(x, str(x)))

rooms = st.sidebar.selectbox("Количество комнат", options=[-1, 1, 2, 3, 4, 5], format_func=lambda x: "Студия" if x == -1 else str(x))

area = st.sidebar.slider("Площадь (м2)", min_value=10, max_value=300, value=50)

building_type = st.sidebar.selectbox("Тип здания", options=[0, 1, 2, 3, 4, 5, 6], format_func=lambda x: {
    0: "Не указан", 1: "Другой", 2: "Панельный", 3: "Монолитный",
    4: "Кирпичный", 5: "Блочный", 6: "Деревянный",
}.get(x, str(x)))

if st.sidebar.button("Оценить стоимость"):
    st.info("Модель будет подключена в чекпоинте 3. Сейчас это заглушка.")
    st.metric("Предсказанная цена", "-- руб.")

st.markdown("---")
st.markdown("*Проект в рамках курса 'Архитектура ИИ' - ИТМО*")
```

**Step 3: Коммит**

```bash
git add src/api/main.py src/app/streamlit_app.py
git commit -m "feat: add FastAPI and Streamlit stubs for future checkpoints"
```

---

### Task 8: Финальный коммит и проверка

**Step 1: Запустить все тесты**

```bash
pytest tests/ -v
# Expected: ALL PASS
```

**Step 2: Проверить структуру проекта**

```bash
find . -type f -not -path './.git/*' -not -path './.venv/*' -not -path './data/*' | sort
```

**Step 3: Финальный коммит (если нужны правки)**

```bash
git add -A
git commit -m "feat: complete Checkpoint 1 - task formalization, EDA, data contract, baseline"
```

---

## Сводка задач

| Task | Описание | Файлы | Время |
|------|---------|-------|-------|
| 1 | Инициализация проекта | .gitignore, requirements.txt, README.md | ~5 мин |
| 2 | Загрузка данных | src/data/download.py | ~10 мин |
| 3 | Очистка данных | src/data/clean.py, tests/test_clean.py | ~15 мин |
| 4 | Data Contract | src/data/contract.py, tests/test_contract.py | ~15 мин |
| 5 | Baseline модель | src/models/baseline.py, tests/test_baseline.py | ~10 мин |
| 6 | EDA Notebook | notebooks/01_eda.ipynb | ~30 мин |
| 7 | API и UI заглушки | src/api/main.py, src/app/streamlit_app.py | ~10 мин |
| 8 | Финальная проверка | - | ~5 мин |
