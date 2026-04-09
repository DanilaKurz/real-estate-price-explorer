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

## Data Pipeline

В проекте реализован ETL-пайплайн для подготовки данных к моделированию:

```bash
python src/data/pipeline.py data/raw/all_v2.csv
```

Пайплайн:
- загружает и нормализует raw CSV;
- валидирует входные данные через Data Contract;
- очищает данные и повторно валидирует cleaned layer;
- строит offline-признаки;
- выполняет region-aware sampling;
- сохраняет `clean`, `features`, `train`, `val`, `test` артефакты в `data/processed/`.

Основные модули:
- `src/data/pipeline.py`
- `src/data/features.py`
- `src/data/clean.py`
- `src/data/contract.py`

Документация Checkpoint 2:
- `docs/checkpoint2_report.md`
- `docs/feature_registry.md`

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
- [x] Чекпоинт 2: Data Engineering и пайплайн данных
- [ ] Чекпоинт 3: Моделирование и эксперименты
- [ ] Чекпоинт 4: Деплой, мониторинг и эксплуатация
