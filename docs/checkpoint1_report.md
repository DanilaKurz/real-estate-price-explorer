# Checkpoint 1 -- Отчет-шпаргалка

**Проект:** RealEstate Price Explorer 2021
**Курс:** Архитектура ИИ, ИТМО
**Задача:** ML-система предсказания цен квартир в России (данные 2021)
**Дата:** 2026-03-18

---

## Сводная таблица по требованиям

### 1. Выбор темы и постановка задачи

| Задание | Что сделано | Результат | Где смотреть |
|---------|-------------|-----------|--------------|
| Тип ML-задачи | Регрессия -- предсказание цены квартиры (целевая: `price`, руб.) | Формализовано | `docs/ML_System_Design_Doc.md`, разделы 1-2 |
| Объект и контекст | Квартиры на рынке РФ, датасет Kaggle 2021, ~11.3 млн записей | Определен | `docs/ML_System_Design_Doc.md` |
| Ключевые признаки | `region`, `object_type`, `building_type`, `rooms`, `area`, `geo_lat`, `geo_lon` | 7 признаков | `docs/ML_System_Design_Doc.md` |
| Метрики | MAPE (основная), MAE, R2. Целевой MAPE <= 25% | Определены | `docs/ML_System_Design_Doc.md` |
| Baseline без ML | MedianBaseline -- медиана цены по группе (region + rooms + building_type) | 3 варианта реализованы и протестированы | `src/models/baseline.py`, `tests/test_baseline.py` |

**Результаты baseline (без ML):**

| Baseline | Группировка | Median APE | MAE |
|----------|-------------|------------|-----|
| Baseline 0 | Глобальная медиана | 43.7% | -- |
| Baseline 1 | region + rooms | 25.3% | 2 245 тыс. руб. |
| Baseline 2 | region + rooms + building_type | **23.5%** | **2 137 тыс. руб.** |

---

### 2. Первичный сбор данных и EDA

| Задание | Что сделано | Результат | Где смотреть |
|---------|-------------|-----------|--------------|
| Источник данных | Kaggle Russia Real Estate 2021 (891 MB CSV, ~11.3 млн записей) | Загружен семпл 1 млн строк | `src/data/download.py`, `data/raw/all_v2.csv` |
| Описание данных | 15 колонок, 86 регионов, топ-5 регионов = 37.2% данных | Описано | `notebooks/01_eda.ipynb` |
| Распределения | Медианная цена 3 550 000 руб., распределение скошено вправо | Нужен log-transform | `data/processed/price_distribution.png` |
| Корреляции | Пирсон: price-area=0.55, price-rooms=0.28; Спирмен: price-area=0.52, price-total_floors=0.44 | Площадь -- главный предиктор | `data/processed/correlation_matrix.png` |
| Пропуски | street_id (32.5%), house_id (24%), postal_code (5%) | Зафиксированы | `data/processed/missing_values.png` |
| Выбросы | 65 цен <= 0, 89 цен > 1 млрд (макс 635 млрд), 110K отриц. kitchen_area | Удалено 3.6% записей | `data/processed/outliers.png`, `src/data/clean.py` |
| Data Leakage | house_id не уникален (ratio=0.161), CV цен в доме=0.244 | **НЕ использовать** house_id, street_id, postal_code как признаки | `notebooks/01_eda.ipynb` |
| Визуализации | 9 графиков + карта РФ + зум 4 городов (Москва, СПб, Крым, Сочи) | Все сохранены | `data/processed/*.png` |
| Очистка данных | Дедупликация + фильтрация выбросов по цене и площади | Реализована программно, покрыта тестами | `src/data/clean.py`, `tests/test_clean.py` |

**Список визуализаций:**

| # | Файл | Содержание |
|---|------|------------|
| 1 | `price_distribution.png` | Распределение цен |
| 2 | `region_distribution.png` | Распределение по регионам |
| 3 | `rooms_distribution.png` | Распределение по числу комнат |
| 4 | `building_type_distribution.png` | Типы зданий |
| 5 | `correlation_matrix.png` | Матрица корреляций |
| 6 | `area_vs_price.png` | Площадь vs цена |
| 7 | `geo_distribution.png` | Карта РФ с границами |
| 8 | `geo_cities_zoom.png` | Зум 4 городов |
| 9 | `missing_values.png` | Пропуски |
| 10 | `outliers.png` | Выбросы |

---

### 3. Data Contract

| Задание | Что сделано | Результат | Где смотреть |
|---------|-------------|-----------|--------------|
| Схема данных | 11 полей: price, region, object_type, building_type, rooms, area, kitchen_area, geo_lat, geo_lon, floor, total_floors | Описана | `docs/ML_System_Design_Doc.md` (Data Contract) |
| Обязательные поля | price, region, object_type, building_type, rooms, area | 6 полей | `src/data/contract.py` |
| Необязательные поля | kitchen_area (до 20% пропусков), geo_lat/geo_lon (до 10%) | 3 поля | `src/data/contract.py` |
| Допустимые диапазоны | price: >0, <10B; region: 1-999; area: 0-1000; geo_lat: 41-82; geo_lon: 19-180 | Определены | `src/data/contract.py` |
| Программная валидация | `validate_schema()` -- возвращает список нарушений | На реальных данных: 2 нарушения (3 цены >10B, 110K kitchen_area<0) | `src/data/contract.py`, `tests/test_contract.py` |
| Актуальность | Статический датасет, обновления не ожидаются | Зафиксировано | `docs/ML_System_Design_Doc.md` |

---

### 4. Архитектура системы

| Задание | Что сделано | Результат | Где смотреть |
|---------|-------------|-----------|--------------|
| Диаграмма архитектуры | Kaggle CSV -> ETL Pipeline -> Cleaned Parquet -> Model Training -> FastAPI -> Streamlit | В Design Doc | `docs/ML_System_Design_Doc.md` |
| API эндпоинты | `POST /predict`, `GET /health`, `POST /retrain`, `GET /stats` | Заглушки созданы | `src/api/main.py` |
| UI | Streamlit-приложение | Заглушка создана | `src/app/streamlit_app.py` |
| Деплой | Streamlit Cloud (UI) + Railway/Render (API), бюджет $0/мес | Спланирован | `docs/ML_System_Design_Doc.md` |
| ML System Design Doc | Заполнен полностью (все разделы) | Готов | `docs/ML_System_Design_Doc.md` |

---

### 5. Риски (v0)

| # | Риск | Причина | Последствие | Митигация |
|---|------|---------|-------------|-----------|
| 1 | Грязные данные | Дубликаты, выбросы в исходном CSV | Смещение модели | Дедупликация, IQR-фильтрация |
| 2 | Data leakage через гео-координаты | Близкие квартиры попадают в train и test | Завышенные метрики | Разделение train/test по географии |
| 3 | Drift данных при дообучении | Изменение рыночных условий | Деградация качества | Искусственный drift для тестирования |
| 4 | RAM-ограничение Streamlit Cloud (1 GB) | Бесплатный хостинг | Падение приложения | Семплирование, формат Parquet |
| 5 | Мультиколлинеарность признаков | Коррелированные фичи (area/rooms) | Нестабильность коэффициентов | VIF-анализ, PCA |
| 6 | Cold start на бесплатном хостинге | Контейнер засыпает после простоя | Долгий первый запрос | Прогрев перед защитой |
| 7 | Неравномерность данных по регионам | Топ-5 регионов = 37% данных | Плохое качество для редких регионов | Стратифицированный сплит |

Файл: `docs/ML_System_Design_Doc.md`, раздел 4.8

---

## Структура проекта

```
real-estate-price-explorer/
+-- README.md
+-- requirements.txt
+-- .gitignore
+-- data/
|   +-- raw/all_v2.csv          (891 MB, в .gitignore)
|   +-- processed/              (графики PNG)
|   +-- geo/                    (шейпфайлы Natural Earth)
+-- notebooks/
|   +-- 01_eda.ipynb
+-- src/
|   +-- data/
|   |   +-- download.py
|   |   +-- clean.py
|   |   +-- contract.py
|   +-- models/
|   |   +-- baseline.py
|   +-- api/
|   |   +-- main.py
|   +-- app/
|       +-- streamlit_app.py
+-- tests/
|   +-- test_clean.py
|   +-- test_contract.py
|   +-- test_baseline.py
+-- docs/
    +-- ML_System_Design_Doc.md
    +-- checkpoint1_report.md   (<-- этот файл)
    +-- plans/
```

---

## Тесты

Все 9 тестов проходят:

| Модуль | Файл тестов | Кол-во |
|--------|-------------|--------|
| `src/data/clean.py` | `tests/test_clean.py` | -- |
| `src/data/contract.py` | `tests/test_contract.py` | -- |
| `src/models/baseline.py` | `tests/test_baseline.py` | -- |
| **Итого** | | **9/9 pass** |

---

## Быстрые ответы на вопросы защиты

| Вопрос | Ответ |
|--------|-------|
| Какой тип задачи? | Регрессия, предсказание цены квартиры |
| Какая основная метрика? | MAPE (Median Absolute Percentage Error) |
| Какой baseline? | Медиана по группе region+rooms+building_type, MAPE=23.5% |
| Сколько данных? | ~11.3 млн записей, 891 MB, семпл 1 млн для EDA |
| Какие пропуски? | street_id 32.5%, house_id 24%, postal_code 5% |
| Есть ли leakage? | Да: house_id, street_id, postal_code -- не использовать как фичи |
| Сколько выбросов удалено? | 3.6% записей |
| Сколько рисков? | 7 штук (минимум требовалось 5) |
| Где архитектура? | `docs/ML_System_Design_Doc.md` |
| Где код? | `src/` (data, models, api, app) |
