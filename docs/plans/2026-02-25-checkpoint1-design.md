# Дизайн-документ: Чекпоинт 1 - Постановка задачи и первичное проектирование

## Обзор проекта

**Название:** RealEstate Price Explorer 2021
**Суть:** ML-система для оценки стоимости квартиры в России по параметрам на основе данных 2021 года.
**Датасет:** https://www.kaggle.com/datasets/mrdaniilak/russia-real-estate-2021 (~11.3 млн записей)
**Команда:** 2 человека
**Стек:** FastAPI + Streamlit + scikit-learn/CatBoost
**Деплой:** Streamlit Cloud

## Архитектура

```
[Kaggle Dataset CSV]
        |
   [ETL Pipeline] --> [Cleaned Data / Parquet]
        |
   [Model Training] --> [Saved Model (.joblib)]
        |
   [FastAPI Service]  <-- REST API (predict, retrain, health)
        |
   [Streamlit UI]  <-- Пользовательский интерфейс
        |
   [Streamlit Cloud]  <-- Деплой
```

## Что входит в Чекпоинт 1

1. **Постановка задачи** - формализация ML-задачи, baseline
2. **EDA** - анализ данных, визуализации, data leakage анализ
3. **Data Contract** - схема данных, ограничения, freshness
4. **Архитектура системы** - диаграмма до выбора инструментов
5. **ML System Design Doc** - заполненный шаблон
6. **Риски (v0)** - минимум 5 рисков

## Подход к дообучению

Датасет будет засплитчен:
- 80% - обучение базовой модели (чекпоинт 2-3)
- 10% - валидация
- 10% - "новые данные" для демонстрации дообучения (чекпоинт 3)
