# Feature Registry

Этот документ фиксирует набор признаков, используемых в проекте `RealEstate Price Explorer 2021`, и разграничивает offline/online использование.

## Правила

- `offline+online`: признак может использоваться и при обучении, и в runtime-инференсе.
- `offline`: признак строится в batch-пайплайне и не обязателен для online inference.
- `excluded`: признак не используется при моделировании из-за риска leakage или низкой надежности.

## Registry

| Feature | Source | Type | Availability | Update Frequency | Leakage Risk | Notes |
|---------|--------|------|--------------|------------------|--------------|-------|
| `region` | raw | categorical[int] | offline+online | on pipeline run / per request | low | Код региона РФ |
| `object_type` | raw | categorical[int] | offline+online | on pipeline run / per request | low | Вторичка / новостройка |
| `building_type` | raw | categorical[int] | offline+online | on pipeline run / per request | low | Тип здания |
| `rooms` | raw | integer | offline+online | on pipeline run / per request | low | `-1` обозначает студию |
| `area` | raw | float | offline+online | on pipeline run / per request | low | Площадь квартиры |
| `kitchen_area` | raw | float | offline+online | on pipeline run / per request | low | Площадь кухни |
| `geo_lat` | raw | float | offline | on pipeline run | medium | Использовать осторожно из-за сильного location signal |
| `geo_lon` | raw | float | offline | on pipeline run | medium | Использовать осторожно из-за сильного location signal |
| `floor` | raw | float | offline | on pipeline run | low | Этаж |
| `total_floors` | raw | float | offline | on pipeline run | low | Этажность здания |
| `is_studio` | derived | int | offline+online | on pipeline run / per request | low | Флаг студии |
| `log_area` | derived | float | offline+online | on pipeline run / per request | low | `log1p(area)` |
| `has_kitchen_area` | derived | int | offline+online | on pipeline run / per request | low | Есть ли информация о кухне |
| `kitchen_to_area_ratio` | derived | float | offline+online | on pipeline run / per request | low | Отношение площади кухни к общей площади |
| `floor_ratio` | derived | float | offline | on pipeline run | low | Относительное положение этажа |
| `is_top_floor` | derived | int | offline | on pipeline run | low | Квартира на последнем этаже |
| `is_first_floor` | derived | int | offline | on pipeline run | low | Квартира на первом этаже |
| `region_listing_count` | aggregate | int | offline | on pipeline run | low | Размер регионального среза в train snapshot |
| `region_median_area` | aggregate | float | offline | on pipeline run | low | Медианная площадь по региону |
| `house_id` | raw | identifier | excluded | never | high | Исключен из-за адресного leakage |
| `street_id` | raw | identifier | excluded | never | high | Исключен из-за адресного leakage |
| `postal_code` | raw | identifier | excluded | never | high | Исключен из-за адресного leakage |

## Комментарии по использованию

- Признаки, связанные с точным адресом, не включаются в модель, даже если они повышают качество на локальном тесте.
- Aggregate-признаки допустимы только при расчете на train snapshot, чтобы не вносить leakage из test-части.
- Online inference в MVP должен опираться на простые детерминированные признаки, которые можно построить прямо из пользовательского запроса.
