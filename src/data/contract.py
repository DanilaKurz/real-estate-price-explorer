"""Data Contract - валидация схемы данных о недвижимости."""

from dataclasses import dataclass
import pandas as pd


@dataclass
class DataContractViolation:
    """Нарушение Data Contract."""
    field: str
    rule: str
    details: str
    severity: str = "error"


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
        "max": 999,
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
    """Валидирует DataFrame по Data Contract."""
    violations = []

    for field, rules in DATA_CONTRACT.items():
        if field not in df.columns:
            if rules.get("required", False):
                violations.append(DataContractViolation(
                    field=field,
                    rule="required_field",
                    details=f"Обязательное поле '{field}' отсутствует",
                ))
            continue

        col = df[field]

        null_pct = col.isna().mean()
        max_null = rules.get("max_null_pct", 1.0)
        if null_pct > max_null:
            violations.append(DataContractViolation(
                field=field,
                rule="null_percentage",
                details=f"Доля пропусков {null_pct:.2%} > допустимых {max_null:.2%}",
                severity="warning" if null_pct < max_null * 2 else "error",
            ))

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
