from __future__ import annotations

from dataclasses import dataclass, asdict
from typing import Any

import numpy as np
import pandas as pd


@dataclass
class ProcessingReport:
    registros_iniciales: int
    columnas_iniciales: int
    duplicados_detectados: int
    duplicados_eliminados: int
    edades_invalidas: int
    edades_imputadas: int
    categorias_imputadas: int
    satisfacciones_imputadas: int
    cantidades_invalidas_eliminadas: int
    precios_invalidos_eliminados: int
    descuentos_invalidos_eliminados: int
    fechas_invalidas: int
    registros_finales: int
    columnas_finales: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def load_data(source: Any) -> pd.DataFrame:
    """
    Carga el archivo CSV desde una ruta local o desde un archivo subido
    con st.file_uploader.
    """
    return pd.read_csv(source)


def process_data(df: pd.DataFrame) -> tuple[pd.DataFrame, ProcessingReport]:
    """
    Replica las transformaciones realizadas en el Colab:

    1. Eliminación de duplicados.
    2. Conversión de la fecha.
    3. Estandarización de ciudades.
    4. Conversión de edades inválidas a valores nulos.
    5. Imputación de edad con la mediana.
    6. Eliminación de cantidades no positivas.
    7. Eliminación de precios unitarios no positivos.
    8. Conservación de descuentos entre 0 y 0.40.
    9. Imputación de categoría con la moda.
    10. Imputación de satisfacción con la mediana.
    11. Creación de variables temporales para el dashboard.
    """
    required_columns = {
        "id_venta",
        "fecha",
        "ciudad",
        "sucursal",
        "cliente_id",
        "genero",
        "edad",
        "categoria",
        "producto",
        "precio_unitario",
        "cantidad",
        "descuento",
        "metodo_pago",
        "tipo_cliente",
        "satisfaccion",
        "total",
    }

    missing_columns = required_columns.difference(df.columns)
    if missing_columns:
        raise ValueError(
            "El archivo no contiene todas las columnas requeridas. "
            f"Faltan: {', '.join(sorted(missing_columns))}"
        )

    clean = df.copy()

    initial_rows, initial_columns = clean.shape
    duplicated_rows = int(clean.duplicated().sum())

    clean = clean.drop_duplicates().copy()

    clean["fecha"] = pd.to_datetime(
        clean["fecha"],
        errors="coerce",
        format="mixed",
        dayfirst=False,
    )
    invalid_dates = int(clean["fecha"].isna().sum())

    city_replacements = {
        "bogota": "Bogotá",
        "Bogota": "Bogotá",
        "BOGOTA": "Bogotá",
        "Bogotá": "Bogotá",
        "medellin": "Medellín",
        "Medellin": "Medellín",
        "MEDELLIN": "Medellín",
        "Medellín": "Medellín",
        "cali": "Cali",
        "CALI": "Cali",
        "Cali": "Cali",
    }
    clean["ciudad"] = clean["ciudad"].replace(city_replacements)

    clean["edad"] = pd.to_numeric(clean["edad"], errors="coerce")
    invalid_age_mask = (clean["edad"] < 18) | (clean["edad"] > 100)
    invalid_ages = int(invalid_age_mask.sum())
    clean.loc[invalid_age_mask, "edad"] = np.nan

    ages_to_impute = int(clean["edad"].isna().sum())
    age_median = clean["edad"].median()
    clean["edad"] = clean["edad"].fillna(age_median)

    clean["cantidad"] = pd.to_numeric(clean["cantidad"], errors="coerce")
    invalid_quantity_mask = clean["cantidad"].isna() | (clean["cantidad"] <= 0)
    invalid_quantities = int(invalid_quantity_mask.sum())
    clean = clean.loc[~invalid_quantity_mask].copy()

    clean["precio_unitario"] = pd.to_numeric(
        clean["precio_unitario"], errors="coerce"
    )
    invalid_price_mask = (
        clean["precio_unitario"].isna()
        | (clean["precio_unitario"] <= 0)
    )
    invalid_prices = int(invalid_price_mask.sum())
    clean = clean.loc[~invalid_price_mask].copy()

    clean["descuento"] = pd.to_numeric(clean["descuento"], errors="coerce")
    invalid_discount_mask = (
        clean["descuento"].isna()
        | (clean["descuento"] < 0)
        | (clean["descuento"] > 0.40)
    )
    invalid_discounts = int(invalid_discount_mask.sum())
    clean = clean.loc[~invalid_discount_mask].copy()

    categories_to_impute = int(clean["categoria"].isna().sum())
    if categories_to_impute:
        category_mode = clean["categoria"].mode(dropna=True)
        if not category_mode.empty:
            clean["categoria"] = clean["categoria"].fillna(category_mode.iloc[0])

    clean["satisfaccion"] = pd.to_numeric(
        clean["satisfaccion"], errors="coerce"
    )
    satisfactions_to_impute = int(clean["satisfaccion"].isna().sum())
    satisfaction_median = clean["satisfaccion"].median()
    clean["satisfaccion"] = clean["satisfaccion"].fillna(satisfaction_median)

    clean["total"] = pd.to_numeric(clean["total"], errors="coerce")
    clean["mes"] = clean["fecha"].dt.month
    clean["mes_nombre"] = clean["fecha"].dt.month_name(locale="C")
    clean["anio"] = clean["fecha"].dt.year
    clean["periodo"] = clean["fecha"].dt.to_period("M").astype("string")
    clean["descuento_porcentaje"] = clean["descuento"] * 100

    month_names = {
        "January": "Enero",
        "February": "Febrero",
        "March": "Marzo",
        "April": "Abril",
        "May": "Mayo",
        "June": "Junio",
        "July": "Julio",
        "August": "Agosto",
        "September": "Septiembre",
        "October": "Octubre",
        "November": "Noviembre",
        "December": "Diciembre",
    }
    clean["mes_nombre"] = clean["mes_nombre"].replace(month_names)

    clean = clean.sort_values("fecha").reset_index(drop=True)

    report = ProcessingReport(
        registros_iniciales=initial_rows,
        columnas_iniciales=initial_columns,
        duplicados_detectados=duplicated_rows,
        duplicados_eliminados=duplicated_rows,
        edades_invalidas=invalid_ages,
        edades_imputadas=ages_to_impute,
        categorias_imputadas=categories_to_impute,
        satisfacciones_imputadas=satisfactions_to_impute,
        cantidades_invalidas_eliminadas=invalid_quantities,
        precios_invalidos_eliminados=invalid_prices,
        descuentos_invalidos_eliminados=invalid_discounts,
        fechas_invalidas=invalid_dates,
        registros_finales=len(clean),
        columnas_finales=clean.shape[1],
    )

    return clean, report


def numeric_summary(df: pd.DataFrame) -> pd.DataFrame:
    numeric = df.select_dtypes(include="number")
    if numeric.empty:
        return pd.DataFrame()

    summary = numeric.describe().T.reset_index()
    summary = summary.rename(
        columns={
            "index": "variable",
            "count": "registros",
            "mean": "media",
            "std": "desviacion",
            "min": "minimo",
            "25%": "percentil_25",
            "50%": "mediana",
            "75%": "percentil_75",
            "max": "maximo",
        }
    )
    return summary


def missing_values_summary(df: pd.DataFrame) -> pd.DataFrame:
    summary = pd.DataFrame(
        {
            "variable": df.columns,
            "valores_nulos": df.isna().sum().values,
            "porcentaje_nulos": (
                df.isna().mean().mul(100).round(2).values
            ),
            "tipo_dato": df.dtypes.astype(str).values,
        }
    )
    return summary.sort_values(
        ["valores_nulos", "variable"],
        ascending=[False, True],
    ).reset_index(drop=True)
