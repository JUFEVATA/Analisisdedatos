from __future__ import annotations

from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

from procesing import (
    load_data,
    missing_values_summary,
    numeric_summary,
    process_data,
)


st.set_page_config(
    page_title="Dashboard EDA de ventas",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');

        html, body, [class*="css"] {
            font-family: "Inter", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 8% 0%, rgba(37, 99, 235, 0.10), transparent 27%),
                radial-gradient(circle at 95% 3%, rgba(124, 58, 237, 0.10), transparent 25%),
                #f7f9fc;
        }

        .main .block-container {
            max-width: 1450px;
            padding-top: 1.6rem;
            padding-bottom: 3rem;
        }

        .dashboard-header {
            border-radius: 24px;
            padding: 2rem 2.2rem;
            color: white;
            background: linear-gradient(125deg, #0f172a, #1d4ed8 58%, #6d28d9);
            box-shadow: 0 22px 50px rgba(15, 23, 42, 0.18);
            margin-bottom: 1.4rem;
        }

        .dashboard-header h1 {
            margin: 0;
            font-size: clamp(2rem, 4vw, 3rem);
            font-weight: 800;
        }

        .dashboard-header p {
            max-width: 900px;
            margin: 0.7rem 0 0;
            color: rgba(255,255,255,0.82);
        }

        .tag {
            display: inline-block;
            padding: 0.35rem 0.7rem;
            margin-bottom: 0.8rem;
            border-radius: 999px;
            background: rgba(255,255,255,0.13);
            border: 1px solid rgba(255,255,255,0.22);
            font-size: 0.75rem;
            font-weight: 700;
        }

        div[data-testid="stMetric"] {
            background: rgba(255,255,255,0.95);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1rem;
            box-shadow: 0 9px 26px rgba(15,23,42,0.06);
        }

        div[data-testid="stPlotlyChart"] {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 20px;
            padding: 0.5rem;
            box-shadow: 0 10px 30px rgba(15,23,42,0.05);
        }

        div[data-testid="stDataFrame"] {
            border: 1px solid #e2e8f0;
            border-radius: 16px;
            overflow: hidden;
        }

        .section-title {
            font-size: 1.35rem;
            font-weight: 800;
            color: #0f172a;
            margin-top: 1rem;
            margin-bottom: 0.15rem;
        }

        .section-description {
            color: #64748b;
            margin-bottom: 1rem;
        }

        .info-card {
            background: rgba(255,255,255,0.96);
            border: 1px solid #e2e8f0;
            border-radius: 18px;
            padding: 1.1rem;
            box-shadow: 0 9px 26px rgba(15,23,42,0.05);
        }

        [data-testid="stSidebar"] {
            background: #ffffff;
            border-right: 1px solid #e2e8f0;
        }

        .small-note {
            font-size: 0.82rem;
            color: #64748b;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def prepare_data(source) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    original = load_data(source)
    clean, report = process_data(original)
    return original, clean, report.to_dict()


def format_currency(value: float) -> str:
    return f"${value:,.0f}".replace(",", ".")


def apply_filters(df: pd.DataFrame) -> pd.DataFrame:
    filtered = df.copy()

    st.sidebar.markdown("## Filtros")

    valid_dates = filtered["fecha"].dropna()
    if not valid_dates.empty:
        min_date = valid_dates.min().date()
        max_date = valid_dates.max().date()

        selected_dates = st.sidebar.date_input(
            "Rango de fechas",
            value=(min_date, max_date),
            min_value=min_date,
            max_value=max_date,
        )

        if isinstance(selected_dates, tuple) and len(selected_dates) == 2:
            start_date, end_date = selected_dates
            filtered = filtered[
                filtered["fecha"].dt.date.between(start_date, end_date)
            ]

    filter_columns = {
        "Ciudad": "ciudad",
        "Sucursal": "sucursal",
        "Categoría": "categoria",
        "Método de pago": "metodo_pago",
        "Tipo de cliente": "tipo_cliente",
        "Género": "genero",
    }

    for label, column in filter_columns.items():
        options = sorted(filtered[column].dropna().astype(str).unique())
        selected = st.sidebar.multiselect(
            label,
            options=options,
            default=options,
        )
        if selected:
            filtered = filtered[filtered[column].astype(str).isin(selected)]
        else:
            filtered = filtered.iloc[0:0]

    if not filtered.empty:
        age_min = int(filtered["edad"].min())
        age_max = int(filtered["edad"].max())
        selected_age = st.sidebar.slider(
            "Rango de edad",
            min_value=age_min,
            max_value=age_max,
            value=(age_min, age_max),
        )
        filtered = filtered[
            filtered["edad"].between(selected_age[0], selected_age[1])
        ]

        satisfaction_min = float(filtered["satisfaccion"].min())
        satisfaction_max = float(filtered["satisfaccion"].max())
        selected_satisfaction = st.sidebar.slider(
            "Satisfacción",
            min_value=satisfaction_min,
            max_value=satisfaction_max,
            value=(satisfaction_min, satisfaction_max),
            step=1.0,
        )
        filtered = filtered[
            filtered["satisfaccion"].between(
                selected_satisfaction[0],
                selected_satisfaction[1],
            )
        ]

    st.sidebar.markdown("---")
    st.sidebar.metric("Registros filtrados", f"{len(filtered):,}".replace(",", "."))
    st.sidebar.caption(
        "Los filtros afectan todos los indicadores, gráficos y tablas."
    )

    return filtered


st.markdown(
    """
    <div class="dashboard-header">
        <div class="tag">ANÁLISIS EXPLORATORIO DE DATOS</div>
        <h1>Dashboard EDA de ventas</h1>
        <p>
            Exploración interactiva del dataset después de aplicar las tareas de
            limpieza, estandarización e imputación desarrolladas en el Colab.
            Esta aplicación no entrena ni consume modelos de machine learning.
        </p>
    </div>
    """,
    unsafe_allow_html=True,
)


st.sidebar.markdown("# Fuente de datos")
uploaded_file = st.sidebar.file_uploader(
    "Cargar un archivo CSV",
    type=["csv"],
    help="Debe conservar las columnas utilizadas en el Colab.",
)

default_path = Path("data.csv")

try:
    if uploaded_file is not None:
        original_df, clean_df, processing_report = prepare_data(uploaded_file)
        source_name = uploaded_file.name
    elif default_path.exists():
        original_df, clean_df, processing_report = prepare_data(str(default_path))
        source_name = "data.csv"
    else:
        st.warning(
            "Carga el archivo CSV desde la barra lateral o agrega `data.csv` "
            "en la misma carpeta de la aplicación."
        )
        st.stop()
except Exception as exc:
    st.error(f"No fue posible procesar el archivo: {exc}")
    st.stop()


filtered_df = apply_filters(clean_df)

st.caption(
    f"Fuente activa: **{source_name}** · "
    f"Datos originales: **{len(original_df):,}** registros · "
    f"Datos limpios: **{len(clean_df):,}** registros"
)


if filtered_df.empty:
    st.warning("No existen registros que coincidan con los filtros seleccionados.")
    st.stop()


total_sales = filtered_df["total"].sum()
average_ticket = filtered_df["total"].mean()
transactions = len(filtered_df)
unique_clients = filtered_df["cliente_id"].nunique()
average_satisfaction = filtered_df["satisfaccion"].mean()
units_sold = filtered_df["cantidad"].sum()

metric_columns = st.columns(6)
metric_columns[0].metric("Ventas totales", format_currency(total_sales))
metric_columns[1].metric("Ticket promedio", format_currency(average_ticket))
metric_columns[2].metric(
    "Transacciones",
    f"{transactions:,}".replace(",", "."),
)
metric_columns[3].metric(
    "Clientes únicos",
    f"{unique_clients:,}".replace(",", "."),
)
metric_columns[4].metric("Unidades vendidas", f"{units_sold:,.0f}".replace(",", "."))
metric_columns[5].metric("Satisfacción media", f"{average_satisfaction:.2f}")


tab_overview, tab_customers, tab_products, tab_quality, tab_data = st.tabs(
    [
        "Resumen general",
        "Clientes",
        "Productos y categorías",
        "Calidad de datos",
        "Explorar datos",
    ]
)


with tab_overview:
    st.markdown(
        '<div class="section-title">Comportamiento general de las ventas</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Evolución temporal, distribución geográfica y medios de pago.</div>',
        unsafe_allow_html=True,
    )

    monthly_sales = (
        filtered_df.dropna(subset=["fecha"])
        .groupby("periodo", as_index=False)
        .agg(
            ventas=("total", "sum"),
            transacciones=("id_venta", "count"),
        )
        .sort_values("periodo")
    )

    figure_monthly = px.line(
        monthly_sales,
        x="periodo",
        y="ventas",
        markers=True,
        title="Ventas totales por mes",
        labels={"periodo": "Periodo", "ventas": "Ventas"},
    )
    figure_monthly.update_traces(
        hovertemplate="Periodo: %{x}<br>Ventas: $%{y:,.0f}<extra></extra>"
    )
    figure_monthly.update_layout(
        hovermode="x unified",
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
    )
    st.plotly_chart(figure_monthly, use_container_width=True)

    col_city, col_payment = st.columns(2)

    city_sales = (
        filtered_df.groupby("ciudad", as_index=False)
        .agg(
            ventas=("total", "sum"),
            transacciones=("id_venta", "count"),
        )
        .sort_values("ventas", ascending=False)
    )

    figure_city = px.bar(
        city_sales,
        x="ciudad",
        y="ventas",
        color="ventas",
        title="Ventas por ciudad",
        labels={"ciudad": "Ciudad", "ventas": "Ventas"},
        text_auto=".3s",
    )
    figure_city.update_layout(
        coloraxis_showscale=False,
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
    )
    col_city.plotly_chart(figure_city, use_container_width=True)

    payment_counts = (
        filtered_df["metodo_pago"]
        .value_counts()
        .rename_axis("metodo_pago")
        .reset_index(name="transacciones")
    )
    figure_payment = px.pie(
        payment_counts,
        names="metodo_pago",
        values="transacciones",
        hole=0.55,
        title="Distribución de métodos de pago",
    )
    figure_payment.update_traces(
        textposition="inside",
        textinfo="percent+label",
    )
    col_payment.plotly_chart(figure_payment, use_container_width=True)

    col_total_hist, col_total_box = st.columns(2)

    figure_total_hist = px.histogram(
        filtered_df,
        x="total",
        nbins=25,
        title="Distribución del total de compra",
        labels={"total": "Total de compra"},
    )
    figure_total_hist.update_layout(
        xaxis_tickprefix="$",
        xaxis_tickformat=",.0f",
    )
    col_total_hist.plotly_chart(figure_total_hist, use_container_width=True)

    figure_total_box = px.box(
        filtered_df,
        y="total",
        points="outliers",
        title="Valores atípicos del total de compra",
        labels={"total": "Total de compra"},
    )
    figure_total_box.update_layout(
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
    )
    col_total_box.plotly_chart(figure_total_box, use_container_width=True)


with tab_customers:
    st.markdown(
        '<div class="section-title">Análisis de los clientes</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Edades, tipos de cliente, género y satisfacción.</div>',
        unsafe_allow_html=True,
    )

    col_age, col_customer_type = st.columns(2)

    figure_age = px.histogram(
        filtered_df,
        x="edad",
        nbins=15,
        title="Distribución de edades",
        labels={"edad": "Edad"},
    )
    col_age.plotly_chart(figure_age, use_container_width=True)

    customer_type_counts = (
        filtered_df["tipo_cliente"]
        .value_counts()
        .rename_axis("tipo_cliente")
        .reset_index(name="ventas")
    )
    figure_customer_type = px.bar(
        customer_type_counts,
        x="tipo_cliente",
        y="ventas",
        color="tipo_cliente",
        title="Ventas por tipo de cliente",
        labels={
            "tipo_cliente": "Tipo de cliente",
            "ventas": "Número de ventas",
        },
        text_auto=True,
    )
    figure_customer_type.update_layout(showlegend=False)
    col_customer_type.plotly_chart(
        figure_customer_type,
        use_container_width=True,
    )

    col_gender, col_satisfaction = st.columns(2)

    gender_summary = (
        filtered_df.groupby("genero", as_index=False)
        .agg(
            ventas=("total", "sum"),
            ticket_promedio=("total", "mean"),
        )
        .sort_values("ventas", ascending=False)
    )
    figure_gender = px.bar(
        gender_summary,
        x="genero",
        y="ventas",
        color="genero",
        title="Ventas totales por género",
        labels={"genero": "Género", "ventas": "Ventas"},
        text_auto=".3s",
    )
    figure_gender.update_layout(
        showlegend=False,
        yaxis_tickprefix="$",
        yaxis_tickformat=",.0f",
    )
    col_gender.plotly_chart(figure_gender, use_container_width=True)

    satisfaction_counts = (
        filtered_df["satisfaccion"]
        .value_counts()
        .sort_index()
        .rename_axis("satisfaccion")
        .reset_index(name="registros")
    )
    figure_satisfaction = px.bar(
        satisfaction_counts,
        x="satisfaccion",
        y="registros",
        color="satisfaccion",
        title="Nivel de satisfacción",
        labels={
            "satisfaccion": "Satisfacción",
            "registros": "Número de registros",
        },
        text_auto=True,
    )
    figure_satisfaction.update_layout(coloraxis_showscale=False)
    col_satisfaction.plotly_chart(
        figure_satisfaction,
        use_container_width=True,
    )


with tab_products:
    st.markdown(
        '<div class="section-title">Productos y categorías</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Categorías con mayor actividad y productos más vendidos.</div>',
        unsafe_allow_html=True,
    )

    col_category, col_top_products = st.columns(2)

    category_summary = (
        filtered_df.groupby("categoria", as_index=False)
        .agg(
            transacciones=("id_venta", "count"),
            unidades=("cantidad", "sum"),
            ventas=("total", "sum"),
        )
        .sort_values("ventas", ascending=False)
    )

    category_metric = col_category.radio(
        "Medida para comparar categorías",
        options=["Ventas", "Transacciones", "Unidades"],
        horizontal=True,
    )
    metric_map = {
        "Ventas": "ventas",
        "Transacciones": "transacciones",
        "Unidades": "unidades",
    }
    selected_metric = metric_map[category_metric]

    figure_category = px.bar(
        category_summary,
        x="categoria",
        y=selected_metric,
        color=selected_metric,
        title=f"{category_metric} por categoría",
        labels={
            "categoria": "Categoría",
            selected_metric: category_metric,
        },
        text_auto=".3s",
    )
    figure_category.update_layout(coloraxis_showscale=False)
    if selected_metric == "ventas":
        figure_category.update_layout(
            yaxis_tickprefix="$",
            yaxis_tickformat=",.0f",
        )
    col_category.plotly_chart(figure_category, use_container_width=True)

    top_n = col_top_products.slider(
        "Cantidad de productos para mostrar",
        min_value=5,
        max_value=20,
        value=10,
    )
    top_products = (
        filtered_df.groupby("producto", as_index=False)
        .agg(
            transacciones=("id_venta", "count"),
            unidades=("cantidad", "sum"),
            ventas=("total", "sum"),
        )
        .sort_values("unidades", ascending=False)
        .head(top_n)
        .sort_values("unidades")
    )

    figure_products = px.bar(
        top_products,
        x="unidades",
        y="producto",
        orientation="h",
        color="unidades",
        title=f"Top {top_n} productos por unidades vendidas",
        labels={"producto": "Producto", "unidades": "Unidades"},
        text_auto=True,
    )
    figure_products.update_layout(coloraxis_showscale=False)
    col_top_products.plotly_chart(
        figure_products,
        use_container_width=True,
    )

    st.dataframe(
        category_summary.style.format(
            {
                "transacciones": "{:,.0f}",
                "unidades": "{:,.0f}",
                "ventas": "${:,.0f}",
            }
        ),
        use_container_width=True,
        hide_index=True,
    )


with tab_quality:
    st.markdown(
        '<div class="section-title">Calidad y transformación de los datos</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Resumen de las operaciones aplicadas en el Colab.</div>',
        unsafe_allow_html=True,
    )

    quality_columns = st.columns(4)
    quality_columns[0].metric(
        "Registros originales",
        f"{processing_report['registros_iniciales']:,}".replace(",", "."),
    )
    quality_columns[1].metric(
        "Registros limpios",
        f"{processing_report['registros_finales']:,}".replace(",", "."),
    )
    quality_columns[2].metric(
        "Duplicados eliminados",
        f"{processing_report['duplicados_eliminados']:,}".replace(",", "."),
    )
    quality_columns[3].metric(
        "Filas eliminadas",
        f"{processing_report['registros_iniciales'] - processing_report['registros_finales']:,}".replace(",", "."),
    )

    report_labels = {
        "duplicados_eliminados": "Duplicados eliminados",
        "edades_invalidas": "Edades fuera del rango 18–100",
        "edades_imputadas": "Edades imputadas con la mediana",
        "categorias_imputadas": "Categorías imputadas con la moda",
        "satisfacciones_imputadas": "Satisfacciones imputadas con la mediana",
        "cantidades_invalidas_eliminadas": "Cantidades no positivas eliminadas",
        "precios_invalidos_eliminados": "Precios no positivos eliminados",
        "descuentos_invalidos_eliminados": "Descuentos fuera de 0–40 % eliminados",
        "fechas_invalidas": "Fechas que no pudieron convertirse",
    }
    report_chart = pd.DataFrame(
        [
            {
                "operacion": label,
                "registros": processing_report[key],
            }
            for key, label in report_labels.items()
        ]
    )
    report_chart = report_chart[report_chart["registros"] > 0]

    if not report_chart.empty:
        figure_report = px.bar(
            report_chart.sort_values("registros"),
            x="registros",
            y="operacion",
            orientation="h",
            color="registros",
            title="Registros afectados por cada operación",
            labels={
                "operacion": "Operación",
                "registros": "Registros",
            },
            text_auto=True,
        )
        figure_report.update_layout(coloraxis_showscale=False)
        st.plotly_chart(figure_report, use_container_width=True)

    col_null_original, col_null_clean = st.columns(2)

    original_nulls = missing_values_summary(original_df)
    clean_nulls = missing_values_summary(clean_df)

    col_null_original.markdown("#### Valores nulos antes de la limpieza")
    col_null_original.dataframe(
        original_nulls,
        use_container_width=True,
        hide_index=True,
    )

    col_null_clean.markdown("#### Valores nulos después de la limpieza")
    col_null_clean.dataframe(
        clean_nulls,
        use_container_width=True,
        hide_index=True,
    )

    st.markdown("#### Matriz de correlación")
    numeric_columns = [
        "edad",
        "precio_unitario",
        "cantidad",
        "descuento",
        "satisfaccion",
        "total",
    ]
    available_numeric = [
        column for column in numeric_columns if column in filtered_df.columns
    ]

    correlation = filtered_df[available_numeric].corr(numeric_only=True)
    figure_correlation = px.imshow(
        correlation,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu_r",
        zmin=-1,
        zmax=1,
        title="Correlación entre variables numéricas",
    )
    st.plotly_chart(figure_correlation, use_container_width=True)


with tab_data:
    st.markdown(
        '<div class="section-title">Exploración de registros</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="section-description">Consulta, ordena y descarga los datos resultantes de los filtros.</div>',
        unsafe_allow_html=True,
    )

    selected_columns = st.multiselect(
        "Columnas visibles",
        options=filtered_df.columns.tolist(),
        default=[
            "id_venta",
            "fecha",
            "ciudad",
            "categoria",
            "producto",
            "cantidad",
            "precio_unitario",
            "descuento",
            "total",
        ],
    )

    if selected_columns:
        st.dataframe(
            filtered_df[selected_columns],
            use_container_width=True,
            hide_index=True,
            height=500,
        )

    col_summary, col_download = st.columns([1.3, 0.7])

    with col_summary:
        st.markdown("#### Resumen estadístico")
        summary = numeric_summary(filtered_df)
        st.dataframe(summary, use_container_width=True, hide_index=True)

    with col_download:
        st.markdown("#### Descargar")
        filtered_csv = filtered_df.to_csv(index=False).encode("utf-8")
        clean_csv = clean_df.to_csv(index=False).encode("utf-8")

        st.download_button(
            "Descargar datos filtrados",
            data=filtered_csv,
            file_name="ventas_filtradas.csv",
            mime="text/csv",
            use_container_width=True,
        )
        st.download_button(
            "Descargar dataset limpio",
            data=clean_csv,
            file_name="ventas_limpias.csv",
            mime="text/csv",
            use_container_width=True,
        )


st.markdown(
    """
    <p class="small-note" style="text-align:center; margin-top:2rem;">
        Dashboard descriptivo para análisis exploratorio. No realiza entrenamiento,
        clasificación, regresión ni predicciones.
    </p>
    """,
    unsafe_allow_html=True,
)
