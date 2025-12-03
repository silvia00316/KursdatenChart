import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ---------------------------------------------------
# 模式开关：
#   True  = 本地开发 / 调试（不用 Databricks，不用 pyspark，用模拟数据）
#   False = 在 Databricks App 里跑（用 Gold-Tabelle + pyspark）
# ---------------------------------------------------
RUN_LOCAL = True

# 只有在 Databricks 模式下才导入 pyspark
if not RUN_LOCAL:
    from pyspark.sql import functions as F
    from pyspark.sql import SparkSession

# ---------- Grundeinstellungen der Seite ----------
st.set_page_config(
    page_title="Portfolio Analyse – Kursdaten aus Gold-Tabelle",
    page_icon="📈",
    layout="wide",
)

# ---------- Hilfsfunktionen für Spark & Datenzugriff ----------

def get_spark():
    """Nur in Databricks-Modus verwendbar."""
    if RUN_LOCAL:
        raise RuntimeError("get_spark wird nur im Databricks-Modus verwendet.")
    try:
        spark  # type: ignore[name-defined]
        return spark  # type: ignore[return-value]
    except NameError:
        return SparkSession.builder.getOrCreate()


@st.cache_data
def load_available_tickers():
    """
    Lädt alle verfügbaren Ticker.
    - Lokal: feste Demo-Liste
    - Databricks: distinct symbol aus der Gold-Tabelle
    """
    if RUN_LOCAL:
        # 本地调试就用几个固定的 Ticker
        return ["AAPL", "MSFT", "GOOG", "SPY"]

    # Databricks-Modus：从 Gold-Tabelle 读取真实的 symbol
    spark = get_spark()
    sdf = (
        spark.table("tud_25.gold.alpha_vantage_marketdata_final")
        .select("symbol")
        .distinct()
        .orderBy("symbol")
    )
    tickers = [row["symbol"] for row in sdf.collect()]
    return tickers


# ---------- 本地模拟数据（Mock） ----------
def load_mock_data(start_date, end_date, freq: str, ticker: str) -> pd.DataFrame:
    """
    本地调试用：模拟 Kurs- und Renditedaten
    """

    if freq == "Täglich":
        pandas_freq = "D"
    elif freq == "Wöchentlich":
        pandas_freq = "W"
    else:
        pandas_freq = "M"

    dates = pd.date_range(start_date, end_date, freq=pandas_freq)
    if len(dates) == 0:
        return pd.DataFrame()

    np.random.seed(42)
    returns = np.random.normal(loc=0.001, scale=0.02, size=len(dates))
    price = 100 * (1 + pd.Series(returns)).cumprod()

    df = pd.DataFrame(
        {
            "Datum": dates,
            "Ticker": ticker,
            "Preis": price,
            "Rendite": returns,
        }
    )

    df["Kumulierte_Rendite"] = (1 + df["Rendite"]).cumprod() - 1
    df.set_index("Datum", inplace=True)
    return df


# ---------- Databricks: aus Gold-Tabelle laden ----------
@st.cache_data
def load_kursdaten_from_gold(
    start_date: date,
    end_date: date,
    freq: str,
    ticker: str,
) -> pd.DataFrame:
    """
    Databricks-Modus：从 Gold-Tabelle 加载真实 Kursdaten
    并计算 Preis, Rendite, kumulierte Rendite.
    """
    if RUN_LOCAL:
        # 安全保护，防止本地误调用
        raise RuntimeError("load_kursdaten_from_gold wird nur im Databricks-Modus verwendet.")

    spark = get_spark()

    # 1) Spark-Query auf die Gold-Tabelle
    sdf = (
        spark.table("tud_25.gold.alpha_vantage_marketdata_final")
        .where(
            (F.col("symbol") == ticker)
            & (F.col("date") >= F.lit(start_date))
            & (F.col("date") <= F.lit(end_date))
        )
        .select("date", "symbol", "close")  # falls Spalten anders heißen → hier anpassen
    )

    if sdf.rdd.isEmpty():
        return pd.DataFrame()

    # 2) nach Pandas holen und Datum setzen
    pdf = sdf.toPandas()
    pdf["Datum"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("Datum").set_index("Datum")

    # 3) Frequenz-Anpassung (Resampling)
    if freq == "Täglich":
        pandas_freq = "D"
    elif freq == "Wöchentlich":
        pandas_freq = "W"
    else:  # "Monatlich"
        pandas_freq = "M"

    pdf_resampled = (
        pdf.resample(pandas_freq)
        .last()              # letzter Kurs im jeweiligen Intervall
        .dropna(subset=["close"])
    )

    if pdf_resampled.empty:
        return pd.DataFrame()

    # 4) DataFrame im Format deiner ursprünglichen App aufbauen
    df = pd.DataFrame(
        {
            "Ticker": ticker,
            "Preis": pdf_resampled["close"],
        }
    )

    # einfache Tages-/Wochen-/Monatsrendite
    df["Rendite"] = df["Preis"].pct_change().fillna(0.0)
    df["Kumulierte_Rendite"] = (1 + df["Rendite"]).cumprod() - 1

    return df


# ---------- 一个统一的数据加载入口 ----------
def load_data(start_date, end_date, freq, ticker) -> pd.DataFrame:
    if RUN_LOCAL:
        return load_mock_data(start_date, end_date, freq, ticker)
    else:
        return load_kursdaten_from_gold(start_date, end_date, freq, ticker)


# ---------- Sidebar: Steuerung / Einstellungen ----------
st.sidebar.title("⚙ Einstellungen")

# 先加载所有 Ticker（本地 = Demo 列表，Databricks = Gold-Tabelle）
all_tickers = load_available_tickers()
if not all_tickers:
    st.sidebar.error("Keine Ticker gefunden.")
    st.stop()

# Auswahl der Frequenz
freq = st.sidebar.selectbox(
    "Frequenz",
    options=["Täglich", "Wöchentlich", "Monatlich"],
    index=0,
)

# Datumsbereich (Standard: letztes Jahr)
default_end = date.today()
default_start = default_end - timedelta(days=365)

start_date = st.sidebar.date_input("Startdatum", value=default_start)
end_date = st.sidebar.date_input("Enddatum", value=default_end)

# Auswahl des Tickers
ticker = st.sidebar.selectbox(
    "Ticker",
    options=all_tickers,
    index=0,
)

# Auswahl der Kennzahl (z. B. Kurs, Rendite)
metric = st.sidebar.selectbox(
    "Kennzahl",
    options=["Kurs", "Rendite", "Kumulierte Rendite"],
    index=0,
)

st.sidebar.markdown("---")
if RUN_LOCAL:
    st.sidebar.caption("Modus: Lokal (Mock-Daten, keine Verbindung zu Databricks).")
else:
    st.sidebar.caption(
        "Modus: Databricks – Datenquelle: `tud_25.gold.alpha_vantage_marketdata_final`."
    )

# ---------- Daten laden ----------
df = load_data(start_date, end_date, freq, ticker)

# ---------- Hauptinhalt ----------
st.title("📈 Portfolio Analyse – Kursdaten")

st.markdown(
    f"""
**Auswahl:**
- Ticker: `{ticker}`
- Frequenz: `{freq}`
- Zeitraum: `{start_date} bis {end_date}`
"""
)

if df.empty:
    st.warning("Im gewählten Zeitraum sind keine Kursdaten vorhanden.")
    st.stop()

# ---------- Kennzahlen (KPI Cards) ----------
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        label="Letzter Preis",
        value=f"{df['Preis'].iloc[-1]:.2f}",
    )

with col2:
    total_return = df["Kumulierte_Rendite"].iloc[-1]
    st.metric(
        label="Gesamtrendite",
        value=f"{total_return * 100:.2f} %",
    )

with col3:
    vol = df["Rendite"].std() * np.sqrt(len(df))
    st.metric(
        label="Volatilität",
        value=f"{vol * 100:.2f} %",
    )

st.markdown("---")

# ---------- Diagramme ----------
left_col, right_col = st.columns((2, 1))

with left_col:
    st.subheader("Kursverlauf")
    if metric == "Kurs":
        st.line_chart(df["Preis"], height=300)
    elif metric == "Rendite":
        st.line_chart(df["Rendite"], height=300)
    else:
        st.line_chart(df["Kumulierte_Rendite"], height=300)

with right_col:
    st.subheader("Verteilung der Renditen")
    st.bar_chart(df["Rendite"], height=300)

st.markdown("---")

# ---------- Tabellarische Darstellung ----------
st.subheader("Tabellarische Daten")
st.dataframe(
    df.reset_index(),  # Index = Datum
    use_container_width=True,
    hide_index=True,
)

