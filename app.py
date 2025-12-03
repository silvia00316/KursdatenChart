import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ---------------------------------------------------
# 模式开关：
# True  = 本地开发（不用 Databricks，不用 Spark，用 Mock 数据）
# False = Databricks App 模式（用 Gold-Tabelle）
# ---------------------------------------------------
RUN_LOCAL = False

# Databricks 模式下导入 Spark
if not RUN_LOCAL:
    from pyspark.sql import functions as F
    from pyspark.sql import SparkSession


# ------------------------- 初始化 -------------------------

st.set_page_config(
    page_title="Portfolio Analyse",
    page_icon="📈",
    layout="wide",
)


def get_spark():
    """提供 SparkSession（仅 Databricks 模式下）"""
    if RUN_LOCAL:
        raise RuntimeError("Spark nur in Databricks-Modus verfügbar.")

    try:
        return SparkSession.builder.getOrCreate()
    except Exception as e:
        st.error(f"⚠ Spark konnte nicht initialisiert werden: {e}")
        st.stop()


# ---------------------- 1) 加载所有 Ticker ----------------------

@st.cache_data
def load_available_tickers():
    if RUN_LOCAL:
        return ["AAPL", "MSFT", "GOOG", "SPY"]

    spark = get_spark()

    try:
        sdf = (
            spark.table("tud_25.gold.alpha_vantage_marketdata_final")
            .select("symbol")
            .distinct()
            .orderBy("symbol")
        )
        tickers = [row["symbol"] for row in sdf.collect()]
        return tickers

    except Exception as e:
        st.sidebar.error(f"⚠ Fehler beim Laden der Ticker: {e}")
        return []


# ---------------------- 2) 本地模拟数据 ----------------------

def load_mock_data(start_date, end_date, freq, ticker):
    if freq == "Täglich":
        pandas_freq = "D"
    elif freq == "Wöchentlich":
        pandas_freq = "W"
    else:
        pandas_freq = "M"

    dates = pd.date_range(start_date, end_date, freq=pandas_freq)
    np.random.seed(42)
    returns = np.random.normal(0.001, 0.02, len(dates))
    price = 100 * (1 + pd.Series(returns)).cumprod()

    df = pd.DataFrame({
        "Datum": dates,
        "Ticker": ticker,
        "Preis": price,
        "Rendite": returns,
    })
    df["Kumulierte_Rendite"] = (1 + df["Rendite"]).cumprod() - 1
    df.set_index("Datum", inplace=True)

    return df


# ---------------------- 3) 从 Databricks Gold-Tabelle 加载数据 ----------------------

@st.cache_data
def load_kursdaten_from_gold(start_date, end_date, freq, ticker):
    if RUN_LOCAL:
        raise RuntimeError(
            "Gold-Tabelle wird nur im Databricks-Modus geladen.")

    spark = get_spark()

    try:
        sdf = (
            spark.table("tud_25.gold.alpha_vantage_marketdata_final")
            .where(
                (F.col("symbol") == ticker)
                & (F.col("date") >= F.lit(start_date))
                & (F.col("date") <= F.lit(end_date))
            )
            .select("date", "symbol", "close")
            .orderBy("date")
        )
    except Exception as e:
        st.error(f"⚠ Fehler beim Spark-Query: {e}")
        return pd.DataFrame()

    if sdf.rdd.isEmpty():
        return pd.DataFrame()

    pdf = sdf.toPandas()
    pdf["Datum"] = pd.to_datetime(pdf["date"])
    pdf = pdf.sort_values("Datum").set_index("Datum")

    # Resampling
    pandas_freq = {"Täglich": "D", "Wöchentlich": "W", "Monatlich": "M"}[freq]
    pdf_res = pdf.resample(pandas_freq).last().dropna(subset=["close"])

    df = pd.DataFrame({
        "Ticker": ticker,
        "Preis": pdf_res["close"],
    })
    df["Rendite"] = df["Preis"].pct_change().fillna(0)
    df["Kumulierte_Rendite"] = (1 + df["Rendite"]).cumprod() - 1

    return df


# ---------------------- 4) 统一的数据加载接口 ----------------------

def load_data(start_date, end_date, freq, ticker):
    try:
        if RUN_LOCAL:
            return load_mock_data(start_date, end_date, freq, ticker)
        else:
            return load_kursdaten_from_gold(start_date, end_date, freq, ticker)
    except Exception as e:
        st.error(f"⚠ Fehler beim Laden der Daten: {e}")
        return pd.DataFrame()


# ---------------------- Sidebar ----------------------

st.sidebar.title("⚙ Einstellungen")

all_tickers = load_available_tickers()
if not all_tickers:
    st.sidebar.error("⚠ Keine Ticker gefunden.")
    st.stop()

freq = st.sidebar.selectbox(
    "Frequenz", ["Täglich", "Wöchentlich", "Monatlich"], index=0)

default_end = date.today()
default_start = default_end - timedelta(days=365)

start_date = st.sidebar.date_input("Startdatum", value=default_start)
end_date = st.sidebar.date_input("Enddatum", value=default_end)

ticker = st.sidebar.selectbox("Ticker", all_tickers, index=0)

metric = st.sidebar.selectbox(
    "Kennzahl", ["Kurs", "Rendite", "Kumulierte Rendite"], index=0)

st.sidebar.markdown("---")
if RUN_LOCAL:
    st.sidebar.caption("Modus: Lokal (Mock-Daten)")
else:
    st.sidebar.caption("Databricks-Modus – Quelle: Gold-Tabelle")


# ---------------------- 主界面 ----------------------

st.title("📈 Portfolio Analyse – Kursdaten")

df = load_data(start_date, end_date, freq, ticker)

if df.empty:
    st.warning("⚠ Keine Kursdaten im gewählten Zeitraum gefunden.")
    st.stop()

st.markdown(
    f"""
**Auswahl:**
- Ticker: `{ticker}`
- Frequenz: `{freq}`
- Zeitraum: `{start_date} bis {end_date}`
"""
)

# KPIs
col1, col2, col3 = st.columns(3)
col1.metric("Letzter Preis", f"{df['Preis'].iloc[-1]:.2f}")
col2.metric("Gesamtrendite",
            f"{df['Kumulierte_Rendite'].iloc[-1] * 100:.2f} %")
col3.metric("Volatilität",
            f"{df['Rendite'].std() * np.sqrt(len(df)) * 100:.2f} %")

st.markdown("---")

# Charts
left, right = st.columns((2, 1))

with left:
    st.subheader("Kursverlauf")
    if metric == "Kurs":
        st.line_chart(df["Preis"])
    elif metric == "Rendite":
        st.line_chart(df["Rendite"])
    else:
        st.line_chart(df["Kumulierte_Rendite"])

with right:
    st.subheader("Rendite-Verteilung")
    st.bar_chart(df["Rendite"])

st.markdown("---")

# Tabelle
st.subheader("Tabellarische Daten")
st.dataframe(df.reset_index(), use_container_width=True)
