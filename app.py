import streamlit as st
import pandas as pd
import numpy as np
from datetime import date, timedelta

# ---------- Grundeinstellungen der Seite ----------
st.set_page_config(
    page_title="Portfolio Analyse",
    page_icon="📈",
    layout="wide",
)

# ---------- Sidebar: Steuerung / Einstellungen ----------
st.sidebar.title("⚙ Einstellungen")

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

# Auswahl des Tickers (später dynamisch aus Databricks laden)
ticker = st.sidebar.selectbox(
    "Ticker",
    options=["AAPL", "MSFT", "GOOG", "SPY"],
    index=0,
)

# Auswahl der Kennzahl (z. B. Kurs, Rendite)
metric = st.sidebar.selectbox(
    "Kennzahl",
    options=["Kurs", "Rendite", "Kumulierte Rendite"],
    index=0,
)

st.sidebar.markdown("---")
st.sidebar.caption(
    "Hinweis: Die Daten werden derzeit simuliert.\n"
    "Später wird hier eine Databricks-Abfrage integriert."
)

# ---------- Datenladen (aktuell simulierte Daten, später Databricks) ----------


def load_mock_data(start_date, end_date, freq: str, ticker: str) -> pd.DataFrame:
    """
    Simuliert Kurs- und Renditedaten für Demonstrationszwecke.
    Später wird diese Funktion ersetzt durch:
      - Laden von Daten aus Databricks
      - Filtern nach Datum und Ticker
      - Berechnung der relevanten Kennzahlen
    """

    # Pandas-Frequenz basierend auf der Auswahl der Benutzer:innen
    if freq == "Täglich":
        pandas_freq = "D"
    elif freq == "Wöchentlich":
        pandas_freq = "W"
    else:
        pandas_freq = "M"

    # Erzeugen einer Zeitreihe basierend auf dem gewählten Zeitraum
    dates = pd.date_range(start_date, end_date, freq=pandas_freq)
    if len(dates) == 0:
        return pd.DataFrame()

    # Simulieren einer Kursentwicklung auf Basis zufälliger Renditen
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

    # Berechnung der kumulierten Rendite
    df["Kumulierte_Rendite"] = (1 + df["Rendite"]).cumprod() - 1

    df.set_index("Datum", inplace=True)
    return df


df = load_mock_data(start_date, end_date, freq, ticker)

# ---------- Hauptinhalt der Seite ----------
st.title("📈 Portfolio Analyse – Visualisierung")

# Informationsbereich oben
st.markdown(
    f"""
**Auswahl:**
- Ticker: `{ticker}`
- Frequenz: `{freq}`
- Zeitraum: `{start_date} bis {end_date}`
"""
)

if df.empty:
    st.warning("Im gewählten Zeitraum sind keine Daten vorhanden.")
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
        label="Volatilität (simuliert)",
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
    df.reset_index(),
    use_container_width=True,
    hide_index=True,
)
