# -----------------------------
# Enhanced VIX Portfolio Sizer with Volatility Analytics
# -----------------------------

import streamlit as st
import yfinance as yf
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import numpy as np

# -----------------------------
# Functions
# -----------------------------

def fetch_vix3m_latest() -> float:
    hist = yf.Ticker("^VIX3M").history(period="5d", interval="1d", auto_adjust=False)
    if hist.empty:
        raise RuntimeError("No recent data returned for ^VIX3M.")
    return float(hist["Close"].iloc[-1])

def fetch_vix3m_ref_one_year() -> float:
    ref_dt = (pd.Timestamp.today() - pd.Timedelta(days=365)).normalize()
    start = (ref_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    end = (ref_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
    df = yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=False)
    if df.empty:
        raise RuntimeError("Failed to fetch historical VIX3M.")
    dates = df.index[df.index <= ref_dt]
    if dates.empty:
        return float(df["Close"].iloc[0])
    return float(df.loc[dates[-1], "Close"])

def compute_multiplier(vix3m_now: float, vix3m_ref: float, min_mult: float = 0.1) -> float:
    if vix3m_now <= 0 or vix3m_ref <= 0:
        raise ValueError("VIX3M values must be positive.")
    return max(vix3m_ref / vix3m_now, min_mult)

@st.cache_data
def fetch_vix_historical_data():
    """Fetch and calculate VIX historical volatility data"""
    # Fetch VIX data for the last 5 years
    vix_data = yf.download("^VIX", period="5y", interval="1d", progress=False, auto_adjust=False)

    # Calculate rolling volatilities
    def calculate_rolling_vol(data, window_days):
        daily_returns = data[('Close', '^VIX')].pct_change()
        rolling_vol = daily_returns.rolling(window=window_days).std() * np.sqrt(252) * 100
        return rolling_vol

    vix_data['3M_Vol'] = calculate_rolling_vol(vix_data, 63)
    vix_data['6M_Vol'] = calculate_rolling_vol(vix_data, 126)
    vix_data['1Y_Vol'] = calculate_rolling_vol(vix_data, 252)
    vix_data['5Y_Vol'] = calculate_rolling_vol(vix_data, 1260)

    # Clean up column names and prepare data
    clean_data = pd.DataFrame()
    clean_data['Close'] = vix_data[('Close', '^VIX')]
    clean_data['3M_Vol'] = vix_data['3M_Vol']
    clean_data['6M_Vol'] = vix_data['6M_Vol']
    clean_data['1Y_Vol'] = vix_data['1Y_Vol']
    clean_data['5Y_Vol'] = vix_data['5Y_Vol']
    clean_data['Date'] = vix_data.index
    clean_data['Year'] = vix_data.index.year

    return clean_data

def create_volatility_stats_table(vix_data):
    """Create volatility statistics table"""
    def get_vol_stats(vol_series, period_name):
        valid_data = vol_series.dropna()
        if len(valid_data) == 0:
            return pd.Series({
                'Period': period_name,
                'Current': np.nan,
                'Mean': np.nan,
                'Median': np.nan,
                'Min': np.nan,
                'Max': np.nan,
                '25th %ile': np.nan,
                '75th %ile': np.nan
            })

        return pd.Series({
            'Period': period_name,
            'Current': float(valid_data.iloc[-1]),
            'Mean': float(valid_data.mean()),
            'Median': float(valid_data.median()),
            'Min': float(valid_data.min()),
            'Max': float(valid_data.max()),
            '25th %ile': float(valid_data.quantile(0.25)),
            '75th %ile': float(valid_data.quantile(0.75))
        })

    stats_list = []
    for period, col in [('3 Months', '3M_Vol'), ('6 Months', '6M_Vol'), ('1 Year', '1Y_Vol'), ('5 Years', '5Y_Vol')]:
        stats_list.append(get_vol_stats(vix_data[col], period))

    return pd.DataFrame(stats_list).round(2)

def create_volatility_chart(vix_data, selected_years):
    """Create interactive volatility chart"""
    # Filter data by selected years
    if selected_years:
        filtered_data = vix_data[vix_data['Year'].isin(selected_years)]
    else:
        filtered_data = vix_data

    # Remove NaN values
    filtered_data = filtered_data.dropna(subset=['3M_Vol', '6M_Vol', '1Y_Vol'])

    fig = go.Figure()

    # Add volatility lines
    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['3M_Vol'],
        mode='lines',
        name='3-Month Volatility',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['6M_Vol'],
        mode='lines',
        name='6-Month Volatility',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=filtered_data['Date'],
        y=filtered_data['1Y_Vol'],
        mode='lines',
        name='1-Year Volatility',
        line=dict(color='red', width=2)
    ))

    # Add current VIX level as reference
    current_vix = float(filtered_data['Close'].iloc[-1])
    fig.add_hline(
        y=current_vix,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Current VIX: {current_vix:.2f}"
    )

    fig.update_layout(
        title="VIX Volatility Analysis",
        xaxis_title="Date",
        yaxis_title="Volatility (%)",
        hovermode='x unified',
        height=600
    )

    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Enhanced VIX Portfolio Sizer", layout="wide")
st.title("📊 Enhanced VIX Portfolio Sizer Dashboard")

st.markdown("""
This enhanced dashboard provides VIX portfolio sizing with comprehensive volatility analytics including:
- Historical volatility statistics across multiple timeframes
- Interactive charts with year filtering
- Portfolio multiplier calculations
""")

# Sidebar for controls
with st.sidebar:
    st.header("Controls")
    override_ref = st.number_input("Override numeric reference (optional)", min_value=0.0, value=0.0, step=0.1)
    override_date = st.date_input("Override reference date (optional)", value=None)

# Main content in columns
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("Portfolio Multiplier")

    # Original VIX3M logic
    try:
        v_now = fetch_vix3m_latest()
    except Exception as e:
        st.error(f"Failed to fetch latest VIX3M: {e}")
        st.stop()

    if override_ref and override_ref > 0:
        v_ref = override_ref
        ref_source = "User-supplied"
    else:
        try:
            if override_date:
                ref_dt = pd.Timestamp(override_date)
                start = (ref_dt - pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                end = (ref_dt + pd.Timedelta(days=10)).strftime("%Y-%m-%d")
                df = yf.download("^VIX3M", start=start, end=end, progress=False, auto_adjust=False)
                if df.empty:
                    st.warning(f"No historical data for {ref_dt.date()}, using 1-year-ago default.")
                    v_ref = fetch_vix3m_ref_one_year()
                else:
                    dates = df.index[df.index <= ref_dt]
                    if len(dates) == 0:
                        v_ref = float(df[('Close', '^VIX3M')].iloc[0])
                    else:
                        v_ref = float(df.loc[dates[-1], ('Close', '^VIX3M')])
                ref_source = f"Historical ({ref_dt.date()})"
            else:
                v_ref = fetch_vix3m_ref_one_year()
                ref_source = "1-year-ago default"
        except Exception as e:
            st.error(f"Failed to fetch reference VIX3M: {e}")
            st.stop()

    mult = compute_multiplier(v_now, v_ref)

    st.metric(label="Latest VIX3M", value=f"{v_now:.2f}")
    st.metric(label=f"Reference VIX3M ({ref_source})", value=f"{v_ref:.2f}")
    st.metric(label="Portfolio Multiplier", value=f"{mult:.4f}")

with col2:
    st.subheader("Current VIX Status")

    # Fetch and display current VIX data
    try:
        vix_historical = fetch_vix_historical_data()
        current_vix = float(vix_historical['Close'].iloc[-1])
        current_3m_vol = float(vix_historical['3M_Vol'].iloc[-1])
        current_6m_vol = float(vix_historical['6M_Vol'].iloc[-1])
        current_1y_vol = float(vix_historical['1Y_Vol'].iloc[-1])

        st.metric(label="Current VIX", value=f"{current_vix:.2f}")
        st.metric(label="3M Realized Vol", value=f"{current_3m_vol:.1f}%")
        st.metric(label="6M Realized Vol", value=f"{current_6m_vol:.1f}%")
        st.metric(label="1Y Realized Vol", value=f"{current_1y_vol:.1f}%")

    except Exception as e:
        st.error(f"Failed to fetch VIX data: {e}")
        vix_historical = None

# Volatility Analysis Section
if vix_historical is not None:
    st.markdown("---")
    st.header("📈 Volatility Analysis")

    # Statistics table
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Historical Volatility Statistics")
        vol_stats = create_volatility_stats_table(vix_historical)
        st.dataframe(vol_stats, use_container_width=True)

    with col4:
        st.subheader("Chart Controls")
        available_years = sorted(vix_historical['Year'].dropna().unique())
        selected_years = st.multiselect(
            "Select years to display:",
            options=available_years,
            default=available_years[-3:] if len(available_years) >= 3 else available_years,
            help="Choose which years to show in the volatility chart"
        )

    # Interactive chart
    st.subheader("Interactive Volatility Chart")
    if selected_years:
        vol_chart = create_volatility_chart(vix_historical, selected_years)
        st.plotly_chart(vol_chart, use_container_width=True)
    else:
        st.info("Please select at least one year to display the chart.")

st.markdown("---")
st.markdown("💡 **Note:** Multiplier = `ref / latest`. Apply to scale your portfolio exposure accordingly.")
