# -----------------------------
# Corrected Enhanced VIX Portfolio Sizer with VIX Level Analytics
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
    """Fetch and calculate VIX historical level data"""
    # Fetch VIX data for the last 5 years
    vix_data = yf.download("^VIX", period="5y", interval="1d", progress=False, auto_adjust=False)

    # Clean up multi-level columns
    clean_data = pd.DataFrame()
    clean_data['Close'] = vix_data[('Close', '^VIX')]
    clean_data['High'] = vix_data[('High', '^VIX')]
    clean_data['Low'] = vix_data[('Low', '^VIX')]
    clean_data['Open'] = vix_data[('Open', '^VIX')]
    clean_data.index = vix_data.index

    # Calculate rolling AVERAGES of VIX levels (not volatility of VIX)
    clean_data['VIX_3M_Avg'] = clean_data['Close'].rolling(window=63, min_periods=30).mean()
    clean_data['VIX_6M_Avg'] = clean_data['Close'].rolling(window=126, min_periods=60).mean()
    clean_data['VIX_1Y_Avg'] = clean_data['Close'].rolling(window=252, min_periods=120).mean()
    clean_data['VIX_5Y_Avg'] = clean_data['Close'].rolling(window=1260, min_periods=600).mean()

    clean_data['Date'] = clean_data.index

    return clean_data

def create_vix_stats_table(vix_data):
    """Create VIX level statistics table"""
    def get_vix_stats(vix_series, period_name):
        valid_data = vix_series.dropna()
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

    current_vix = vix_data['Close'].iloc[-1]
    stats_list = []
    stats_list.append(get_vix_stats(pd.Series([current_vix]), 'Current VIX'))
    stats_list.append(get_vix_stats(vix_data['VIX_3M_Avg'], '3M Average'))
    stats_list.append(get_vix_stats(vix_data['VIX_6M_Avg'], '6M Average'))
    stats_list.append(get_vix_stats(vix_data['VIX_1Y_Avg'], '1Y Average'))
    stats_list.append(get_vix_stats(vix_data['VIX_5Y_Avg'], '5Y Average'))

    return pd.DataFrame(stats_list).round(2)

def create_vix_chart(vix_data):
    """Create comprehensive VIX time series chart"""

    fig = go.Figure()

    # Add VIX close (daily) - most prominent
    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['Close'],
        mode='lines',
        name='VIX Daily Close',
        line=dict(color='black', width=1.5),
        opacity=0.7
    ))

    # Add rolling averages
    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['VIX_3M_Avg'],
        mode='lines',
        name='3-Month Average',
        line=dict(color='blue', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['VIX_6M_Avg'],
        mode='lines',
        name='6-Month Average',
        line=dict(color='green', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['VIX_1Y_Avg'],
        mode='lines',
        name='1-Year Average',
        line=dict(color='red', width=2)
    ))

    fig.add_trace(go.Scatter(
        x=vix_data['Date'],
        y=vix_data['VIX_5Y_Avg'],
        mode='lines',
        name='5-Year Average',
        line=dict(color='purple', width=2)
    ))

    # Add current VIX level as reference
    current_vix = float(vix_data['Close'].iloc[-1])
    fig.add_hline(
        y=current_vix,
        line_dash="dash",
        line_color="orange",
        annotation_text=f"Current VIX: {current_vix:.2f}",
        annotation_position="top right"
    )

    fig.update_layout(
        title="VIX Levels Over Time - Daily Close vs Rolling Averages",
        xaxis_title="Date",
        yaxis_title="VIX Level",
        hovermode='x unified',
        height=600,
        showlegend=True,
        legend=dict(
            yanchor="top",
            y=0.99,
            xanchor="left",
            x=0.01
        )
    )

    return fig

# -----------------------------
# Streamlit UI
# -----------------------------

st.set_page_config(page_title="Corrected VIX Portfolio Sizer", layout="wide")
st.title("📊 VIX Portfolio Sizer with VIX Level Analytics")

st.markdown("""
This dashboard provides VIX portfolio sizing with comprehensive VIX level analytics including:
- Historical VIX level statistics across multiple timeframes (not volatilities)
- Complete time series chart showing VIX trends
- Portfolio multiplier calculations based on VIX3M
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
        current_3m_avg = float(vix_historical['VIX_3M_Avg'].iloc[-1])
        current_6m_avg = float(vix_historical['VIX_6M_Avg'].iloc[-1])
        current_1y_avg = float(vix_historical['VIX_1Y_Avg'].iloc[-1])

        st.metric(label="Current VIX", value=f"{current_vix:.2f}")
        st.metric(label="3M Average VIX", value=f"{current_3m_avg:.2f}")
        st.metric(label="6M Average VIX", value=f"{current_6m_avg:.2f}")
        st.metric(label="1Y Average VIX", value=f"{current_1y_avg:.2f}")

    except Exception as e:
        st.error(f"Failed to fetch VIX data: {e}")
        vix_historical = None

# VIX Analysis Section
if vix_historical is not None:
    st.markdown("---")
    st.header("📈 VIX Level Analysis")

    # Statistics table
    col3, col4 = st.columns([1, 1])

    with col3:
        st.subheader("Historical VIX Level Statistics")
        vix_stats = create_vix_stats_table(vix_historical)
        st.dataframe(vix_stats, use_container_width=True)
        st.caption("Note: These are VIX levels (typically 10-40), not volatility percentages")

    with col4:
        st.subheader("Key Insights")
        st.write("**VIX Interpretation:**")
        st.write("• VIX < 20: Low volatility/fear")
        st.write("• VIX 20-30: Moderate volatility")  
        st.write("• VIX > 30: High volatility/fear")
        st.write("• VIX > 40: Extreme fear")

        st.write("**Current Assessment:**")
        if current_vix < 20:
            st.success(f"Current VIX ({current_vix:.1f}) indicates low market fear")
        elif current_vix < 30:
            st.warning(f"Current VIX ({current_vix:.1f}) indicates moderate concern")
        else:
            st.error(f"Current VIX ({current_vix:.1f}) indicates high market fear")

    # Complete time series chart
    st.subheader("Complete VIX Time Series")
    vix_chart = create_vix_chart(vix_historical)
    st.plotly_chart(vix_chart, use_container_width=True)

st.markdown("---")
st.markdown("💡 **Note:** Multiplier = `ref / latest`. Apply to scale your portfolio exposure accordingly.")
